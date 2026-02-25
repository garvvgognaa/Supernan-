"""
modules/audio_sync.py – Duration matching and time-stretching of TTS audio.

Problem:
  TTS-generated speech for a translation is often shorter or longer than the
  original English audio segment. For lip sync to work well, each Hindi TTS
  clip must match the duration of the corresponding original segment exactly.

Solution:
  Use ffmpeg's 'atempo' audio filter to stretch or compress each TTS clip to
  the target duration, then concatenate all clips into a single audio track.

  atempo supports ratios between 0.5× and 2.0×. For values outside this range,
  we chain multiple atempo filters (e.g. 0.25× = atempo=0.5,atempo=0.5).
"""

from __future__ import annotations

import os
import math
from typing import List, Dict, Any

import config
from modules.utils import (
    get_logger, temp_path, ensure_dirs,
    run_ffmpeg, get_audio_duration,
)

log = get_logger("audio_sync")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sync_audio_to_video(
    segments: List[Dict[str, Any]],
    total_duration_sec: float,
    output_path: str | None = None,
) -> str:
    """
    Time-stretch each per-segment TTS audio to match its original timing,
    then concatenate into one complete audio track ready for lip sync.

    Args:
        segments:          Segments with 'tts_audio', 'start', 'end' keys.
        total_duration_sec: Total duration of the source video segment.
        output_path:       Output WAV path. Defaults to temp/hindi_audio_synced.wav.

    Returns:
        Path to the concatenated, time-synced Hindi audio WAV.
    """
    output_path = output_path or temp_path("hindi_audio_synced.wav")
    ensure_dirs(config.TEMP_DIR)

    stretched_paths: List[str] = []

    for i, seg in enumerate(segments):
        tts_audio = seg.get("tts_audio")
        if not tts_audio or not os.path.isfile(tts_audio):
            log.warning("Segment %d has no TTS audio, inserting silence.", i)
            silence_path = _generate_silence(
                duration_sec=max(0.1, seg["end"] - seg["start"]),
                output_path=temp_path(f"silence_{i:04d}.wav"),
            )
            stretched_paths.append(silence_path)
            continue

        original_duration = seg["end"] - seg["start"]
        tts_duration = get_audio_duration(tts_audio)

        if tts_duration < 0.05:
            log.warning("Segment %d TTS output is near-empty, skipping.", i)
            continue

        ratio = original_duration / tts_duration

        log.info(
            "Segment %d: orig=%.3fs tts=%.3fs → tempo ratio %.3f",
            i, original_duration, tts_duration, ratio,
        )

        stretched_path = temp_path(f"stretched_{i:04d}.wav")
        _apply_tempo(tts_audio, ratio, stretched_path)
        stretched_paths.append(stretched_path)

    if not stretched_paths:
        raise RuntimeError("No TTS audio segments to concatenate.")

    merged = _concatenate_wavs(stretched_paths, output_path)
    _pad_or_trim(merged, total_duration_sec, output_path)

    log.info("Synced Hindi audio → %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Tempo adjustment
# ─────────────────────────────────────────────────────────────────────────────

def _apply_tempo(
    input_wav: str,
    ratio: float,
    output_wav: str,
) -> None:
    """
    Apply an atempo filter chain to scale the audio duration by `ratio`.

    ratio < 1.0 → slow down (stretch)   e.g. 0.5 → 2× longer
    ratio > 1.0 → speed up (compress)   e.g. 2.0 → 2× shorter

    ffmpeg atempo is constrained to [0.5, 2.0]. We chain filters for ratios
    outside this range.
    """
    # Clamp to avoid extreme distortion
    ratio = max(0.25, min(ratio, 4.0))

    # Build atempo chain
    atempo_filters = _build_atempo_chain(ratio)
    filter_str = ",".join(atempo_filters)

    run_ffmpeg([
        "-i", input_wav,
        "-filter:a", filter_str,
        "-acodec", "pcm_s16le",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", "1",
        output_wav,
    ], log=log)


def _build_atempo_chain(ratio: float) -> List[str]:
    """
    Decompose a tempo ratio into a chain of atempo filters,
    each in the valid [0.5, 2.0] range.

    Example:
        ratio=4.0 → ["atempo=2.0", "atempo=2.0"]
        ratio=0.25 → ["atempo=0.5", "atempo=0.5"]
        ratio=1.5 → ["atempo=1.5"]
    """
    filters = []
    remaining = ratio

    if ratio >= 1.0:
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        if remaining != 1.0:
            filters.append(f"atempo={remaining:.6f}")
    else:  # ratio < 1.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        if remaining != 1.0:
            filters.append(f"atempo={remaining:.6f}")

    return filters or ["atempo=1.0"]


# ─────────────────────────────────────────────────────────────────────────────
# Silence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_silence(duration_sec: float, output_path: str) -> str:
    """Generate a WAV file of silence with the specified duration."""
    run_ffmpeg([
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=mono:sample_rate={config.AUDIO_SAMPLE_RATE}",
        "-t", str(duration_sec),
        "-acodec", "pcm_s16le",
        output_path,
    ], log=log)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Concatenation
# ─────────────────────────────────────────────────────────────────────────────

def _concatenate_wavs(wav_paths: List[str], output_path: str) -> str:
    """
    Concatenate multiple WAV files into one using ffmpeg concat demuxer.
    All inputs must have identical sample rate and channel layout.
    """
    # Write a concat list file
    list_file = temp_path("concat_list.txt")
    with open(list_file, "w") as f:
        for p in wav_paths:
            f.write(f"file '{p}'\n")

    run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-acodec", "pcm_s16le",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", "1",
        output_path,
    ], log=log)

    return output_path


def _pad_or_trim(audio_path: str, target_sec: float, output_path: str) -> None:
    """
    Ensure the final audio track is exactly target_sec long:
    - Pad with silence if audio is shorter.
    - Trim if audio is longer.

    Works in-place (overwrites output_path).
    """
    actual = get_audio_duration(audio_path)
    log.debug("pad_or_trim: actual=%.3fs target=%.3fs", actual, target_sec)

    if abs(actual - target_sec) < 0.05:
        return  # Close enough, no adjustment needed

    # Use apad + atrim to force exact duration
    run_ffmpeg([
        "-i", audio_path,
        "-filter_complex",
        f"apad=pad_dur={target_sec},atrim=end={target_sec}",
        "-acodec", "pcm_s16le",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", "1",
        output_path + ".padded.wav",
    ], log=log)

    os.replace(output_path + ".padded.wav", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Testing _build_atempo_chain:")
    for r in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        chain = _build_atempo_chain(r)
        print(f"  ratio={r:.2f} → {chain}")

    print("\nGenerate 2s silence test:")
    out = _generate_silence(2.0, temp_path("test_silence.wav"))
    dur = get_audio_duration(out)
    print(f"  Generated {dur:.3f}s at {out}")
