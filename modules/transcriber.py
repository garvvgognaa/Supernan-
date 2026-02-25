"""
modules/transcriber.py – Audio transcription using OpenAI Whisper (local).

Responsibilities:
  - Load a configurable Whisper model (tiny/base/small/medium/large).
  - Transcribe an audio file and return word-level timed segments.
  - Handle long audio by chunking on silence boundaries for memory efficiency.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import config
from modules.utils import get_logger, resolve_device, split_into_chunks

log = get_logger("transcriber")


# ─────────────────────────────────────────────────────────────────────────────
# Core transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: str,
    model_size: str | None = None,
    language: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file using Whisper and return sentence-level segments.

    Each segment is a dict with:
        {
            "text":  str,           # The transcribed sentence
            "start": float,         # Start time in seconds
            "end":   float,         # End time in seconds
            "words": List[dict],    # Word-level timing (if word_timestamps=True)
        }

    Args:
        audio_path:  Path to WAV/MP3 audio file.
        model_size:  Whisper model size (overrides config.WHISPER_MODEL_SIZE).
        language:    Source language code (overrides config.WHISPER_LANGUAGE).

    Returns:
        List of segment dicts, ordered by start time.
    """
    import whisper

    model_size = model_size or config.WHISPER_MODEL_SIZE
    language = language or config.WHISPER_LANGUAGE
    device = resolve_device(config.WHISPER_DEVICE)

    log.info(
        "Loading Whisper model '%s' on device '%s'", model_size, device
    )
    model = whisper.load_model(model_size, device=device)

    log.info("Transcribing '%s' (language=%s)...", audio_path, language)
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=config.WHISPER_WORD_TIMESTAMPS,
        verbose=False,
    )

    segments = _normalise_segments(result.get("segments", []))
    log.info("Transcription complete: %d segments", len(segments))

    if log.isEnabledFor(10):  # DEBUG
        for s in segments:
            log.debug("[%.2f → %.2f] %s", s["start"], s["end"], s["text"])

    return segments


def transcribe_long_audio(
    audio_path: str,
    chunk_duration_sec: float = 60.0,
    model_size: str | None = None,
    language: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Transcribe very long audio files by first splitting on silence boundaries,
    then transcribing each chunk separately and re-assembling with corrected
    absolute timestamps.

    This allows processing of 500+ hour audio on limited VRAM by keeping model
    memory usage constant regardless of total file length.

    Args:
        audio_path:        Path to the audio file.
        chunk_duration_sec: Target max duration of each chunk in seconds.
        model_size:        Whisper model size.
        language:          Source language code.

    Returns:
        Merged list of segments with corrected absolute timestamps.
    """
    import whisper
    import librosa
    import soundfile as sf
    import numpy as np
    from modules.utils import temp_path, ensure_dirs

    model_size = model_size or config.WHISPER_MODEL_SIZE
    language = language or config.WHISPER_LANGUAGE
    device = resolve_device(config.WHISPER_DEVICE)

    log.info("Loading audio for chunking: '%s'", audio_path)
    audio_array, sr = librosa.load(audio_path, sr=None, mono=True)
    total_duration = len(audio_array) / sr

    log.info("Total audio duration: %.2fs. Splitting into %.0fs chunks...",
             total_duration, chunk_duration_sec)

    # Split on silence
    chunks = _split_audio_on_silence(audio_array, sr, chunk_duration_sec)
    log.info("Created %d audio chunks", len(chunks))

    # Load model once
    model = whisper.load_model(model_size, device=device)

    ensure_dirs(config.TEMP_DIR)
    all_segments: List[Dict[str, Any]] = []
    offset = 0.0

    for i, (chunk_array, chunk_start) in enumerate(chunks):
        chunk_path = temp_path(f"chunk_{i:04d}.wav")
        sf.write(chunk_path, chunk_array, sr)

        log.info("Transcribing chunk %d/%d (offset=%.2fs)...",
                 i + 1, len(chunks), chunk_start)

        result = model.transcribe(
            chunk_path,
            language=language,
            word_timestamps=config.WHISPER_WORD_TIMESTAMPS,
            verbose=False,
        )

        for seg in _normalise_segments(result.get("segments", [])):
            seg["start"] += chunk_start
            seg["end"] += chunk_start
            if seg.get("words"):
                for w in seg["words"]:
                    w["start"] += chunk_start
                    w["end"] += chunk_start
            all_segments.append(seg)

        os.remove(chunk_path)

    log.info("Long-audio transcription complete: %d segments", len(all_segments))
    return sorted(all_segments, key=lambda s: s["start"])


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_segments(raw_segments: list) -> List[Dict[str, Any]]:
    """Convert Whisper's raw segment dicts into our canonical format."""
    out = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        out.append({
            "text":  text,
            "start": float(seg.get("start", 0)),
            "end":   float(seg.get("end", 0)),
            "words": seg.get("words", []),
        })
    return out


def _split_audio_on_silence(
    audio: "np.ndarray",
    sr: int,
    target_chunk_sec: float,
) -> List[tuple]:
    """
    Split a numpy audio array into (chunk_array, chunk_start_seconds) tuples
    by finding silence boundaries near each target_chunk_sec interval.

    Falls back to hard chunking if no silence is found.
    """
    import numpy as np
    import librosa

    chunk_samples = int(target_chunk_sec * sr)
    total_samples = len(audio)
    chunks = []
    pos = 0

    # Compute RMS energy in 50ms frames for silence detection
    frame_length = int(0.05 * sr)
    hop_length = frame_length // 2
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    silence_thresh = 10 ** (config.SILENCE_THRESHOLD_DB / 20)

    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)

        # Search for a silence point in the last 10s of the chunk
        search_start = max(pos, end - int(10 * sr))
        search_frame = int(search_start / hop_length)
        end_frame = int(end / hop_length)

        silent_frames = np.where(rms[search_frame:end_frame] < silence_thresh)[0]

        if len(silent_frames) > 0:
            # Jump to the last silent frame before `end`
            split_frame = search_frame + silent_frames[-1]
            split_sample = split_frame * hop_length
            end = min(split_sample, total_samples)

        chunk_start_sec = pos / sr
        chunks.append((audio[pos:end], chunk_start_sec))
        pos = end

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m modules.transcriber <audio_path> [model_size]")
        sys.exit(1)

    audio = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "base"

    segs = transcribe_audio(audio, model_size=size)
    print(json.dumps(segs, indent=2, ensure_ascii=False))
