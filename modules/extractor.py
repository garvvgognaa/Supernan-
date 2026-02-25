"""
modules/extractor.py – Video/audio segment extraction using ffmpeg.

Responsibilities:
  - Cut a specific [start, end] time window from the source video.
  - Extract the corresponding audio track as a WAV file.
  - Extract a short speaker reference clip for XTTS voice cloning.
"""

from __future__ import annotations

import os
from modules.utils import get_logger, ensure_dirs, run_ffmpeg, temp_path
import config

log = get_logger("extractor")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_segment(
    input_video: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
    output_video: str | None = None,
    output_audio: str | None = None,
) -> tuple[str, str]:
    """
    Extract a video/audio segment from input_video.

    Args:
        input_video:  Absolute path to the source video.
        start_sec:    Segment start in seconds. Defaults to config.SEGMENT_START_SEC.
        end_sec:      Segment end in seconds.   Defaults to config.SEGMENT_END_SEC.
        output_video: Destination path for the video clip.
        output_audio: Destination path for the audio WAV file.

    Returns:
        (output_video_path, output_audio_path)
    """
    start_sec = start_sec if start_sec is not None else config.SEGMENT_START_SEC
    end_sec = end_sec if end_sec is not None else config.SEGMENT_END_SEC
    duration = end_sec - start_sec

    if duration <= 0:
        raise ValueError(f"Segment duration must be > 0, got {duration}s")

    ensure_dirs(config.TEMP_DIR)

    out_vid = output_video or temp_path("segment.mp4")
    out_aud = output_audio or temp_path("segment_audio.wav")

    log.info(
        "Extracting segment %.2fs → %.2fs (%.2fs) from '%s'",
        start_sec, end_sec, duration, input_video,
    )

    # ── 1. Video clip (re-encode to ensure frame accuracy) ──────────────────
    run_ffmpeg([
        "-ss", str(start_sec),
        "-i", input_video,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-an",                  # no audio; we handle audio separately
        out_vid,
    ], log=log)
    log.info("Video clip saved → %s", out_vid)

    # ── 2. Audio WAV (mono, 16 kHz for Whisper + Wav2Lip compatibility) ─────
    run_ffmpeg([
        "-ss", str(start_sec),
        "-i", input_video,
        "-t", str(duration),
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # uncompressed PCM
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", "1",              # mono
        out_aud,
    ], log=log)
    log.info("Audio WAV saved → %s", out_aud)

    return out_vid, out_aud


def extract_speaker_ref(
    input_audio: str,
    duration_sec: float | None = None,
    output_path: str | None = None,
) -> str:
    """
    Extract a clean speaker reference audio clip for XTTS voice cloning.

    Takes the first `duration_sec` seconds of `input_audio` and saves it as a
    high-quality WAV at 22 050 Hz (XTTS v2 recommended sample rate).

    Args:
        input_audio:  Path to the extracted segment WAV.
        duration_sec: Length of the reference clip. Defaults to
                      config.SPEAKER_REF_DURATION.
        output_path:  Destination path.

    Returns:
        Path to the speaker reference WAV.
    """
    duration_sec = duration_sec or config.SPEAKER_REF_DURATION
    out = output_path or temp_path("speaker_ref.wav")

    log.info("Extracting %.1fs speaker reference from '%s'", duration_sec, input_audio)

    run_ffmpeg([
        "-i", input_audio,
        "-t", str(duration_sec),
        "-acodec", "pcm_s16le",
        "-ar", "22050",          # XTTS v2 native sample rate
        "-ac", "1",
        out,
    ], log=log)

    log.info("Speaker reference saved → %s", out)
    return out


def merge_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """
    Merge a new audio track into a silent video file, producing the final output.

    Args:
        video_path:  Path to the video (no embedded audio).
        audio_path:  Path to the new audio track.
        output_path: Destination of the combined video.

    Returns:
        output_path
    """
    log.info("Merging audio into video → %s", output_path)
    ensure_dirs(os.path.dirname(output_path))

    run_ffmpeg([
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", config.OUTPUT_AUDIO_CODEC,
        "-shortest",             # trim to the shorter stream
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path,
    ], log=log)

    log.info("Merged video saved → %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m modules.extractor <video_path> [start] [end]")
        sys.exit(1)

    video = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else config.SEGMENT_START_SEC
    end = float(sys.argv[3]) if len(sys.argv) > 3 else config.SEGMENT_END_SEC

    vid_out, aud_out = extract_segment(video, start, end)
    ref_out = extract_speaker_ref(aud_out)
    print(f"Video  : {vid_out}")
    print(f"Audio  : {aud_out}")
    print(f"SpeakerRef: {ref_out}")
