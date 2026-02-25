#!/usr/bin/env python3
"""
dub_video.py – Supernan Hindi Video Dubbing Pipeline (Main Orchestrator)
=========================================================================

Usage
-----
  # Full run with defaults from config.py:
  python dub_video.py --input path/to/video.mp4

  # Custom segment (30s → 45s):
  python dub_video.py --input video.mp4 --start 30 --end 45

  # Skip face restoration for speed:
  python dub_video.py --input video.mp4 --no-face-restore

  # Use base Whisper model (faster on CPU):
  python dub_video.py --input video.mp4 --whisper-model base

  # Process full video (no segment trimming):
  python dub_video.py --input video.mp4 --full-video

Pipeline Stages
---------------
  1. extract    – Cut the 15-second (or custom) segment using ffmpeg
  2. transcribe – Transcribe English speech with OpenAI Whisper
  3. translate  – Translate to Hindi with Helsinki-NLP/opus-mt-en-hi
  4. tts        – Synthesize Hindi speech via Coqui XTTS v2 (voice cloning)
  5. sync       – Time-stretch audio to match original lip timings
  6. lipsync    – Run Wav2Lip to create mouth movement matching Hindi audio
  7. restore    – Apply GFPGAN to fix Wav2Lip face blurriness
  8. output     – Merge everything and save final dubbed video

Architecture for Scale (500-hour batch)
----------------------------------------
  If given a budget and GPU fleet, this pipeline can scale to 500h/night by:
  1. Splitting the input into N-second chunks (configurable).
  2. Distributing chunks across GPU workers using a job queue (Celery / Ray).
  3. Each worker runs stages 1-7 independently.
  4. A merge step stitches all chunks back together.
  See README.md § "Scaling Architecture" for details.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import config
from modules.utils import get_logger, ensure_dirs, clean_temp, temp_path, output_path
from modules.extractor import extract_segment, extract_speaker_ref, merge_audio_into_video
from modules.transcriber import transcribe_audio, transcribe_long_audio
from modules.translator import translate_segments
from modules.tts import synthesize_hindi
from modules.audio_sync import sync_audio_to_video
from modules.lipsync import run_lipsync
from modules.face_restore import restore_faces

log = get_logger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supernan Hindi Dubbing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="Path to source video file (MP4 / MKV / AVI).")
    p.add_argument("--start", type=float, default=config.SEGMENT_START_SEC,
                   help="Segment start time in seconds.")
    p.add_argument("--end",   type=float, default=config.SEGMENT_END_SEC,
                   help="Segment end time in seconds.")
    p.add_argument("--output", "-o", default=None,
                   help="Output video file path. Default: output/final_dubbed.mp4")
    p.add_argument("--whisper-model", default=config.WHISPER_MODEL_SIZE,
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (larger = more accurate, slower).")
    p.add_argument("--no-face-restore", action="store_true",
                   help="Skip GFPGAN face restoration (faster, lower visual quality).")
    p.add_argument("--full-video", action="store_true",
                   help="Process the entire video (ignores --start / --end).")
    p.add_argument("--skip-lipsync", action="store_true",
                   help="Skip Wav2Lip inference (for quick translation-only testing).")
    p.add_argument("--keep-temp", action="store_true",
                   help="Do not delete temporary files after completion.")
    p.add_argument("--long-audio", action="store_true",
                   help="Use chunked transcription for very long audio (>10 min).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> str:
    """
    Execute all pipeline stages in sequence.
    Returns the path to the final output video.
    """
    t0 = time.time()

    # Apply CLI overrides to config
    config.WHISPER_MODEL_SIZE = args.whisper_model
    if args.no_face_restore:
        config.ENABLE_FACE_RESTORE = False

    # Resolve output path
    final_output = args.output or output_path(config.OUTPUT_VIDEO_NAME)

    # Compute effective segment bounds
    if args.full_video:
        start_sec, end_sec = None, None
        log.info("Processing full video.")
    else:
        start_sec = args.start
        end_sec = args.end
        log.info("Processing segment %.2fs ~ %.2fs", start_sec, end_sec)

    ensure_dirs(config.OUTPUT_DIR, config.TEMP_DIR)

    log.info("=" * 60)
    log.info(" SUPERNAN HINDI DUBBING PIPELINE")
    log.info("=" * 60)

    # ── Stage 1: Extract ────────────────────────────────────────────────────
    log.info("\n[Stage 1/7] Extracting video/audio segment...")
    if args.full_video:
        from modules.utils import get_video_duration
        total_dur = get_video_duration(args.input)
        vid_clip = temp_path("segment.mp4")
        aud_clip = temp_path("segment_audio.wav")
        # Copy original video as-is
        from modules.utils import run_ffmpeg
        run_ffmpeg(["-i", args.input, "-c:v", "libx264", "-an", vid_clip], log=log)
        run_ffmpeg(["-i", args.input, "-vn", "-acodec", "pcm_s16le",
                    "-ar", str(config.AUDIO_SAMPLE_RATE), "-ac", "1", aud_clip], log=log)
        segment_duration = total_dur
    else:
        vid_clip, aud_clip = extract_segment(args.input, start_sec, end_sec)
        segment_duration = end_sec - start_sec

    speaker_ref = extract_speaker_ref(aud_clip)
    _stage_ok(1, "extract")

    # ── Stage 2: Transcribe ─────────────────────────────────────────────────
    log.info("\n[Stage 2/7] Transcribing English audio with Whisper '%s'...",
             config.WHISPER_MODEL_SIZE)
    if args.long_audio:
        segments = transcribe_long_audio(aud_clip)
    else:
        segments = transcribe_audio(aud_clip)

    if not segments:
        log.error("Transcription returned no segments. Aborting.")
        sys.exit(1)

    log.info("Transcript preview:")
    for seg in segments[:3]:
        log.info("  [%.2f→%.2f] %s", seg["start"], seg["end"], seg["text"])
    _stage_ok(2, "transcribe")

    # ── Stage 3: Translate ──────────────────────────────────────────────────
    log.info("\n[Stage 3/7] Translating to Hindi...")
    segments = translate_segments(segments)
    log.info("Translation preview:")
    for seg in segments[:3]:
        log.info("  EN: %s", seg["text"])
        log.info("  HI: %s", seg.get("hindi_text", ""))
    _stage_ok(3, "translate")

    # ── Stage 4: TTS ────────────────────────────────────────────────────────
    log.info("\n[Stage 4/7] Synthesizing Hindi speech (XTTS v2 voice cloning)...")
    segments = synthesize_hindi(segments, speaker_ref_wav=speaker_ref)
    tts_count = sum(1 for s in segments if s.get("tts_audio"))
    log.info("TTS generated %d/%d clips.", tts_count, len(segments))
    _stage_ok(4, "tts")

    # ── Stage 5: Audio Sync ─────────────────────────────────────────────────
    log.info("\n[Stage 5/7] Syncing audio durations to match original timing...")
    synced_audio = sync_audio_to_video(segments, total_duration_sec=segment_duration)
    _stage_ok(5, "audio_sync")

    # ── Stage 6: Lip Sync ───────────────────────────────────────────────────
    if args.skip_lipsync:
        log.info("\n[Stage 6/7] Lip sync SKIPPED (--skip-lipsync).")
        lipsync_video = merge_audio_into_video(vid_clip, synced_audio, temp_path("no_lipsync.mp4"))
    else:
        log.info("\n[Stage 6/7] Running Wav2Lip lip sync...")
        lipsync_video = run_lipsync(
            face_video=vid_clip,
            audio_path=synced_audio,
        )
    _stage_ok(6, "lipsync")

    # ── Stage 7: Face Restore ───────────────────────────────────────────────
    if config.ENABLE_FACE_RESTORE and not args.skip_lipsync:
        log.info("\n[Stage 7/7] Applying GFPGAN face restoration...")
        restored_video = restore_faces(lipsync_video)
    else:
        log.info("\n[Stage 7/7] Face restoration SKIPPED.")
        restored_video = lipsync_video
    _stage_ok(7, "face_restore")

    # ── Final Output ────────────────────────────────────────────────────────
    log.info("\nCopying final output → %s", final_output)
    import shutil
    ensure_dirs(os.path.dirname(final_output))
    shutil.copy2(restored_video, final_output)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(" PIPELINE COMPLETE in %.1f seconds", elapsed)
    log.info(" Output: %s", final_output)
    log.info("=" * 60)

    if not args.keep_temp:
        log.info("Cleaning up temporary files...")
        clean_temp()

    return final_output


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stage_ok(num: int, name: str) -> None:
    log.info("✓ Stage %d (%s) complete.", num, name)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    output_file = run_pipeline(args)
    print(f"\nDone! Final video: {output_file}")
