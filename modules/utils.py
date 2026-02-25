"""
modules/utils.py – Shared utilities: logging, paths, batching, progress.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

import config


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and to pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:         # Avoid duplicate handlers on re-import
        return logger

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(config.LOG_FILE)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not already exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def temp_path(filename: str) -> str:
    """Return an absolute path inside TEMP_DIR."""
    ensure_dirs(config.TEMP_DIR)
    return os.path.join(config.TEMP_DIR, filename)


def output_path(filename: str) -> str:
    """Return an absolute path inside OUTPUT_DIR."""
    ensure_dirs(config.OUTPUT_DIR)
    return os.path.join(config.OUTPUT_DIR, filename)


def clean_temp() -> None:
    """Remove all temporary files."""
    if os.path.isdir(config.TEMP_DIR):
        shutil.rmtree(config.TEMP_DIR)
    ensure_dirs(config.TEMP_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_ffmpeg(args: List[str], log: logging.Logger | None = None) -> None:
    """
    Run ffmpeg with the given argument list.
    Always prepends '-y' (overwrite) and '-hide_banner' for clean output.
    Raises RuntimeError on non-zero exit.
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    if log:
        log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {result.returncode}):\n{result.stderr}"
        )


def get_video_duration(video_path: str) -> float:
    """Return the duration of a video file in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def get_audio_duration(audio_path: str) -> float:
    """Return the duration of an audio file in seconds."""
    return get_video_duration(audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# Batching helpers (for long-form processing)
# ─────────────────────────────────────────────────────────────────────────────

def batch_list(items: list, batch_size: int) -> List[list]:
    """
    Split a list into batches of at most batch_size.

    Example:
        batch_list([1,2,3,4,5], 2) → [[1,2], [3,4], [5]]
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def split_into_chunks(
    segments: List[dict],
    max_duration_sec: float = 60.0,
) -> List[List[dict]]:
    """
    Group word/sentence segments into time-based chunks no longer than
    max_duration_sec. Useful for batching TTS on very long videos.

    Each segment must have keys: 'text', 'start', 'end'.
    """
    chunks: List[List[dict]] = []
    current_chunk: List[dict] = []
    chunk_start = 0.0

    for seg in segments:
        seg_duration = seg["end"] - seg.get("start", 0)
        if current_chunk and (seg["end"] - chunk_start) > max_duration_sec:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_start = seg.get("start", 0)
        current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(device_setting: str) -> str:
    """
    Resolve 'auto' to 'cuda' (if available) or 'cpu'.
    Accepts 'cpu' and 'cuda' as explicit overrides.
    """
    if device_setting.lower() == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_setting.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log = get_logger("utils")
    ensure_dirs(config.OUTPUT_DIR, config.TEMP_DIR)
    log.info("Directories ensured: OUTPUT_DIR=%s, TEMP_DIR=%s",
             config.OUTPUT_DIR, config.TEMP_DIR)
    log.info("Device resolution: 'auto' → %s", resolve_device("auto"))
    log.info("batch_list test: %s", batch_list(list(range(7)), 3))
    log.info("utils.py OK")
