"""
modules/face_restore.py – GFPGAN face restoration post-processing.

Problem:
  Wav2Lip often produces slightly blurry face regions around the mouth because
  it generates low-resolution lip patches and blends them back in. GFPGAN is a
  GAN-based face restoration model that sharpens faces and removes artifacts.

Approach:
  1. Process the lip-synced video frame-by-frame using GFPGAN.
  2. Reassemble frames back into a video with the original audio track reattached.

Note:
  GFPGAN is optional. Toggle via config.ENABLE_FACE_RESTORE.
  On free Colab (T4), processing 15 seconds at 25 FPS ≈ 375 frames ≈ ~3 minutes.
"""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional

import config
from modules.utils import get_logger, ensure_dirs, temp_path, run_ffmpeg

log = get_logger("face_restore")

GFPGAN_REPO_URL = "https://github.com/TencentARC/GFPGAN.git"
GFPGAN_WEIGHTS_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def restore_faces(
    input_video: str,
    output_path: Optional[str] = None,
    upscale: int | None = None,
) -> str:
    """
    Apply GFPGAN face restoration to every frame of input_video.

    Args:
        input_video: Path to the lip-synced video.
        output_path: Destination for the restored video.
        upscale:     Upscale factor (1 = original resolution). Overrides config.

    Returns:
        Path to the face-restored video.
    """
    if not config.ENABLE_FACE_RESTORE:
        log.info("Face restoration disabled (ENABLE_FACE_RESTORE=False). Skipping.")
        return input_video

    output_path = output_path or temp_path("face_restored.mp4")
    upscale = upscale if upscale is not None else config.GFPGAN_UPSCALE

    _ensure_gfpgan_repo()
    _ensure_gfpgan_weights()

    # GFPGAN expects an image directory, not a video.
    # Step 1: Extract frames from the lip-synced video.
    frames_dir = temp_path("frames_input")
    restored_dir = temp_path("frames_restored")
    ensure_dirs(frames_dir, restored_dir)

    log.info("Extracting frames from '%s'...", input_video)
    _extract_frames(input_video, frames_dir)

    # Step 2: Run GFPGAN on the frames directory.
    log.info("Running GFPGAN on %s...", frames_dir)
    _run_gfpgan(frames_dir, restored_dir, upscale)

    # GFPGAN outputs to a 'restored_imgs' subfolder inside output dir
    gfpgan_out_dir = os.path.join(restored_dir, "restored_imgs")

    # Step 3: Re-assemble frames into a video (no audio yet).
    log.info("Re-assembling frames into video...")
    silent_video = temp_path("restored_silent.mp4")
    _assemble_frames(gfpgan_out_dir, silent_video, input_video)

    # Step 4: Re-attach the audio from the lip-synced video.
    log.info("Re-attaching audio...")
    _attach_audio(silent_video, input_video, output_path)

    log.info("Face restoration complete → %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frames(video_path: str, frames_dir: str) -> None:
    """Extract all frames from a video as numbered PNG files."""
    run_ffmpeg([
        "-i", video_path,
        "-q:v", "2",            # High quality JPEG equivalent
        os.path.join(frames_dir, "frame_%06d.png"),
    ], log=log)


def _run_gfpgan(
    input_dir: str,
    output_dir: str,
    upscale: int = 1,
) -> None:
    """Run GFPGAN restore script as a subprocess."""
    gfpgan_dir = config.GFPGAN_DIR
    script = os.path.join(gfpgan_dir, "inference_gfpgan.py")

    cmd = [
        sys.executable, script,
        "--input", input_dir,
        "--output", output_dir,
        "--version", "1.4",
        "--upscale", str(upscale),
        "--arch", config.GFPGAN_ARCH,
        "--model_path", config.GFPGAN_CHECKPOINT,
        "--bg_upsampler", "None",   # Skip background upsampling for speed
        "--only_center_face",       # Only restore the detected face region
    ]

    log.debug("GFPGAN command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=gfpgan_dir, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"GFPGAN failed (code {result.returncode})")


def _assemble_frames(frames_dir: str, output_video: str, ref_video: str) -> None:
    """
    Re-assemble frames into a video, matching the FPS of the reference video.
    Assumes frames are named in sorted alphabetical order (e.g. frame_000001.png).
    """
    import subprocess as sp

    # Get source FPS
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        ref_video,
    ]
    result = sp.run(probe_cmd, capture_output=True, text=True)
    fps_raw = result.stdout.strip()  # e.g. "25/1" or "30000/1001"
    fps = eval(fps_raw) if "/" in fps_raw else float(fps_raw)
    fps = fps or config.OUTPUT_FPS

    run_ffmpeg([
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(frames_dir, "*.png"),
        "-c:v", config.OUTPUT_VIDEO_CODEC,
        "-crf", str(config.OUTPUT_VIDEO_CRF),
        "-pix_fmt", "yuv420p",
        output_video,
    ], log=log)


def _attach_audio(silent_video: str, audio_source: str, output_path: str) -> None:
    """Copy audio from audio_source and mux with silent_video."""
    run_ffmpeg([
        "-i", silent_video,
        "-i", audio_source,
        "-c:v", "copy",
        "-c:a", config.OUTPUT_AUDIO_CODEC,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ], log=log)


def _ensure_gfpgan_repo() -> None:
    """Clone GFPGAN repo if not present."""
    repo_dir = config.GFPGAN_DIR
    if os.path.isdir(repo_dir) and os.path.isfile(
        os.path.join(repo_dir, "inference_gfpgan.py")
    ):
        return

    ensure_dirs(config.MODELS_DIR)
    log.info("Cloning GFPGAN into %s...", repo_dir)
    subprocess.run(["git", "clone", GFPGAN_REPO_URL, repo_dir], check=True)

    req = os.path.join(repo_dir, "requirements.txt")
    if os.path.isfile(req):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req, "-q"],
            check=True,
        )
    # Also install basicsr separately (common missing dep)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "basicsr", "facexlib", "-q"],
        check=True,
    )


def _ensure_gfpgan_weights() -> None:
    """Download GFPGAN v1.4 weights if not present."""
    checkpoint = config.GFPGAN_CHECKPOINT
    if os.path.isfile(checkpoint):
        return

    ensure_dirs(os.path.dirname(checkpoint))
    log.info("Downloading GFPGANv1.4.pth (~332 MB)...")

    import urllib.request
    urllib.request.urlretrieve(GFPGAN_WEIGHTS_URL, checkpoint)
    log.info("GFPGAN weights → %s", checkpoint)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m modules.face_restore <lipsync_video>")
        sys.exit(1)

    result = restore_faces(sys.argv[1])
    print(f"Face-restored video: {result}")
