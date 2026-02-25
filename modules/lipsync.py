"""
modules/lipsync.py – Wav2Lip lip-sync wrapper.

Responsibilities:
  - Clone the Wav2Lip repository and download model weights if not present.
  - Run Wav2Lip inference to produce a lip-synced video.
  - Support both Wav2Lip (faster) and Wav2Lip_GAN (higher fidelity) checkpoints.

Architecture note:
  Wav2Lip is run as a subprocess because it ships its own inference script
  with hardcoded import paths. This is the standard integration approach.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import config
from modules.utils import get_logger, ensure_dirs, temp_path

log = get_logger("lipsync")

WAV2LIP_REPO_URL = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_GAN_URL = (
    "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/"
    "_layouts/15/download.aspx?share=EdjI7bZlgApMqX_-6T5GqksBAAGxIpiqeKMnBNFneFM2UA"
)
# Fallback public mirror (Hugging Face hosted)
WAV2LIP_GAN_HF = "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2Lip/wav2lip_gan.pth"
WAV2LIP_HF     = "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2Lip/wav2lip.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_lipsync(
    face_video: str,
    audio_path: str,
    output_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    use_gan: bool = True,
) -> str:
    """
    Run Wav2Lip to produce a lip-synced video.

    Args:
        face_video:   Path to video clip (face visible, no audio needed).
        audio_path:   Path to the new Hindi audio track (WAV).
        output_path:  Destination of the lip-synced video.
        checkpoint:   Path to Wav2Lip model checkpoint (.pth).
        use_gan:      Use wav2lip_gan.pth (higher fidelity) vs wav2lip.pth.

    Returns:
        Path to the lip-synced output video.
    """
    output_path = output_path or temp_path("lipsync_out.mp4")
    checkpoint = checkpoint or _get_checkpoint(use_gan)

    _ensure_wav2lip_repo()
    _ensure_checkpoint(checkpoint, use_gan)

    wav2lip_repo = config.WAV2LIP_DIR
    inference_script = os.path.join(wav2lip_repo, "inference.py")

    log.info("Running Wav2Lip inference...")
    log.info("  face    : %s", face_video)
    log.info("  audio   : %s", audio_path)
    log.info("  model   : %s", checkpoint)
    log.info("  output  : %s", output_path)

    cmd = [
        sys.executable, inference_script,
        "--checkpoint_path", checkpoint,
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_path,
        "--resize_factor", str(config.WAV2LIP_RESIZE_FACTOR),
    ]

    result = subprocess.run(
        cmd,
        cwd=wav2lip_repo,
        capture_output=False,   # Let output stream to console for progress
    )

    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip inference failed (code {result.returncode})")

    if not os.path.isfile(output_path):
        raise RuntimeError(f"Wav2Lip output not found at {output_path}")

    log.info("Lip sync complete → %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_checkpoint(use_gan: bool) -> str:
    """Return the expected checkpoint path based on use_gan flag."""
    name = "wav2lip_gan.pth" if use_gan else "wav2lip.pth"
    return os.path.join(config.WAV2LIP_DIR, "checkpoints", name)


def _ensure_wav2lip_repo() -> None:
    """Clone the Wav2Lip repository if it doesn't exist."""
    repo_dir = config.WAV2LIP_DIR
    if os.path.isdir(repo_dir) and os.path.isfile(
        os.path.join(repo_dir, "inference.py")
    ):
        log.debug("Wav2Lip repo already present at %s", repo_dir)
        return

    ensure_dirs(config.MODELS_DIR)
    log.info("Cloning Wav2Lip repository into %s...", repo_dir)
    subprocess.run(
        ["git", "clone", WAV2LIP_REPO_URL, repo_dir],
        check=True,
    )

    # Install Wav2Lip's requirements
    req_file = os.path.join(repo_dir, "requirements.txt")
    if os.path.isfile(req_file):
        log.info("Installing Wav2Lip requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "-q"],
            check=True,
        )

    log.info("Wav2Lip repository ready.")


def _ensure_checkpoint(checkpoint_path: str, use_gan: bool) -> None:
    """Download the Wav2Lip checkpoint if it's not already present."""
    if os.path.isfile(checkpoint_path):
        log.debug("Checkpoint already exists: %s", checkpoint_path)
        return

    ensure_dirs(os.path.dirname(checkpoint_path))
    url = WAV2LIP_GAN_HF if use_gan else WAV2LIP_HF
    log.info("Downloading Wav2Lip checkpoint from %s...", url)

    try:
        import urllib.request
        urllib.request.urlretrieve(url, checkpoint_path)
        log.info("Checkpoint downloaded → %s", checkpoint_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Wav2Lip checkpoint from {url}.\n"
            f"Please download it manually and place it at:\n  {checkpoint_path}\n"
            f"Error: {e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m modules.lipsync <face_video> <audio_wav>")
        sys.exit(1)

    face = sys.argv[1]
    audio = sys.argv[2]
    out = run_lipsync(face, audio)
    print(f"Lip-synced video: {out}")
