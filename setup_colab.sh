#!/usr/bin/env bash
# setup_colab.sh – One-click setup for Google Colab (Free T4 GPU)
# ─────────────────────────────────────────────────────────────────
# Run this cell FIRST in your Colab notebook:
#   !bash setup_colab.sh
# ─────────────────────────────────────────────────────────────────

set -e   # Exit on any error

echo "============================================================"
echo "  Supernan Hindi Dubbing Pipeline – Colab Setup"
echo "============================================================"

# ── 0. Check GPU ─────────────────────────────────────────────────
echo "[0/7] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  && echo "GPU OK" || echo "WARNING: No GPU detected, running on CPU (slow)"

# ── 1. Install system packages ────────────────────────────────────
echo "[1/7] Installing ffmpeg and git..."
apt-get update -qq
apt-get install -y -qq ffmpeg git libsndfile1

# ── 2. Upgrade pip ───────────────────────────────────────────────
echo "[2/7] Upgrading pip..."
pip install --upgrade pip -q

# ── 3. Install Python requirements ───────────────────────────────
echo "[3/7] Installing Python requirements..."
pip install -r requirements.txt -q

# ── 4. Verify Whisper ────────────────────────────────────────────
echo "[4/7] Verifying Whisper..."
python -c "import whisper; print('Whisper OK:', whisper.__file__)"

# ── 5. Verify HuggingFace Transformers ───────────────────────────
echo "[5/7] Verifying Transformers..."
python -c "import transformers; print('Transformers OK:', transformers.__version__)"

# ── 6. Verify Coqui TTS ──────────────────────────────────────────
echo "[6/7] Verifying TTS..."
python -c "from TTS.api import TTS; print('Coqui TTS OK')"

# ── 7. Verify ffmpeg binary ──────────────────────────────────────
echo "[7/7] Verifying ffmpeg..."
ffmpeg -version | head -1

echo ""
echo "✅ Setup complete! Run the pipeline with:"
echo "   python dub_video.py --input your_video.mp4 --start 15 --end 30"
echo ""
