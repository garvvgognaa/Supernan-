"""
config.py – Central configuration for the Hindi Dubbing Pipeline.

All tunable parameters are defined here. Edit this file to change model sizes,
paths, timestamps, or feature flags before running dub_video.py.
"""

import os

# ─────────────────────────────────────────────
# Segment Configuration
# ─────────────────────────────────────────────
SEGMENT_START_SEC: float = 15.0   # Start of the 15-second clip (seconds)
SEGMENT_END_SEC: float = 30.0     # End of the clip (seconds)
SEGMENT_DURATION: float = SEGMENT_END_SEC - SEGMENT_START_SEC

# Duration (seconds) of the speaker reference audio used for XTTS voice cloning.
# Taken from the beginning of the clip.
SPEAKER_REF_DURATION: float = 6.0

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODELS_DIR = os.path.join(BASE_DIR, "models")
WAV2LIP_DIR = os.path.join(MODELS_DIR, "Wav2Lip")
GFPGAN_DIR = os.path.join(MODELS_DIR, "GFPGAN")

# ─────────────────────────────────────────────
# Whisper (Transcription)
# ─────────────────────────────────────────────
# Model sizes: "tiny", "base", "small", "medium", "large"
# Free Colab T4: use "small" or "medium" for good accuracy/speed tradeoff
# CPU-only fallback: use "base" or "tiny"
WHISPER_MODEL_SIZE: str = "small"
WHISPER_LANGUAGE: str = "en"            # Source language
WHISPER_DEVICE: str = "auto"            # "auto" | "cpu" | "cuda"
WHISPER_WORD_TIMESTAMPS: bool = True    # Enable word-level timing

# ─────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────
# Primary: Helsinki-NLP (runs on CPU, no internet needed after first download)
# Alternative: IndicTrans2 (better quality, needs more VRAM)
TRANSLATION_MODEL: str = "Helsinki-NLP/opus-mt-en-hi"
TRANSLATION_BATCH_SIZE: int = 8          # Sentences per batch
USE_INDIC_TRANS2: bool = False           # Switch to IndicTrans2 if available

# ─────────────────────────────────────────────
# TTS / Voice Cloning (Coqui XTTS v2)
# ─────────────────────────────────────────────
XTTS_MODEL_NAME: str = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE: str = "hi"               # Hindi
XTTS_DEVICE: str = "auto"              # "auto" | "cpu" | "cuda"
XTTS_SAMPLE_RATE: int = 24000          # XTTS v2 native sample rate

# ─────────────────────────────────────────────
# Audio Sync
# ─────────────────────────────────────────────
# Maximum tempo ratio allowed when stretching audio.
# ffmpeg atempo supports 0.5–2.0 per filter (chained for more range).
AUDIO_TEMPO_MIN: float = 0.5
AUDIO_TEMPO_MAX: float = 2.0
AUDIO_SAMPLE_RATE: int = 16000          # Working sample rate for intermediate audio
SILENCE_THRESHOLD_DB: float = -40.0    # dB below this = silence (for chunking)
MIN_SILENCE_DURATION_MS: int = 500     # Min silence length for split point

# ─────────────────────────────────────────────
# Lip Sync (Wav2Lip)
# ─────────────────────────────────────────────
WAV2LIP_CHECKPOINT: str = os.path.join(
    WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth"
)
WAV2LIP_FACE_DETECT_BATCH: int = 16    # Batch size for face detection
WAV2LIP_INFERENCE_BATCH: int = 128     # Batch size for lip-sync inference
WAV2LIP_RESIZE_FACTOR: int = 1         # Downscale factor (1 = no downscale)

# ─────────────────────────────────────────────
# Face Restoration (GFPGAN)
# ─────────────────────────────────────────────
ENABLE_FACE_RESTORE: bool = True        # Toggle GFPGAN post-processing
GFPGAN_CHECKPOINT: str = os.path.join(
    GFPGAN_DIR, "experiments", "pretrained_models", "GFPGANv1.4.pth"
)
GFPGAN_UPSCALE: int = 1                # 1 = no upscale (keeps original resolution)
GFPGAN_ARCH: str = "clean"

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
OUTPUT_VIDEO_NAME: str = "final_dubbed.mp4"
OUTPUT_VIDEO_CODEC: str = "libx264"
OUTPUT_AUDIO_CODEC: str = "aac"
OUTPUT_VIDEO_CRF: int = 18             # Lower = better quality (18 is near-lossless)
OUTPUT_FPS: int = 25                   # Output FPS (set to None to copy source FPS)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL: str = "INFO"                # "DEBUG" | "INFO" | "WARNING" | "ERROR"
LOG_FILE: str = os.path.join(BASE_DIR, "pipeline.log")
