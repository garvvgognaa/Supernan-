# 🎬 Supernan Hindi Video Dubbing Pipeline

A modular, zero-cost Python pipeline that takes a source video and produces a **Hindi-dubbed, lip-synced, voice-cloned output clip** — built entirely on open-source models.

---

## Architecture Overview

```
Source Video (MP4)
       │
       ▼
┌──────────────┐     ffmpeg
│ 1. Extract   │──────────────► segment.mp4 + segment_audio.wav
└──────────────┘
       │
       ▼
┌──────────────┐     OpenAI Whisper (local)
│ 2. Transcribe│──────────────► [{text, start, end, words}, ...]
└──────────────┘
       │
       ▼
┌──────────────┐     Helsinki-NLP/opus-mt-en-hi
│ 3. Translate │──────────────► [{..., hindi_text}, ...]
└──────────────┘
       │
       ▼
┌──────────────┐     Coqui XTTS v2 (voice cloning)
│  4. TTS      │──────────────► [tts_0000.wav, tts_0001.wav, ...]
└──────────────┘
       │
       ▼
┌──────────────┐     ffmpeg atempo filter
│ 5. AudioSync │──────────────► hindi_audio_synced.wav
└──────────────┘
       │
       ▼
┌──────────────┐     Wav2Lip + wav2lip_gan.pth
│ 6. Lip Sync  │──────────────► lipsync_out.mp4
└──────────────┘
       │
       ▼
┌──────────────┐     GFPGAN v1.4
│ 7. FaceRestore──────────────► face_restored.mp4
└──────────────┘
       │
       ▼
   final_dubbed.mp4  ✅
```

---

## Setup

### Option A: Google Colab (Recommended — Free T4 GPU)

1. Upload your source video and this repo to Colab.
2. Run setup:
   ```bash
   !bash setup_colab.sh
   ```
3. Run the pipeline:
   ```bash
   !python dub_video.py --input video.mp4 --start 15 --end 30
   ```

Alternatively, open **`notebooks/dub_pipeline_colab.ipynb`** which has all steps pre-filled.

### Option B: Local Machine (CPU or CUDA)

**Prerequisites:**
- Python 3.9–3.11
- `ffmpeg` installed and on PATH: `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux)
- CUDA-capable GPU optional (pipeline runs on CPU; slower)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/supernan-hindi-dubbing.git
cd supernan-hindi-dubbing

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python dub_video.py --input video.mp4 --start 15 --end 30
```

---

## Usage

```
python dub_video.py [OPTIONS]

Options:
  --input      -i  PATH         Source video file (required)
  --start           FLOAT        Segment start time in seconds [default: 15.0]
  --end             FLOAT        Segment end time in seconds   [default: 30.0]
  --output     -o  PATH         Output file path
  --whisper-model  STR          Whisper size: tiny/base/small/medium/large [default: small]
  --no-face-restore             Skip GFPGAN (faster, slightly lower quality)
  --full-video                  Process the entire video
  --skip-lipsync                Skip Wav2Lip (translation+TTS only)
  --keep-temp                   Do not delete temp files after completion
  --long-audio                  Use chunked transcription for audio > 10 min
```

### Examples

```bash
# Standard run (segment 0:15 – 0:30):
python dub_video.py --input training.mp4

# Custom segment (0:45 – 1:00), medium Whisper:
python dub_video.py --input training.mp4 --start 45 --end 60 --whisper-model medium

# Quick test – skip lip sync:
python dub_video.py --input training.mp4 --skip-lipsync --no-face-restore

# Process full video (production mode):
python dub_video.py --input training.mp4 --full-video --whisper-model large
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `openai-whisper` | ≥20231117 | English ASR / transcription |
| `transformers` | ≥4.38 | Helsinki-NLP translation model |
| `TTS` (Coqui) | ≥0.22 | XTTS v2 voice cloning |
| `torch` | ≥2.0 | GPU inference backend |
| `librosa` | ≥0.10 | Audio analysis & chunking |
| `soundfile` | ≥0.12 | WAV I/O |
| `opencv-python-headless` | ≥4.8 | Frame extraction |
| `basicsr`, `facexlib` | latest | GFPGAN face restoration |
| `ffmpeg` (binary) | ≥6.0 | Video/audio muxing & tempo |

Wav2Lip and GFPGAN repos are cloned automatically on first run.

---

## Cost Analysis

### Free Tier (Google Colab T4 / Kaggle P100)

| Stage | Time for 15s clip | Cost |
|---|---|---|
| Extract (ffmpeg) | ~2s | ₹0 |
| Transcribe (Whisper small) | ~8s | ₹0 |
| Translate (Helsinki CPU) | ~3s | ₹0 |
| TTS (XTTS v2, GPU) | ~20s | ₹0 |
| Audio sync (ffmpeg) | ~2s | ₹0 |
| Lip sync (Wav2Lip GAN) | ~90s | ₹0 |
| Face restore (GFPGAN) | ~60s | ₹0 |
| **Total** | **~3 min** | **₹0** |

### Estimated Cost Per Minute of Video (at scale)

| Infrastructure | GPU | Cost/hr | Throughput | **Cost/min video** |
|---|---|---|---|---|
| Colab Free | T4 | ₹0 | ~5 min/hr | ₹0 |
| Vast.ai spot | A100 | ~₹100/hr | ~60 min/hr | ~₹1.67 |
| AWS p3.2xlarge | V100 | ~₹220/hr | ~60 min/hr | ~₹3.67 |
| RunPod community | A40 | ~₹80/hr | ~60 min/hr | ~₹1.33 |

**Bottom line:** This pipeline can produce 1 minute of dubbed video for **≈ ₹1–3 on budget cloud GPUs**, and ₹0 on Colab free tier.

---

## Scaling Architecture (500 Hours Overnight)

Given a budget and GPU fleet, the pipeline can be parallelized as follows:

```
500 hours of video
        │
        ▼
  ┌───────────────────┐
  │  Chunking Service │  Split all videos into N-second segments
  └────────┬──────────┘
           │ Job queue (Redis + Celery / Ray Datasets)
    ┌──────┴──────────────────────────┐
    │                                 │
  Worker 1                        Worker N
  (GPU A100)                      (GPU A100)
  stages 1–7                      stages 1–7
    │                                 │
    └──────────────┬──────────────────┘
                   │
           ┌───────▼────────┐
           │  Merge Service │  Stitch segments, upload to S3
           └────────────────┘
```

**Engineering changes for scale:**
1. Replace `subprocess` calls with async task queues (Celery workers per GPU)
2. Use `librosa.effects.split` for smarter silence-based chunking to avoid mid-word cuts
3. Model caching: load Whisper/XTTS/Wav2Lip once per worker, process N chunks
4. Checkpointing: save state after each stage so crashed jobs resume without reprocessing
5. Use `ffmpeg` segment concat (not re-encode) for final merge to save time

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| XTTS v2 requires 3s clean audio reference | Poor voice cloning with noisy input | Denoise with `noisereduce` first |
| Wav2Lip blurs mouth region | Visible quality drop | GFPGAN post-processing (already included) |
| Helsinki-NLP is literal; misses idioms | Unnatural Hindi in some phrases | Use IndicTrans2 for better contextual translation |
| `atempo` distorts voice at extreme ratios | Robotic audio if TTS is 4× longer | Cap ratio at 2.0×; add pause handling |
| XTTS v2 ~1.8 GB download on first run | Slow start on Colab | Pre-cache weights in Google Drive |
| Wav2Lip requires frontal face | Fails on profile/obscured faces | VideoReTalking handles more poses (swap in `lipsync.py`) |

---

## What I'd Improve With More Time

1. **IndicTrans2 as default** — 30% better Hindi quality than Helsinki-NLP, especially for domain-specific content
2. **VideoReTalking instead of Wav2Lip** — Handles non-frontal faces, produces sharper results
3. **Denoising pre-processing** — `noisereduce` or `deepfilternet` before Whisper for better transcription on noisy recordings
4. **Prosody alignment** — Use forced alignment (Montreal Forced Aligner) to synchronize translated text phoneme-by-phoneme
5. **Speaker diarisation** — Handle multiple speakers with `pyannote.audio`
6. **Real-time progress UI** — FastAPI backend + React frontend for drag-and-drop video upload
7. **GPU-aware batching** — Dynamic batch sizes based on available VRAM (query `nvidia-smi` at startup)
8. **Background music separation** — `demucs` to extract and re-mix background audio into dubbed output

---

## Project Structure

```
├── dub_video.py           # Main orchestrator (CLI entry point)
├── config.py              # All tuneable parameters in one place
├── requirements.txt       # Python dependencies
├── setup_colab.sh         # One-click Colab environment setup
├── .gitignore
├── README.md
├── modules/
│   ├── __init__.py
│   ├── extractor.py       # ffmpeg: clip extraction + speaker ref
│   ├── transcriber.py     # Whisper: English ASR + chunking
│   ├── translator.py      # Helsinki-NLP / IndicTrans2: en→hi
│   ├── tts.py             # Coqui XTTS v2: Hindi voice cloning
│   ├── audio_sync.py      # ffmpeg atempo: duration matching
│   ├── lipsync.py         # Wav2Lip: lip-sync inference
│   ├── face_restore.py    # GFPGAN: face sharpening
│   └── utils.py           # Logging, paths, batching, ffmpeg wrappers
├── models/                # Auto-downloaded model weights (gitignored)
│   ├── Wav2Lip/
│   └── GFPGAN/
├── output/                # Final output videos (gitignored)
├── temp/                  # Intermediate files, auto-cleaned (gitignored)
└── notebooks/
    └── dub_pipeline_colab.ipynb
```

---

## License

MIT — use freely, attribution appreciated.
