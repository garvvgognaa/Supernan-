"""
modules/tts.py – Hindi TTS with voice cloning using Coqui XTTS v2.

Responsibilities:
  - Load XTTS v2 model (lazy, once per process).
  - Generate per-segment Hindi speech that clones the original speaker's voice.
  - Return a list of per-segment WAV file paths.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import config
from modules.utils import get_logger, temp_path, ensure_dirs, resolve_device

log = get_logger("tts")

_xtts_model = None   # lazy-loaded


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_hindi(
    segments: List[Dict[str, Any]],
    speaker_ref_wav: str,
    language: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Synthesize Hindi speech for each segment using Coqui XTTS v2 voice cloning.

    For each segment, the model clones the speaker's voice from `speaker_ref_wav`
    and generates Hindi speech for `segment['hindi_text']`.

    Args:
        segments:        Translated segments (must have 'hindi_text', 'start', 'end').
        speaker_ref_wav: Path to 3–10s reference WAV of the target speaker.
        language:        TTS language code. Defaults to config.XTTS_LANGUAGE ('hi').

    Returns:
        Same segments list with added 'tts_audio' key (path to per-segment WAV).
    """
    global _xtts_model

    language = language or config.XTTS_LANGUAGE
    device = resolve_device(config.XTTS_DEVICE)

    if _xtts_model is None:
        _xtts_model = _load_xtts(device)

    ensure_dirs(config.TEMP_DIR)

    for i, seg in enumerate(segments):
        hindi = seg.get("hindi_text", "").strip()
        if not hindi:
            log.warning("Segment %d has no hindi_text, skipping TTS.", i)
            seg["tts_audio"] = None
            continue

        out_path = temp_path(f"tts_{i:04d}.wav")
        log.info(
            "Synthesizing segment %d/%d: '%s' → %s",
            i + 1, len(segments), hindi[:60], out_path,
        )

        _synthesize_segment(
            model=_xtts_model,
            text=hindi,
            speaker_wav=speaker_ref_wav,
            language=language,
            output_path=out_path,
        )
        seg["tts_audio"] = out_path

    log.info("TTS complete. %d files generated.", len(segments))
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_xtts(device: str):
    """
    Load Coqui XTTS v2 and return the model object.
    Downloads weights on first call (~1.8 GB).
    """
    try:
        from TTS.api import TTS
    except ImportError as e:
        raise ImportError(
            "Coqui TTS not installed. Run: pip install TTS"
        ) from e

    log.info("Loading XTTS v2 model on device='%s' (first run downloads ~1.8 GB)…", device)
    tts = TTS(
        model_name=config.XTTS_MODEL_NAME,
        progress_bar=True,
        gpu=(device == "cuda"),
    )
    log.info("XTTS v2 model loaded.")
    return tts


def _synthesize_segment(
    model,
    text: str,
    speaker_wav: str,
    language: str,
    output_path: str,
) -> None:
    """
    Run XTTS v2 inference for a single text segment.

    XTTS v2 accepts a reference WAV and clones prosody + timbre from it.
    """
    model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m modules.tts <speaker_ref_wav>")
        sys.exit(1)

    ref = sys.argv[1]
    test_segments = [
        {
            "text": "Hello world.",
            "hindi_text": "नमस्ते दुनिया।",
            "start": 0.0,
            "end": 2.0,
        },
        {
            "text": "Welcome to Supernan training.",
            "hindi_text": "सुपरनैन प्रशिक्षण में आपका स्वागत है।",
            "start": 2.0,
            "end": 5.0,
        },
    ]
    result = synthesize_hindi(test_segments, speaker_ref_wav=ref)
    for seg in result:
        print(f"HI: {seg['hindi_text']}")
        print(f"File: {seg.get('tts_audio')}")
        print()
