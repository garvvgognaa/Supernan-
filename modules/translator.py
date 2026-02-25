"""
modules/translator.py – Context-aware English → Hindi translation.

Primary model:   Helsinki-NLP/opus-mt-en-hi  (runs on CPU, no API key needed)
Fallback model:  IndicTrans2 (ai4bharat)     (better quality, requires GPU)

Responsibilities:
  - Load the translation model once (lazy) and cache it.
  - Translate a list of segment dicts, preserving timing metadata.
  - Process in configurable batches to avoid OOM on long videos.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

import config
from modules.utils import get_logger, batch_list

log = get_logger("translator")

# Lazy global model cache — loaded once per process
_pipeline = None
_indic_model = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def translate_segments(
    segments: List[Dict[str, Any]],
    batch_size: int | None = None,
    use_indic: bool | None = None,
) -> List[Dict[str, Any]]:
    """
    Translate a list of transcription segments from English to Hindi.

    Input segments format:
        [{"text": "Hello", "start": 0.0, "end": 1.5, ...}, ...]

    Output: Same list with an added 'hindi_text' key in each segment.

    Args:
        segments:    List of segment dicts from transcriber.py.
        batch_size:  Override config.TRANSLATION_BATCH_SIZE.
        use_indic:   Override config.USE_INDIC_TRANS2.

    Returns:
        Segments with 'hindi_text' populated.
    """
    batch_size = batch_size or config.TRANSLATION_BATCH_SIZE
    use_indic = use_indic if use_indic is not None else config.USE_INDIC_TRANS2

    texts = [seg["text"] for seg in segments]
    log.info(
        "Translating %d segments en→hi using %s (batch_size=%d)...",
        len(texts),
        "IndicTrans2" if use_indic else "Helsinki-NLP/opus-mt-en-hi",
        batch_size,
    )

    if use_indic:
        translated = _translate_indictrans2(texts)
    else:
        translated = _translate_helsinki(texts, batch_size)

    for seg, hi_text in zip(segments, translated):
        seg["hindi_text"] = hi_text
        log.debug("[%.2f→%.2f] EN: %s | HI: %s",
                  seg["start"], seg["end"], seg["text"], hi_text)

    log.info("Translation complete.")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Helsinki-NLP backend
# ─────────────────────────────────────────────────────────────────────────────

def _translate_helsinki(texts: List[str], batch_size: int) -> List[str]:
    """Translate using Helsinki-NLP/opus-mt-en-hi via HuggingFace Transformers."""
    global _pipeline

    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        log.info("Loading Helsinki-NLP/opus-mt-en-hi...")
        _pipeline = hf_pipeline(
            task="translation_en_to_hi",
            model=config.TRANSLATION_MODEL,
            device=-1,     # CPU; change to 0 for CUDA
            max_length=512,
        )
        log.info("Helsinki-NLP model loaded.")

    translated: List[str] = []
    batches = batch_list(texts, batch_size)

    for i, batch in enumerate(batches):
        log.debug("Translating batch %d/%d (%d items)...", i + 1, len(batches), len(batch))
        results = _pipeline(batch)
        translated.extend(r["translation_text"] for r in results)

    return translated


# ─────────────────────────────────────────────────────────────────────────────
# IndicTrans2 backend (optional, higher quality)
# ─────────────────────────────────────────────────────────────────────────────

def _translate_indictrans2(texts: List[str]) -> List[str]:
    """
    Translate using ai4bharat/indictrans2-en-indic-1B.

    This produces significantly higher quality Hindi than Helsinki-NLP
    but requires ~4 GB GPU memory and the IndicTrans2 dependencies
    (IndicNLP, sentencepiece, etc.).

    Falls back to Helsinki-NLP if the package is not installed.
    """
    try:
        from IndicTransToolkit import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError:
        log.warning(
            "IndicTrans2 dependencies not found. Falling back to Helsinki-NLP."
        )
        return _translate_helsinki(texts, config.TRANSLATION_BATCH_SIZE)

    global _indic_model

    if _indic_model is None:
        model_name = "ai4bharat/indictrans2-en-indic-1B"
        log.info("Loading IndicTrans2 model '%s'...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        processor = IndicProcessor(inference=True)
        _indic_model = (tokenizer, model, processor)
        log.info("IndicTrans2 loaded.")

    tokenizer, model, processor = _indic_model
    import torch

    # IndicTrans2 expects source/target language codes
    src_lang = "eng_Latn"
    tgt_lang = "hin_Deva"

    preprocessed = processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        preprocessed,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=1,
            max_length=256,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    postprocessed = processor.postprocess_batch(decoded, lang=tgt_lang)
    return postprocessed


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    test_segments = [
        {"text": "Welcome to the Supernan training program.", "start": 0.0, "end": 3.0},
        {"text": "Today we will learn how to care for children.", "start": 3.0, "end": 6.5},
        {"text": "Safety is our top priority at all times.", "start": 6.5, "end": 10.0},
    ]

    result = translate_segments(test_segments)
    for seg in result:
        print(f"EN: {seg['text']}")
        print(f"HI: {seg.get('hindi_text', 'N/A')}")
        print()
