"""modules/__init__.py"""
from modules.extractor import extract_segment, extract_speaker_ref
from modules.transcriber import transcribe_audio
from modules.translator import translate_segments
from modules.tts import synthesize_hindi
from modules.audio_sync import sync_audio_to_video
from modules.lipsync import run_lipsync
from modules.face_restore import restore_faces
from modules.utils import get_logger, ensure_dirs

__all__ = [
    "extract_segment",
    "extract_speaker_ref",
    "transcribe_audio",
    "translate_segments",
    "synthesize_hindi",
    "sync_audio_to_video",
    "run_lipsync",
    "restore_faces",
    "get_logger",
    "ensure_dirs",
]
