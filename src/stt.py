import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _model


def transcribe(audio: np.ndarray) -> str:
    segments, _info = get_model().transcribe(audio, language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
