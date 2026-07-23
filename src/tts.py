from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

from config import KOKORO_VOICE

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_kokoro: Kokoro | None = None


def get_kokoro() -> Kokoro:
    global _kokoro
    if _kokoro is None:
        _kokoro = Kokoro(
            str(MODELS_DIR / "kokoro-v1.0.int8.onnx"),
            str(MODELS_DIR / "voices-v1.0.bin"),
        )
    return _kokoro


def synthesize(text: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = get_kokoro().create(text, voice=KOKORO_VOICE)
    return audio, sample_rate
