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


_SENTENCE_ENDINGS = (".", "!", "?", "\n")


def synthesize_sentence_stream(text_chunks):
    """
    Consume an iterator of text deltas (as they stream in from the LLM) and
    yield (audio, sample_rate) per completed sentence, as soon as each
    sentence is complete, instead of waiting for the full reply. This is what
    lets playback of sentence 1 start while the LLM is still generating
    sentence 3.
    """
    buffer = ""
    for delta in text_chunks:
        buffer += delta
        while True:
            cut = -1
            for ending in _SENTENCE_ENDINGS:
                idx = buffer.find(ending)
                if idx != -1 and (cut == -1 or idx < cut):
                    cut = idx
            if cut == -1:
                break
            sentence, buffer = buffer[:cut + 1].strip(), buffer[cut + 1:]
            if sentence:
                yield synthesize(sentence)
    remainder = buffer.strip()
    if remainder:
        yield synthesize(remainder)
