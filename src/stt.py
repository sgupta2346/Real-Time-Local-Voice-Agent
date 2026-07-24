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


NO_SPEECH_PROB_THRESHOLD = 0.6


def transcribe(audio: np.ndarray) -> str:
    """
    Whisper (like most ASR models) hallucinates plausible-sounding stock
    phrases ("thank you", "thanks for watching") on silence or near-silence,
    because that's genuinely what's in a lot of its training data at the
    quiet parts of a clip. This matters more here than in a batch-transcribe
    setting, VAD occasionally passes through a short low-quality clip (echo
    residue, background noise) that isn't real speech, and every one of
    those was showing up as a confidently wrong transcript instead of an
    empty one. faster-whisper exposes a per-segment no_speech_prob; drop any
    segment where the model itself is signaling "this probably wasn't
    speech" rather than trusting the text it produced anyway.
    """
    segments, _info = get_model().transcribe(audio, language="en", beam_size=1)
    kept = [seg.text.strip() for seg in segments if seg.no_speech_prob < NO_SPEECH_PROB_THRESHOLD]
    return " ".join(kept).strip()
