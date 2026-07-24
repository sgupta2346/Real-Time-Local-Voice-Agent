"""
VAD-based turn-taking: replaces "record for N fixed seconds" with "record
until the user actually stops talking."

Silero VAD's VADIterator is a streaming endpointer: feed it 32ms frames, it
tells you when speech starts and when it ends (after a trailing-silence
window). That's the real turn-taking signal a fixed timer can't give you.
"""
import numpy as np
from silero_vad import VADIterator, load_silero_vad

from config import SAMPLE_RATE
from frame_source import FRAME_SIZE

_model = None


def get_vad_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


MIN_UTTERANCE_SECONDS = 0.25


def record_utterance(
    frame_source,
    min_silence_duration_ms: int = 400,
    max_seconds: float = 15.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Consume frames from `frame_source` until Silero VAD detects the end of an
    utterance (speech followed by min_silence_duration_ms of silence).
    Returns the concatenated speech audio. Frames before speech starts are
    discarded, so callers don't pay STT cost transcribing dead air.

    A genuine "start" can still fire on a brief noise blip (leftover echo
    the AEC didn't fully cancel, a cough, a chair creak), not just real
    speech. Anything shorter than MIN_UTTERANCE_SECONDS gets treated as no
    speech rather than handed to Whisper, which will happily hallucinate a
    confident, wrong transcript out of a fraction of a second of noise.
    """
    vad_iterator = VADIterator(
        get_vad_model(),
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    collected = []
    speech_started = False
    max_frames = int(max_seconds * SAMPLE_RATE / FRAME_SIZE)

    for i, frame in enumerate(frame_source):
        event = vad_iterator(frame, return_seconds=False)
        if speech_started:
            collected.append(frame)
        if event and "start" in event:
            speech_started = True
            collected.append(frame)
        if event and "end" in event and speech_started:
            break
        if i >= max_frames:
            break

    vad_iterator.reset_states()
    if not collected:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(collected)
    if len(audio) < int(MIN_UTTERANCE_SECONDS * SAMPLE_RATE):
        return np.zeros(0, dtype=np.float32)
    return audio
