import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import audio_io
import frame_source
import llm
import stt
import tts
import vad

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _padded_speech():
    speech = audio_io.load_wav(str(SAMPLES_DIR / "sample_question_apollo11.wav"))
    silence = np.zeros(int(0.8 * 16000), dtype=np.float32)
    return np.concatenate([silence, speech, silence])


def test_vad_strips_silence_and_keeps_speech_intact():
    padded = _padded_speech()
    frames = frame_source.iter_frames_from_array(padded, realtime=False)
    utterance = vad.record_utterance(frames)

    assert len(utterance) > 0
    # should be shorter than the padded input (silence discarded) ...
    assert len(utterance) < len(padded)
    # ... but the speech itself has to survive, or transcription would fail
    text = stt.transcribe(utterance)
    assert "apollo" in text.lower() or "moon" in text.lower()


def test_vad_returns_empty_for_pure_silence():
    silence = np.zeros(16000 * 2, dtype=np.float32)
    frames = frame_source.iter_frames_from_array(silence, realtime=False)
    utterance = vad.record_utterance(frames, max_seconds=2.0)
    assert len(utterance) == 0


def test_llm_reply_stream_yields_multiple_chunks_and_reassembles_correctly():
    chunks = list(llm.reply_stream("What is the capital of France? Answer in one short sentence."))
    assert len(chunks) > 1
    assert "paris" in "".join(chunks).lower()


def test_sentence_chunked_tts_yields_one_chunk_per_sentence():
    def fake_stream():
        yield "First sentence. "
        yield "Second sentence."

    results = list(tts.synthesize_sentence_stream(fake_stream()))
    assert len(results) == 2
    for audio, sr in results:
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr > 0
