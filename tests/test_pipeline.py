r"""
Integration smoke test for the voice pipeline.

Requires Ollama running locally with the model pulled (see scripts/setup.ps1).
Run with: ..\.venv\Scripts\python.exe -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import audio_io
import stt
import llm
import tts

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_stt_transcribes_known_sample():
    audio = audio_io.load_wav(str(SAMPLES_DIR / "sample_question_capital_of_france.wav"))
    text = stt.transcribe(audio)
    assert "france" in text.lower() or "capital" in text.lower()


def test_llm_answers_the_transcribed_question():
    reply = llm.reply("What is the capital of France?")
    assert "paris" in reply.lower()


def test_tts_produces_nonempty_audio():
    audio, sample_rate = tts.synthesize("This is a test.")
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert sample_rate > 0
