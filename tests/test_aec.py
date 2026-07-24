"""
Synthetic validation of the NLMS echo canceller: without it, a VAD can't
tell the agent's own echo apart from real user speech (both read as
speech). With it, echo should read as near-silence and real speech should
still read clearly as speech. No live mic involved, the echo path and the
"user interrupting" event are both synthesized.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import frame_source
import nlms_aec
import vad as vad_module

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SR = 16000
FRAME = 512


def _load_16k(path):
    audio, sr = sf.read(str(path))
    idx = np.arange(0, len(audio), sr / SR).astype(np.int64)
    idx = idx[idx < len(audio)]
    return audio[idx].astype(np.float64)


def _build_scenario():
    ref = np.tile(_load_16k(SAMPLES_DIR / "sample_question_apollo11.wav"), 3)

    rir = np.zeros(400)
    rir[0], rir[50], rir[120], rir[300] = 0.6, 0.25, 0.1, 0.05
    echo = np.convolve(ref, rir)[: len(ref)]

    near = _load_16k(SAMPLES_DIR / "sample_question_capital_of_france.wav") * 0.8
    near_start = len(echo) - len(near) - 8000
    near_full = np.zeros_like(echo)
    near_full[near_start:near_start + len(near)] = near

    rng = np.random.default_rng(0)
    mic = (echo + near_full + rng.normal(0, 0.001, len(echo))).astype(np.float32)
    return ref, mic, near_start


def _speech_probs(signal):
    model = vad_module.get_vad_model()
    model.reset_states()
    probs = []
    for frame in frame_source.iter_frames_from_array(signal, realtime=False):
        probs.append(model(torch.from_numpy(frame), SR).item())
    return np.array(probs)


def test_aec_lets_vad_tell_echo_from_real_speech():
    ref, mic, near_start = _build_scenario()

    aec = nlms_aec.NLMSEchoCanceller(filter_length=512, mu=0.5)
    residual = aec.process(ref, mic.astype(np.float64)).astype(np.float32)

    near_start_frame = near_start // FRAME
    late_echo_only = slice(near_start_frame - 100, near_start_frame - 20)
    near_speech = slice(near_start_frame + 10, near_start_frame + 60)

    p_raw = _speech_probs(mic)
    p_residual = _speech_probs(residual)

    # Without AEC: echo alone already looks like speech to the VAD.
    assert p_raw[late_echo_only].mean() > 0.8

    # With AEC: echo (after the filter has converged) should read as
    # near-silence, while real near-end speech still reads as speech.
    assert p_residual[late_echo_only].mean() < 0.3
    assert p_residual[near_speech].mean() > 0.5
