import numpy as np
import sounddevice as sd
import soundfile as sf

from config import SAMPLE_RATE, CHANNELS


def record_fixed(seconds: float) -> np.ndarray:
    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def play(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def load_wav(path: str, target_sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sample_rate:
        idx = np.arange(0, len(audio), sr / target_sample_rate).astype(np.int64)
        idx = idx[idx < len(audio)]
        audio = audio[idx]
    return audio.astype(np.float32)


def save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio, sample_rate)
