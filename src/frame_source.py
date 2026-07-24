"""
Frame sources: anything that yields fixed-size 16kHz mono float32 frames.

The point of this abstraction is that VAD-based endpointing shouldn't care
whether frames are coming from a live microphone or a pre-recorded file being
replayed. Same downstream code, two sources: one for real use, one for
testing without a microphone.
"""
import queue
import time

import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE

FRAME_SIZE = 512  # required by Silero VAD at 16kHz (32ms)


def iter_frames_from_array(audio: np.ndarray, realtime: bool = False):
    """Yield fixed-size frames from an in-memory audio array, zero-padding the tail."""
    n = len(audio)
    for start in range(0, n, FRAME_SIZE):
        chunk = audio[start:start + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
        if realtime:
            time.sleep(FRAME_SIZE / SAMPLE_RATE)
        yield chunk.astype(np.float32)


def iter_frames_from_file(path: str, realtime: bool = False):
    import audio_io
    audio = audio_io.load_wav(path)
    yield from iter_frames_from_array(audio, realtime=realtime)


class MicFrameSource:
    """Live microphone input, chunked into VAD-sized frames."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._q: queue.Queue = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata[:, 0].copy())

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.stop()
        self._stream.close()

    def __iter__(self):
        while True:
            yield self._q.get()
