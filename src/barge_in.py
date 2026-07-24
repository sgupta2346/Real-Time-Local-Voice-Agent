"""
Live barge-in: play the agent's reply while simultaneously listening, and if
the user starts talking over it, stop immediately and start capturing what
they're saying instead of finishing the sentence.

This needs a duplex audio stream (input and output open at the same time,
same callback) so the reference signal (what we're playing) and the mic
signal (what's being captured) are sample-aligned for the echo canceller.
The audio callback itself has to stay fast and non-blocking (PortAudio will
glitch if it doesn't get fed on time), so it does the minimum: run AEC, feed
the residual to VAD, set a flag if speech starts while playing. Everything
else, stopping playback, deciding what to do next, happens in the calling
thread.
"""
import collections
import queue

import numpy as np
from silero_vad import VADIterator

from config import SAMPLE_RATE
from frame_source import FRAME_SIZE
from nlms_aec import NLMSEchoCanceller
from vad import get_vad_model

try:
    import sounddevice as sd
except OSError:
    sd = None


class ResidualFrameQueue:
    """Frame source fed by the duplex callback's echo-cancelled residual.

    Shares the same iterator interface as frame_source.MicFrameSource, so
    vad.record_utterance() can consume it unchanged once a barge-in has
    stopped playback.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def push(self, frame: np.ndarray) -> None:
        self._q.put(frame)

    def drain(self) -> None:
        """Discard whatever's queued up (stale echo residue, dead air from
        processing time) so the next listen starts from *now*, not from
        whatever accumulated while nobody was draining the queue."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def __iter__(self):
        while True:
            yield self._q.get()


class BargeInStream:
    def __init__(self, doubletalk_threshold: float = 1.8):
        if sd is None:
            raise RuntimeError("sounddevice/PortAudio not available on this machine")

        self.aec = NLMSEchoCanceller(filter_length=512, mu=0.5, doubletalk_threshold=doubletalk_threshold)
        self.vad_iterator = VADIterator(get_vad_model(), sampling_rate=SAMPLE_RATE, min_silence_duration_ms=300)
        self.residual_frames = ResidualFrameQueue()

        self._playback: collections.deque = collections.deque()
        self._interrupted = False
        self._playing = False
        self._chunks_remaining = 0

        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=self._callback,
        )

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop()
        self.stream.close()

    def _callback(self, indata, outdata, frames, time_info, status):
        if self._playback:
            ref_block = self._playback.popleft()
            self._chunks_remaining -= 1
        else:
            ref_block = np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = ref_block

        mic_block = indata[:, 0].astype(np.float64)
        residual = self.aec.process(ref_block.astype(np.float64), mic_block).astype(np.float32)

        if self._playing:
            event = self.vad_iterator(residual)
            if event and "start" in event:
                self._interrupted = True
                self._playback.clear()
                self._chunks_remaining = 0

        self.residual_frames.push(residual)

    def speak(self, audio: np.ndarray, sr: int) -> bool:
        """
        Play `audio` (resampled to SAMPLE_RATE by the caller) while watching
        for barge-in. Returns True if the user interrupted, False if it
        played to completion.
        """
        assert sr == SAMPLE_RATE, "resample before calling speak()"
        self._interrupted = False
        self.vad_iterator.reset_states()

        chunks = []
        for start in range(0, len(audio), FRAME_SIZE):
            chunk = audio[start:start + FRAME_SIZE]
            if len(chunk) < FRAME_SIZE:
                chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
            chunks.append(chunk)
        self._chunks_remaining = len(chunks)
        self._playing = True
        for chunk in chunks:
            self._playback.append(chunk)

        import time
        while self._chunks_remaining > 0 and not self._interrupted:
            time.sleep(0.01)

        self._playing = False
        return self._interrupted
