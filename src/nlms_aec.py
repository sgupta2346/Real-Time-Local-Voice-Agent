"""
Acoustic echo cancellation, from scratch, via NLMS (Normalized Least Mean
Squares).

The problem barge-in has to solve: while the agent is talking, the mic
still picks up the agent's own voice coming back through the speakers. Naive
VAD on raw mic input would trigger "the user is talking" constantly, on the
agent's own audio. AEC subtracts a predicted copy of the echo from the mic
signal before VAD ever sees it.

Went with a from-scratch NLMS filter instead of a pip package: the obvious
one (speexdsp-python) needs a compiled C extension built against
libspeexdsp-dev, which isn't a clean Windows install and would break the
one-command setup this repo otherwise has. NLMS is~40 lines of numpy and is
the same algorithm underneath anyway.

How it works: keep an adaptive FIR filter that models the echo path
(speaker -> air -> mic). At each sample, predict the echo from the known
reference signal (the audio we're playing), subtract the prediction from
the real mic signal, and nudge the filter based on how wrong the prediction
was. Given enough samples it converges onto the actual echo path (room
acoustics, speaker/mic response) and the residual left over is (ideally)
just whatever the mic picked up that ISN'T an echo of what we played, i.e.
the user's own voice.
"""
import numpy as np


class NLMSEchoCanceller:
    def __init__(
        self,
        filter_length: int = 2048,
        mu: float = 0.5,
        eps: float = 1e-6,
        doubletalk_threshold: float = 1.8,
    ):
        """
        doubletalk_threshold: a basic Geigel double-talk detector. Echo can
        only ever be quieter than (or comparable to) the reference signal
        that caused it, an acoustic path doesn't amplify. So if the mic
        sample is louder than `doubletalk_threshold` times the loudest
        recent reference sample, that excess can't be echo, it means the
        near end is also talking. When that fires, freeze adaptation for
        that sample: keep cancelling with the last good filter instead of
        letting an unexplainable near-end signal drag the weights around.
        Without this, the filter actively diverges the moment someone
        barges in, which is exactly the moment barge-in needs it to work.
        """
        self.filter_length = filter_length
        self.mu = mu
        self.eps = eps
        self.doubletalk_threshold = doubletalk_threshold
        self.weights = np.zeros(filter_length, dtype=np.float64)
        self._history = np.zeros(filter_length, dtype=np.float64)

    def reset(self):
        self.weights[:] = 0
        self._history[:] = 0

    def process(self, reference: np.ndarray, mic: np.ndarray) -> np.ndarray:
        """
        reference: what we played (the echo source), same length as mic.
        mic: what the microphone captured (echo + near-end speech + noise).
        Returns the echo-cancelled residual, same length as the inputs.
        """
        n = len(reference)
        residual = np.empty(n, dtype=np.float64)
        w = self.weights
        hist = self._history
        L = self.filter_length

        for i in range(n):
            hist[1:] = hist[:-1]
            hist[0] = reference[i]

            echo_estimate = w @ hist
            error = mic[i] - echo_estimate
            residual[i] = error

            ref_peak = np.max(np.abs(hist))
            doubletalk = abs(mic[i]) > self.doubletalk_threshold * ref_peak + self.eps
            if not doubletalk:
                norm = hist @ hist + self.eps
                w += (self.mu * error / norm) * hist

        return residual
