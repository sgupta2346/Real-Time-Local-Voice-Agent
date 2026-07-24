"""
Full live loop with barge-in: listen, reply sentence by sentence, and if you
start talking while it's mid-reply, it stops and listens to you instead.

Unlike everything else in this repo, this can't be exercised with
synthetic/file audio, it needs a real microphone and real speakers open at
the same time, because the whole point is the echo cancellation between
them. Run it and just talk over the agent partway through a reply.
"""
import audio_io
import barge_in
import llm
import stt
import tts
import vad
from config import SAMPLE_RATE


def run_session():
    with barge_in.BargeInStream() as stream:
        print("[session started, speak whenever you like]")
        audio = vad.record_utterance(stream.residual_frames)

        while True:
            if len(audio) == 0:
                print("[no speech detected, still listening]")
                audio = vad.record_utterance(stream.residual_frames)
                continue

            text = stt.transcribe(audio)
            print(f"you said: {text!r}")
            if not text:
                audio = vad.record_utterance(stream.residual_frames)
                continue

            interrupted = False
            for sentence_audio, sr in tts.synthesize_sentence_stream(llm.reply_stream(text)):
                sentence_16k = audio_io.resample(sentence_audio, sr, SAMPLE_RATE)
                if stream.speak(sentence_16k, SAMPLE_RATE):
                    print("[barge-in: stopped talking, listening to you]")
                    interrupted = True
                    break

            if interrupted:
                # Don't drain here: the frames that triggered the interrupt
                # (and whatever you kept saying right after) are already
                # queued and are exactly what we want to capture next.
                audio = vad.record_utterance(stream.residual_frames)
            else:
                print("[done speaking, your turn]")
                # Discard whatever queued up during STT/LLM/TTS processing
                # and the agent's own (echo-cancelled) playback, so the next
                # listen starts from now, not from a backlog of dead air.
                stream.residual_frames.drain()
                audio = vad.record_utterance(stream.residual_frames)


if __name__ == "__main__":
    run_session()
