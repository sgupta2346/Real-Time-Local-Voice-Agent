"""
Streaming version of the pipeline: VAD-based end-of-turn detection instead of
a fixed recording window, and sentence-chunked TTS that starts playing the
first sentence while the LLM is still generating the rest (and while later
sentences are still being synthesized), instead of waiting for the entire
reply before saying anything.

STT is still one batch transcription call once VAD detects the user has
stopped talking; faster-whisper doesn't do token-level partial streaming the
way the LLM and TTS stages do here, and for a voice agent the STT stage
isn't in the way, once the turn ends, there's nothing to stream, the whole
utterance needs to be there before it can be transcribed. What streaming
buys here is a much shorter gap between the user finishing and hearing a
reply, not partial live captions.
"""
import argparse
import queue
import threading
import time

import audio_io
import frame_source
import llm
import stt
import tts
import vad


def run_streaming_turn(input_wav: str | None, output_prefix: str | None, play_output: bool):
    t_start = time.time()

    if input_wav:
        print(f"[reading audio from {input_wav}, simulating live arrival]")
        frames = frame_source.iter_frames_from_file(input_wav, realtime=False)
    else:
        print("[listening: speak now, a pause ends your turn]")
        mic = frame_source.MicFrameSource()
        with mic:
            audio = vad.record_utterance(mic)
        t_end_of_turn = time.time()
        return _finish_turn(audio, t_start, t_end_of_turn, output_prefix, play_output)

    audio = vad.record_utterance(frames)
    t_end_of_turn = time.time()
    return _finish_turn(audio, t_start, t_end_of_turn, output_prefix, play_output)


def _finish_turn(audio, t_start, t_end_of_turn, output_prefix, play_output) -> dict:
    if len(audio) == 0:
        print("[no speech detected]")
        return {}

    text = stt.transcribe(audio)
    t_stt_done = time.time()
    print(f"you said: {text!r}")
    if not text:
        print("[no speech detected]")
        return {}

    audio_queue: queue.Queue = queue.Queue()
    first_audio_time = None
    sentence_count = 0

    def produce():
        nonlocal sentence_count
        for out_audio, sr in tts.synthesize_sentence_stream(llm.reply_stream(text)):
            sentence_count += 1
            audio_queue.put((out_audio, sr))
        audio_queue.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    played_chunks = []
    while True:
        item = audio_queue.get()
        if item is None:
            break
        out_audio, sr = item
        if first_audio_time is None:
            first_audio_time = time.time()
            print(f"[first audio ready {first_audio_time - t_end_of_turn:.2f}s after end of turn]")
        played_chunks.append((out_audio, sr))
        if play_output:
            audio_io.play(out_audio, sr)

    t_all_done = time.time()

    if output_prefix and played_chunks:
        import numpy as np
        sr = played_chunks[0][1]
        full = np.concatenate([c[0] for c in played_chunks])
        audio_io.save_wav(f"{output_prefix}.wav", full, sr)
        print(f"[wrote reply audio to {output_prefix}.wav]")

    print(
        f"[timing] end_of_turn->stt_done={t_stt_done - t_end_of_turn:.2f}s "
        f"stt_done->first_audio={(first_audio_time - t_stt_done) if first_audio_time else float('nan'):.2f}s "
        f"first_audio->all_done={(t_all_done - first_audio_time) if first_audio_time else float('nan'):.2f}s "
        f"({sentence_count} sentences) total_end_of_turn->done={t_all_done - t_end_of_turn:.2f}s"
    )
    return {
        "stt": t_stt_done - t_end_of_turn,
        "first_audio": (first_audio_time - t_end_of_turn) if first_audio_time else float("nan"),
        "total": t_all_done - t_end_of_turn,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to a wav file to simulate as live mic input")
    parser.add_argument("--output-prefix", help="Write the full reply audio to <prefix>.wav")
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    run_streaming_turn(args.input, args.output_prefix, play_output=not args.no_play)
