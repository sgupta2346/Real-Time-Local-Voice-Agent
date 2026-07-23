import argparse
import time

import audio_io
import stt
import llm
import tts


def run_turn(input_wav: str | None, record_seconds: float, output_wav: str | None, play_output: bool) -> None:
    t0 = time.time()
    if input_wav:
        print(f"[reading audio from {input_wav}]")
        audio = audio_io.load_wav(input_wav)
    else:
        print(f"[listening for {record_seconds}s...]")
        audio = audio_io.record_fixed(record_seconds)
    t1 = time.time()

    text = stt.transcribe(audio)
    t2 = time.time()
    print(f"you said: {text!r}")
    if not text:
        print("[no speech detected]")
        return

    answer = llm.reply(text)
    t3 = time.time()
    print(f"agent: {answer!r}")

    out_audio, sr = tts.synthesize(answer)
    t4 = time.time()

    if output_wav:
        audio_io.save_wav(output_wav, out_audio, sr)
        print(f"[wrote reply audio to {output_wav}]")
    if play_output:
        audio_io.play(out_audio, sr)
    t5 = time.time()

    print(
        f"[timing] input={t1 - t0:.2f}s stt={t2 - t1:.2f}s llm={t3 - t2:.2f}s "
        f"tts={t4 - t3:.2f}s output={t5 - t4:.2f}s pipeline(stt+llm+tts)="
        f"{t4 - t1:.2f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to a wav file to use instead of live mic input")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds to record if no --input given")
    parser.add_argument("--output", help="Path to write the reply audio to")
    parser.add_argument("--no-play", action="store_true", help="Don't play the reply through speakers")
    args = parser.parse_args()

    run_turn(args.input, args.seconds, args.output, play_output=not args.no_play)
