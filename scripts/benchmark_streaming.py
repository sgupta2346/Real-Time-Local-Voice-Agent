r"""
Fair, apples-to-apples comparison of the baseline (non-streaming) pipeline
against the streaming one, in a single warmed-up process so neither result
is skewed by a cold model load. Run from src/:

    ..\.venv\Scripts\python.exe ..\scripts\benchmark_streaming.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import audio_io
import llm
import stt
import tts
import main as baseline_main
import main_streaming

SAMPLE = str(Path(__file__).resolve().parent.parent / "samples" / "sample_question_apollo11.wav")


def warm_up():
    print("--- warming up all three models ---")
    audio = audio_io.load_wav(SAMPLE)
    stt.transcribe(audio)
    llm.reply("Say hi.")
    tts.synthesize("Warming up.")
    print("--- warm ---\n")


def main():
    warm_up()

    print("=== baseline (non-streaming) ===")
    baseline = baseline_main.run_turn(SAMPLE, 0, None, play_output=False)

    print("\n=== streaming (VAD + streaming LLM + sentence-chunked TTS) ===")
    streaming = main_streaming.run_streaming_turn(SAMPLE, None, play_output=False)

    print("\n=== comparison (same warm models, same input) ===")
    print(f"{'':30s} {'baseline':>10s} {'streaming':>10s}")
    print(f"{'time to first audio':30s} {baseline.get('first_audio', float('nan')):>9.2f}s {streaming.get('first_audio', float('nan')):>9.2f}s")
    print(f"{'total time':30s} {baseline.get('total', float('nan')):>9.2f}s {streaming.get('total', float('nan')):>9.2f}s")


if __name__ == "__main__":
    main()
