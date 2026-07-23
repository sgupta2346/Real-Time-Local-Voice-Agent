import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort
from werkzeug.utils import secure_filename

import audio_io
import stt
import llm
import tts

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
GENERATED_DIR = SAMPLES_DIR / "_generated"
GENERATED_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "static"))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/samples")
def list_samples():
    files = sorted(SAMPLES_DIR.glob("sample_question_*.wav"))
    samples = [
        {
            "id": f.stem.replace("sample_question_", ""),
            "filename": f.name,
            "label": f.stem.replace("sample_question_", "").replace("_", " ").title(),
        }
        for f in files
    ]
    return jsonify(samples)


@app.get("/audio/<path:filename>")
def serve_audio(filename: str):
    safe_name = secure_filename(filename)
    for directory in (SAMPLES_DIR, GENERATED_DIR):
        candidate = directory / safe_name
        if candidate.is_file():
            return send_from_directory(directory, safe_name)
    abort(404)


@app.post("/api/run")
def run_pipeline():
    payload = request.get_json(force=True)
    filename = secure_filename(payload.get("filename", ""))
    input_path = SAMPLES_DIR / filename
    if not input_path.is_file():
        abort(400, "unknown sample")

    t0 = time.time()
    audio = audio_io.load_wav(str(input_path))
    t1 = time.time()

    transcript = stt.transcribe(audio)
    t2 = time.time()

    reply_text = llm.reply(transcript) if transcript else ""
    t3 = time.time()

    reply_filename = f"reply_{filename}"
    if reply_text:
        out_audio, sr = tts.synthesize(reply_text)
        audio_io.save_wav(str(GENERATED_DIR / reply_filename), out_audio, sr)
    t4 = time.time()

    return jsonify(
        {
            "transcript": transcript,
            "reply_text": reply_text,
            "input_audio_url": f"/audio/{filename}",
            "reply_audio_url": f"/audio/{reply_filename}" if reply_text else None,
            "timings": {
                "load": round(t1 - t0, 3),
                "stt": round(t2 - t1, 3),
                "llm": round(t3 - t2, 3),
                "tts": round(t4 - t3, 3),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
