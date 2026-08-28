"""
app.py — Flask web server for the offline video/audio translation tool.

Routes:
  GET  /                      -> UI
  POST /api/upload            -> accept file + params, start background job, return job_id
  GET  /api/status/<job_id>   -> poll progress
  GET  /api/download/<job_id>/<kind> -> download an output file
  GET  /api/archive           -> reuse dashboard data
"""
import os
import threading
import uuid

from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

from config import (
    HOST, PORT, ALLOWED_VIDEO_EXT, ALLOWED_AUDIO_EXT,
    MAX_VIDEO_SIZE_MB, MAX_AUDIO_SIZE_MB, JOBS_DIR,
    ENGINE_CHOICES, ASR_ENGINE_CHOICES,
)
from pipeline.orchestrator import run_job, JobError
from pipeline.stage5_archive import get_archive_stats
from pipeline.logging_setup import get_logger

log = get_logger("app")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = max(MAX_VIDEO_SIZE_MB, MAX_AUDIO_SIZE_MB) * 1024 * 1024

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads_tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# in-memory job registry: job_id -> {"progress": {...}, "result": {...} or None, "error": str or None}
JOBS = {}
JOBS_LOCK = threading.Lock()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ALLOWED_VIDEO_EXT:
        max_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    elif ext in ALLOWED_AUDIO_EXT:
        max_bytes = MAX_AUDIO_SIZE_MB * 1024 * 1024
    else:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. "
                     f"Allowed video: {sorted(ALLOWED_VIDEO_EXT)}, audio: {sorted(ALLOWED_AUDIO_EXT)}"
        }), 400

    temp_id = uuid.uuid4().hex[:10]
    saved_path = os.path.join(UPLOAD_DIR, f"{temp_id}_{filename}")
    f.save(saved_path)

    if os.path.getsize(saved_path) > max_bytes:
        os.remove(saved_path)
        return jsonify({"error": f"File exceeds size limit ({max_bytes // (1024*1024)} MB)."}), 400

    source_lang_hint = request.form.get("source_lang") or None
    target_lang = request.form.get("target_lang", "hin")
    want_burned_in = request.form.get("burned_in") == "true"
    want_voiceover = request.form.get("voiceover") == "true"

    engine_override = request.form.get("engine", "auto")
    if engine_override not in ENGINE_CHOICES:
        engine_override = "auto"

    asr_engine = request.form.get("asr_engine", "whisper")
    if asr_engine not in ASR_ENGINE_CHOICES:
        asr_engine = "whisper"

    # Voiceover TTS is AI4Bharat/Indic-TTS (Piper, IndicF5, and Indic
    # Parler-TTS were all tried and removed). Optional male/female override;
    # falls back to config.INDIC_TTS_DEFAULT_SPEAKER if omitted.
    tts_speaker = request.form.get("tts_speaker") or None
    if tts_speaker not in (None, "male", "female"):
        tts_speaker = None

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"progress": {"stage": "queued", "message": "Queued...", "pct": 0},
                         "result": None, "error": None}

    log.info(f"Upload accepted: {filename} ({ext}) -> queued job {job_id}")

    def _progress_cb(update):
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = update

    def _worker():
        try:
            result = run_job(
                saved_path, source_lang_hint, target_lang,
                want_burned_in, want_voiceover, progress_cb=_progress_cb,
                engine_override=engine_override,
                asr_engine=asr_engine,
                tts_speaker=tts_speaker,
            )
            with JOBS_LOCK:
                JOBS[job_id]["result"] = result
        except (JobError, Exception) as e:
            log.exception(f"Job {job_id} (upload {filename}) failed")
            with JOBS_LOCK:
                JOBS[job_id]["error"] = str(e)
        finally:
            if os.path.exists(saved_path):
                os.remove(saved_path)

    threading.Thread(target=_worker, daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404

    resp = {"progress": job["progress"]}
    if job["error"]:
        resp["error"] = job["error"]
    if job["result"]:
        result = job["result"]
        resp["result"] = {
            "job_id": result["job_id"],
            "available_downloads": [
                k for k in ["srt", "vtt", "job_report_docx", "burned_in_mp4", "voiceover_mp4"]
                if result.get(k)
            ],
            "stats": result["stats"],
        }
    return jsonify(resp)


DOWNLOAD_KEY_MAP = {
    "srt": "srt", "vtt": "vtt", "report": "job_report_docx",
    "burned_in": "burned_in_mp4", "voiceover": "voiceover_mp4",
}


@app.route("/api/download/<job_id>/<kind>")
def download(job_id, kind):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or not job.get("result"):
        return jsonify({"error": "Job not found or not finished."}), 404

    result_key = DOWNLOAD_KEY_MAP.get(kind)
    if not result_key:
        return jsonify({"error": f"Unknown download kind '{kind}'."}), 400

    path = job["result"].get(result_key)
    if not path or not os.path.exists(path):
        return jsonify({"error": "Requested output not available for this job."}), 404

    return send_file(path, as_attachment=True)


@app.route("/api/archive")
def archive():
    return jsonify({"jobs": get_archive_stats()})


if __name__ == "__main__":
    print("=" * 70)
    print("Offline Field Translator")
    print("-" * 70)
    print("First run notice: the first time each model/tool is actually used")
    print("(ASR, translation, TTS, ffmpeg), it auto-downloads and caches itself.")
    print("That first job may take a while depending on your connection; every")
    print("job after that runs fully offline.")
    if not os.environ.get("HF_TOKEN"):
        print()
        print("NOTE: IndicTrans2 (used for Indic<->Indic and English<->Indic")
        print("translation) is a free but 'gated' model on Hugging Face. If a job")
        print("using it fails, accept the terms once while logged in at:")
        print("  https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M")
        print("  https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M")
        print("  https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M")
        print("then set:  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   and restart.")
        print("Translations into/from other languages via NLLB-200 don't need this.")
    print()
    print("NOTE: Voiceover uses AI4Bharat/Indic-TTS (FastPitch+HiFi-GAN, MIT")
    print("licensed, no HF login needed) in a SEPARATE venv (.venv-tts) to avoid")
    print("a transformers version conflict with IndicTrans2/NLLB. One-time setup:")
    print("  python3 -m venv .venv-tts")
    print("  .venv-tts/bin/pip install -r tts_worker/requirements.txt")
    print("Checkpoints (~1.5GB/language) download automatically on first use.")
    print("=" * 70)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
