"""
app.py — Flask web server for the offline video/audio translation tool.

Routes:
  GET  /                         -> UI
  POST /api/upload               -> accept file + params, start background job
  GET  /api/status/<job_id>      -> poll progress
  GET  /api/download/<job_id>/<kind>
                                -> download USER-FACING output files only
  GET  /api/archive              -> reuse dashboard data

IMPORTANT
---------
The pipeline continues generating ALL internal output files:

    SRT
    VTT
    Subtitle PDF
    Job Report DOCX
    Voiceover MP4
    Burned-in MP4
    Other pipeline files

However, the dashboard intentionally exposes ONLY:

    1. Burned-in Voiceover + Subtitles MP4
    2. Subtitle PDF

The other files remain safely stored inside the job's output directory.
"""

import os
import threading
import uuid

from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

from config import (
    HOST,
    PORT,
    ALLOWED_VIDEO_EXT,
    ALLOWED_AUDIO_EXT,
    MAX_VIDEO_SIZE_MB,
    MAX_AUDIO_SIZE_MB,
    JOBS_DIR,
    ENGINE_CHOICES,
    ASR_ENGINE_CHOICES,
)

from pipeline.orchestrator import (
    run_job,
    JobError,
)

from pipeline.stage5_archive import (
    get_archive_stats,
)

from pipeline.logging_setup import (
    get_logger,
)

from pipeline.cancellation import (
    JobCancelled,
)


log = get_logger("app")


# ============================================================
# Flask application
# ============================================================

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
)

app.config["MAX_CONTENT_LENGTH"] = (
    max(
        MAX_VIDEO_SIZE_MB,
        MAX_AUDIO_SIZE_MB,
    )
    * 1024
    * 1024
)


# ============================================================
# Temporary upload directory
# ============================================================

UPLOAD_DIR = os.path.join(
    os.path.dirname(__file__),
    "uploads_tmp",
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)


# ============================================================
# In-memory job registry
# ============================================================

JOBS = {}

JOBS_LOCK = threading.Lock()


# ============================================================
# Dashboard
# ============================================================

@app.route("/")
def index():
    return app.send_static_file(
        "index.html"
    )


# ============================================================
# Upload
# ============================================================

@app.route(
    "/api/upload",
    methods=["POST"],
)
def upload():

    if "file" not in request.files:
        return jsonify({
            "error": "No file part in request."
        }), 400

    f = request.files["file"]

    if f.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    filename = secure_filename(
        f.filename
    )

    ext = os.path.splitext(
        filename
    )[1].lower()

    # --------------------------------------------------------
    # Validate extension / size
    # --------------------------------------------------------

    if ext in ALLOWED_VIDEO_EXT:

        max_bytes = (
            MAX_VIDEO_SIZE_MB
            * 1024
            * 1024
        )

    elif ext in ALLOWED_AUDIO_EXT:

        max_bytes = (
            MAX_AUDIO_SIZE_MB
            * 1024
            * 1024
        )

    else:

        return jsonify({
            "error": (
                f"Unsupported file type '{ext}'. "
                f"Allowed video: "
                f"{sorted(ALLOWED_VIDEO_EXT)}, "
                f"audio: "
                f"{sorted(ALLOWED_AUDIO_EXT)}"
            )
        }), 400

    # --------------------------------------------------------
    # Save upload temporarily
    # --------------------------------------------------------

    temp_id = uuid.uuid4().hex[:10]

    saved_path = os.path.join(
        UPLOAD_DIR,
        f"{temp_id}_{filename}",
    )

    f.save(
        saved_path
    )

    if os.path.getsize(
        saved_path
    ) > max_bytes:

        os.remove(
            saved_path
        )

        return jsonify({
            "error": (
                "File exceeds size limit "
                f"({max_bytes // (1024 * 1024)} MB)."
            )
        }), 400

    # --------------------------------------------------------
    # Job parameters
    # --------------------------------------------------------

    source_lang_hint = (
        request.form.get(
            "source_lang"
        )
        or None
    )

    target_lang = (
        request.form.get(
            "target_lang",
            "hin",
        )
    )

    want_burned_in = (
        request.form.get(
            "burned_in"
        )
        == "true"
    )

    want_voiceover = (
        request.form.get(
            "voiceover"
        )
        == "true"
    )

    engine_override = (
        request.form.get(
            "engine",
            "auto",
        )
    )

    if engine_override not in ENGINE_CHOICES:
        engine_override = "auto"

    asr_engine = (
        request.form.get(
            "asr_engine",
            "indic_conformer",
        )
    )

    if asr_engine not in ASR_ENGINE_CHOICES:
        asr_engine = "indic_conformer"

    tts_speaker = (
        request.form.get(
            "tts_speaker"
        )
        or None
    )

    if tts_speaker not in (
        None,
        "male",
        "female",
    ):
        tts_speaker = None

    # --------------------------------------------------------
    # Create job
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex[:12]

    with JOBS_LOCK:

        JOBS[job_id] = {
            "progress": {
                "stage": "queued",
                "message": "Queued...",
                "pct": 0,
            },

            "result": None,

            "error": None,

            "cancelled": False,

            "cancel_event": threading.Event(),
        }

    log.info(
        f"Upload accepted: {filename} ({ext}) "
        f"-> queued job {job_id}"
    )

    # --------------------------------------------------------
    # Progress callback
    # --------------------------------------------------------

    def _progress_cb(update):

        with JOBS_LOCK:

            if JOBS[job_id][
                "cancel_event"
            ].is_set():

                raise JobCancelled(
                    "Cancelled by user."
                )

            JOBS[job_id][
                "progress"
            ] = update

    # --------------------------------------------------------
    # Background worker
    # --------------------------------------------------------

    def _worker():

        try:

            result = run_job(
                saved_path,
                source_lang_hint,
                target_lang,
                want_burned_in,
                want_voiceover,
                progress_cb=_progress_cb,
                engine_override=engine_override,
                asr_engine=asr_engine,
                tts_speaker=tts_speaker,
                cancel_event=JOBS[job_id]["cancel_event"],
            )

            with JOBS_LOCK:

                JOBS[job_id][
                    "result"
                ] = result

        except JobCancelled:

            log.info(
                f"Job {job_id} "
                f"(upload {filename}) cancelled by user"
            )

            with JOBS_LOCK:

                JOBS[job_id][
                    "cancelled"
                ] = True

                JOBS[job_id][
                    "progress"
                ] = {
                    "stage": "cancelled",
                    "message": "Cancelled by user.",
                    "pct": JOBS[job_id][
                        "progress"
                    ].get("pct", 0),
                }

        except (
            JobError,
            Exception,
        ) as e:

            log.exception(
                f"Job {job_id} "
                f"(upload {filename}) failed"
            )

            with JOBS_LOCK:

                JOBS[job_id][
                    "error"
                ] = str(e)

        finally:

            if os.path.exists(
                saved_path
            ):

                os.remove(
                    saved_path
                )

    threading.Thread(
        target=_worker,
        daemon=True,
    ).start()

    return jsonify({
        "job_id": job_id
    })


# ============================================================
# Job status
# ============================================================

@app.route(
    "/api/status/<job_id>"
)
def status(job_id):

    with JOBS_LOCK:

        job = JOBS.get(
            job_id
        )

    if job is None:

        return jsonify({
            "error": "Unknown job_id"
        }), 404

    resp = {
        "progress": job[
            "progress"
        ]
    }

    # --------------------------------------------------------
    # Cancelled
    # --------------------------------------------------------

    if job["cancelled"]:

        resp["cancelled"] = True

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    if job["error"]:

        resp["error"] = (
            job["error"]
        )

    # --------------------------------------------------------
    # Completed result
    # --------------------------------------------------------

    if job["result"]:

        result = job[
            "result"
        ]

        # IMPORTANT:
        #
        # Only these two outputs are exposed to the dashboard.
        #
        # Everything else continues to exist inside:
        #
        # jobs/<job_id>/output/
        #
        # Examples of hidden files:
        #
        #   *_subtitles.srt
        #   *_subtitles.vtt
        #   *_voiceover.mp4
        #   *_job_report.docx
        #
        # They are NOT deleted.
        # They are simply not advertised to the frontend.

        available_downloads = [
            key
            for key in [
                "burned_in_mp4",
                "subtitles_pdf",
            ]
            if result.get(key)
        ]

        resp["result"] = {

            "job_id":
                result["job_id"],

            "available_downloads":
                available_downloads,

            "stats":
                result["stats"],
        }

    return jsonify(
        resp
    )


# ============================================================
# Cancel a running job
# ============================================================

@app.route(
    "/api/cancel/<job_id>",
    methods=["POST"],
)
def cancel(job_id):

    with JOBS_LOCK:

        job = JOBS.get(
            job_id
        )

        if job is None:

            return jsonify({
                "error": "Unknown job_id"
            }), 404

        if (
            job["result"]
            or job["error"]
            or job["cancelled"]
        ):

            return jsonify({
                "error": (
                    "Job has already "
                    "finished."
                )
            }), 400

        job["cancel_event"].set()

    log.info(
        f"Cancel requested for job {job_id}"
    )

    return jsonify({
        "ok": True
    })


# ============================================================
# USER-FACING DOWNLOAD MAP
# ============================================================

"""
Only these download routes are exposed.

Frontend:

    burned_in
        ↓
    burned_in_mp4

    subtitles_pdf
        ↓
    subtitles_pdf

There are intentionally NO routes here for:

    srt
    vtt
    report
    voiceover

Those files still remain in output/.
"""

DOWNLOAD_KEY_MAP = {

    "burned_in":
        "burned_in_mp4",

    "subtitles_pdf":
        "subtitles_pdf",
}


# ============================================================
# Download
# ============================================================

@app.route(
    "/api/download/<job_id>/<kind>"
)
def download(
    job_id,
    kind,
):

    with JOBS_LOCK:

        job = JOBS.get(
            job_id
        )

    if (
        job is None
        or not job.get("result")
    ):

        return jsonify({
            "error": (
                "Job not found "
                "or not finished."
            )
        }), 404

    # --------------------------------------------------------
    # Only approved user-facing download types are accepted.
    # --------------------------------------------------------

    result_key = (
        DOWNLOAD_KEY_MAP.get(
            kind
        )
    )

    if not result_key:

        return jsonify({
            "error": (
                f"Unknown download kind "
                f"'{kind}'."
            )
        }), 400

    # --------------------------------------------------------
    # Locate generated file
    # --------------------------------------------------------

    path = job[
        "result"
    ].get(
        result_key
    )

    if (
        not path
        or not os.path.exists(path)
    ):

        return jsonify({
            "error": (
                "Requested output "
                "not available for "
                "this job."
            )
        }), 404

    return send_file(
        path,
        as_attachment=True,
    )


# ============================================================
# Archive
# ============================================================

@app.route(
    "/api/archive"
)
def archive():

    return jsonify({
        "jobs":
            get_archive_stats()
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "Offline Field Translator"
    )

    print("=" * 70)

    print(
        f"Server: "
        f"http://{HOST}:{PORT}"
    )

    print(
        "Dashboard user-facing outputs:"
    )

    print(
        "  - Burned-in Voiceover + "
        "Subtitles (.mp4)"
    )

    print(
        "  - Subtitle PDF (.pdf)"
    )

    print(
        "Internal SRT/VTT/DOCX/"
        "voiceover files remain "
        "stored in the job output."
    )

    print("=" * 70)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
    )