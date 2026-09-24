"""
POST /api/ai/transcribe

Accepts multipart/form-data with a single `file` field containing audio
recorded in the browser (typically audio/webm from MediaRecorder).

Forwards the audio to Groq Whisper via _ai_utils.transcribe_with_groq,
validates and cleans the response, and returns a predictable JSON contract.

Never exposes API keys. Never fabricates a transcript.
"""

import os
import sys
import time

from flask import Flask, request, jsonify

# Ensure the sibling _ai_utils.py in this directory is importable both
# locally (uv run ...) and on Vercel (where the function is invoked
# directly from this file's directory).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _ai_utils import (  # noqa: E402
    generate_request_id,
    transcribe_with_groq,
    log_ai_event,
    ALLOWED_CONTENT_TYPES,
    MAX_AUDIO_SIZE_BYTES,
    GROQ_MODEL,
)


app = Flask(__name__)


def _err(request_id, message, code, http_status):
    return (
        jsonify(
            {
                "success": False,
                "error": message,
                "code": code,
                "request_id": request_id,
            }
        ),
        http_status,
    )


# Vercel forwards the full path (e.g. /api/ai/transcribe) to the WSGI app.
# Locally, running `app.run()` and hitting "/" should also work.
# The catch-all route handles both.
@app.route("/", methods=["POST"], defaults={"subpath": ""})
@app.route("/<path:subpath>", methods=["POST"])
def transcribe(subpath=""):  # noqa: ARG001
    request_id = generate_request_id()
    started_at = time.time()

    # 1. Audio present?
    audio = request.files.get("file")
    if audio is None:
        return _err(request_id, "No audio file provided.", "MISSING_AUDIO", 400)

    # 2. Content type acceptable?
    raw_content_type = (audio.mimetype or audio.content_type or "").strip()
    base_content_type = raw_content_type.split(";")[0].strip().lower()
    if base_content_type not in ALLOWED_CONTENT_TYPES:
        return _err(
            request_id,
            "Unsupported audio format.",
            "UNSUPPORTED_FORMAT",
            400,
        )

    # 3. Read + size checks
    try:
        audio_bytes = audio.read()
    except Exception:
        return _err(request_id, "Could not read audio upload.", "READ_FAILED", 400)

    audio_size = len(audio_bytes)
    if audio_size == 0:
        return _err(request_id, "Audio file is empty.", "EMPTY_AUDIO", 400)

    if audio_size > MAX_AUDIO_SIZE_BYTES:
        return _err(
            request_id,
            "Audio file is too large.",
            "AUDIO_TOO_LARGE",
            413,
        )

    # 4. Send to Groq with key rotation
    filename = audio.filename or "audio.webm"
    success, result, code, _info = transcribe_with_groq(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=base_content_type,
    )

    duration = time.time() - started_at

    # 5. Response
    if success:
        log_ai_event(
            request_id=request_id,
            status="success",
            duration_seconds=duration,
            audio_size=audio_size,
            model=GROQ_MODEL,
        )
        return (
            jsonify(
                {
                    "success": True,
                    "text": result,
                    "request_id": request_id,
                }
            ),
            200,
        )

    http_status = 500
    if code == "NO_API_KEYS":
        http_status = 503
    elif code == "PROVIDER_REJECTED":
        http_status = 502
    elif code == "TIMEOUT":
        http_status = 504
    elif code in ("INVALID_PROVIDER_RESPONSE", "INVALID_TRANSCRIPT", "EMPTY_TRANSCRIPT"):
        http_status = 502

    log_ai_event(
        request_id=request_id,
        status="failure",
        duration_seconds=duration,
        audio_size=audio_size,
        code=code or "UNKNOWN",
        model=GROQ_MODEL,
    )

    return _err(
        request_id,
        result or "AI transcription failed.",
        code or "UNKNOWN",
        http_status,
    )


# Vercel's Python runtime looks for `app` (Flask WSGI) or `handler`.
# Expose both for compatibility.
handler = app
