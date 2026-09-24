"""
GET /api/ai/health

Lightweight liveness probe. Confirms the Python AI service is deployed and
responding. Does NOT call Groq and does NOT expose any secret.

Reports only a boolean flag for whether at least one GROQ_API_KEY_* is
configured — never the key value itself.
"""

import os
import sys

from flask import Flask, jsonify

# Same sibling-import safety net as transcribe.py, so _ai_utils is importable
# from this directory under Vercel's invocation model.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _ai_utils import get_api_keys  # noqa: E402


app = Flask(__name__)


@app.route("/", methods=["GET"], defaults={"subpath": ""})
@app.route("/<path:subpath>", methods=["GET"])
def health(subpath=""):  # noqa: ARG001
    keys = get_api_keys()
    return (
        jsonify(
            {
                "success": True,
                "service": "ai",
                "status": "healthy",
                # Safe, non-sensitive indicator only. Never the key value.
                "groq_configured": len(keys) > 0,
            }
        ),
        200,
    )


handler = app
