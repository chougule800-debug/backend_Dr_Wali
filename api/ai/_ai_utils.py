"""
Shared utilities for the Neuro Specialities AI service.

Responsibilities:
- Load GROQ_API_KEY_1..3 from environment variables (and from a local .env
  file at the service root, for convenience during development).
- Generate request IDs for every AI call.
- Call Groq Whisper with controlled API-key rotation.
- Validate the provider response.
- Clean up the transcript with the SAME normalization the current
  frontend uses (useMic.ts: "em ar eye" / "mr i" -> "MRI").
- Safe logging. NEVER log or return API keys. NEVER fabricate a transcript.

This module intentionally mirrors the current frontend Groq settings so
transcription quality does not regress when we cut over in Phase 3:

    model       = whisper-large-v3
    temperature = 0
    language    = en
    prompt      = "Medical terms: MRI, CT Scan, EEG, ECG, NCV, Doppler, GTCS, BPPV, "
"""

import os
import re
import uuid
import logging
import requests

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
# When running locally, we want the service to pick up GROQ_API_KEY_* from a
# local .env file at the service root (backend/.env), regardless of the
# caller's current working directory. On Vercel there is no .env file, so
# the env vars come from Vercel's environment configuration — same code
# path, no branching.
#
# override=False means a real shell export still wins over the .env value.
# That is the conventional, safe behavior: CI / Vercel always take priority,
# and a developer's shell can intentionally override for a one-off test.

def _load_env_file():
    """
    Walk up from this file's location looking for a .env file, and load it
    into os.environ if found. Returns the path that was loaded, or None.

    Layout:
        backend/api/ai/_ai_utils.py    <- __file__
        backend/api/ai/                <- dirname(__file__)
        backend/api/                   <- 1 level up
        backend/                       <- 2 levels up  (this is where .env lives)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, ".env"),                          # api/ai/.env
        os.path.join(here, "..", ".env"),                    # api/.env
        os.path.join(here, "..", "..", ".env"),              # backend/.env   <- intended
        os.path.join(here, "..", "..", "..", ".env"),        # repo root .env
    ]
    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.isfile(normalized):
            load_dotenv(normalized, override=False)
            return normalized
    return None


_ENV_LOADED_FROM = _load_env_file()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("neuro_ai")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

if _ENV_LOADED_FROM:
    logger.info("ai_env_loaded path=%s", _ENV_LOADED_FROM)
else:
    logger.info("ai_env_loaded path=- (no .env file, using process environment)")


# ---------------------------------------------------------------------------
# Configuration (kept identical to the current frontend useMic.ts)
# ---------------------------------------------------------------------------

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"
GROQ_LANGUAGE = "en"
GROQ_TEMPERATURE = "0"
MEDICAL_PROMPT = "Medical terms: MRI, CT Scan, EEG, ECG, NCV, Doppler, GTCS, BPPV, "

# MediaRecorder on the frontend produces audio/webm. Accept a few common
# audio types as a safety net without loosening the check too far.
ALLOWED_CONTENT_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
    "audio/aac",
    "application/octet-stream",  # some browsers send this instead
}

# Vercel serverless request body cap is ~4.5 MB. Stay comfortably under it.
MAX_AUDIO_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB

# Hard cap on the total time we are willing to spend talking to Groq for
# a single user recording.
REQUEST_TIMEOUT_SECONDS = 55

# HTTP status classification for retry/key-rotation decisions.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
PERMANENT_STATUSES = {400, 401, 403}


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

def generate_request_id() -> str:
    return f"AI-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def get_api_keys():
    """
    Read Groq keys from environment variables, in order.
    Expected names: GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3.
    Blank / whitespace entries are skipped.
    NEVER log, return, or expose the values.
    """
    keys = []
    for i in (1, 2, 3):
        value = os.environ.get(f"GROQ_API_KEY_{i}")
        if value and value.strip():
            keys.append(value.strip())
    return keys


# ---------------------------------------------------------------------------
# Transcript cleaning (NO fabrication — only known normalizations)
# ---------------------------------------------------------------------------

_MRI_PATTERN = re.compile(r"\b(em ar eye|mr i|m r i)\b", re.IGNORECASE)


def clean_transcript(text: str) -> str:
    """
    Normalize common Whisper mishearings of medical abbreviations.
    Mirrors the current frontend post-processing in useMic.ts.
    Does NOT invent words. Only rewrites patterns we know are wrong.
    """
    if not isinstance(text, str):
        return ""
    normalized = _MRI_PATTERN.sub("MRI", text)
    return normalized.strip()


def validate_transcript(text):
    """
    Returns (is_valid: bool, error_message_or_None).
    Rejects malformed / empty / non-string provider responses.
    """
    if text is None:
        return False, "Transcript missing."
    if not isinstance(text, str):
        return False, "Transcript is not a string."
    if not text.strip():
        return False, "Transcript is empty."
    return True, None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_status(status_code: int) -> str:
    if status_code in PERMANENT_STATUSES:
        return "permanent"
    if status_code in RETRYABLE_STATUSES:
        return "retryable"
    return "unknown"


# ---------------------------------------------------------------------------
# Groq call with key rotation
# ---------------------------------------------------------------------------

def transcribe_with_groq(audio_bytes, filename, content_type):
    """
    Attempt transcription across the configured API keys.

    Returns:
        (success: bool, text_or_error: str, code_or_None: str, info: dict)

        success case -> (True,  "<transcript>", None,        {...})
        failure case -> (False, "<error msg>",  "<ERR_CODE>", {...})
    """
    keys = get_api_keys()
    if not keys:
        return (
            False,
            "AI service is not configured.",
            "NO_API_KEYS",
            {"attempts": []},
        )

    attempts = []
    last_error_message = "AI transcription failed."
    last_error_code = "UNKNOWN"

    for idx, key in enumerate(keys, start=1):
        files = {"file": (filename or "audio.webm", audio_bytes, content_type)}
        data = {
            "model": GROQ_MODEL,
            "temperature": GROQ_TEMPERATURE,
            "language": GROQ_LANGUAGE,
            "prompt": MEDICAL_PROMPT,
        }
        headers = {"Authorization": f"Bearer {key}"}

        try:
            resp = requests.post(
                GROQ_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout:
            attempts.append({"key_index": idx, "result": "timeout"})
            last_error_message = "AI request timed out."
            last_error_code = "TIMEOUT"
            continue
        except requests.exceptions.RequestException:
            attempts.append({"key_index": idx, "result": "network_error"})
            last_error_message = "Could not reach AI provider."
            last_error_code = "NETWORK_ERROR"
            continue

        attempts.append({"key_index": idx, "http_status": resp.status_code})

        # ----- Success -----
        if resp.status_code == 200:
            try:
                payload = resp.json()
            except ValueError:
                last_error_message = "AI provider returned invalid JSON."
                last_error_code = "INVALID_PROVIDER_RESPONSE"
                continue

            if not isinstance(payload, dict):
                last_error_message = "AI provider returned unexpected payload."
                last_error_code = "INVALID_PROVIDER_RESPONSE"
                continue

            raw_text = payload.get("text")
            ok, err = validate_transcript(raw_text)
            if not ok:
                last_error_message = err or "Invalid transcript."
                last_error_code = "INVALID_TRANSCRIPT"
                continue

            cleaned = clean_transcript(raw_text)
            if not cleaned:
                last_error_message = "Transcript was empty after cleaning."
                last_error_code = "EMPTY_TRANSCRIPT"
                continue

            return True, cleaned, None, {"attempts": attempts}

        # ----- Permanent: do NOT rotate keys -----
        if classify_status(resp.status_code) == "permanent":
            return (
                False,
                "AI provider rejected the request.",
                "PROVIDER_REJECTED",
                {"attempts": attempts},
            )

        # ----- Retryable / unknown: try the next key -----
        last_error_message = "AI provider temporarily unavailable."
        last_error_code = "PROVIDER_UNAVAILABLE"
        continue

    return False, last_error_message, last_error_code, {"attempts": attempts}


# ---------------------------------------------------------------------------
# Safe logging
# ---------------------------------------------------------------------------

def log_ai_event(request_id, status, duration_seconds, audio_size, code="", model=GROQ_MODEL):
    """
    Log only safe diagnostic metadata.
    NEVER logs audio bytes, transcripts, or API keys.
    """
    logger.info(
        "ai_event request_id=%s status=%s duration=%.2fs size=%d model=%s code=%s",
        request_id,
        status,
        duration_seconds,
        audio_size,
        model,
        code or "-",
    )
