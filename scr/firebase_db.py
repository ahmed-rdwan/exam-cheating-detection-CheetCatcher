# ---- firebase_db.py — Firebase Firestore + Local JSON Fallback ----
# This module provides a unified interface for storing and reading alerts.
# The key design rule:  the function signatures are identical regardless
# of which backend is actually used — so the rest of the app (Tkinter UI,
# scoring engine) never needs to know or care.
#
# Backend decision (Firestore only, no Storage):
#   Screenshots stay saved locally in alerts/screenshots/.
#   The Firestore document stores the LOCAL file path of the screenshot,
#   not a cloud URL.  This keeps things simple and avoids needing
#   Firebase Storage (which requires a Blaze billing plan for heavy use).

import os
import json
from datetime import datetime
from config import (
    FIREBASE_CREDENTIALS_PATH,
    FIREBASE_COLLECTION,
    LOCAL_ALERTS_JSON,
)


# ---- Firebase initialization ----
# We try to initialize Firebase once at import time.
# If credentials are missing or invalid, we set _firestore_db to None
# and all writes silently fall back to local JSON.

_firestore_db = None

def _init_firebase():
    """
    Attempt to connect to Firebase Firestore using the service-account
    credentials file.  If anything goes wrong (file missing, invalid JSON,
    network error), we just print a notice and continue — the app
    will use local JSON instead, and nothing crashes.
    """
    global _firestore_db

    if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
        print(f"[Firebase] Credentials file not found at '{FIREBASE_CREDENTIALS_PATH}'.")
        print("[Firebase] Falling back to local JSON storage.")
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # Check if Firebase is already initialized (prevents double-init errors)
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)

        _firestore_db = firestore.client()
        print("[Firebase] Connected to Firestore successfully.")

    except Exception as e:
        print(f"[Firebase] Initialization failed: {e}")
        print("[Firebase] Falling back to local JSON storage.")
        _firestore_db = None

# Run initialization when this module is first imported
_init_firebase()


# ---- Public API ----

def add_alert(record):
    """
    Save an alert record to the database.

    Args:
        record: dict with keys:
            - 'track_id':        int
            - 'behavior':        str (e.g. "Phone Detected", "Looking Away")
            - 'confidence':      float (the student's current score)
            - 'screenshot_path': str (local file path to the saved screenshot)
            - 'timestamp':       str (ISO format, auto-added if missing)

    The function tries Firestore first;  if that fails for any reason,
    it appends the same record to the local JSON file instead.
    The caller never needs to handle this distinction.
    """
    # Ensure a timestamp exists
    if "timestamp" not in record:
        record["timestamp"] = datetime.now().isoformat()

    # Try Firestore first
    if _firestore_db is not None:
        try:
            _firestore_db.collection(FIREBASE_COLLECTION).add(record)
            print(f"[Firebase] Alert saved to Firestore: Student #{record['track_id']}")
            return
        except Exception as e:
            print(f"[Firebase] Write failed: {e}  →  falling back to local JSON.")

    # Fallback: append to local JSON file
    _write_local(record)


def get_all_alerts():
    """
    Read all stored alerts.

    Returns:
        list of dicts, each with the same keys as the record passed to add_alert().

    Tries Firestore first;  if unavailable, reads from local JSON.
    """
    if _firestore_db is not None:
        try:
            docs = _firestore_db.collection(FIREBASE_COLLECTION).order_by("timestamp").stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            print(f"[Firebase] Read failed: {e}  →  reading from local JSON.")

    return _read_local()


# ---- Local JSON helpers ----

def _write_local(record):
    """
    Append a single alert record to the local JSON file.
    The file stores a JSON array of records.  We read it, append, and
    write it back — not the most efficient for thousands of records,
    but perfectly fine for the scale of an exam session.
    """
    os.makedirs(os.path.dirname(LOCAL_ALERTS_JSON), exist_ok=True)

    existing = _read_local()
    existing.append(record)

    with open(LOCAL_ALERTS_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[Local] Alert saved to {LOCAL_ALERTS_JSON}: Student #{record['track_id']}")


def _read_local():
    """
    Read all alerts from the local JSON file.
    Returns an empty list if the file doesn't exist yet.
    """
    if not os.path.exists(LOCAL_ALERTS_JSON):
        return []

    try:
        with open(LOCAL_ALERTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []
