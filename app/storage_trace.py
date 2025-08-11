# app/storage_trace.py
import json
import os
from typing import Optional, Dict, Any

_USE_FIRESTORE = False
try:
    from google.cloud import firestore
    _USE_FIRESTORE = True
except Exception:
    _USE_FIRESTORE = False

_COLLECTION = "traces"
_LOCAL_FILE = "data/traces.json"

def _fs_client():
    return firestore.Client()

def _load_local() -> Dict[str, Any]:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(_LOCAL_FILE):
        return {}
    with open(_LOCAL_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def _save_local(data: Dict[str, Any]):
    os.makedirs("data", exist_ok=True)
    with open(_LOCAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_trace(doc: Dict[str, Any]):
    """Upsert a trace by id."""
    tid = doc.get("id")
    if not tid:
        raise ValueError("Trace must include 'id'")

    if _USE_FIRESTORE:
        db = _fs_client()
        db.collection(_COLLECTION).document(tid).set(doc)
    else:
        data = _load_local()
        data[tid] = doc
        _save_local(data)

def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    if _USE_FIRESTORE:
        db = _fs_client()
        snap = db.collection(_COLLECTION).document(trace_id).get()
        return snap.to_dict() if snap.exists else None
    else:
        data = _load_local()
        return data.get(trace_id)
