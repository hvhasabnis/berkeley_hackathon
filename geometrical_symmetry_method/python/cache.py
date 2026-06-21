"""
Lightweight Redis cache for the completion pipeline.

Goals:
  * Skip re-parsing the same point-cloud CSV (numpy array cache).
  * Skip re-running pipeline() for identical input + parameters (result cache).

Safety:
  * If Redis is not installed or not running, EVERYTHING degrades to a no-op.
    The pipeline still runs exactly as before, just without caching. Nothing
    here ever raises into the caller.

Run Redis locally (macOS):
    brew install redis
    brew services start redis      # or just: redis-server
    redis-cli ping                 # -> PONG
    pip install redis
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

_NS = "gsm:"                 # key namespace so we can flush only our keys
_TTL = int(os.environ.get("GSM_CACHE_TTL", 24 * 3600))   # 1 day default

try:
    import redis
    _client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "127.0.0.1"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        socket_connect_timeout=0.3,
    )
    _client.ping()
    print("[cache] Redis connected")
except Exception as e:  # noqa: BLE001 - any failure => caching disabled
    _client = None
    print(f"[cache] Redis disabled ({type(e).__name__}); running without cache")


def available() -> bool:
    return _client is not None


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def file_signature(path) -> str:
    """A signature that changes whenever the file changes (mtime + size)."""
    p = Path(path)
    try:
        st = p.stat()
        return f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return str(p)


def _k(key: str) -> str:
    return _NS + key


def get_json(key: str):
    if _client is None:
        return None
    try:
        raw = _client.get(_k(key))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set_json(key: str, value, ttl: int = _TTL) -> None:
    if _client is None:
        return
    try:
        _client.set(_k(key), json.dumps(value), ex=ttl)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Numpy array cache (point clouds). Uses .npy bytes, no pickle.
# --------------------------------------------------------------------------- #
def get_array(key: str):
    if _client is None:
        return None
    try:
        raw = _client.get(_k(key))
        if raw is None:
            return None
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        return None


def set_array(key: str, arr: np.ndarray, ttl: int = _TTL) -> None:
    if _client is None or arr is None:
        return
    try:
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
        _client.set(_k(key), buf.getvalue(), ex=ttl)
    except Exception:
        pass


def get_labeled_cloud(path):
    """Return (points, labels) or None on miss. labels may be None."""
    sig = file_signature(path)
    base = "lc:" + hashlib.md5(sig.encode()).hexdigest()
    pts = get_array(base + ":p")
    if pts is None:
        return None
    lab = get_array(base + ":l")
    labels = lab.astype(object) if lab is not None else None
    return pts, labels


def set_labeled_cloud(path, points, labels, ttl: int = _TTL) -> None:
    sig = file_signature(path)
    base = "lc:" + hashlib.md5(sig.encode()).hexdigest()
    set_array(base + ":p", np.asarray(points, dtype=np.float64), ttl)
    if labels is not None:
        # store as fixed-width unicode array so np.save needs no pickle
        set_array(base + ":l", np.asarray(labels, dtype=str), ttl)


_PIPELINE_ARG_KEYS = (
    "voxel", "fine_voxel", "knn", "std_ratio", "angle_step_deg", "fine_step_deg",
    "offset_fraction", "plane_trim", "score_sample", "normal_angle_deg",
    "icp_iterations", "icp_trim", "icp_tolerance", "max_correspondence_distance",
    "seed",
)


def pipeline_key(input_path, args) -> str:
    params = {k: getattr(args, k, None) for k in _PIPELINE_ARG_KEYS}
    blob = json.dumps(
        {"sig": file_signature(input_path), "params": params},
        sort_keys=True,
    )
    return "pipe:" + hashlib.md5(blob.encode()).hexdigest()
