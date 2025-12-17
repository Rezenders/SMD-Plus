#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ----------------------------
# Configuration (hardcode here)
# ----------------------------
INPUT_DIR = Path("/datasets/smd_plus/VIS_Onboard/ObjectGT")  # <- change this
OUTPUT_DIR = INPUT_DIR                   # write CSVs next to MATs


# ----------------------------
# Helpers
# ----------------------------
def _is_internal_key(k: str) -> bool:
    return k.startswith("__")


def _unwrap(x):
    """Unwrap scipy/matlab containers (singletons, nested object arrays)."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        if x.size == 1:
            return _unwrap(x.item())
        return x
    return x


def _to_str(x):
    x = _unwrap(x)
    if x is None:
        return None
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray):
        # array of strings/cells
        if x.dtype == object:
            return [_to_str(v) for v in x.ravel().tolist()]
        if x.dtype.kind in ("U", "S"):
            return [str(v) for v in x.ravel().tolist()]
        return x
    return str(x)


def _to_1d_list(x, n_expected: int | None = None):
    """Convert MATLAB field to a Python list (length n_expected if given)."""
    x = _unwrap(x)
    if x is None:
        out = []
    elif isinstance(x, np.ndarray):
        if x.dtype == object:
            out = [_to_str(v) for v in x.ravel().tolist()]
        else:
            out = [v.item() if hasattr(v, "item") else v for v in x.ravel().tolist()]
    else:
        out = [_to_str(x)] if isinstance(_to_str(x), str) else [x]

    if n_expected is None:
        return out
    if len(out) == 0:
        return [None] * n_expected
    if len(out) == 1 and n_expected > 1:
        return out * n_expected
    if len(out) < n_expected:
        out = out + [None] * (n_expected - len(out))
    return out[:n_expected]


def _normalize_bb(bb):
    """
    BB is [x, y, w, h] per object.
    Return shape (K,4) float32.
    """
    bb = _unwrap(bb)
    if bb is None:
        return np.zeros((0, 4), dtype=np.float32)
    bb = np.array(bb)
    if bb.ndim == 1 and bb.shape[0] == 4:
        return bb.reshape(1, 4).astype(np.float32)
    if bb.ndim == 2 and bb.shape[1] == 4:
        return bb.astype(np.float32)
    if bb.ndim == 2 and bb.shape[0] == 4:
        return bb.T.astype(np.float32)
    return np.zeros((0, 4), dtype=np.float32)


def _iter_records(obj):
    """Iterate MATLAB struct array / structured array / object array."""
    # MATLAB struct object (scipy) has _fieldnames
    if hasattr(obj, "_fieldnames"):
        yield obj
        return

    # Structured ndarray
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        for rec in obj.ravel():
            yield rec
        return

    # Object ndarray (possibly structs)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for rec in obj.ravel():
            if rec is not None:
                yield rec
        return

    yield obj


def _get_field(rec, name: str):
    if rec is None:
        return None
    if hasattr(rec, "_fieldnames") and name in rec._fieldnames:
        return getattr(rec, name)
    if isinstance(rec, np.void) and rec.dtype.names and name in rec.dtype.names:
        return rec[name]
    if isinstance(rec, dict) and name in rec:
        return rec[name]
    return None


def _choose_annotation_key(mat: dict) -> str:
    """Pick the variable that most likely contains the annotations."""
    keys = [k for k in mat.keys() if not _is_internal_key(k)]
    if not keys:
        raise RuntimeError("No non-internal variables found in .mat")

    # Prefer variables whose records contain a 'BB' field
    candidates = []
    for k in keys:
        v = mat[k]
        try:
            rec0 = next(_iter_records(v))
            has_bb = _get_field(rec0, "BB") is not None
        except Exception:
            has_bb = False
        size = v.size if isinstance(v, np.ndarray) else 1
        candidates.append((has_bb, size, k))

    # sort: has_bb first, then largest size
    candidates.sort(reverse=True)
    return candidates[0][2]


def mat_to_dataframe(mat_path: Path) -> pd.DataFrame:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    key = _choose_annotation_key(mat)
    ann = mat[key]

    rows = []
    for frame_id, rec in enumerate(_iter_records(ann)):
        bb = _normalize_bb(_get_field(rec, "BB"))
        k = int(bb.shape[0])

        motion = _to_1d_list(_get_field(rec, "Motion"), k)
        obj = _to_1d_list(_get_field(rec, "Object"), k)
        dist = _to_1d_list(_get_field(rec, "Distance"), k)

        motion_type = _to_1d_list(_get_field(rec, "MotionType"), k)
        object_type = _to_1d_list(_get_field(rec, "ObjectType"), k)
        distance_type = _to_1d_list(_get_field(rec, "DistanceType"), k)

        for obj_id in range(k):
            x, y, w, h = bb[obj_id].tolist()
            rows.append(
                {
                    "frame_id": frame_id,
                    "obj_id": obj_id,
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "Motion": motion[obj_id],
                    "Object": obj[obj_id],
                    "Distance": dist[obj_id],
                    "MotionType": motion_type[obj_id],
                    "ObjectType": object_type[obj_id],
                    "DistanceType": distance_type[obj_id],
                }
            )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(INPUT_DIR.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {INPUT_DIR}")

    for mat_path in mat_files:
        df = mat_to_dataframe(mat_path)
        out_csv = OUTPUT_DIR / (mat_path.stem + ".csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] {mat_path.name} -> {out_csv.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
