#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# -----------------------------
# Hardcode paths + image size
# -----------------------------
# MAT_DIR = Path("/datasets/smd_plus/VIS_Onboard/ObjectGT")
MAT_DIR = Path("/datasets/smd_plus/VIS_Onshore/ObjectGT")
OUTPUT_DIR = Path("/datasets/smd_plus/labels")    # output root

IMG_W = 1920  # set to your frame width
IMG_H = 1080  # set to your frame height

# MATLAB rectangle is often 1-based (x,y start at 1). Keep True unless you confirm 0-based.
SUBTRACT_ONE = True

# -----------------------------
# Fixed classes (0-based IDs)
# -----------------------------
CLASS_MAP = {
    "Ferry": 0,
    "Buoy": 1,
    "Vessel/ship": 2,
    "Boat": 3,
    "Kayak": 4,
    "Sailboat": 5,
    "Other": 6,
}

MAT_VAR_PREFERRED = "structXML"  # in your example file


def unwrap_scalar(v):
    """Unwrap nested 1-element arrays/cells into a Python scalar."""
    while isinstance(v, np.ndarray) and v.size == 1:
        v = v.item()
    return v


def to_str(v) -> str:
    v = unwrap_scalar(v)
    if v is None:
        return ""
    return str(v)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def yolo_line_from_xywh(x, y, w, h, class_id: int) -> str:
    xc = (x + w / 2.0) / IMG_W
    yc = (y + h / 2.0) / IMG_H
    wn = w / IMG_W
    hn = h / IMG_H
    xc, yc, wn, hn = map(clamp01, (xc, yc, wn, hn))
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def pick_mat_var(mat: dict) -> str:
    if MAT_VAR_PREFERRED in mat:
        return MAT_VAR_PREFERRED
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise RuntimeError("No usable variables found in .mat file.")
    return keys[0]


def main():
    if IMG_W <= 0 or IMG_H <= 0:
        raise ValueError("Set IMG_W and IMG_H to the frame resolution.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: save classes for your YOLO data.yaml
    classes_txt = OUTPUT_DIR / "classes.txt"
    class_names = [name for name, idx in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]
    classes_txt.write_text("\n".join(class_names) + "\n")

    mat_files = sorted(MAT_DIR.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {MAT_DIR}")

    for mat_path in mat_files:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        var = pick_mat_var(mat)
        records = mat[var]

        # records is typically an object array of MATLAB structs
        if isinstance(records, np.ndarray):
            records_iter = records.ravel()
        else:
            records_iter = [records]

        out_dir = OUTPUT_DIR / mat_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        n_frames = 0
        n_boxes = 0
        for frame_id, rec in enumerate(records_iter):
            bb = np.array(getattr(rec, "BB"))
            bb = bb.reshape(-1, 4) if bb.size else bb.reshape(0, 4)

            obj_types = getattr(rec, "ObjectType", [])
            obj_types = np.array(obj_types).ravel() if isinstance(obj_types, np.ndarray) else [obj_types]

            lines = []
            for i in range(bb.shape[0]):
                x, y, w, h = map(float, bb[i].tolist())

                if SUBTRACT_ONE:
                    x -= 1.0
                    y -= 1.0

                if w <= 1.0 or h <= 1.0:
                    continue

                label = to_str(obj_types[i] if i < len(obj_types) else "")
                if label not in CLASS_MAP:
                    continue

                lines.append(yolo_line_from_xywh(x, y, w, h, CLASS_MAP[label]))
                n_boxes += 1

            (out_dir / f"{frame_id:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_frames += 1

        print(f"[OK] {mat_path.name} -> {out_dir} | frames: {n_frames}, boxes: {n_boxes}")

    print(f"[OK] Wrote classes to: {classes_txt}")


if __name__ == "__main__":
    main()
