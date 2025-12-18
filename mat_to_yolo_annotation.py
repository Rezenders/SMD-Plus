#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# -----------------------------
# Hardcode paths + image size
# -----------------------------
MAT_DIRS = [Path("/datasets/smd_plus/VIS_Onshore/ObjectGT"),
            Path("/datasets/smd_plus/VIS_Onboard/ObjectGT")]
OUT_ROOT = Path("/datasets/smd_plus/labels")

TRAINING_VIDEOS_PREFIXES = [
    'MVI_1451', 'MVI_1452', 'MVI_1470', 'MVI_1471', 'MVI_1478', 'MVI_1479',
    'MVI_1481', 'MVI_1482', 'MVI_1483', 'MVI_1484', 'MVI_1485', 'MVI_1486',
    'MVI_1578', 'MVI_1582', 'MVI_1583', 'MVI_1584', 'MVI_1609', 'MVI_1610',
    'MVI_1612', 'MVI_1617', 'MVI_1619', 'MVI_1620', 'MVI_1622', 'MVI_1623',
    'MVI_1624', 'MVI_1625', 'MVI_1626', 'MVI_1627', 'MVI_0788', 'MVI_0789',
    'MVI_0790', 'MVI_0792', 'MVI_0794', 'MVI_0795', 'MVI_0796', 'MVI_0797',
    'MVI_0801' ]

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

MAT_VAR = "structXML"  # in your example file


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


def main():
    if IMG_W <= 0 or IMG_H <= 0:
        raise ValueError("Set IMG_W and IMG_H to the frame resolution.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_path = OUT_ROOT / 'train'
    val_path = OUT_ROOT / 'val'

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    classes_txt = OUT_ROOT / "classes.txt"
    class_names = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]
    classes_txt.write_text("\n".join(class_names) + "\n")

    mat_files = []
    for MAT_DIR in MAT_DIRS:
        mat_files.extend(MAT_DIR.glob("*.mat"))
    mat_files = sorted(mat_files)

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {MAT_DIRS}")

    frame_id = 0
    skipped_count = 0
    skipped_files = []
    for mat_path in mat_files:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        records = mat[MAT_VAR]

        # records is typically an object array of MATLAB structs
        if isinstance(records, np.ndarray):
            records_iter = records.ravel()
        else:
            records_iter = [records]

        out_dir = train_path if any(mat_path.stem.startswith(prefix) for prefix in TRAINING_VIDEOS_PREFIXES) else val_path

        n_frames = 0
        n_boxes = 0
        for rec in records_iter:
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
            frame_id += 1

        print(f"[OK] {mat_path.name} -> {out_dir} | frames: {n_frames}, boxes: {n_boxes}")

    print(f"[OK] Wrote classes to: {classes_txt}")
    print(f"total frames processed: {frame_id}")
    # print(f"[INFO] Skipped {skipped_count} boxes in total.")
    # for fname, fid in skipped_files:
    #     print(f"       - {fname}, frame ID: {fid}")


if __name__ == "__main__":
    main()
