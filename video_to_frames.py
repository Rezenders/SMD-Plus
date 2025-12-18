#!/usr/bin/env python3
from pathlib import Path
import cv2

VIDEO_DIRS = [Path("/datasets/smd_plus/VIS_Onboard/Videos"),
              Path("/datasets/smd_plus/VIS_Onshore/Videos")]
OUT_ROOT  = Path("/datasets/smd_plus/images")

TRAINING_VIDEOS_PREFIXES = [
    'MVI_1451', 'MVI_1452', 'MVI_1470', 'MVI_1471', 'MVI_1478', 'MVI_1479',
    'MVI_1481', 'MVI_1482', 'MVI_1483', 'MVI_1484', 'MVI_1485', 'MVI_1486',
    'MVI_1578', 'MVI_1582', 'MVI_1583', 'MVI_1584', 'MVI_1609', 'MVI_1610',
    'MVI_1612', 'MVI_1617', 'MVI_1619', 'MVI_1620', 'MVI_1622', 'MVI_1623',
    'MVI_1624', 'MVI_1625', 'MVI_1626', 'MVI_1627', 'MVI_0788', 'MVI_0789',
    'MVI_0790', 'MVI_0792', 'MVI_0794', 'MVI_0795', 'MVI_0796', 'MVI_0797',
    'MVI_0801' ]

def extract_all_frames(video_path: Path, out_dir: Path, frame_start_id: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_id = frame_start_id
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = out_dir / f"{frame_id:06d}.png"
        ok = cv2.imwrite(str(out_path), frame)  # PNG is lossless by default
        if not ok:
            raise RuntimeError(f"Failed to write frame: {out_path}")

        frame_id += 1

    cap.release()
    print(f"[OK] {video_path.name}: saved {frame_id} frames to {out_dir}")
    return frame_id


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_path = OUT_ROOT / 'train'
    val_path = OUT_ROOT / 'val'

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    videos = []
    for VIDEO_DIR in VIDEO_DIRS:
        videos.extend(VIDEO_DIR.glob("*.avi"))
    videos.sort()
    if not videos:
        raise FileNotFoundError(f"No .avi videos found in: {VIDEO_DIR}")

    frame_start_id = 0
    for vp in videos:
        out_dir = train_path if any(vp.stem.startswith(prefix) for prefix in TRAINING_VIDEOS_PREFIXES) else val_path
        print(f"[INFO] Processing {vp.name}: saving frames to {out_dir} set.")
        frame_start_id = extract_all_frames(vp, out_dir, frame_start_id)


if __name__ == "__main__":
    main()
