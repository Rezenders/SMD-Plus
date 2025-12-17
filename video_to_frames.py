#!/usr/bin/env python3
from pathlib import Path
import cv2

# VIDEO_DIR = Path("/datasets/smd_plus/VIS_Onboard/Videos")
VIDEO_DIR = Path("/datasets/smd_plus/VIS_Onshore/Videos")
OUT_ROOT  = Path("/datasets/smd_plus/images")   # output: OUT_ROOT/<video_stem>/000000.png ...


def extract_all_frames(video_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_id = 0
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


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.avi"))
    if not videos:
        raise FileNotFoundError(f"No .avi videos found in: {VIDEO_DIR}")

    for vp in videos:
        extract_all_frames(vp, OUT_ROOT / vp.stem)


if __name__ == "__main__":
    main()
