from ultralytics import YOLO
from pathlib import Path


def main():
    dataset_root = Path("/datasets/smd_plus/MVI_0788_VIS")
    data_yaml = dataset_root / "smd_plus_MVI_0788.yaml"  # must already exist

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model = YOLO("yolov5l6u.pt")  # pretrained weights

    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=1280,
        batch=2,
        device=0,
        workers=8,
    )


if __name__ == "__main__":
    main()
