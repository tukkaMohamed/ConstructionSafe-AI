from ultralytics import YOLO
from pathlib import Path


def main():
    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = (
        BASE_DIR
        / "runs"
        / "detect"
        / "runs"
        / "construction_yolo11n"
        / "weights"
        / "best.pt"
    )

    DATA_YAML = BASE_DIR / "data.yaml"

    print("Loading trained YOLO11n model...")
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded from: {MODEL_PATH}")

    print("\nStarting evaluation on TEST set...\n")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project="runs",
        name="test_evaluation",
        exist_ok=True
    )

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)

    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    print("\nPer-class mAP50:")

    for class_name, map50 in zip(
        model.names.values(),
        metrics.box.ap50
    ):
        print(f"{class_name:20s}: {map50:.4f}")

    print("=" * 50)
    print("Evaluation completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()