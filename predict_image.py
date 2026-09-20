from ultralytics import YOLO
from pathlib import Path


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

IMAGE_PATH = BASE_DIR / "test" / "images"

print("Loading trained YOLO11n model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")
print("Running prediction on test images...\n")

results = model.predict(
    source=str(IMAGE_PATH),
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    project="runs",
    name="test_predictions",
    exist_ok=True
)

print("\nPrediction completed!")
print("Results saved to:")
print(BASE_DIR / "runs" / "detect" / "test_predictions")