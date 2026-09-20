from ultralytics import YOLO
from pathlib import Path
from collections import Counter


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

    IMAGE_PATH = (
        BASE_DIR
        / "test"
        / "images"
        / "-2025-03-31-23_49_47_png.rf.86734ad31d659d17fbeed1299be0f795.jpg"
    )

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("Running detection...\n")

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 3)
        })

    # Count detected objects
    counts = Counter(
        detection["class"]
        for detection in detections
    )

    # Safety-related detections
    safety_hazards = [
        detection
        for detection in detections
        if detection["class"] in [
            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest"
        ]
        and detection["confidence"] >= 0.50
    ]

    print("=" * 60)
    print("DETECTED OBJECTS")
    print("=" * 60)

    for class_name, count in counts.items():
        print(f"{class_name:20s}: {count}")

    print("\n" + "=" * 60)
    print("SAFETY-RELATED DETECTIONS")
    print("=" * 60)

    if safety_hazards:
        for hazard in safety_hazards:
            print(
                f"{hazard['class']:20s} "
                f"confidence={hazard['confidence']:.3f}"
            )
    else:
        print("No high-confidence PPE violations detected.")

    # Build RAG context
    detected_classes = ", ".join(
        f"{class_name} ({count})"
        for class_name, count in counts.items()
    )

    hazard_context = ", ".join(
        f"{hazard['class']} "
        f"(confidence {hazard['confidence']:.2f})"
        for hazard in safety_hazards
    )

    rag_context = f"""
Construction site visual analysis:

Detected objects:
{detected_classes}

Safety-related detections:
{hazard_context if hazard_context else "None detected with confidence >= 0.50"}

Use these visual detections as context.
Safety requirements must be retrieved from the provided safety documents.
Do not invent requirements that are not supported by the retrieved documents.
"""

    print("\n" + "=" * 60)
    print("RAG CONTEXT")
    print("=" * 60)
    print(rag_context)

    print("=" * 60)


if __name__ == "__main__":
    main()