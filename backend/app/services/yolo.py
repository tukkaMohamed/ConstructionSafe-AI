from pathlib import Path

from ultralytics import YOLO


class YOLOService:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.30,
    ):
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold

        print("Loading YOLO model...")

        self.model = YOLO(self.model_path)

        print(
            f"YOLO service ready. "
            f"Model: {self.model_path}"
        )

    def detect(self, image_path: Path) -> list[dict]:
        """
        Run YOLO inference on an image and return
        structured detections.
        """

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            names = result.names

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = names[class_id]

                xyxy = box.xyxy[0].tolist()

                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [
                            round(value, 2)
                            for value in xyxy
                        ],
                    }
                )

        return detections

    def build_safety_context(
        self,
        detections: list[dict],
    ) -> str:
        """
        Convert YOLO detections into a textual context
        that can be provided to the LLM.
        """

        if not detections:
            return (
                "Computer vision observations from the uploaded "
                "image: No objects were detected above the "
                "confidence threshold."
            )

        counts = {}

        for detection in detections:
            class_name = detection["class_name"]
            counts[class_name] = counts.get(class_name, 0) + 1

        lines = [
            "Computer vision observations from the uploaded image:"
        ]

        for class_name, count in counts.items():
            lines.append(
                f"- {class_name}: {count} detection(s)"
            )

        lines.append(
            "\nImportant: These are visual detections only. "
            "They should not automatically be interpreted as "
            "proof of compliance or non-compliance with a "
            "safety requirement."
        )

        return "\n".join(lines)