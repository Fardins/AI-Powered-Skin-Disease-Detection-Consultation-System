from ultralytics import YOLO
from config import TRAINED_MODEL_PATH

class Predictor:
    def __init__(self, model_path=TRAINED_MODEL_PATH):
        self.model = YOLO(model_path)

    def predict(self, image_path):
        results = self.model.predict(
            source=image_path,
            save=False,
            imgsz=224,
            conf=0.25
        )

        probs = results[0].probs

        class_idx = probs.top1
        confidence = probs.top1conf.item()

        raw_name = self.model.names[class_idx]

        # 🔥 Clean disease name
        disease_name = raw_name.split(" - ")[0]
        disease_name = disease_name.split(". ", 1)[-1]
        disease_name = disease_name.replace("pictures", "").strip()

        return disease_name, confidence
