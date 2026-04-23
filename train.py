from ultralytics import YOLO
from config import OUTPUT_PATH, MODEL_NAME, EPOCHS, IMG_SIZE, BATCH_SIZE

def train_model():
    print(" Training started...") 

    model = YOLO(MODEL_NAME)

    model.train(
        data=OUTPUT_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        task="classify",
        device="cpu"
    )

    print("✅ Training completed!")


# ✅ Run directly
if __name__ == "__main__":
    train_model()
