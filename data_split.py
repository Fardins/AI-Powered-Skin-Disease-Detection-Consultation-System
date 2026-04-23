import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import random
from pathlib import Path

from config import DATASET_PATH, OUTPUT_PATH, SPLIT_RATIO


def split_dataset():
    print("Dataset Path:", DATASET_PATH)

    # Create folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_PATH, split), exist_ok=True)

    # Loop over each class folder
    for class_name in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(class_path):
            continue

        # 🔥 Recursive image loading (important for local)
        images = list(Path(class_path).rglob("*.*"))
        images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        print(f"{class_name}: {len(images)} images")

        if len(images) == 0:
            continue

        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class folders
        os.makedirs(os.path.join(OUTPUT_PATH, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "val", class_name), exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy2(
                str(img),
                os.path.join(OUTPUT_PATH, "train", class_name, os.path.basename(img))
            )

        for img in val_images:
            shutil.copy2(
                str(img),
                os.path.join(OUTPUT_PATH, "val", class_name, os.path.basename(img))
            )

    print("✅ Dataset split completed!")


# ✅ Run directly
if __name__ == "__main__":
    split_dataset()