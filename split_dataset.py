from pathlib import Path
import random
import shutil
import yaml

# =========================
# SETTINGS
# =========================

DATASET_DIR = Path(__file__).resolve().parent

SOURCE_IMAGES = DATASET_DIR / "train" / "images"
SOURCE_LABELS = DATASET_DIR / "train" / "labels"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

# =========================
# CHECK DATASET
# =========================

if not SOURCE_IMAGES.exists():
    raise FileNotFoundError(f"Images folder not found: {SOURCE_IMAGES}")

if not SOURCE_LABELS.exists():
    raise FileNotFoundError(f"Labels folder not found: {SOURCE_LABELS}")

# Get all images
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in SOURCE_IMAGES.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

print(f"Found {len(images)} images.")

if len(images) == 0:
    raise RuntimeError("No images found!")

# =========================
# SHUFFLE
# =========================

random.seed(SEED)
random.shuffle(images)

total = len(images)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

print(f"Train: {len(train_images)}")
print(f"Val:   {len(val_images)}")
print(f"Test:  {len(test_images)}")

# =========================
# CREATE FOLDERS
# =========================

for split in ["train", "val", "test"]:
    (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# =========================
# MOVE FILES
# =========================

def move_split(image_list, split_name):
    for image_path in image_list:

        # Corresponding YOLO label
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

        destination_image = (
            DATASET_DIR / split_name / "images" / image_path.name
        )

        destination_label = (
            DATASET_DIR / split_name / "labels" / f"{image_path.stem}.txt"
        )

        shutil.move(str(image_path), str(destination_image))

        if label_path.exists():
            shutil.move(str(label_path), str(destination_label))
        else:
            print(f"WARNING: Label not found for {image_path.name}")

# Move validation and test first
# Then leave the remaining images as train

move_split(val_images, "val")
move_split(test_images, "test")

# =========================
# UPDATE YAML
# =========================

yaml_path = DATASET_DIR / "data.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

data["train"] = "train/images"
data["val"] = "val/images"
data["test"] = "test/images"

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

# =========================
# FINAL RESULT
# =========================

remaining_train = len(list(
    (DATASET_DIR / "train" / "images").glob("*")
))

print("\n==============================")
print("DATASET SPLIT COMPLETED")
print("==============================")
print(f"Train images: {remaining_train}")
print(f"Val images:   {len(val_images)}")
print(f"Test images:  {len(test_images)}")
print("==============================")
print("data.yaml updated successfully!")