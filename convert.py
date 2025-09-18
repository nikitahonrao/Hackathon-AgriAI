import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
ROOT = r"C:\Hackathon-AgriAI\global-wheat-detection"
IMG_DIR = os.path.join(ROOT, "train")   # where original images are stored
CSV_PATH = os.path.join(ROOT, "train.csv")

# Output dirs
OUT_TRAIN_IMG = os.path.join(ROOT, "images", "train")
OUT_TRAIN_LBL = os.path.join(ROOT, "labels", "train")
OUT_VAL_IMG = os.path.join(ROOT, "images", "val")
OUT_VAL_LBL = os.path.join(ROOT, "labels", "val")

os.makedirs(OUT_TRAIN_IMG, exist_ok=True)
os.makedirs(OUT_TRAIN_LBL, exist_ok=True)
os.makedirs(OUT_VAL_IMG, exist_ok=True)
os.makedirs(OUT_VAL_LBL, exist_ok=True)

# Load annotations
df = pd.read_csv(CSV_PATH)

# Helper: convert bbox → YOLO format
def convert_bbox(width, height, bbox):
    x, y, w, h = bbox
    x_c = (x + w / 2) / width
    y_c = (y + h / 2) / height
    w = w / width
    h = h / height
    return x_c, y_c, w, h

# Get all unique images
all_images = df["image_id"].unique()
train_ids, val_ids = train_test_split(all_images, test_size=0.2, random_state=42)

# Process images
for img_set, ids, out_img_dir, out_lbl_dir in [
    ("train", train_ids, OUT_TRAIN_IMG, OUT_TRAIN_LBL),
    ("val", val_ids, OUT_VAL_IMG, OUT_VAL_LBL),
]:
    for img_id in ids:
        img_file = os.path.join(IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_file):
            continue

        # Copy image
        shutil.copy(img_file, os.path.join(out_img_dir, f"{img_id}.jpg"))

        # Get labels
        rows = df[df["image_id"] == img_id]
        label_path = os.path.join(out_lbl_dir, f"{img_id}.txt")

        with open(label_path, "w") as f:
            for _, row in rows.iterrows():
                bbox = [float(x) for x in row["bbox"].strip("[]").split(",")]
                # YOLO expects: class x_center y_center w h
                # Assume class 0 (flagleaf/wheat head)
                x_c, y_c, w, h = convert_bbox(row["width"], row["height"], bbox)
                f.write(f"0 {x_c} {y_c} {w} {h}\n")

print("✅ Dataset converted and split into YOLO format!")
