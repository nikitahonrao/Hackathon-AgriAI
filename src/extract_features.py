# src/extract_features.py
import os, glob
import pandas as pd
from infer_and_measure import detect_and_measure  # your detector
from tqdm import tqdm

def features_for_image(image_path, gsd=0.2):
    dets = detect_and_measure(image_path, gsd)
    if len(dets) == 0:
        return {
            "leaf_count": 0,
            "mean_area": 0,
            "total_area": 0,
            "mean_length": 0,
            "mean_width": 0
        }
    areas = [d['area_cm2'] for d in dets]
    lengths = [d['length_cm'] for d in dets]
    widths = [d['width_cm'] for d in dets]
    return {
        "leaf_count": len(dets),
        "mean_area": float(sum(areas)/len(areas)),
        "total_area": float(sum(areas)),
        "mean_length": float(sum(lengths)/len(lengths)),
        "mean_width": float(sum(widths)/len(widths))
    }

def run_extract(data_dir="../global-wheat-detection/train", out_csv="../image_features.csv", gsd=0.2):
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    rows = []
    for p in tqdm(image_paths, desc="Extracting features"):
        img_id = os.path.splitext(os.path.basename(p))[0]  # save without extension
        feats = features_for_image(p, gsd)
        feats.update({"image_id": img_id})
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("✅ Saved features to", out_csv)


if __name__ == "__main__":
    run_extract()
