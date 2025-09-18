# src/infer_and_measure.py
import cv2, numpy as np
from ultralytics import YOLO

MODEL_PATH = "C://Hackathon-AgriAI//runs//detect//flagleaf_det4//weights//best.pt"  # after training
model = YOLO(MODEL_PATH)

def detect_and_measure(image_path, gsd_cm_per_px):
    """
    Returns:
      results: list of dicts for each detected leaf with keys:
        bbox (x1,y1,x2,y2), confidence, mask (binary numpy), length_cm, width_cm, area_cm2
    """
    image = cv2.imread(image_path)
    res = model.predict(source=image, conf=0.25, imgsz=1024, verbose=False)[0]
    out = []
    h, w = image.shape[:2]
    for i, box in enumerate(res.boxes.data.tolist()):
        # box format: [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # mask if segmentation model
        mask = None
        if hasattr(res, 'masks') and res.masks is not None:
            # masks.data is list of boolean masks aligned with boxes
            mask = res.masks.data[i].cpu().numpy().astype('uint8')  # HxW mask
            # Crop mask to bbox for faster measurement
            crop_mask = mask[y1:y2+1, x1:x2+1]
        else:
            # fallback: use bounding box as mask
            crop_mask = np.ones((y2-y1+1, x2-x1+1), dtype='uint8')

        # compute pixel length/width: approximate length = max bbox side in px
        length_px = max(crop_mask.shape)
        width_px  = min(crop_mask.shape)
        # Better: skeletonize mask and measure major axis — for brevity we approximate
        length_cm = length_px * gsd_cm_per_px
        width_cm  = width_px  * gsd_cm_per_px

        # area in pixels
        area_px = (crop_mask > 0).sum()
        area_cm2 = area_px * (gsd_cm_per_px**2)

        out.append({
            "bbox": (x1,y1,x2,y2),
            "conf": float(conf),
            "length_cm": float(length_cm),
            "width_cm": float(width_cm),
            "area_cm2": float(area_cm2),
            "mask": crop_mask
        })
    return out
