import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import os
import pandas as pd
from src.infer_and_measure import detect_and_measure
from src.wheather_utils import get_weather   # ✅ fixed spelling

st.set_page_config(layout="wide", page_title="🌾 Flag Leaf Measurement & Yield Prediction")
st.title("🌾🌾Flag Leaf Measurement & Yield Prediction (Wheat)🌾🌾")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Image & Location")
    location = st.text_input("Location (name/coords):", "Unknown location")
    gsd = st.number_input(
        "GSD (cm/pixel)", min_value=0.01, value=0.2, step=0.01,
        help="Ground Sampling Distance (cm per pixel). Provide correct GSD for accurate measurements."
    )
    uploaded = st.file_uploader(
        "Upload image", type=['jpg','jpeg','png'], accept_multiple_files=False
    )

col1, col2 = st.columns([1,1])

# ---------------------------
# Process uploaded image
# ---------------------------
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img_arr = np.array(img)
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Save temporarily
    temp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded.name)
    img.save(temp_path)

    st.info("Running detection... (this may take a few secs)")
    results = detect_and_measure(temp_path, gsd)

    # ---------------------------
    # Draw boxes and overlay masks
    # ---------------------------
    vis = img_arr.copy()
    for idx, r in enumerate(results, 1):
        x1, y1, x2, y2 = r["bbox"]
        color = tuple(np.random.randint(0,255,3).tolist())  # random color per leaf
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if r.get("mask") is not None:
            mask = r["mask"]
            mh, mw = mask.shape
            roi = vis[y1:y1+mh, x1:x1+mw]
            if roi.shape[:2] != mask.shape:
                mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
            overlay = np.zeros_like(roi)
            overlay[mask > 0] = color
            alpha = 0.35
            vis[y1:y1+overlay.shape[0], x1:x1+overlay.shape[1]] = cv2.addWeighted(
                roi, 1 - alpha, overlay, alpha, 0
            )

    st.subheader("Detections (overlay)")
    st.image(vis, use_container_width=True)

    # ---------------------------
    # Measurements summary
    # ---------------------------
    st.subheader("Measurements")
    if len(results) == 0:
        st.warning("No flag leaves detected.")
    else:
        rows = []
        for i, r in enumerate(results, 1):
            rows.append({
                "leaf": i,
                "length_cm": round(r["length_cm"], 2),
                "width_cm": round(r["width_cm"], 2),
                "area_cm2": round(r["area_cm2"], 2),
                "confidence": round(r["conf"], 3)
            })
        df = pd.DataFrame(rows)
        st.table(df)
        st.success(f"Detected {len(results)} flag leaves")

        # ---------------------------
        # Leaf feature extraction
        # ---------------------------
        total_area = df["area_cm2"].sum()
        mean_area = df["area_cm2"].mean()
        mean_length = df["length_cm"].mean()
        mean_width = df["width_cm"].mean()
        leaf_count = len(results)

        feat_dict = {
            "leaf_count": leaf_count,
            "mean_area": mean_area,
            "total_area": total_area,
            "mean_length": mean_length,
            "mean_width": mean_width
        }

        # ---------------------------
        # Add weather features
        # ---------------------------
     #   weather_data = get_weather(location)
     #   if weather_data:
      #      st.subheader("🌦 Weather Data")
      #      st.json(weather_data)
      #      feat_dict.update({
       #         "temp_c": weather_data.get("temp_c", 0),
       #         "humidity": weather_data.get("humidity", 0),
      #        "rainfall": weather_data.get("rainfall", 0)
      #      })

        feat_df = pd.DataFrame([feat_dict])

        # ---------------------------
        # Load model & predict yield
        # ---------------------------
        # ---------------------------
if 'feat_df' in locals():
    # Try multiple possible locations for yield_model.pkl
    possible_paths = [
        os.path.join(os.getcwd(), "models", "yield_model.pkl"),           # root folder
        os.path.join(os.path.dirname(__file__), "..", "models", "yield_model.pkl"),  # relative path
        os.path.join(os.path.dirname(__file__), "models", "yield_model.pkl")         # inside src/
    ]

    model_path = None
    for path in possible_paths:
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            model_path = norm_path
            break

    if model_path:
        model = joblib.load(model_path)
        pred = model.predict(feat_df)[0]
        st.metric("🌾 Predicted Yield", f"{pred:.2f} kg/ha")
    else:
        st.info("⚠️ Yield model not found. Run training (src/train_yield_model.py) to create models/yield_model.pkl")
else:
    st.warning("Cannot predict yield because no flag leaves were detected.")