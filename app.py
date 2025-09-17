# ====================================
# Streamlit App: Flag Leaf Measurement & Yield Prediction
# ====================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
import tempfile

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Flag Leaf Yield Prediction", layout="wide")
st.title("🌾 AI-Assisted Flag Leaf Measurement & Yield Prediction")
st.write("Upload UAV crop field images to detect flag leaves and predict yield.")

# Load YOLO model (use pretrained or your fine-tuned one)
model = YOLO("C:/HACKATHON-AGRIAI/global-wheat_dataset/runs/detect/train/weights/best.pt")
   # replace with your trained model file if available

# Load regression dataset (replace with your real dataset)
# Must have columns: Width_cm, Length_cm, Area_cm2, Yield
try:
    df_train = pd.read_csv("leaf_yield_dataset.csv")
    X = df_train[["Width_cm","Length_cm","Area_cm2"]]
    y = df_train["Yield"]
    reg_model = LinearRegression().fit(X, y)
    st.success("✅ Regression model trained successfully from dataset.")
except Exception as e:
    st.warning("⚠️ No dataset found. Using demo model.")
    data = {
        "Width_cm": np.random.uniform(1,5,20),
        "Length_cm": np.random.uniform(5,20,20),
        "Area_cm2": np.random.uniform(20,100,20),
        "Yield": np.random.uniform(2,6,20)  # tons/hectare
    }
    df_train = pd.DataFrame(data)
    X = df_train[["Width_cm","Length_cm","Area_cm2"]]
    y = df_train["Yield"]
    reg_model = LinearRegression().fit(X, y)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload UAV Image", type=["jpg","png","jpeg"])

# Ground Sampling Distance (metadata)
GSD = st.number_input("Ground Sampling Distance (cm/pixel)", value=0.2, step=0.01)

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img_path = tfile.name
    
    # Read image
    img = cv2.imread(img_path)
    
    # Run YOLO detection
    results = model.predict(img, conf=0.25)
    
    leaf_data = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Convert pixels → cm
        width_cm = float(width_px) * GSD
        height_cm = float(height_px) * GSD
        area_cm2 = width_cm * height_cm
        
        leaf_data.append([width_cm, height_cm, area_cm2])
        
        # Draw box + text
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, f"L:{height_cm:.1f}cm W:{width_cm:.1f}cm",
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 2)
    
    # Show annotated image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Leaves", use_column_width=True)
    
    if leaf_data:
        df = pd.DataFrame(leaf_data, columns=["Width_cm","Length_cm","Area_cm2"])
        st.write("📏 Leaf Measurements (cm):")
        st.dataframe(df)
        
        # Predict yield
        X_new = df[["Width_cm","Length_cm","Area_cm2"]]
        yield_pred = reg_model.predict(X_new)
        st.success(f"🌾 Predicted Yield: {np.mean(yield_pred):.2f} tons/hectare")
    else:
        st.warning("No leaves detected. Try another image.")
