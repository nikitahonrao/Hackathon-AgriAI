# src/train_detector.py
# Train YOLOv8 segmentation on your flag leaf dataset (Ultralytics)
from ultralytics import YOLO
import os

# Create YAML file describing dataset (one-time). Example dataset.yaml:
# train: /full/path/to/data/train
# val: /full/path/to/data/val
# nc: 1
# names: ['flagleaf']

DATA_YAML = "C:\Hackathon-AgriAI\global-wheat-detection\dataset.yaml"  # create or replace path

# Model: start from yolov8n-seg for speed (or yolov8m-seg for more accuracy)
model = YOLO("yolov8n-seg.pt")  # pretrained seg model

# Train (adjust epochs, imgsz, batch)
model.train(data=DATA_YAML, epochs=50, imgsz=640, batch=8, name="flagleaf_seg")
# outputs will be in runs/detect/flagleaf_seg
