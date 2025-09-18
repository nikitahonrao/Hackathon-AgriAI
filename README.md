<h1>🌾 AI-Assisted Flag Leaf Measurement & Yield Prediction</h1>

This project leverages YOLOv8 and Machine Learning to automate flag leaf detection and assist in crop yield prediction for wheat. By analyzing UAV/drone field images, our system detects wheat flag leaves and provides insights that help farmers and researchers make informed decisions about crop productivity.

<h4>🚀 Features</h4>

Flag Leaf Detection: Uses YOLOv8 object detection to identify wheat flag leaves in UAV images.

Automated Measurement: Calculates flag leaf size and area from bounding boxes.

Yield Prediction: Trains a regression model to predict wheat yield based on extracted features.

Streamlit Web App: User-friendly interface to upload images, visualize detections, and get yield predictions.

<h4>📂 Project Structure </h4>
├── data/                 # Dataset (images + annotations)

│   ├── train/            # Training images

│   ├── val/              # Validation images

│   ├── labels/           # YOLO label files

│   └── train.csv         # Annotation file (image_id, bbox, etc.)

│
├── src/                  # Source code

│   ├── train_yolo.py     # Train YOLOv8 model

│   ├── split_data.py     # Split dataset into train/val

│   ├── train_yield_model.py # Train regression model for yield prediction

│   ├── inference.py      # Run YOLO detection + measurement

│   └── app.py            # Streamlit web app

│
├── models/               # Saved YOLO + ML models

├── results/              # Detection and training results

├── wheat.yaml            # Dataset config for YOLO

├── requirements.txt      # Dependencies

└── README.md             # Project documentation


<h4>⚙️ Installation</h4>
Clone this repository:

git clone https://github.com/nikitahonrao/Hackathon-AgriAI
cd flag-leaf-ai

Create a virtual environment & install dependencies:

pip install -r requirements.txt


<h4>🏋️ Training</h4>
1. Train YOLOv8 on Flag Leaf Dataset
yolo detect train data=wheat.yaml model=yolov8n.pt epochs=50 imgsz=640

2. Train Yield Prediction Model
python src/train_yield_model.py

<h4>📊 Running the App</h4>

Start the Streamlit App:

streamlit run src/app.py


Upload a UAV field image.

The app detects flag leaves, measures their size, and predicts yield.

<h4>📈 Example Results</h4>

Flag leaf detection visualization with bounding boxes.

Yield prediction chart comparing estimated vs actual yield.

<h4>🛠️ Tech Stack</h4>

YOLOv8 (Ultralytics)

Streamlit

OpenCV & NumPy

scikit-learn (Linear Regression)

Pandas

<h4>🎯 Use Cases</h4>

Agricultural research for wheat productivity.

Early yield prediction for farmers.

Precision agriculture & crop monitoring.

<h4>📌 Future Improvements</h4>

Support for multi-class detection (flag leaves, spikes, etc.).

Integration with satellite/UAV pipelines.

More advanced deep learning models for yield prediction.

<h4>👩‍💻 Contributors</h4>

Your Name – AI/ML Implementation, Streamlit App

Team members (if any)

<h4>📜 License</h4>

This project is licensed under the MIT License – feel free to use and modify.
