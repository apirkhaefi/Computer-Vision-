# Computer-Vision
Glass detection using Resnet18 + Yolov8 + and my own dataset 
---
# 🕶️ Profile-Aware Glasses Detection

A multi-stage deep learning pipeline for detecting whether a person in an image is wearing glasses, with **profile-aware thresholding** to improve robustness across different face angles.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Features](#key-features)
- [Model Details](#model-details)
- [Limitations & Future Work](#limitations--future-work)

---

## 🔍 Overview

This project implements an end-to-end **glasses detection system** that:

1. **Detects people** in an image using YOLOv8n
2. **Extracts faces** using MediaPipe Face Detection
3. **Classifies each face** as "wearing glasses" or "not wearing glasses" using a fine-tuned ResNet18
4. Applies a **profile-aware threshold** based on face aspect ratio to improve accuracy on side-profile faces

The system is designed to be robust across frontal, semi-profile, and profile face angles.

---

## 🏗️ Pipeline Architecture

```
Input Image
    │
    ▼
┌─────────────────┐
│   YOLOv8n       │  Person Detection
│   (Person Det.) │
└────────┬────────┘
         │  Bounding boxes (people)
         ▼
┌─────────────────┐
│   Crop Person   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MediaPipe     │  Face Detection
│   Face Det.     │
└────────┬────────┘
         │  Face crops
         ▼
┌─────────────────┐
│   Quality Gate  │  Reject small (<48px) or dark (mean <45) faces
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ResNet18      │  Glasses / No Glasses
│   Classifier    │  (2-class output)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Profile-Aware       │  FRONTAL_THRESH=0.65
│ Threshold Decision  │  PROFILE_THRESH=0.50
└─────────────────────┘
         │
         ▼
   Final Prediction

---

## 📁 Project Structure


glasses-detection/
├── models/
│   ├── resnet18_glasses.pth     # Trained ResNet18 weights
│   └── yolov8n.pt               # YOLOv8n pretrained weights
├── datasets/
│   └── images/
│       ├── train/
│       └── val/
│           ├── glasses/
│           └── noglasses/
├── train.py                     # Training script for ResNet18
├── inference.py                 # Full pipeline inference
├── requirements.txt
└── README.md

---

## ⚙️ Installation

bash
# Clone the repository
git clone https://github.com/yourusername/glasses-detection.git
cd glasses-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

**`requirements.txt`:**

torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
mediapipe>=0.10.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

---

## 🚀 Usage

### Single Image Inference

python
from inference import GlassesDetector

# Initialize detector
detector = GlassesDetector(
    glasses_model_path="models/resnet18_glasses.pth",
    device="cuda"  # or "cpu"
)

# Run inference
result = detector.predict("path/to/image.jpg")

# Output: list of dicts with face bbox, glasses prediction, and confidence
for face in result:
    print(f"Face at {face['bbox']}: {'Glasses' if face['glasses'] else 'No Glasses'}")

### Command Line

bash
python inference.py --image path/to/image.jpg --model models/resnet18_glasses.pth --device cuda

---

## 📊 Results

### ResNet18 Classifier (Standalone)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **95.12%** |
| **Train Accuracy** | 100.00% |
| **Epochs** | 10 |

### Confusion Matrix (Validation Set)

| | Predicted: No Glasses | Predicted: Glasses |
|---|---|---|
| **Actual: No Glasses** | 42 ✅ | 2 ❌ (FP) |
| **Actual: Glasses** | 2 ❌ (FN) | 36 ✅ |

- **False Positives:** 2
- **False Negatives:** 2

### Inference Speed (per image, 384×640)

| Stage | Time |
|-------|------|
| Preprocessing | 43.8 ms |
| Inference | 103.4 ms |
| Postprocessing | 2.6 ms |
| **Total** | **~150 ms** |

---

## ⭐ Key Features

### 1. Quality Gate
Filters out low-quality face crops before classification:
- Minimum face size: **48×48 pixels**
- Minimum mean brightness: **45**

### 2. Profile-Aware Thresholding
The system adjusts classification confidence thresholds based on face aspect ratio:

| Face Orientation | Threshold | Aspect Ratio |
|------------------|-----------|--------------|
| **Frontal** | 0.65 | ~1.0 |
| **Profile** | 0.50 | < 0.85 |

This prevents false negatives on side-profile faces where glasses features are less visible.

### 3. Multi-Stage Detection
Separating person detection, face detection, and glasses classification into distinct stages allows:
- Independent optimization of each component
- Easy swapping of models (e.g., YOLOv8 → YOLOv9)
- Fine-grained error analysis at each stage

---

## 🧠 Model Details

### ResNet18 Classifier

| Component | Detail |
|-----------|--------|
| **Architecture** | ResNet18 (pretrained: No — trained from scratch on glasses dataset) |
| **Input Size** | 224×224×3 |
| **Normalization** | ImageNet mean/std: `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` |
| **Output** | 2 classes: `[No Glasses, Glasses]` |
| **Final Layer** | `nn.Linear(512, 2)` |

### YOLOv8n (Person Detector)

| Component | Detail |
|-----------|--------|
| **Model** | YOLOv8n (nano) |
| **Weights** | Pretrained on COCO |
| **Classes Used** | Class 0 (person) only |

### MediaPipe (Face Detector)

| Component | Detail |
|-----------|--------|
| **Model Selection** | 1 (short-range, within 2m) |
| **Min Detection Confidence** | 0.6 |

---

## ⚠️ Limitations & Future Work
- [ ] **Occlusion handling**: Performance degrades when glasses are partially occluded
- [ ] **Real-time video**: Currently ~6-7 FPS; optimization needed for real-time applications

---

## 📝 License

This project is for academic/research purposes.

---

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google/mediapipe)
- [PyTorch](https://pytorch.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) — He et al., 2015

---

*Made with ❤️ and PyTorch*

---
