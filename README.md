---
title: Recyclable Object Detection
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# ♻️ Recyclable Object Detection

YOLOv8s model for detecting recyclable objects in images.

## Model Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 63.4% |
| Precision | 71.8% |
| Recall | 70.0% |
| F1-Score | 0.709 |

## Usage

1. Upload an image containing recyclable objects
2. Adjust confidence threshold as needed
3. View detection results

## Tech Stack

- **Model**: YOLOv8s (fine-tuned)
- **Framework**: Ultralytics
- **Frontend**: Streamlit
