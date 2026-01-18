# Smart Scanner

Real-time object monitoring system using YOLOv8 and OpenCV.  
Detects phones, wallets, bottles, laptops, and more, and logs detections locally.

## Features

- Real-time multi-object detection
- Corner-style aesthetic bounding boxes
- Semi-transparent labels
- Duration tracking per object
- CSV logging (`detections.csv`) — ignored in GitHub

## Requirements

- Python 3.10+
- Mac M1/M2 recommended for faster GPU detection (`device="mps"`)
- Install dependencies:

```bash
pip install -r requirements.txt
