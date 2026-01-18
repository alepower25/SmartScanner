# Smart Scanner

A real-time object monitoring system using YOLOv8 and OpenCV.  
Detects phones, wallets, bottles, laptops, and more. Logs detections to a CSV file with timestamps and duration.

## Features

- Real-time multi-object detection
- Corner-style aesthetic bounding boxes
- Semi-transparent labels
- Duration tracking per object
- CSV logging (`detections.csv`)

## Requirements

- Python 3.10+
- Mac M1/M2 recommended for faster GPU detection (using `device="mps"`)
- Packages:
