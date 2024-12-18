### README

**SmartShelf AI: Real-Time Inventory Monitoring System**

Welcome to the SmartShelf AI project! This document provides an overview of how to set up and use the prototype system designed to detect and count products on a retail shelf using computer vision and machine learning techniques.

#### Overview

SmartShelf AI aims to optimize inventory management by employing real-time stock monitoring through camera feeds and machine learning models. This prototype script uses a pretrained SSD MobileNet model to detect, identify, and count objects in a video stream. The setup is indicative of how retail shelves can automatically keep track of product availability, aiding in efficient stock management.

#### Features

- **Real-Time Monitoring**: Captures live video feed to monitor stock levels.
- **Object Detection**: Uses a pre-trained deep learning model to detect specific objects (e.g., bottles, cans) on the shelf.
- **Stock Counting**: Identifies and counts detected objects in real time, updating the inventory status dynamically.
- **User Interface**: Displays detected and counted products in a video window with bounding boxes and labels for visualization.

#### Directory Structure

- `SmartShelfAI/` – Main project directory.
  - `detect_stock.py` – Python script for detecting and counting products.
  - `models/` – Directory for storing pretrained models.
  - `requirements.txt` – Text file listing dependencies.

#### Getting Started

1. **Dependencies**: To run the project, install necessary dependencies using the `requirements.txt` file. This installation can be done via pip:
