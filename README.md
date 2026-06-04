## PPE-detection-kit: YOLOv11 PPE Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)

PPE-detection-kit is a realtime computer vision system designed to monitor construction environments and ensure compliance with safety regulations. Powered by **YOLOv11** and wrapped in a responsive **Streamlit** dashboard, it actively detects whether personnel are wearing mandatory Personal Protective Equipment (PPE) such as hardhats and high visibility vests.

## ✨ Features

* **Interactive Admin Dashboard:** A complete Streamlit web UI featuring dual mode viewing (Admin Dashboard and Full Screen Site View).
* **Live Violation Logging:** Automatically captures snapshots of safety violations (e.g., missing hardhats) and queues them in a verification sidebar for admin review.
* **Smart Filtering:** Utilizes distinct confidence thresholds for standard detections versus violations to minimize false alarms on the job site.
* **Dynamic Sensitivity:** Adjust the model's confidence threshold on the fly using the dashboard slider without restarting the system.
* **HighSpeed Object Detection:** Utilizes YOLOv11 for rapid, high accuracy inference on live camera feeds.

## 🛠️ Prerequisites

Ensure you have Python 3.8+ installed. If you intend to run inference on a GPU for maximum frames per second (FPS), ensure you have the appropriate NVIDIA drivers and CUDA toolkit installed for PyTorch.

## 🚀 Installation & Setup

1. **Clone the repository--->**

    <ins>cmd/bash:</ins>

   ***git clone [https://github.com/frakxzo/PPE-detection-kit.git](https://github.com/frakxzo/PPE-detection-kit.git)***

   ***cd PPE detection kit***

2. **Install the required dependencies--->**

    <ins>cmd/bash:</ins>

   ***pip install  r requirements.txt*** 

3. **Model Weights--->**
    
    <ins>Ensure your custom trained YOLOv11 weights file (best.pt) is placed in the root directory before running the application.</ins>


## 🎮 Usage Guide

To launch the interactive dashboard, use the Streamlit CLI command in your terminal:

  **streamlit run dashboard.py**

Once the local server starts, your web browser will automatically open the dashboard. Use the Navigator in the sidebar to toggle between the Admin View and the Live Site Monitor, and click START SYSTEM to initialize the camera feed.


## 🗺️ Roadmap & Future Development

**Multi Cam Integration:** Expanding the video capture logic to process parallel RTSP streams from multiple IP cameras simultaneously.

**UI Enhancements:** Refining the Streamlit layout for better mobile responsiveness and optimizing the verification queue rendering.

**Database Integration:** Moving the session state alert logging to a persistent SQLite or PostgreSQL database for long term compliance tracking.
      






                                       

