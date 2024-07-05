# Driver Monitoring System

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a driver monitoring system using facial landmarks detection to detect drowsiness and monitor eye movements.

## Demo
https://github.com/aLok-1105/driver-monitoring-system/assets/106423643/31ef5b3a-0cc2-43c6-986f-51d76ab827ba

![Demo](https://github.com/aLok-1105/driver-monitoring-system/assets/106423643/9e36f405-80a5-4f30-8cf3-72523425f6a8)


## Features

- **Facial Landmarks Detection**: Uses MediaPipe FaceMesh to detect and track facial landmarks, including eye landmarks for monitoring.
- **Drowsiness Detection**: Monitors eye aspect ratio (EAR) to detect drowsiness based on predefined thresholds.
- **Blink Detection**: Counts blinks and calculates blink rate.
- **Excel Data Logging**: Timestamp, Eye Aspect Ratio, Blink Rate, Blink Count, Drowsy Time, Regio and Fixation Time data to an Excel spreadsheet.
- **Position Estimation**: Estimates eye position (LEFT, CENTER, RIGHT) based on landmark analysis.
- **Yawn Detection**: Detects yawns based on lip landmarks.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Mediapipe (`pip install mediapipe`)
- OpenPyXL (`pip install openpyxl`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aLok-1105/driver-monitoring-system.git
   cd driver-monitoring-system
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

1. Run the main script `main.py`:
   ```bash
   python main.py
2. Adjust parameters in `main.py` such as video input source, thresholds, and tracking settings.
   ```bash
   cap = cv2.VideoCapture('file_name')

## File Structure

1. main.py: Main script implementing drowsiness detection and eye tracking.
2. helper.py: Helper functions for distance calculation, eye position estimation, and facial landmark processing.
3. monitoring.py: Module for real-time monitoring and visualization using OpenCV.
