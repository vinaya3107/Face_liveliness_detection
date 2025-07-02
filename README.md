# Face Liveliness Detection

This project implements a **Face Liveliness Detection System** using Python and OpenCV. It aims to differentiate between a **real human face** and a **spoofed image or video**, helping prevent unauthorized access through facial spoofing (e.g., photos, videos, or masks).

## 🧠 Project Objective

To build a computer vision-based solution that can detect whether the captured face is live or spoofed using techniques like **eye blink detection**, **face movement analysis**, or **texture-based detection**. This is useful in security systems such as facial login, e-KYC, and biometric authentication.

---

## 🔍 Features

- Real-time face detection using OpenCV.
- Eye blink detection to check for liveliness.
- Drowsiness or spoof detection based on facial cues.
- Webcam integration for live testing.
- Lightweight and fast.

---

## 🛠️ Tech Stack

- Python 3.x
- OpenCV
- Dlib / Mediapipe (optional, depending on your implementation)
- NumPy
- yolo

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3 and pip installed. Then install the required dependencies:

```bash
pip install -r requirements.txt
