# AI-Powered Vision Assistant

A Python prototype of an AI-powered assistant for visually impaired people. It runs entirely on a laptop using the built-in webcam and speakers — no cloud, no internet required after the first run.

## Features

| Feature | How it works |
|---|---|
| **Face Recognition** | Detects faces with OpenCV Haar Cascade, then identifies *who* they are using an LBPH recognizer trained on photos in `known_faces/` |
| **Obstacle Detection** | Uses a pretrained YOLOv5s model to detect chairs, people, bottles, laptops, and more |
| **Voice Alerts** | Spoken alerts via `pyttsx3` on a background thread — won't lag the video feed |
| **Smart Cooldowns** | Each alert category has a 6-second cooldown so the same message isn't repeated every frame |

## Installation

**Requirements**: Python 3.8+

```bash
# 1. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **First run only**: YOLOv5s weights (~15 MB) are downloaded automatically from the internet.
> After that the model is cached locally and no internet is needed.

## Setting Up Known Faces

To let the system greet people by name:

```
known_faces/
    Ganesh/          ← one subfolder per person (folder name = spoken name)
        photo1.jpg
        photo2.jpg    ← 5–10 clear, well-lit photos recommended
    Mom/
        photo1.jpg
```

- The recognizer trains automatically every time the script starts.
- **Green box** = recognized person (name shown), **Red box** = unknown face.
- If a known person keeps showing as "Unknown", add more photos of them.

## Running

```bash
python main.py
```

- A webcam window opens and audio alerts begin immediately.
- Press **`q`** in the video window to quit.

**Test initialization only (no webcam)**:
```bash
python main.py --test-init
```

## Project Structure

```
vision/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── yolov5s.pt           # Cached YOLOv5 weights (auto-downloaded)
└── known_faces/
    ├── keep.txt         # Instructions for adding face photos
    └── <PersonName>/    # One folder per person with their photos
```

## Architecture

```
Webcam Frame
    │
    ├─► Haar Cascade ──► Face detected?
    │       └─► LBPH Recognizer ──► Who is it? ──► TTS Alert ("Ranjith is nearby")
    │
    └─► YOLOv5s ──► Object detected? ──► TTS Alert ("Chair is nearby")
                            │
                   Cooldown filter (6s per category)
                            │
                   TTS Worker Thread (pyttsx3)
```

## Future Development

1. **Raspberry Pi** — Lightweight MobileNet-SSD or Coral Edge TPU for portable use with a Pi Camera and earpiece.
2. **Wearable Glasses** — Mini-camera on frames feeding a Pi Zero in a pocket.
3. **Smartphone App** — Swift/CoreML (iOS) or Kotlin/TFLite (Android) for native performance.
4. **Distance Estimation** — Stereo cameras or LiDAR to announce *how far* an obstacle is.
