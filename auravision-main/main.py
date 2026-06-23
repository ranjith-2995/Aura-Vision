# type: ignore  # noqa: PGH003
import cv2  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import pyttsx3  # type: ignore
import numpy as np  # type: ignore
import os
import threading
import typing
import queue
import time
import argparse
from torchvision import transforms, models  # type: ignore

# ─── TTS Worker (Dedicated Thread) ─────────────────────────────────────────────
# pyttsx3 is NOT thread-safe. The engine MUST be created and used in the same thread.
# We use a queue: detection logic puts messages in, worker thread speaks them.

tts_queue: queue.Queue = queue.Queue()

# Cooldown: track last time we announced each *category* to avoid spamming.
announcement_cooldowns: typing.Dict[str, float] = {}
COOLDOWN_SECONDS: float = 12.0  # Increased from 4.0 to reduce irritating spam
MIRROR_CAMERA: bool = False   # Set to False to match training orientation (straight view)


def tts_worker() -> None:
    """Dedicated thread that speaks queued messages. Uses win32com for robust threading on Windows."""
    use_sapi = False
    engine = None

    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore
        pythoncom.CoInitialize()
        engine = win32com.client.Dispatch("SAPI.SpVoice")
        use_sapi = True
        print("[TTS] Windows SAPI Voice engine ready.")
    except Exception:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        print("[TTS] Fallback pyttsx3 Voice engine ready.")

    while True:
        text = tts_queue.get()  # blocks until a message arrives
        if text is None:        # None = shutdown signal
            break
        try:
            print(f"[ALERT] {text}")
            if use_sapi:
                engine.Speak(text)
            else:
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")
        tts_queue.task_done()


# Start the dedicated TTS thread at module level so it's always ready
_tts_thread = threading.Thread(target=tts_worker, daemon=True, name="TTS-Worker")
_tts_thread.start()


def announce(text: str, cooldown_key: typing.Optional[str] = None) -> None:
    """Queue a voice alert, skipping if the same *category* was said recently.

    Args:
        text:         The sentence to speak.
        cooldown_key: A stable key for the cooldown bucket (defaults to ``text``).
                      Use this so that related messages like "2 people are nearby"
                      and "3 people are nearby" share the same cooldown bucket.
    """
    global announcement_cooldowns
    key = cooldown_key if cooldown_key is not None else text
    now = time.time()
    if key in announcement_cooldowns and now - announcement_cooldowns[key] < COOLDOWN_SECONDS:
        return
    announcement_cooldowns[key] = now
    # Drop the alert if the TTS thread is already backed up (> 5 items waiting)
    # to avoid speaking a flood of stale messages.
    if tts_queue.qsize() > 5:
        print(f"[ALERT DROPPED – queue full] {text}")
        return
    tts_queue.put(text)


# ─── Face Recognition ───────────────────────────────────────────────────────────

def load_known_faces(
    known_faces_dir: str,
    face_cascade: "cv2.CascadeClassifier",
) -> typing.Tuple[typing.Any, typing.Dict[int, str]]:
    """
    Load and train an LBPH face recognizer from the known_faces directory.

    Expected folder layout:
        known_faces/
            Ganesh/          <- subfolder name = person's name
                photo1.jpg
                photo2.jpg
            Mom/
                photo1.jpg
            ...

    Returns:
        recognizer  - trained cv2.face.LBPHFaceRecognizer (or None if no data)
        label_map   - dict mapping integer label -> person name
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    label_map: typing.Dict[int, str] = {}
    face_samples: typing.List[typing.Any] = []
    labels: typing.List[int] = []

    if not os.path.isdir(known_faces_dir):
        print(f"[FACE-TRAIN] known_faces directory not found: {known_faces_dir}")
        return None, {}

    label_id = 0
    for person_name in sorted(os.listdir(known_faces_dir)):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue  # skip keep.txt or any loose files

        label_map[label_id] = person_name
        img_count = 0

        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [FACE-TRAIN] Could not read image: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(40, 40))
            if len(faces) == 0:
                print(f"  [FACE-TRAIN] ⚠ No face found in '{img_file}' – skipping")
                continue

            # ── CRITICAL CHECK: skip group photos ──────────────────────────────
            # If multiple faces are found, the photo is a group shot.
            # Training on it would pollute the model (strangers labelled as this person).
            if len(faces) > 1:
                print(f"  [FACE-TRAIN] ⚠ '{img_file}' has {len(faces)} faces (group photo) – SKIPPED.")
                print(f"               Use solo photos of {person_name} only.")
                continue

            (x, y, w, h) = faces[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            roi = cv2.equalizeHist(roi)
            face_samples.append(roi)
            labels.append(label_id)
            img_count += 1  # type: ignore
            print(f"  [FACE-TRAIN] ✓ '{img_file}' – 1 face loaded for '{person_name}'")

        if img_count > 0:
            print(f"  [FACE-TRAIN] '{person_name}': {img_count} usable image(s) loaded (label {label_id})")
            label_id += 1  # type: ignore
        else:
            print(f"  [FACE-TRAIN] ✗ '{person_name}': 0 usable images — skipping this person entirely.")
            print(f"               All photos were either group shots or had no detectable face.")
            label_map.pop(label_id, None)

    if len(face_samples) == 0:
        print("[FACE-TRAIN] No training images found. Recognition disabled.")
        print("             Add subfolders to known_faces/ with SOLO face photos (one person per photo).")
        return None, {}

    recognizer.train(face_samples, np.array(labels))
    print(f"[FACE-TRAIN] ✓ Trained on {len(face_samples)} face sample(s) across {len(label_map)} person(s).")
    return recognizer, label_map


def recognize_face(
    gray_roi: typing.Any,
    recognizer: typing.Any,
    label_map: typing.Dict[int, str],
    confidence_threshold: float = 75,
) -> typing.Tuple[str, float]:
    """
    Run LBPH recognition on a single grayscale face ROI.

    LBPH confidence = distance (LOWER = better match):
      0-50   -> strong match
      50-100 -> likely match
      100+   -> probably unknown

    Returns:
        name        - person's name, or "Unknown" if confidence too high / no model
        confidence  - raw LBPH distance value
    """
    if recognizer is None:
        return "Unknown", 999.0

    roi_resized = cv2.resize(gray_roi, (200, 200))
    roi_equalized = cv2.equalizeHist(roi_resized)
    label, confidence = recognizer.predict(roi_equalized)

    if confidence < confidence_threshold:
        name = label_map.get(label, "Unknown")
        print(f"  [FACE-RECOG] MATCH  -> '{name}'  (LBPH dist: {confidence:.1f}, threshold: {confidence_threshold})")
    else:
        name = "Unknown"
        best_label = label_map.get(label, '?')
        print(f"  [FACE-RECOG] UNKNOWN (best guess: '{best_label}', LBPH dist: {confidence:.1f} exceeds threshold: {confidence_threshold})")

    return name, confidence


# ─── Currency Recognition ───────────────────────────────────────────────────────

# Friendly spoken names for each folder label
CURRENCY_SPOKEN: typing.Dict[str, str] = {
    "ten_new":      "Ten rupees",
    "ten_old":      "Ten rupees",
    "twenty_new":   "Twenty rupees",
    "twenty_old":   "Twenty rupees",
    "fifty_new":    "Fifty rupees",
    "fifty_old":    "Fifty rupees",
    "hundred_new":  "One hundred rupees",
    "hundred_old":  "One hundred rupees",
    "two_hundred":  "Two hundred rupees",
    "five_hundred": "Five hundred rupees",
    "two_thousand": "Two thousand rupees",
}

CURRENCY_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_currency_model(
    model_path: str,
    labels_path: str,
) -> typing.Tuple[typing.Any, typing.List[str]]:
    """Load the trained MobileNetV2 currency classifier."""
    if not os.path.exists(model_path):
        print(f"[CURRENCY] No trained model found at '{model_path}'.")
        print("[CURRENCY] Run:  python train_currency.py  to train it first.")
        return None, []

    checkpoint = torch.load(model_path, map_location="cpu")
    class_names: typing.List[str] = checkpoint.get("class_names", [])
    num_classes: int = checkpoint.get("num_classes", len(class_names))

    # Rebuild the same architecture used in training
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[CURRENCY] Model loaded. Classes: {class_names}")
    return model, class_names


def predict_currency(
    frame_rgb: typing.Any,
    currency_model: typing.Any,
    class_names: typing.List[str],
    threshold: float = 0.65,
) -> typing.Tuple[typing.Optional[str], float]:
    """
    Run currency classification on an RGB frame.
    Returns (label, confidence) or (None, 0) if below threshold.
    """
    if currency_model is None:
        return None, 0.0
    try:
        tensor = CURRENCY_TRANSFORM(frame_rgb).unsqueeze(0)  # (1, C, H, W)
        with torch.no_grad():
            logits = currency_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = probs.max(0)
            conf_val: float = float(conf.item())
        if conf_val < threshold:
            return None, conf_val
        label: str = class_names[int(idx.item())]
        return label, conf_val
    except Exception as e:
        print(f"[CURRENCY] Inference error: {e}")
        return None, 0.0


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(test_init: bool = False) -> None:
    print("Initializing Vision Assistant...")

    # --- 1. Load Object Detection Model ---
    print("Loading YOLOv5 Object Detection model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,
                                    trust_repo=True, _verbose=False)
        yolo_model.to(device)
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        print("Please ensure you have internet connection for the first run.")
        return

    # --- 2. Initialize Face Detector (OpenCV Haar Cascade) ---
    print("Loading OpenCV Face Detector...")
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Error: Could not load face cascade. Check OpenCV installation.")
        return

    # --- 3. Train Face Recognizer from known_faces/ ---
    print("Loading known faces for recognition...")
    known_faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'known_faces')
    recognizer, label_map = load_known_faces(known_faces_dir, face_cascade)

    if recognizer is not None:
        print(f"[FACE] Recognition ACTIVE. Known people: {list(label_map.values())}")
    else:
        print("[FACE] Recognition DISABLED (no training data). Will say 'a person' for all faces.")

    announce("System initialized.")
    if test_init:
        print("Test initialization successful. Exiting...")
        return

    # --- 4. Load Indian Currency Classifier ---
    currency_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "currency_model.pt")
    currency_labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "currency_labels.txt")
    currency_model, currency_class_names = load_currency_model(currency_model_path, currency_labels_path)
    if currency_model is not None:
        print("[CURRENCY] Recognition ACTIVE. Hold a note in front of the camera!")

    # --- 5. Start Webcam Video Capture ---
    print("Starting webcam...")
    video_capture = cv2.VideoCapture(0)  # 0 = default laptop webcam

    if not video_capture.isOpened():
        print("Error: Could not access the webcam.")
        return

    print('System Running. Press "q" to quit in the video window.')

    # Frame counter to skip frames for faster processing
    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame

    # Initialize defaults so display code works before first detection cycle
    yolo_results = None
    face_boxes = []
    face_names = []

    while True:
        ret, frame = video_capture.read()

        # Mirror the frame horizontally if enabled (natural/selfie view)
        if ret and MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        if not ret:
            print("Failed to grab frame. Exiting loop.")
            break

        frame_count += 1  # type: ignore

        if frame_count % process_every_n_frames == 0:

            # ------------------------------------------------------------------
            # PART A: FACE DETECTION + RECOGNITION
            # ------------------------------------------------------------------
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            face_boxes = []
            face_names = []

            for (x, y, w, h) in detected_faces:
                face_boxes.append((x, y, x + w, y + h))

                # --- Recognition ---
                face_roi_gray = gray_frame[y:y + h, x:x + w]
                name, conf = recognize_face(face_roi_gray, recognizer, label_map)
                face_names.append(name)

                print(f"  [FACE]  Detected: {name:<20} confidence: {conf:.1f}")

            # --- Announce who is present ---
            if len(face_boxes) > 0:
                known_people = [n for n in face_names if n != "Unknown"]
                unknown_count = face_names.count("Unknown")

                # Announce each known person by name (separate cooldown per person)
                for person in set(known_people):
                    announce(f"{person} is nearby", cooldown_key=f"person_{person.lower()}")

                # Announce unknown faces with a clear warning
                if unknown_count == 1 and len(known_people) == 0:
                    announce("Unknown person is detected", cooldown_key="unknown_person")
                elif unknown_count > 1 and len(known_people) == 0:
                    announce(f"{unknown_count} unknown persons are detected", cooldown_key="unknown_person")
                elif unknown_count > 0:
                    # Mix of known and unknown
                    announce("Unknown person is also detected", cooldown_key="unknown_person")

            # ------------------------------------------------------------------
            # PART B: OBJECT DETECTION (Obstacles via YOLOv5)
            # ------------------------------------------------------------------
            yolo_results = yolo_model(frame)
            detections = yolo_results.pandas().xyxy[0]  # xmin,ymin,xmax,ymax,conf,class,name

            if not detections.empty:
                print(f"\n--- Frame {frame_count} Detections ---")

            for index, row in detections.iterrows():
                class_name = row['name']
                confidence = float(row['confidence'])

                print(f"  Detected: {class_name:<20} confidence: {confidence:.0%}")

                if confidence > 0.5:
                    # Filtered down to essential necessary objects requested by user to reduce spam
                    obstacle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 
                                        'person', 'chair', 'cell phone', 'bench', 'bed', 'dining table']

                    if class_name in obstacle_classes:
                        label = class_name.replace('dining table', 'table').capitalize()
                        if class_name == 'person':
                            if len(face_boxes) == 0:
                                # Person detected by YOLO but no face visible (back/side view)
                                announce("Unknown person is detected", cooldown_key="unknown_person")
                            # (known/unknown face announcements already handled in PART A)
                        else:
                            announce(f"{label} is detected", cooldown_key=f"obj_{class_name}")

            # ------------------------------------------------------------------
            # PART C: CURRENCY RECOGNITION
            # ------------------------------------------------------------------
            if currency_model is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                currency_label, currency_conf = predict_currency(
                    frame_rgb, currency_model, currency_class_names, threshold=0.98
                )
                if currency_label is not None and currency_label != "background":
                    spoken = CURRENCY_SPOKEN.get(currency_label, currency_label.replace("_", " "))
                    print(f"  [CURRENCY] {spoken}  ({currency_conf:.0%} confidence)")
                    announce(f"{spoken} detected", cooldown_key="currency")

        # ------------------------------------------------------------------
        # DISPLAY FOR DEBUGGING
        # ------------------------------------------------------------------
        if yolo_results is not None:
            rendered_frame = np.squeeze(yolo_results.render())  # type: ignore
        else:
            rendered_frame = frame.copy()

        # Draw face boxes with recognized name label
        for i, (left, top, right, bottom) in enumerate(face_boxes):
            face_name = face_names[i] if i < len(face_names) else "Unknown"  # type: ignore
            color = (0, 200, 0) if face_name != "Unknown" else (0, 0, 255)  # green=known, red=unknown

            cv2.rectangle(rendered_frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(rendered_frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(rendered_frame, face_name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Vision Assistant Prototype (Press "q" to Quit)', rendered_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visually Impaired AI Assistant")
    parser.add_argument('--test-init', action='store_true',
                        help="Run initialization and model loading test without webcam.")
    args = parser.parse_args()
    main(test_init=args.test_init)
