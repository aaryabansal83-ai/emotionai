# app/utils.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# ── Constants ────────────────────────────────────────────────────
IMG_SIZE = 48
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Angry':    '#E74C3C',
    'Disgust':  '#8E44AD',
    'Fear':     '#3498DB',
    'Happy':    '#F1C40F',
    'Neutral':  '#95A5A6',
    'Sad':      '#2980B9',
    'Surprise': '#E67E22'
}

EMOTION_EMOJIS = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😄',
    'Neutral':  '😐',
    'Sad':      '😢',
    'Surprise': '😲'
}

# Haar cascade for face detection
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


@st.cache_resource
def load_model():
    """
    Load and cache the trained Keras model.
    Tries multiple paths for compatibility with
    both local development and cloud deployment.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    candidate_paths = [
        # Local: final saved model
        os.path.join(base_dir, '..', 'models', 'saved_model', 'emotion_model.keras'),
        # Local: best checkpoint
        os.path.join(base_dir, '..', 'models', 'checkpoints', 'best_model.keras'),
        # Cloud / alternate working directory
        'models/saved_model/emotion_model.keras',
        'models/checkpoints/best_model.keras',
        # Same folder fallback
        os.path.join(base_dir, 'emotion_model.keras'),
        os.path.join(base_dir, 'best_model.keras'),
    ]

    for path in candidate_paths:
        resolved = os.path.normpath(path)
        if os.path.exists(resolved):
            try:
                model = tf.keras.models.load_model(resolved)
                return model
            except Exception as e:
                st.warning(f"Found model at {resolved} but failed to load: {e}")
                continue

    return None  # No model found


def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """
    Preprocess a face ROI (grayscale or BGR) for model inference.
    Returns tensor shaped (1, 48, 48, 1).
    """
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    resized    = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Convert any full image (RGB/RGBA/Gray) to
    a model-ready tensor shaped (1, 48, 48, 1).
    """
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    resized    = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_emotion(model, tensor: np.ndarray) -> tuple:
    """
    Run inference on a preprocessed tensor.

    Returns:
        emotion    (str)   - top predicted emotion label
        confidence (float) - confidence percentage (0-100)
        all_probs  (dict)  - {emotion: confidence%} for all 7 classes
    """
    probs      = model.predict(tensor, verbose=0)[0]
    top_idx    = int(np.argmax(probs))
    emotion    = EMOTIONS[top_idx]
    confidence = float(probs[top_idx]) * 100
    all_probs  = {
        EMOTIONS[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(EMOTIONS))
    }
    return emotion, confidence, all_probs


def detect_and_predict(model, img_rgb: np.ndarray) -> list:
    """
    Detect ALL faces in an RGB image and predict
    emotion for each one independently.

    Returns:
        List of dicts, one per face:
        {
            'bbox'      : (x, y, w, h),
            'emotion'   : str,
            'confidence': float,
            'all_probs' : dict
        }
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    results = []
    for (x, y, w, h) in faces:
        face_roi                       = gray[y:y+h, x:x+w]
        tensor                         = preprocess_face(face_roi)
        emotion, confidence, all_probs = predict_emotion(model, tensor)
        results.append({
            'bbox':       (x, y, w, h),
            'emotion':    emotion,
            'confidence': confidence,
            'all_probs':  all_probs
        })

    return results


def draw_face_boxes(img: np.ndarray, results: list) -> np.ndarray:
    """
    Draw colour-coded bounding boxes + emotion labels
    on an RGB image. Works for 1 or many faces.

    Returns annotated RGB image (numpy array).
    """
    output = img.copy()

    for idx, r in enumerate(results):
        x, y, w, h = r['bbox']
        emotion     = r['emotion']
        conf        = r['confidence']

        hex_col = EMOTION_COLORS[emotion].lstrip('#')
        rgb     = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))

        cv2.rectangle(output, (x, y), (x+w, y+h), rgb, 2)

        label       = f"#{idx+1} {emotion} {conf:.1f}%"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(output,
                      (x, y - th - 14),
                      (x + tw + 8, y),
                      rgb, -1)
        cv2.putText(output, label,
                    (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return output


def process_webcam_frame(model, frame_bgr: np.ndarray) -> tuple:
    """
    Process a single raw webcam frame (BGR numpy array):
      1. Detect all faces
      2. Predict emotion for each face
      3. Draw annotated bounding boxes

    Returns:
        annotated_frame (BGR numpy array) - ready for display
        results         (list of dicts)   - predictions per face
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    results = []
    output  = frame_bgr.copy()

    for idx, (x, y, w, h) in enumerate(faces):
        face_roi                       = gray[y:y+h, x:x+w]
        tensor                         = preprocess_face(face_roi)
        emotion, confidence, all_probs = predict_emotion(model, tensor)

        results.append({
            'bbox':       (x, y, w, h),
            'emotion':    emotion,
            'confidence': confidence,
            'all_probs':  all_probs
        })

        hex_col = EMOTION_COLORS[emotion].lstrip('#')
        bgr_col = tuple(int(hex_col[i:i+2], 16) for i in (4, 2, 0))

        cv2.rectangle(output, (x, y), (x+w, y+h), bgr_col, 2)

        label       = f"#{idx+1} {emotion} {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(output,
                      (x, y - th - 14),
                      (x + tw + 8, y),
                      bgr_col, -1)
        cv2.putText(output, label,
                    (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return output, results