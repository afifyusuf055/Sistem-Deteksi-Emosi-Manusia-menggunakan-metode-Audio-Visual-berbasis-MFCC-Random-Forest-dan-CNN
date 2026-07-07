"""
=========================================================
VISUAL ENGINE
CNN + Haar Cascade
=========================================================
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import sys


# ======================================================
# RESOURCE PATH
# ======================================================

def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS

    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ======================================================
# MODEL
# ======================================================

MODEL_PATH = resource_path(
    "models/cnn_visual_strong.h5"
)

visual_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Visual Model Loaded")


# ======================================================
# LABEL
# ======================================================

EMOTIONS = [

    "angry",

    "happy",

    "neutral",

    "sad"

]


# ======================================================
# FACE DETECTOR
# ======================================================

face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades +

    "haarcascade_frontalface_default.xml"

)

if face_cascade.empty():

    print("❌ Haar Cascade gagal dimuat")

else:

    print("✅ Haar Cascade berhasil dimuat")

# ======================================================
# PREDICT FRAME
# ======================================================

def predict_frame(frame):

    emotion = "neutral"

    confidence = 0.0

    probability = np.zeros(4)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80,80)
    )

    # Tidak ada wajah
    if len(faces) == 0:

         return (
            frame,
            emotion,
            confidence,
            probability,
            faces
        )

    # Ambil wajah pertama
    (x, y, w, h) = faces[0]

    cv2.rectangle(
        frame,
        (x, y),
        (x+w, y+h),
        (0,255,0),
        2
    )

    # Crop wajah
    face = gray[y:y+h, x:x+w]

    if face.size == 0:

        return (
            frame,
            emotion,
            confidence,
            probability,
            faces
        )

    # Resize
    face = cv2.resize(
        face,
        (96,96)
    )

    face = face.astype("float32") / 255.0

    face = np.expand_dims(
        face,
        axis=-1
    )

    face = np.expand_dims(
        face,
        axis=0
    )

    # Prediksi CNN
    probability = visual_model.predict(
        face,
        verbose=0
    )[0]

    idx = np.argmax(
        probability
    )

    emotion = EMOTIONS[idx]

    confidence = float(
        probability[idx] * 100
    )

    # Tampilkan label di kamera
    cv2.putText(
        frame,
        f"{emotion} ({confidence:.1f}%)",
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )
    print("DEBUG RETURN:", len(probability), len(faces))
    return (
         frame,
         emotion,
         confidence,
         probability,
         faces
)