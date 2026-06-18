import cv2
import numpy as np
import tensorflow as tf

# ===============================
# PATH MODEL
# ===============================
MODEL_PATH = "models/cnn_visual_strong.h5"

# ===============================
# PARAMETER
# ===============================
IMG_SIZE = 96   # SESUAI TRAIN TERAKHIR KAMU
EMOTIONS = ["angry", "happy", "neutral", "sad"]

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model visual berhasil dimuat")

# ===============================
# FACE DETECTOR
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

print("🎥 Tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        preds = model.predict(face, verbose=0)
        emotion_idx = np.argmax(preds)
        emotion = EMOTIONS[emotion_idx]
        confidence = np.max(preds) * 100

        label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Visual Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
