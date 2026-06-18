from email.mime import audio

import cv2
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import queue
import threading
import time
import joblib

# ======================================================
# LOAD MODEL
# ======================================================
VISUAL_MODEL_PATH = "models/cnn_visual_strong.h5"
AUDIO_MODEL_PATH = "models/rf_audio_mfcc.pkl"
SCALER_PATH = "models/audio_scaler.pkl"

visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)

audio_model = joblib.load(AUDIO_MODEL_PATH)
audio_scaler = joblib.load(SCALER_PATH)

print("✅ Semua model berhasil dimuat")

# ======================================================
# LABEL EMOSI
# ======================================================
EMOTIONS = ["angry", "happy", "neutral", "sad"]

# ======================================================
# FACE DETECTOR
# ======================================================
face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

# ======================================================
# CEK DEVICE AUDIO
# ======================================================
print("\n🎤 LIST MICROPHONE:")
print(sd.query_devices())

# GANTI SESUAI NOMOR MIC EXTERNAL
MIC_DEVICE = 1

# ======================================================
# AUDIO VARIABLE
# ======================================================
audio_q = queue.Queue()

audio_emotion = "neutral"
audio_confidence = 0

# REAL AUDIO PROBABILITY
audio_probabilities = np.zeros(4)

# SMOOTHING
audio_history = []

# ======================================================
# AUDIO CALLBACK
# ======================================================
def audio_callback(indata, frames, time_, status):

    if status:
        print(status)

    audio_q.put(indata.copy())

# ======================================================
# AUDIO THREAD
# ======================================================
def audio_process():

    global audio_emotion
    global audio_confidence
    global audio_probabilities
    global audio_history

    with sd.InputStream(
        device=MIC_DEVICE,
        channels=1,
        samplerate=16000,
        blocksize=16000,
        callback=audio_callback
    ):

        print("🎤 Audio aktif")

        while True:

            if not audio_q.empty():

                try:

                    audio = audio_q.get().flatten()

                    # ==================================================
                    # NORMALISASI
                    # ==================================================
                    audio = audio / (np.max(np.abs(audio)) + 1e-6)

                    # ==================================================
                    # NOISE REDUCTION
                    # ==================================================
                    audio = librosa.effects.preemphasis(audio)

                    # ==================================================
                    # MFCC
                    # ==================================================
                    mfcc = librosa.feature.mfcc(
                         y=audio,
                         sr=16000,
                         n_mfcc=40
                    )

                    # ==================================================
                    # DELTA
                    # ==================================================
                    delta = librosa.feature.delta(mfcc)

                    # ==================================================
                    # DELTA 2
                    # ==================================================
                    delta2 = librosa.feature.delta(
                        mfcc,
                        order=2
                    )

                    # ==================================================
                    # CHROMA
                    # ==================================================
                    chroma = librosa.feature.chroma_stft(
                        y=audio,
                         sr=16000
                    )
                    
                    # ==================================================
                    # FEATURE GABUNGAN
                    # TOTAL = 132 FEATURE
                    # ==================================================
                    feat = np.hstack([
                        
                        np.mean(mfcc.T, axis=0),

                        np.mean(delta.T, axis=0),

                        np.mean(delta2.T, axis=0),

                        np.mean(chroma.T, axis=0)

                    ])

                    feat = np.nan_to_num(feat)

                    feat = feat.reshape(1, -1)

                    # ==================================================
                    # SCALER
                    # ==================================================
                    feat = audio_scaler.transform(feat)

                    # ==================================================
                    # AUDIO PREDICT
                    # ==================================================
                    prob = audio_model.predict_proba(feat)[0]

                    # simpan probability asli
                    audio_probabilities = prob

                    # ==================================================
                    # SMOOTHING
                    # ==================================================
                    audio_history.append(prob)

                    if len(audio_history) > 3:
                        audio_history.pop(0)

                    avg_prob = np.mean(audio_history, axis=0)

                    idx = np.argmax(avg_prob)

                    audio_emotion = EMOTIONS[idx]

                    audio_confidence = (
                        avg_prob[idx] * 100
                    )

                except Exception as e:
                    print("❌ Audio error:", e)

            time.sleep(0.1)

# ======================================================
# START AUDIO THREAD
# ======================================================
threading.Thread(
    target=audio_process,
    daemon=True
).start()

# ======================================================
# WEBCAM
# ======================================================
cap = cv2.VideoCapture(0)

print("🎥 Tekan tombol Q untuk keluar")

# ======================================================
# MAIN LOOP
# ======================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # ==================================================
        # FACE PREPROCESS
        # ==================================================
        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (96, 96))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=-1)

        face = np.expand_dims(face, axis=0)

        # ==================================================
        # VISUAL PREDICT
        # ==================================================
        visual_prob = visual_model.predict(
            face,
            verbose=0
        )[0]

        visual_idx = np.argmax(visual_prob)

        visual_emotion = EMOTIONS[visual_idx]

        visual_confidence = (
            visual_prob[visual_idx] * 100
        )

        # ==================================================
        # FUSION
        # VISUAL = 75%
        # AUDIO  = 25%
        # ==================================================
        fusion_prob = (
            visual_prob * 0.75 +
            audio_probabilities * 0.25
        )

        fusion_idx = np.argmax(fusion_prob)

        fusion_emotion = EMOTIONS[fusion_idx]

        fusion_confidence = (
            fusion_prob[fusion_idx] * 100
        )

        # ==================================================
        # DRAW FACE BOX
        # ==================================================
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        # ==================================================
        # DISPLAY TEXT
        # ==================================================
        cv2.putText(
            frame,
            f"Fusion : {fusion_emotion} ({fusion_confidence:.1f}%)",
            (x, y-45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Visual : {visual_emotion} ({visual_confidence:.1f}%)",
            (x, y-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Audio : {audio_emotion} ({audio_confidence:.1f}%)",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,255),
            2
        )

    # ======================================================
    # SHOW WINDOW
    # ======================================================
    cv2.imshow(
        "Fusion Emotion Recognition",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# ======================================================
# RELEASE
# ======================================================
cap.release()

cv2.destroyAllWindows()