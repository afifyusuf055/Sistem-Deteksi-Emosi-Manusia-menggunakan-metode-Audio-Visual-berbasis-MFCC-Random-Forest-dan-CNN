import tensorflow as tf
import librosa
import sounddevice as sd
import queue
import threading
import time
import joblib
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pygame
import sys
import os
import cv2.data

# ======================================================
# RESOURCE PATH
# ======================================================
def resource_path(relative_path):

    try:

        base_path = sys._MEIPASS

    except Exception:

        base_path = os.path.abspath(".")

    return os.path.join(
        base_path,
        relative_path
    )


# ======================================================
# LOAD MODEL
# ======================================================

VISUAL_MODEL_PATH = resource_path(
    "models/cnn_visual_strong.h5"
)

AUDIO_MODEL_PATH = resource_path(
    "models/rf_audio_mfcc.pkl"
)

SCALER_PATH = resource_path(
    "models/audio_scaler.pkl"
)

CASCADE_PATH = resource_path(
    "models/haarcascade_frontalface_default.xml"
)

# ======================================================
# LOAD VISUAL MODEL
# ======================================================
visual_model = tf.keras.models.load_model(
    VISUAL_MODEL_PATH
)

# ======================================================
# LOAD AUDIO MODEL
# ======================================================
audio_model = joblib.load(
    AUDIO_MODEL_PATH
)

# ======================================================
# LOAD SCALER
# ======================================================
audio_scaler = joblib.load(
    SCALER_PATH
)


# ======================================================
# LABEL EMOSI
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

HAAR_PATH = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

print("HAAR:", HAAR_PATH)

face_cascade = cv2.CascadeClassifier(
    HAAR_PATH
)

if face_cascade.empty():

    print("❌ Haarcascade gagal dimuat")

else:

    print("✅ Haarcascade berhasil dimuat")


# ======================================================
# AUDIO VARIABLE
# ======================================================

audio_q = queue.Queue()

audio_emotion = "neutral"
audio_confidence = 0

audio_probabilities = np.zeros(4)

audio_history = []

# ======================================================
# AUDIO CALLBACK
# ======================================================
def audio_callback(indata, frames, time_, status):

    global audio_q

    if status:
        print(status)

    volume = np.max(np.abs(indata))

    print("MIC:", volume)

    audio_q.put(indata.copy())

# ======================================================
# AUDIO THREAD
# ======================================================
def audio_process():

    global audio_emotion
    global audio_confidence
    global audio_probabilities
    global audio_history

    device_index = int(
    selected_device.get().split(" - ")[0]
)
    with sd.InputStream(
        channels=1,
        samplerate=16000,
        blocksize=16000,
        device=None,
        callback=audio_callback
    ):

        print("🎤 Audio aktif")

        while True:

            if not audio_q.empty():

                try:

                    audio = audio_q.get().flatten()
                    print("Volume:", np.max(np.abs(audio)))

                    # FILTER NOISE
                    if np.max(np.abs(audio)) < 0.01:
                        continue

                    # NORMALISASI
                    audio = audio / (
                        np.max(np.abs(audio)) + 1e-6
                    )

                    # NOISE REDUCTION
                    audio = librosa.effects.preemphasis(audio)

                    # MFCC
                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=16000,
                        n_mfcc=40
                    )
                    # DELTA
                    delta = librosa.feature.delta(mfcc)
                    # DELTA2
                    delta2 = librosa.feature.delta(
                        mfcc,
                        order=2
                    )
                    # CHROMA
                    chroma = librosa.feature.chroma_stft(
                        y=audio,
                        sr=16000
                    )
                    # FEATURE
                    feat = np.hstack([

                        np.mean(mfcc.T, axis=0),

                        np.mean(delta.T, axis=0),

                        np.mean(delta2.T, axis=0),

                        np.mean(chroma.T, axis=0)

                    ])

                    feat = np.nan_to_num(feat)
                    feat = feat.reshape(1, -1)

                
                    
                    feat = audio_scaler.transform(feat)

                    prob = audio_model.predict_proba(feat)[0]

                    audio_probabilities = prob

                    audio_history.append(prob)

                    if len(audio_history) > 3:
                        audio_history.pop(0)

                    avg_prob = np.mean(
                        audio_history,
                        axis=0
                    )

                    idx = np.argmax(avg_prob)

                    audio_emotion = EMOTIONS[idx]

                    audio_confidence = (
                        avg_prob[idx] * 100
                    )
                    if audio_confidence < 25:
                        audio_emotion = "neutral"

                except Exception as e:
                    print("Audio Error:", e)

            time.sleep(0.1)

# ======================================================
# WINDOW
# ======================================================
root = tk.Tk()

root.title("Emotion Recognition GUI")
root.geometry("1200x900")
root.configure(bg="#1e1e1e")

# ======================================================
# AUDIO OUTPUT
# ======================================================
pygame.mixer.init()

tts_enabled = True

last_speak_time = 0


# ======================================================
# TITLE
# ======================================================
title = tk.Label(
    root,
    text="SISTEM DETEKSI EMOSI AUDIO-VISUAL",
    font=("Helvetica", 22, "bold"),
    fg="cyan",
    bg="#1e1e1e"
)

title.pack(pady=20)

# ======================================================
# CAMERA LABEL
# ======================================================
camera_label = tk.Label(root)
camera_label.pack()


# ======================================================
# EMOTION LABEL
# ======================================================
emotion_label = tk.Label(
    root,
    text="😐 NEUTRAL",
    font=("Helvetica", 28, "bold"),
    fg="yellow",
    bg="#1e1e1e"
)

emotion_label.pack(pady=10)

# ======================================================
# CONFIDENCE LABEL
# ======================================================
confidence_label = tk.Label(
    root,
    text="Confidence : 0%",
    font=("Helvetica", 18),
    fg="white",
    bg="#1e1e1e"
)

confidence_label.pack()

# ======================================================
# MODE LABEL
# ======================================================
mode_label = tk.Label(
    root,
    text="MODE : FUSION",
    font=("Helvetica", 18, "bold"),
    fg="lime",
    bg="#1e1e1e"
)

mode_label.pack(pady=10)

# ======================================================
# AUDIO DEVICE
# ======================================================
audio_devices = []

devices = sd.query_devices()

for i, dev in enumerate(devices):

    if dev['max_input_channels'] > 0:

        audio_devices.append(
            f"{i} - {dev['name']}"
        )

# ======================================================
# DROPDOWN VARIABLE
# ======================================================
selected_device = tk.StringVar()

selected_device.set(audio_devices[0])



# ======================================================
# BUTTON FUNCTION
# ======================================================
CURRENT_MODE = "fusion"

def set_mode(mode):
    if mode == "audio":
        print("MODE AUDIO AKTIF")

    global CURRENT_MODE

    CURRENT_MODE = mode

    if mode == "visual":
        color = "lime"
    elif mode == "audio":
            color = "orange"
    else:
        color = "Violet"
    mode_label.config(
        text=f"MODE : {mode.upper()}",
        fg=color
        
    )

# ======================================================
# SPEAK EMOTION
# ======================================================
def speak_emotion(emotion):

    global tts_enabled
    global last_speak_time

    if not tts_enabled:
        return

    current_time = time.time()

    # OUTPUT SETIAP 5 DETIK
    if current_time - last_speak_time < 5:
        return

    last_speak_time = current_time

    try:

        if emotion == "happy":

            pygame.mixer.music.load(
                resource_path(
                    "sounds/happy.wav"
    )
            )

        elif emotion == "sad":

            pygame.mixer.music.load(
                resource_path(
                    "sounds/sad.wav"
                )
            )

        elif emotion == "angry":

            pygame.mixer.music.load(
                resource_path(
                    "sounds/angry.wav"
                )
            )

        else:

            pygame.mixer.music.load(
                resource_path(
                    "sounds/neutral.wav"
                )
            )

        pygame.mixer.music.play()

    except Exception as e:

        print("Sound Error:", e)


# ======================================================
# TOGGLE SPEAKER
# ======================================================
def toggle_tts():

    global tts_enabled

    tts_enabled = not tts_enabled

    if tts_enabled:

        speaker_btn.config(
            text="🔊 SOUND ON",
            bg="green"
        )

    else:

        speaker_btn.config(
            text="🔇 SOUND OFF",
            bg="red"
        )

# ======================================================
# SPEAKER BUTTON
# ======================================================
speaker_btn = tk.Button(

    root,

    text="🔊 SOUND ON",

    command=toggle_tts,

    font=("Helvetica", 12, "bold"),

    bg="green",

    fg="white",

    width=15
)

speaker_btn.place(
    x=980,
    y=680
)

# ======================================================
# BUTTON FRAME
# ======================================================
button_frame = tk.Frame(
    root,
    bg="#1e1e1e"
)

button_frame.pack(pady=30)

# ======================================================
# CREATE BUTTON
# ======================================================
def create_button(text, command, color):

    return tk.Button(
        button_frame,
        text=text,
        command=command,
        font=("Helvetica", 14, "bold"),
        bg=color,
        fg="white",
        width=14,
        height=2,
        relief="flat",
        cursor="hand2"
    )

# ======================================================
# BUTTONS
# ======================================================
visual_btn = create_button(
    "MODE VISUAL",
    lambda: set_mode("visual"),
    "blue"
)

visual_btn.grid(row=0, column=0, padx=10)

audio_btn = create_button(
    "MODE AUDIO",
    lambda: set_mode("audio"),
    "orange"
)

audio_btn.grid(row=0, column=1, padx=10)

fusion_btn = create_button(
    "MODE FUSION",
    lambda: set_mode("fusion"),
    "purple"
)

fusion_btn.grid(row=0, column=2, padx=10)

# ======================================================
# DROPDOWN MENU
# ======================================================
device_menu = tk.OptionMenu(
    root,
    selected_device,
    *audio_devices
)

device_menu.config(
    font=("Helvetica", 12),
    bg="#2e2e2e",
    fg="white",
    width=30
)
device_menu.place(
    x=900,
    y=740
)
#device_menu.pack(pady=10)

# ======================================================
# WEBCAM
# ======================================================
cap = cv2.VideoCapture(0)

# ======================================================
# AUDIO CALLBACK GUI
# ======================================================
def update_audio_gui(emotion, confidence):

    global audio_emotion
    global audio_confidence

    audio_emotion = emotion
    audio_confidence = confidence


# ======================================================
# UPDATE CAMERA
# ======================================================
def update_frame():

    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # =====================================
        # MODE AUDIO
        # =====================================
        if CURRENT_MODE == "audio":


            # update text audio
            emotion_label.config(
                text=f"🔊 {audio_emotion.upper()}"
            )

            confidence_label.config(
                text=f"Confidence : {audio_confidence:.1f}%"
            )

        else:


            # =====================================
            # FACE DETECTION
            # =====================================
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (x, y, w, h) in faces:

                # =====================================
                # FACE PREPROCESS
                # =====================================
                face = gray[y:y+h, x:x+w]

                face = cv2.resize(
                    face,
                    (96,96)
                )

                face = face.astype(
                    "float32"
                ) / 255.0

                face = np.expand_dims(
                    face,
                    axis=-1
                )

                face = np.expand_dims(
                    face,
                    axis=0
                )

                # =====================================
                # VISUAL PREDICT
                # =====================================
                visual_prob = visual_model.predict(
                    face,
                    verbose=0
                )[0]

                visual_idx = np.argmax(
                    visual_prob
                )

                visual_emotion = EMOTIONS[
                    visual_idx
                ]

                visual_confidence = (
                    visual_prob[visual_idx] * 100
                )

                # =====================================
                # FUSION
                # =====================================
                fusion_prob = (
                    visual_prob * 0.75 +
                    audio_probabilities * 0.25
                )

                fusion_idx = np.argmax(
                    fusion_prob
                )

                fusion_emotion = EMOTIONS[
                    fusion_idx
                ]

                fusion_confidence = (
                    fusion_prob[fusion_idx] * 100
                )

                # =====================================
                # MODE
                # =====================================
                if CURRENT_MODE == "visual":

                    final_emotion = visual_emotion
                    final_conf = visual_confidence

                else:

                    final_emotion = fusion_emotion
                    final_conf = fusion_confidence

                # =====================================
                # EMOJI
                # =====================================
                emoji = "😐"

                if final_emotion == "happy":
                    emoji = "😊"

                elif final_emotion == "sad":
                    emoji = "😢"

                elif final_emotion == "angry":
                    emoji = "😠"

                # =====================================
                # FACE BOX
                # =====================================
                cv2.rectangle(
                    frame,
                    (x,y),
                    (x+w,y+h),
                    (0,255,0),
                    3
                )

                # =====================================
                # TEXT CAMERA
                # =====================================
                cv2.putText(
                    frame,
                    f"{final_emotion} ({final_conf:.1f}%)",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                # =====================================
                # GUI TEXT
                # =====================================
                emotion_label.config(
                    text=f"{emoji} {final_emotion.upper()}"
                )

                confidence_label.config(
                    text=f"Confidence : {final_conf:.1f}%"
                )
                # =====================================
                # SPEAK
                # =====================================
                speak_emotion(final_emotion)

        # =====================================
        # RGB
        # =====================================
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(frame)

        img = img.resize((650,420))

        imgtk = ImageTk.PhotoImage(
            image=img
        )

        camera_label.imgtk = imgtk

        camera_label.configure(
            image=imgtk
        )

    root.after(10, update_frame)

# ======================================================
# START
# =====================================================
update_frame()
threading.Thread(
    target=audio_process,
    daemon=True
).start()


# ======================================================
# CLOSE
# ======================================================
def on_closing():

    cap.release()

    root.destroy()

root.protocol(
    "WM_DELETE_WINDOW",
    on_closing
)

# ======================================================
# MAINLOOP
# ======================================================
root.mainloop()