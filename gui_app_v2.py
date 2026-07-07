"""
=========================================================
GUI V2
Human Emotion Detection System
Audio - Visual - Fusion
Author : M. Afif Fuadie Yusuf
=========================================================
"""

import tkinter as tk
from tkinter import ttk

from PIL import Image
from PIL import ImageTk
from datetime import datetime

import cv2
import numpy as np
from audio_engine import (
    start_audio,
    get_audio_result
)
from visual_engine import predict_frame
from fusion_engine import predict_fusion

import threading
import queue
import sounddevice as sd

import pygame
import librosa


import time
import sys
import os

# ======================================================
# INIT SPEAKER
# ======================================================

pygame.mixer.init()
# ======================================================
# SPEAKER VARIABLE
# ======================================================

last_emotion = ""
last_play_time = 0

PLAY_DELAY = 3      # detik

# ======================================================
# GET AUDIO DEVICES
# ======================================================

def load_audio_devices():

    input_devices = []
    output_devices = []

    devices = sd.query_devices()

    for i, dev in enumerate(devices):

        if dev["max_input_channels"] > 0:
            input_devices.append(f"{i} - {dev['name']}")

        if dev["max_output_channels"] > 0:
            output_devices.append(f"{i} - {dev['name']}")

    return input_devices, output_devices


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
# LABEL
# ======================================================

EMOTIONS = [

    "angry",

    "happy",

    "neutral",

    "sad"

]

# ======================================================
# MODE
# ======================================================

current_mode = "fusion"

# ======================================================
# AUDIO BUFFER
# ======================================================

audio_buffer = np.zeros(70)


# ======================================================
# CAMERA
# ======================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera tidak ditemukan")

# ======================================================
# ROOT
# ======================================================

root = tk.Tk()

root.title("Deteksi Emosi Audio Visual")

root.geometry("1360x760")

root.configure(
    bg="#edf2f7"
)

root.resizable(False,False)

# ======================================================
# STYLE
# ======================================================

style = ttk.Style()

style.theme_use("clam")

style.configure(

    "Card.TFrame",

    background="white"

)

style.configure(

    "Title.TLabel",

    font=("Segoe UI",18,"bold"),

    background="#0d1b52",

    foreground="white"

)

style.configure(

    "Header.TLabel",

    font=("Segoe UI",13,"bold"),

    background="white"

)

style.configure(

    "Normal.TLabel",

    font=("Segoe UI",11),

    background="white"

)

# ======================================================
# HEADER
# ======================================================

header = tk.Frame(

    root,

    bg="#0d1b52",

    height=50

)

header.pack(

    fill="x"

)

header.pack_propagate(False)

title = tk.Label(

    header,

    text="📄 Deteksi Emosi Audio-Visual",

    bg="#0d1b52",

    fg="white",

    font=("Segoe UI",17,"bold")

)

title.pack(

    side="left",

    padx=20,

    pady=8

)

# ======================================================
# MAIN
# ======================================================

main = tk.Frame(

    root,

    bg="#edf2f7"

)

main.pack(

    fill="both",

    expand=True,

    padx=15,

    pady=15

)

# ======================================================
# LEFT PANEL
# ======================================================

left_panel = ttk.Frame(

    main,

    style="Card.TFrame",

    width=180

)

left_panel.pack(

    side="left",

    fill="y"

)

left_panel.pack_propagate(False)

# ======================================================
# MODE
# ======================================================

mode_title = tk.Label(

    left_panel,

    text="MODE",

    font=("Segoe UI",12,"bold"),

    bg="white",

    fg="#0d1b52"

)

mode_title.pack(

    pady=(20,10)

)

audio_btn = tk.Button(

    left_panel,

    text="🎤  Audio",

    command=lambda: change_mode("audio"),

    font=("Segoe UI",12),

    bg="white",

    width=16,

    relief="groove",

    cursor="hand2"

)

audio_btn.pack(

    pady=5

)

visual_btn = tk.Button(

    left_panel,

    text="📷  Visual",

    command=lambda: change_mode("visual"),

    font=("Segoe UI",12),

    bg="white",

    width=16,

    relief="groove",

    cursor="hand2"

)

visual_btn.pack(

    pady=5

)

fusion_btn = tk.Button(

    left_panel,

    text="🧠  Fusion",

    command=lambda: change_mode("fusion"),

    font=("Segoe UI",12,"bold"),

    bg="#008080",

    fg="white",

    width=16,

    relief="flat",

    cursor="hand2"

)

fusion_btn.pack(

    pady=5

)

mic_title = tk.Label(

    left_panel,

    text="MICROPHONE",

    font=("Segoe UI",11,"bold"),

    bg="white"

)

mic_title.pack(

    pady=(25,10)

)

mic_combo = ttk.Combobox(

    left_panel,

    width=18,

    state="readonly"

)

# ======================================================
# TOGGLE SPEAKER
# ======================================================

speaker_enabled = True

def toggle_speaker():

    global speaker_enabled

    speaker_enabled = not speaker_enabled

    if speaker_enabled:

        speaker_btn.config(
            text="ON",
            bg="#009966",
            fg="white"
        )

        print("🔊 Speaker ON")

    else:

        speaker_btn.config(
            text="OFF",
            bg="red",
            fg="white"
        )

        print("🔇 Speaker OFF")

# ============================================
# LOAD MICROPHONE LIST
# ============================================

input_devices, output_devices = load_audio_devices()

mic_combo["values"] = input_devices

if input_devices:
    mic_combo.current(0)

mic_combo.pack()


mic_combo.pack()

speaker_title = tk.Label(

    left_panel,

    text="SPEAKER",

    font=("Segoe UI",11,"bold"),

    bg="white"

)

speaker_title.pack(

    pady=(25,10)

)

speaker_btn = tk.Button(

    left_panel,

    text="ON",

    bg="#009966",

    fg="white",

    width=10,

    relief="flat",

    command= toggle_speaker

)

speaker_btn.pack()


# ======================================================
# CHANGE MODE
# ======================================================

def change_mode(mode):

    global current_mode

    current_mode = mode

    print("MODE :", current_mode)

    # Reset semua tombol
    audio_btn.config(
        bg="white",
        fg="black",
        relief="groove"
    )

    visual_btn.config(
        bg="white",
        fg="black",
        relief="groove"
    )

    fusion_btn.config(
        bg="white",
        fg="black",
        relief="groove"
    )

    # Highlight tombol aktif
    if mode == "audio":

        audio_btn.config(
            bg="#008080",
            fg="white",
            relief="flat"
        )

    elif mode == "visual":

        visual_btn.config(
            bg="#008080",
            fg="white",
            relief="flat"
        )

    elif mode == "fusion":

        fusion_btn.config(
            bg="#008080",
            fg="white",
            relief="flat"
        )

# ======================================================
# DEFAULT MODE
# ======================================================

current_mode = "fusion"

change_mode("fusion")

# ======================================================
# SPEAKER STATUS
# ======================================================

speaker_enabled = True

# ======================================================
# AUDIO DEVICE
# ======================================================

selected_output_device = None

# ======================================================
# CENTER PANEL
# ======================================================

center_panel = ttk.Frame(

    main,

    style="Card.TFrame"

)

center_panel.pack(

    side="left",

    fill="both",

    expand=True,

    padx=15

)

# ======================================================
# CAMERA FRAME
# ======================================================

camera_frame = tk.LabelFrame(

    center_panel,

    text="Camera",

    font=("Segoe UI",12,"bold"),

    bg="white",

    fg="#0d1b52",

    padx=10,

    pady=10

)

camera_frame.pack(

    fill="both",

    expand=True,

    pady=(0,10)

)

# ======================================================
# CAMERA LABEL
# ======================================================

camera_label = tk.Label(

    camera_frame,

    bg="black",

    width=640,

    height=380

)

camera_label.pack(
    padx=10,
    pady=10
)

# ======================================================
# WAVEFORM FRAME
# ======================================================

wave_frame = tk.LabelFrame(

    center_panel,

    text="Realtime Audio Waveform",

    font=("Segoe UI",11,"bold"),

    bg="white",

    fg="#0d1b52"

)

wave_frame.pack(

    fill="x",

    pady=(0,5)

)

wave_canvas = tk.Canvas(

    wave_frame,

    width=760,

    height=70,          # Sebelumnya 120

    bg="#101010",

    highlightthickness=0

)

wave_canvas.pack(

    fill="x",

    padx=10,

    pady=8

)

# ======================================================
# WAVEFORM
# ======================================================

wave_items = []

def draw_waveform(levels):

    wave_canvas.delete("all")

    width = 760

    height = 120

    bar_width = width / len(levels)

    for i, level in enumerate(levels):

        x = i * bar_width

        h = level * 80

        wave_canvas.create_rectangle(

            x,

            height/2-h,

            x+bar_width-2,

            height/2+h,

            fill="#00ff66",

            outline=""

        )

# ======================================================
# AUDIO CALLBACK
# ======================================================

def audio_callback(indata, frames, time, status):

    global audio_buffer

    if status:
        print(status)

    audio = indata[:,0]

    # Bagi audio menjadi 70 bagian
    chunk = np.array_split(audio, 70)

    audio_buffer = np.array([

        np.max(np.abs(c))

        for c in chunk

    ])

# ======================================================
# START AUDIO STREAM
# ======================================================

stream = sd.InputStream(

    samplerate=16000,

    channels=1,

    callback=audio_callback

)

stream.start()

# ======================================================
# UPDATE WAVEFORM
# ======================================================

def update_waveform():

    draw_waveform(audio_buffer)

    root.after(
        30,
        update_waveform
    )

# ======================================================
# AUDIO THREAD
# ======================================================

def audio_thread():

    global audio_emotion
    global audio_confidence
    global audio_probability

    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        callback=audio_callback
    )

    stream.start()

    print("🎤 Audio Thread Running")

    while True:

        time.sleep(0.1)

# ======================================================
# UPDATE AUDIO GUI
# ======================================================

audio_emotion = "Neutral"
audio_confidence = 0


def update_audio_gui(emotion, confidence):

    global audio_emotion
    global audio_confidence

    if current_mode == "visual":
         return

    audio_emotion = emotion
    audio_confidence = confidence

    audio_result_label.config(
        text=f"🎤 {emotion.upper()}"
    )

    audio_conf_label.config(
        text=f"Confidence : {confidence:.2f}%"
    )

    audio_bar["value"] = confidence

# ======================================================
# UPDATE VISUAL GUI
# ======================================================

def update_visual_gui(emotion, confidence, probability, faces):

    # ======================================================
    # TIDAK ADA WAJAH
    # ======================================================

    if faces is not None and len(faces) == 0:

        emotion_label.config(
            text="😐 NO FACE",
            fg="gray"
        )

        confidence_label.config(
            text="Confidence : 0 %"
        )

        confidence_bar["value"] = 0

        happy_value.config(
            text="0 %"
        )

        sad_value.config(
            text="0 %"
        )

        angry_value.config(
            text="0 %"
        )

        neutral_value.config(
            text="0 %"
        )

        return

    # ======================================================
    # EMOJI
    # ======================================================

    if emotion == "happy":

        emoji = "😊"
        color = "#00aa00"

    elif emotion == "sad":

        emoji = "😢"
        color = "#0066ff"

    elif emotion == "angry":

        emoji = "😠"
        color = "#ff0000"

    else:

        emoji = "😐"
        color = "#008800"

    # ======================================================
    # HASIL PREDIKSI
    # ======================================================

    emotion_label.config(
        text=f"{emoji} {emotion.upper()}",
        fg=color
    )

    confidence_label.config(
        text=f"Confidence : {confidence:.2f}%"
    )

    confidence_bar["value"] = confidence

    # ======================================================
    # PROBABILITAS
    # ======================================================

    # Reset warna default
    happy_label.config(fg="#00A651")
    happy_value.config(
        text=f"{probability[1]*100:.2f}%",
        fg="#00A651"
    )

    sad_label.config(fg="#1E88E5")
    sad_value.config(
        text=f"{probability[3]*100:.2f}%",
        fg="#1E88E5"
    )

    angry_label.config(fg="#E53935")
    angry_value.config(
        text=f"{probability[0]*100:.2f}%",
        fg="#E53935"
    )

    neutral_label.config(fg="#616161")
    neutral_value.config(
        text=f"{probability[2]*100:.2f}%",
        fg="#616161"
    )

    # Highlight emosi yang terdeteksi
    if emotion == "happy":

        happy_label.config(
            font=("Segoe UI",12,"bold"),
            bg="#DFFFE2"
        )

    elif emotion == "sad":

        sad_label.config(
            font=("Segoe UI",12,"bold"),
            bg="#DCEEFF"
        )

    elif emotion == "angry":

        angry_label.config(
            font=("Segoe UI",12,"bold"),
            bg="#FFE3E3"
        )

    else:

        neutral_label.config(
            font=("Segoe UI",12,"bold"),
            bg="#EEEEEE"
        )
# ======================================================
# RUN AUDIO MODE
# ======================================================

def run_audio_mode():

    emotion, confidence, probability = get_audio_result()

    update_audio_gui(
        emotion,
        confidence
    )

    emotion_label.config(
        text=f"🎤 {emotion.upper()}",
        fg="blue"
    )

    confidence_label.config(
        text=f"Confidence : {confidence:.2f}%"
    )

    confidence_bar["value"] = confidence

    happy_value.config(
        text=f"{probability[1]*100:.2f}%"
    )

    sad_value.config(
        text=f"{probability[3]*100:.2f}%"
    )

    angry_value.config(
        text=f"{probability[0]*100:.2f}%"
    )

    neutral_value.config(
        text=f"{probability[2]*100:.2f}%"
    )

    # ==============================
    # Tambahkan Log Prediksi
    # ==============================
    add_log(
        "AUDIO",
        emotion,
        confidence
    )

    # ==============================
    # Output Speaker
    # ==============================
    play_emotion_sound(
        emotion
    )

    # ==============================
    # Nonaktifkan Panel Visual
    # ==============================
    disable_visual_panel()

# ======================================================
# RUN VISUAL MODE
# ======================================================

def run_visual_mode(frame):

    (
        frame,
        emotion,
        confidence,
        probability,
        faces
    ) = predict_frame(frame)

    print("===== VISUAL MODE =====")
    print("Emotion :", emotion)
    print("Confidence :", confidence)

    update_visual_gui(
        emotion,
        confidence,
        probability,
        faces
    )
    add_log(
        "VISUAL",
        emotion,
        confidence
    )
    play_emotion_sound(emotion)
    disable_audio_panel()

    # ======================================================
    # UPDATE VISUAL RESULT PANEL
    # ======================================================

    visual_result_label.config(
    text=f"Visual : {emotion.upper()}"
    )

    visual_confidence_label.config(
    text=f"Confidence : {confidence:.2f}%"
    )

    visual_confidence_bar["value"] = confidence

    return frame

# ======================================================
# RUN FUSION MODE
# ======================================================

def run_fusion_mode(frame):

    (
        frame,
        fusion_emotion,
        fusion_conf,
        fusion_prob,
        visual_emotion,
        visual_conf,
        audio_emotion,
        audio_conf,
        faces
    ) = predict_fusion(frame)

    update_visual_gui(
        fusion_emotion,
        fusion_conf,
        fusion_prob,
        faces
    )
    add_log(
        "FUSION",
        fusion_emotion,
        fusion_conf
    )
    play_emotion_sound(fusion_emotion)

    # ======================================================
    # UPDATE VISUAL RESULT
    # ======================================================

    visual_result_label.config(
    text=f"Visual : {visual_emotion.upper()}"
    )

    visual_confidence_label.config(
        text=f"Confidence : {visual_conf:.2f}%"
    )

    visual_confidence_bar["value"] = visual_conf
    
    return frame

# ======================================================
# RESET VISUAL PANEL
# ======================================================

def disable_visual_panel():

    visual_result_label.config(
        text="Visual : Not Active"
    )

    visual_confidence_label.config(
        text="Confidence : 0 %"
    )

    visual_confidence_bar["value"] = 0


# ======================================================
# RESET AUDIO PANEL
# ======================================================

def disable_audio_panel():

    audio_result_label.config(
        text="Audio : Not Active"
    )

    audio_conf_label.config(
        text="Confidence : 0 %"
    )

    audio_bar["value"] = 0

# ======================================================
# TOGGLE SPEAKER
# ======================================================

def toggle_speaker():

    global speaker_enabled

    speaker_enabled = not speaker_enabled

    if speaker_enabled:

        speaker_btn.config(
            text="ON",
            bg="#009966",
            fg="white"
        )

        print("🔊 Speaker ON")

    else:

        speaker_btn.config(
            text="OFF",
            bg="red",
            fg="white"
        )

        print("🔇 Speaker OFF")

# ======================================================
# PLAY EMOTION SOUND
# ======================================================

last_emotion = ""
last_play_time = 0
PLAY_DELAY = 3      # detik

def play_emotion_sound(emotion):

    global speaker_enabled
    global last_emotion
    global last_play_time

    if not speaker_enabled:
        return

    current_time = time.time()

    # Jika speaker masih memutar suara, jangan putar lagi
    if pygame.mixer.music.get_busy():
        return

    # Jika emosinya sama dan belum 3 detik, jangan ulangi
    if emotion == last_emotion:

        if current_time - last_play_time < PLAY_DELAY:
            return

    sound_file = os.path.join(
        "sounds",
        f"{emotion}.wav"
    )

    if not os.path.exists(sound_file):

        print("❌ File tidak ditemukan :", sound_file)
        return

    try:

        pygame.mixer.music.load(sound_file)

        pygame.mixer.music.play()

        last_emotion = emotion
        last_play_time = current_time

    except Exception as e:

        print("Speaker Error :", e)

# ======================================================
# ADD LOG
# ======================================================

last_log = ""

def add_log(mode, emotion, confidence):

    global last_log

    log = f"{mode}-{emotion}"

    if log == last_log:
        return

    last_log = log

    waktu = datetime.now().strftime("%H:%M:%S")

    teks = f"[{waktu}] {mode:<7} | {emotion.upper():<8} | {confidence:.2f}%"

    log_list.insert(tk.END, teks)

    # Auto scroll ke bawah
    log_list.yview(tk.END)

    # Maksimal simpan 100 log
    if log_list.size() > 100:
        log_list.delete(0)


# ======================================================
# UPDATE CAMERA
# ======================================================

def update_camera():

    global current_mode

    ret, frame = cap.read()

    if not ret:

        root.after(15, update_camera)
        return

    # Mirror Camera
    frame = cv2.flip(frame, 1)

    # ======================================================
    # PILIH MODE
    # ======================================================

    if current_mode == "audio":

        run_audio_mode()

    elif current_mode == "visual":

        frame = run_visual_mode(frame)

    elif current_mode == "fusion":

        frame = run_fusion_mode(frame)

    # ======================================================
    # TAMPILKAN KAMERA
    # ======================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(rgb)

    img = img.resize((700,420))

    imgtk = ImageTk.PhotoImage(
        image=img
    )

    camera_label.imgtk = imgtk

    camera_label.configure(
        image=imgtk
    )

    root.after(
        15,
        update_camera
    )

# ======================================================
# RIGHT PANEL
# ======================================================

right_panel = ttk.Frame(

    main,

    style="Card.TFrame",

    width=320

)

right_panel.pack(

    side="right",

    fill="y"

)

# ======================================================
# RESULT FRAME
# ======================================================

result_frame = tk.LabelFrame(

    right_panel,

    text="Hasil Prediksi",

    font=("Segoe UI",12,"bold"),

    bg="white",

    fg="#0d1b52"

)

result_frame.pack(

    fill="x",

    padx=10,

    pady=10

)

emotion_label = tk.Label(

    result_frame,

    text="😐 Neutral",

    font=("Segoe UI",24,"bold"),

    fg="green",

    bg="white"

)

emotion_label.pack(
    pady=(15,10)
)

confidence_label = tk.Label(

    result_frame,

    text="Confidence : 0 %",

    font=("Segoe UI",16),

    bg="white"

)

confidence_label.pack(
    pady=5
)

confidence_bar = ttk.Progressbar(

    result_frame,

    orient="horizontal",

    length=220,

    mode="determinate"

)

confidence_bar.pack(
    pady=15
)

# ======================================================
# PROBABILITY FRAME
# ======================================================

prob_frame = tk.LabelFrame(
    right_panel,
    text="PROBABILITAS",
    font=("Segoe UI", 11, "bold"),
    bg="white",
    fg="#0d1b52"
)

prob_frame.pack(
    fill="x",
    padx=10,
    pady=(5,10)
)

happy_label = tk.Label(
    prob_frame,
    text="🟢 Happy",
    bg="white",
    font=("Segoe UI",11)
)
happy_label.grid(row=0,column=0,sticky="w",padx=10,pady=5)

happy_value = tk.Label(
    prob_frame,
    text="0 %",
    bg="white",
    font=("Segoe UI",11,"bold")
)
happy_value.grid(row=0,column=1,sticky="e",padx=10)

sad_label = tk.Label(
    prob_frame,
    text="🔵 Sad",
    bg="white",
    font=("Segoe UI",11)
)
sad_label.grid(row=1,column=0,sticky="w",padx=10,pady=5)

sad_value = tk.Label(
    prob_frame,
    text="0 %",
    bg="white",
    font=("Segoe UI",11,"bold")
)
sad_value.grid(row=1,column=1,sticky="e",padx=10)

angry_label = tk.Label(
    prob_frame,
    text="🔴 Angry",
    bg="white",
    font=("Segoe UI",11)
)
angry_label.grid(row=2,column=0,sticky="w",padx=10,pady=5)

angry_value = tk.Label(
    prob_frame,
    text="0 %",
    bg="white",
    font=("Segoe UI",11,"bold")
)
angry_value.grid(row=2,column=1,sticky="e",padx=10)

neutral_label = tk.Label(
    prob_frame,
    text="⚪ Neutral",
    bg="white",
    font=("Segoe UI",11)
)
neutral_label.grid(row=3,column=0,sticky="w",padx=10,pady=5)

neutral_value = tk.Label(
    prob_frame,
    text="0 %",
    bg="white",
    font=("Segoe UI",11,"bold")
)
neutral_value.grid(row=3,column=1,sticky="e",padx=10)


# ======================================================
# VISUAL RESULT
# ======================================================

visual_frame = tk.LabelFrame(
    right_panel,
    text="VISUAL RESULT",
    font=("Segoe UI",12,"bold")
)

visual_frame.pack(
    fill="x",
    padx=10,
    pady=5
)

# ======================================================
# VISUAL LABEL
# ======================================================

visual_result_label = tk.Label(
    visual_frame,
    text="Visual : -",
    font=("Segoe UI",16,"bold"),
    fg="darkgreen"
)

visual_result_label.pack(
    pady=(10,5)
)

# ======================================================
# VISUAL CONFIDENCE
# ======================================================

visual_confidence_label = tk.Label(
    visual_frame,
    text="Confidence : 0 %",
    font=("Segoe UI",12)
)

visual_confidence_label.pack()

# ======================================================
# VISUAL BAR
# ======================================================

visual_confidence_bar = ttk.Progressbar(
    visual_frame,
    orient="horizontal",
    mode="determinate",
    length=220
)

visual_confidence_bar.pack(
    pady=10
)

# ======================================================
# AUDIO RESULT
# ======================================================

audio_frame = tk.LabelFrame(

    right_panel,

    text="AUDIO RESULT",

    font=("Segoe UI",11,"bold"),

    bg="white",

    fg="#0d1b52"

)

audio_frame.pack(

    fill="x",

    padx=10,

    pady=10

)

audio_result_label = tk.Label(

    audio_frame,

    text="🎤 Waiting...",

    font=("Segoe UI",18,"bold"),

    bg="white",

    fg="blue"

)

audio_result_label.pack(

    pady=(15,5)

)

audio_conf_label = tk.Label(

    audio_frame,

    text="Confidence : 0 %",

    font=("Segoe UI",11),

    bg="white"

)

audio_conf_label.pack(

    pady=5

)

audio_bar = ttk.Progressbar(

    audio_frame,

    orient="horizontal",

    mode="determinate",

    length=220

)

audio_bar.pack(

    pady=(5,15)

)

right_panel.pack_propagate(False)


# ======================================================
# LOG FRAME
# ======================================================

log_frame = tk.LabelFrame(

    center_panel,

    text="LOG PREDIKSI",

    font=("Segoe UI",11,"bold"),

    bg="white",

    fg="#0d1b52"

)

log_frame.pack(

    fill="both",

    expand=True,

    pady=(5,0)

)

# Scrollbar
log_scroll = tk.Scrollbar(log_frame)

log_scroll.pack(
    side="right",
    fill="y"
)

# List Log
log_list = tk.Listbox(

    log_frame,

    height=8,

    font=("Consolas",10),

    yscrollcommand=log_scroll.set,

    borderwidth=0

)

log_list.pack(

    side="left",

    fill="both",

    expand=True,

    padx=10,

    pady=10

)

log_scroll.config(
    command=log_list.yview
)

# Tombol Clear
clear_btn = tk.Button(

    log_frame,

    text="Clear",

    font=("Segoe UI",10,"bold"),

    bg="white",

    width=8,

    relief="groove",

    command=lambda: log_list.delete(0, tk.END)

)

clear_btn.pack(

    anchor="e",

    padx=10,

    pady=(0,10)

)

# ======================================================
# LOG PREDIKSI
# ======================================================

log_text = tk.Text(

    log_frame,

    height=7,

    font=("Consolas",10),

    bg="white",

    fg="black"

)

log_text.pack(

    fill="both",

    expand=True,

    padx=5,

    pady=5

)

log_text.config(state="disabled")



# ======================================================
# MAIN LOOP
# ======================================================

update_camera()

update_waveform()

threading.Thread(

    target=lambda: start_audio(update_audio_gui),

    daemon=True

).start()

root.mainloop()