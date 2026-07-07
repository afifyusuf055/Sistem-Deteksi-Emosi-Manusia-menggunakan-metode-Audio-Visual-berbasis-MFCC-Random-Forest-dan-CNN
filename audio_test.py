"""
=========================================================
AUDIO TEST
MFCC + RANDOM FOREST
=========================================================
"""

import sounddevice as sd
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

# ======================================================
# PATH
# ======================================================

MODEL_PATH = "models/rf_audio_mfcc.pkl"
SCALER_PATH = "models/audio_scaler.pkl"

# ======================================================
# AUDIO
# ======================================================

SAMPLE_RATE = 16000
DURATION = 3

# ======================================================
# LOAD MODEL
# ======================================================

print("Loading Audio Model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Random Forest Loaded")

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
# LIST MICROPHONE
# ======================================================

print("\n========== MICROPHONE ==========\n")

devices = sd.query_devices()

for i, dev in enumerate(devices):

    if dev["max_input_channels"] > 0:

        print(f"[{i}] {dev['name']}")

mic = int(input("\nPilih Microphone : "))

print()

# ======================================================
# LOOP
# ======================================================

while True:

    input("Tekan ENTER untuk mulai merekam...")

    print("\nRecording...")

    audio = sd.rec(

        int(SAMPLE_RATE * DURATION),

        samplerate=SAMPLE_RATE,

        channels=1,

        dtype="float32",

        device=mic

    )

    sd.wait()

    audio = audio.flatten()

    print("Jumlah Sample :", len(audio))

  # ==================================================
    # AUDIO ANALYSIS
    # ==================================================

    duration = len(audio) / SAMPLE_RATE

    max_amp = np.max(audio)

    min_amp = np.min(audio)

    rms = np.sqrt(np.mean(audio**2))

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    time_axis = np.linspace(0, duration, len(audio))

    plt.figure(figsize=(12,6))

    plt.clf()

    plt.plot(
        time_axis,
        audio,
        color="royalblue",
        linewidth=1
    )

    plt.title(
        "Audio Emotion Analysis",
        fontsize=16,
        weight="bold"
    )

    plt.xlabel("Time (second)")

    plt.ylabel("Amplitude")

    plt.grid(True)

    info = (
        f"Emotion    : {emotion.upper()}\n"
        f"Confidence : {confidence:.2f}%\n\n"
        f"Sample Rate : {SAMPLE_RATE} Hz\n"
        f"Duration    : {duration:.2f} s\n"
        f"Samples     : {len(audio)}\n\n"
        f"Max Amp     : {max_amp:.3f}\n"
        f"Min Amp     : {min_amp:.3f}\n"
        f"RMS Energy  : {rms:.4f}\n"
        f"ZCR         : {zcr:.4f}"
    )

    plt.gcf().text(
        0.72,
        0.55,
        info,
        fontsize=11,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round"
        )
    )

    plt.tight_layout()

    plt.show()

    # ==================================================
    # NORMALISASI
    # ==================================================

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # ==================================================
    # PREEMPHASIS
    # ==================================================

    audio = librosa.effects.preemphasis(audio)

    # ==================================================
    # MFCC
    # ==================================================

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=SAMPLE_RATE,

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

        sr=SAMPLE_RATE

    )

    # ==================================================
    # FEATURE 132
    # ==================================================

    feature = np.hstack([

        np.mean(mfcc.T, axis=0),

        np.mean(delta.T, axis=0),

        np.mean(delta2.T, axis=0),

        np.mean(chroma.T, axis=0)

    ])

    feature = np.nan_to_num(feature)

    feature = feature.reshape(1, -1)

    feature = scaler.transform(feature)

    # ==================================================
    # PREDIKSI
    # ==================================================

    probability = model.predict_proba(feature)[0]

    idx = np.argmax(probability)

    emotion = EMOTIONS[idx]

    confidence = probability[idx] * 100

    print("\n===============================")
    print("HASIL PREDIKSI")
    print("===============================")

    print("Emotion    :", emotion.upper())
    print("Confidence :", f"{confidence:.2f}%")

    print("\nProbability")

    for i, emo in enumerate(EMOTIONS):

        print(f"{emo:<8} : {probability[i]*100:.2f}%")

    print("===============================\n")