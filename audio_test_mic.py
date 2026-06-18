import sounddevice as sd
import numpy as np
import librosa
import joblib
import queue
import time

# ======================================================
# PATH
# ======================================================
MODEL_PATH = "models/rf_audio_mfcc.pkl"
SCALER_PATH = "models/audio_scaler.pkl"

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
# VARIABLE
# ======================================================
audio_q = queue.Queue()

RUNNING = False

# ======================================================
# LOAD MODEL
# ======================================================
def load_model():

    try:

        model = joblib.load(
            MODEL_PATH
        )

        scaler = joblib.load(
            SCALER_PATH
        )

        print("✅ Model audio berhasil dimuat")

        return model, scaler

    except Exception as e:

        print("❌ Gagal load model:", e)

        return None, None

# ======================================================
# CALLBACK AUDIO
# ======================================================
def callback(indata, frames, time_, status):

    if status:
        print(status)

    audio_q.put(
        indata.copy()
    )

# ======================================================
# PREPROCESS AUDIO
# ======================================================
def preprocess_audio(audio):

    # =====================================
    # NORMALISASI
    # =====================================
    audio = audio / (
        np.max(np.abs(audio)) + 1e-6
    )

    # =====================================
    # PREEMPHASIS
    # =====================================
    audio = np.append(
        audio[0],
        audio[1:] - 0.97 * audio[:-1]
    )

    return audio

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_feature(audio):

    # =====================================
    # MFCC
    # =====================================
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40
    )

    # =====================================
    # DELTA
    # =====================================
    delta = librosa.feature.delta(
        mfcc
    )

    # =====================================
    # DELTA2
    # =====================================
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # =====================================
    # CHROMA
    # =====================================
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=16000
    )

    # =====================================
    # GABUNG FEATURE
    # =====================================
    feature = np.hstack([

        np.mean(mfcc.T, axis=0),

        np.mean(delta.T, axis=0),

        np.mean(delta2.T, axis=0),

        np.mean(chroma.T, axis=0)

    ])

    feature = np.nan_to_num(
        feature
    )

    return feature.reshape(1, -1)

# ======================================================
# START AUDIO
# ======================================================
def start_audio(callback_gui):

    global RUNNING

    RUNNING = True

    # =====================================
    # LOAD MODEL
    # =====================================
    model, scaler = load_model()

    if model is None:
        return

    # =====================================
    # HISTORY
    # =====================================
    history = []

    try:

        with sd.InputStream(

            channels=1,

            samplerate=16000,

            blocksize=48000,  # 3 DETIK

            callback=callback

        ):

            print("🎤 Audio aktif... (3 detik window)")

            while RUNNING:

                if not audio_q.empty():

                    # =====================================
                    # GET AUDIO
                    # =====================================
                    audio = audio_q.get().flatten()

                    # =====================================
                    # FILTER SUARA KECIL
                    # =====================================
                    if np.max(np.abs(audio)) < 0.03:
                        continue

                    # =====================================
                    # PREPROCESS
                    # =====================================
                    audio = preprocess_audio(
                        audio
                    )

                    # =====================================
                    # FEATURE
                    # =====================================
                    feat = extract_feature(
                        audio
                    )

                    # =====================================
                    # SCALER
                    # =====================================
                    feat = scaler.transform(
                        feat
                    )

                    # =====================================
                    # PREDICT PROBA
                    # =====================================
                    prob = model.predict_proba(
                        feat
                    )[0]

                    # =====================================
                    # SMOOTHING
                    # =====================================
                    history.append(prob)

                    if len(history) > 2:
                        history.pop(0)

                    avg_prob = np.mean(
                        history,
                        axis=0
                    )

                    # =====================================
                    # HASIL
                    # =====================================
                    idx = np.argmax(
                        avg_prob
                    )

                    confidence = (
                        avg_prob[idx] * 100
                    )

                    emotion = EMOTIONS[idx]

                    # =====================================
                    # THRESHOLD
                    # =====================================
                    if confidence < 50:

                        emotion = "neutral"

                    # =====================================
                    # PRINT
                    # =====================================
                    print(
                        f"🎧 {emotion} ({confidence:.2f}%)"
                    )

                    # =====================================
                    # CALLBACK GUI
                    # =====================================
                    callback_gui(
                        emotion,
                        confidence
                    )

                time.sleep(0.2)

    except Exception as e:

        print("❌ Error audio:", e)

# ======================================================
# STOP AUDIO
# ======================================================
def stop_audio():

    global RUNNING

    RUNNING = False

    print("🛑 Audio dihentikan")

# ======================================================
# TEST TERMINAL
# ======================================================
if __name__ == "__main__":

    def tampil(emotion, confidence):

        print(
            f"🎧 {emotion} ({confidence:.2f}%)"
        )

    try:

        start_audio(tampil)

    except KeyboardInterrupt:

        stop_audio()

        print("🛑 Program dihentikan")