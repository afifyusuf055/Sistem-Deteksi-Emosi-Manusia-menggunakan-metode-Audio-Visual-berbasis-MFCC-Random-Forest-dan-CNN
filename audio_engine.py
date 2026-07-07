"""
=========================================================
AUDIO TEST V3
Emotion Recognition Audio Engine
Author : M. Afif Fuadie Yusuf
=========================================================
"""

import sounddevice as sd
import numpy as np
import librosa
import joblib
import queue
import time


# =====================================================
# PATH
# =====================================================

MODEL_PATH = "models/rf_audio_mfcc.pkl"
SCALER_PATH = "models/audio_scaler.pkl"

# =====================================================
# PARAMETER
# =====================================================

SAMPLE_RATE = 16000

RECORD_SECONDS = 3

BLOCKSIZE = SAMPLE_RATE * RECORD_SECONDS

N_MFCC = 40

HISTORY_SIZE = 2

VOICE_THRESHOLD = 0.01

# =====================================================
# LABEL
# =====================================================

EMOTIONS = [
    "angry",
    "happy",
    "neutral",
    "sad"
]

# =====================================================
# VARIABLE
# =====================================================

audio_q = queue.Queue()

RUNNING = False

history = []

# =====================================================
# LAST RESULT
# =====================================================

last_emotion = "neutral"

last_confidence = 0.0

last_probability = np.zeros(4)

# =====================================================
# WAVEFORM
# =====================================================



# =====================================================
# LOAD MODEL
# =====================================================

def load_model():

    print("Loading Audio Model...")

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    print("Random Forest Loaded")

    return model, scaler

# =====================================================
# CALLBACK
# =====================================================

def callback(indata, frames, time_info, status):

    if status:

        print(status)

    audio_q.put(indata.copy())

# =====================================================
# ENERGY
# =====================================================

def audio_energy(audio):

    return np.sqrt(np.mean(audio**2))

# =====================================================
# VOICE DETECTION
# =====================================================

def has_voice(audio):

    return audio_energy(audio) > VOICE_THRESHOLD

# =====================================================
# AUTO GAIN CONTROL
# =====================================================

def automatic_gain(audio):

    rms = np.sqrt(np.mean(audio**2))

    if rms < 1e-6:

        return audio

    target = 0.12     

    gain = target / rms

    audio = audio * gain

    audio = np.clip(audio,-1,1)

    return audio

# =====================================================
# NORMALIZE
# =====================================================

def normalize(audio):
    

    peak = np.max(np.abs(audio))

    if peak > 0:

        audio = audio / peak

    return audio

# =====================================================
# UPDATE WAVEFORM
# =====================================================



# =====================================================
# PRE EMPHASIS
# =====================================================

def pre_emphasis(audio):

    return np.append(

        audio[0],

        audio[1:] - 0.97*audio[:-1]

    )

# =====================================================
# REMOVE SILENCE
# =====================================================

def remove_silence(audio):

    audio,_ = librosa.effects.trim(

        audio,

        top_db=20

    )

    return audio

# =====================================================
# FIX LENGTH
# =====================================================

def fix_length(audio):

    target = SAMPLE_RATE * RECORD_SECONDS

    if len(audio) < target:

        audio = np.pad(

            audio,

            (0,target-len(audio))

        )

    else:

        audio = audio[:target]

    return audio

# =====================================================
# PREPROCESS
# =====================================================

def preprocess_audio(audio):

    # =====================================
    # NORMALISASI
    # Sama seperti audio_train.py
    # =====================================
    audio = audio.astype(np.float32)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # =====================================
    # PRE-EMPHASIS
    # Sama seperti audio_train.py
    # =====================================
    audio = np.append(

        audio[0],

        audio[1:] - 0.97 * audio[:-1]

    )

    return audio

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_feature(audio):

    # ===============================
    # MFCC
    # ===============================
    if len(audio) < SAMPLE_RATE:

        return np.zeros((1,132))

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=SAMPLE_RATE,

        n_mfcc=N_MFCC

    )

    # ===============================
    # DELTA
    # ===============================
    delta = librosa.feature.delta(mfcc)

    # ===============================
    # DELTA DELTA
    # ===============================
    if mfcc.shape[1] >= 9:

        delta = librosa.feature.delta(mfcc)

        delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    else:

        delta = np.zeros_like(mfcc)

        delta2 = np.zeros_like(mfcc)

    # ===============================
    # CHROMA
    # ===============================
    chroma = librosa.feature.chroma_stft(

        y=audio,

        sr=SAMPLE_RATE

    )

    # ===============================
    # FEATURE VECTOR
    # ===============================
    feature = np.hstack([

        np.mean(mfcc.T,axis=0),

        np.mean(delta.T,axis=0),

        np.mean(delta2.T,axis=0),

        np.mean(chroma.T,axis=0)

    ])

    feature = np.nan_to_num(feature)
    
    return feature.reshape(1,-1)

# =====================================================
# SMOOTHING
# =====================================================

def smooth_probability(probability):

    global history

    history.append(probability)

    if len(history) > HISTORY_SIZE:
        history.pop(0)

    probability = np.mean(history, axis=0)

    # Hilangkan probabilitas yang terlalu kecil
    probability[probability < 0.03] = 0

    # Normalisasi kembali
    probability = probability / (np.sum(probability) + 1e-8)

    return probability

# =====================================================
# PREDICT
# =====================================================

def predict_audio(

    audio,

    model,

    scaler

):

    # ===============================
    # PREPROCESS
    # ===============================
    audio = preprocess_audio(audio)

    # ===============================
    # FEATURE
    # ===============================
    feature = extract_feature(audio)

    # ===============================
    # SCALE
    # ===============================
    feature = scaler.transform(feature)

    # ===============================
    # PREDICT
    # ===============================
    probability = model.predict_proba(feature)[0]

    # ==================================
    # HISTORY SMOOTHING
    # ==================================
    probability = smooth_probability(probability)

    # ==================================
    # CONFIDENCE CALIBRATION
    #  Membuat probabilitas lebih tegas
    # tanpa mengubah kelas prediksi
    # ==================================
    temperature = 0.60

    probability = probability ** (1.0 / temperature)
    probability = probability / np.sum(probability)

    # ==================================
    # HASIL AKHIR
    # ==================================
    idx = np.argmax(probability)

    emotion = EMOTIONS[idx]

    confidence = float(probability[idx] * 100)

    return (

    emotion,

    confidence,

    probability

        )

# =====================================================
# START AUDIO
# =====================================================

def start_audio(callback_gui):

    global RUNNING

    RUNNING = True

    model, scaler = load_model()

    print("===================================")
    print(" AUDIO ENGINE STARTED")
    print("===================================")
    print("Sample Rate :", SAMPLE_RATE)
    print("Record Time :", RECORD_SECONDS, "detik")
    print("===================================")

    try:

        with sd.InputStream(

             device=1,

             channels=1,

             samplerate=SAMPLE_RATE,

             blocksize = 48000,

                dtype="float32",

                latency="high",

                callback=callback
            
        ):

            print("🎤 Menunggu suara...")

            while RUNNING:

                if audio_q.empty():

                    time.sleep(0.05)

                    continue

                # ==================================
                # GET AUDIO
                # ==================================
                audio = audio_q.get().flatten()
                
                print("Jumlah Sample :", len(audio))

                # ==================================
                # DETEKSI SUARA
                # ==================================
                if not has_voice(audio):

                    continue

                # ==================================
                # PREDIKSI
                # ==================================
                emotion, confidence, probability = predict_audio(

                    audio,

                    model,

                    scaler

                )

                # ==================================
                # SIMPAN HASIL TERAKHIR
                # ==================================

                global last_emotion
                global last_confidence
                global last_probability

                last_emotion = emotion
                last_confidence = confidence
                last_probability = probability

                # ==================================
                # PRINT
                # ==================================
                print("--------------------------------------")

                print(f"Emotion     : {emotion}")

                print(f"Confidence  : {confidence:.2f}%")

                print("Probability :")

                for emo, prob in zip(EMOTIONS, probability):

                    print(f"   {emo:8s}: {prob*100:.2f}%")

                print("--------------------------------------")

                # ==================================
                # CALLBACK GUI
                # ==================================
                callback_gui(

                    emotion,

                    confidence

                )

    except Exception as e:

         import traceback
         traceback.print_exc()

# =====================================================
# STOP AUDIO
# =====================================================

def stop_audio():

    global RUNNING

    RUNNING = False

    print("🛑 Audio dihentikan")

# =====================================================
# GET LAST AUDIO RESULT
# =====================================================

def get_audio_result():

    global last_emotion
    global last_confidence
    global last_probability

    return (
        last_emotion,
        last_confidence,
        last_probability
    )
# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    def tampil(

        emotion,

        confidence

    ):

        print(

            f"\n🎧 {emotion} ({confidence:.2f}%)"

        )

    try:

        start_audio(tampil)

    except KeyboardInterrupt:

        stop_audio()