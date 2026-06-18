import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# ======================================================
# PATH
# ======================================================
DATASET_PATH = "dataset/audio/RAVDESS"

MODEL_PATH = "models/rf_audio_mfcc.pkl"
SCALER_PATH = "models/audio_scaler.pkl"

CM_PATH = "results/confusion_matrix_audio.png"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ======================================================
# LABEL
# ======================================================
emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

labels_order = [
    "angry",
    "happy",
    "neutral",
    "sad"
]

# ======================================================
# SPLIT AUDIO
# ======================================================
def split_audio(signal, sr, duration=3, overlap=1):

    step = int((duration - overlap) * sr)

    size = int(duration * sr)

    segments = []

    # AUDIO PENDEK
    if len(signal) < size:

        padding = size - len(signal)

        signal = np.pad(
            signal,
            (0, padding),
            mode='constant'
        )

        segments.append(signal)

        return segments

    # SPLIT AUDIO
    for i in range(
        0,
        len(signal) - size + 1,
        step
    ):

        segments.append(
            signal[i:i+size]
        )

    return segments

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_feature(signal, sr):

    # =====================================
    # NORMALISASI
    # =====================================
    signal = signal / (
        np.max(np.abs(signal)) + 1e-6
    )

    # =====================================
    # PREEMPHASIS
    # =====================================
    signal = np.append(
        signal[0],
        signal[1:] - 0.97 * signal[:-1]
    )

    # =====================================
    # MFCC
    # =====================================
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40
    )

    # =====================================
    # DELTA
    # =====================================
    delta = librosa.feature.delta(mfcc)

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
        y=signal,
        sr=sr
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

    feature = np.nan_to_num(feature)

    return feature

# ======================================================
# LOAD DATASET
# ======================================================
X = []
y = []

print("📂 Loading dataset...")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        emotion_code = file.split("-")[2]

        if emotion_code not in emotion_map:
            continue

        path = os.path.join(
            actor_path,
            file
        )

        # =====================================
        # LOAD AUDIO
        # =====================================
        signal, sr = librosa.load(
            path,
            sr=16000
        )

        # =====================================
        # SPLIT AUDIO
        # =====================================
        segments = split_audio(
            signal,
            sr
        )

        for seg in segments:

            feat = extract_feature(
                seg,
                sr
            )

            X.append(feat)

            y.append(
                emotion_map[emotion_code]
            )

X = np.array(X)

y = np.array(y)

print(f"✅ Total data : {len(X)}")

print("📊 Distribusi :")

print(Counter(y))

# ======================================================
# SCALER
# ======================================================
scaler = StandardScaler()

X = scaler.fit_transform(X)

# ======================================================
# SPLIT DATA
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# RANDOM FOREST
# ======================================================
model = RandomForestClassifier(

    n_estimators=300,

    max_depth=20,

    min_samples_split=2,

    min_samples_leaf=1,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)

print("🚀 Training model...")

model.fit(
    X_train,
    y_train
)

# ======================================================
# EVALUASI
# ======================================================
y_pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    y_pred
)

print(f"\n🔥 Akurasi : {acc*100:.2f}%\n")

print(classification_report(
    y_test,
    y_pred
))

# ======================================================
# CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels_order
)

cm_percent = cm.astype('float') / (
    cm.sum(axis=1)[:, np.newaxis]
)

labels = np.array([

    f"{cm[i,j]}\n({cm_percent[i,j]*100:.1f}%)"

    for i in range(len(labels_order))

    for j in range(len(labels_order))

]).reshape(
    len(labels_order),
    len(labels_order)
)

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=labels,

    fmt="",

    cmap="Blues",

    xticklabels=labels_order,

    yticklabels=labels_order
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Confusion Matrix Audio Emotion"
)

plt.tight_layout()

plt.savefig(
    CM_PATH,
    dpi=300
)

plt.show()

print(f"📸 Confusion matrix disimpan : {CM_PATH}")

# ======================================================
# SAVE MODEL
# ======================================================
joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print("✅ Model berhasil disimpan")