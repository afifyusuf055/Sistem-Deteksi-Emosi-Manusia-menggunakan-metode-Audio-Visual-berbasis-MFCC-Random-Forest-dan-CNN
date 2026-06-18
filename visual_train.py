import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# PATH
# ===============================
DATASET_PATH = "dataset/visual/train"
MODEL_PATH = "models/cnn_visual_strong.h5"
CM_PATH = "results/confusion_matrix_visual.png"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ===============================
# PARAMETER
# ===============================
IMG_SIZE = 96          # ⬅️ KUNCI
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-4

EMOTIONS = ["angry", "happy", "neutral", "sad"]

# ===============================
# DATA AUGMENTATION
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ===============================
# CNN STRONG
# ===============================
model = Sequential([
    Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(128, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Conv2D(256, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(4, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACK
# ===============================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
]

# ===============================
# TRAIN
# ===============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===============================
# EVALUATION
# ===============================
val_gen.reset()
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTIONS))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=EMOTIONS,
    yticklabels=EMOTIONS
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Strong CNN")
plt.tight_layout()
plt.savefig(CM_PATH)
plt.close()

print("\nConfusion matrix disimpan di:", CM_PATH)
print("Model CNN visual berhasil disimpan di:", MODEL_PATH)
