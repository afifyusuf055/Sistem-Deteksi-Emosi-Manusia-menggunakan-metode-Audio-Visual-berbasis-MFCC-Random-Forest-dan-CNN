# Sistem Deteksi Emosi Manusia Menggunakan Metode Audio-Visual Berbasis MFCC, Random Forest, dan CNN

## Deskripsi

Sistem Deteksi Emosi Manusia merupakan aplikasi berbasis Python yang dirancang untuk mengenali emosi manusia secara real-time menggunakan kombinasi informasi audio dan visual.

Sistem ini mengintegrasikan metode:

- Mel Frequency Cepstral Coefficients (MFCC) untuk ekstraksi fitur suara
- Random Forest untuk klasifikasi emosi berdasarkan audio
- Convolutional Neural Network (CNN) untuk klasifikasi emosi berdasarkan ekspresi wajah
- Audio-Visual Fusion untuk menggabungkan hasil prediksi audio dan visual

Aplikasi mampu mengenali empat kategori emosi yaitu:

- 😄 Happy (Senang)
- 😠 Angry (Marah)
- 😢 Sad (Sedih)
- 😐 Neutral (Netral)

---

## Latar Belakang

Pengenalan emosi manusia merupakan salah satu bidang penting dalam Human Computer Interaction (HCI). Kemampuan sistem untuk memahami kondisi emosional pengguna dapat meningkatkan kualitas interaksi antara manusia dan komputer.

Pada penelitian ini dikembangkan sistem deteksi emosi berbasis audio-visual yang memanfaatkan suara dan ekspresi wajah secara bersamaan. Pendekatan multimodal ini diharapkan mampu memberikan hasil deteksi yang lebih akurat dibandingkan penggunaan satu modalitas saja.

---

## Fitur Utama

### Mode Visual
Mendeteksi emosi berdasarkan ekspresi wajah pengguna menggunakan webcam secara real-time.

### Mode Audio
Mendeteksi emosi berdasarkan suara pengguna menggunakan mikrofon.

### Mode Fusion
Menggabungkan hasil prediksi audio dan visual untuk memperoleh keputusan emosi yang lebih stabil.

### Real-Time Detection
Sistem melakukan deteksi secara langsung tanpa proses upload file.

### Voice Feedback
Aplikasi dapat memberikan output suara sesuai emosi yang terdeteksi.

---

## Metode Penelitian

### Audio Processing

1. Akuisisi suara menggunakan mikrofon
2. Ekstraksi fitur menggunakan MFCC
3. Normalisasi fitur menggunakan StandardScaler
4. Klasifikasi menggunakan Random Forest

### Visual Processing

1. Akuisisi citra menggunakan webcam
2. Deteksi wajah menggunakan Haar Cascade
3. Preprocessing citra wajah
4. Klasifikasi menggunakan CNN

### Audio-Visual Fusion

Prediksi audio dan visual digabungkan menggunakan metode voting untuk menghasilkan keputusan emosi akhir.

---

## Dataset

### Dataset Audio

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Emosi yang digunakan:

- Angry
- Happy
- Neutral
- Sad

### Dataset Visual

FER2013 (Facial Expression Recognition 2013)

Emosi yang digunakan:

- Angry
- Happy
- Neutral
- Sad

---

## Struktur Project

```text
Github/
│
├── models/
│   ├── audio_scaler.pkl
│   ├── rf_audio_mfcc.pkl
│   ├── cnn_visual_strong.h5
│   ├── haarcascade_frontalface_default.xml
│   └── speaker.png
│
├── sounds/
│   ├── angry.wav
│   ├── happy.wav
│   ├── neutral.wav
│   └── sad.wav
│
├── results/
│   ├── confusion_matrix_audio.png
│   └── confusion_matrix_visual.png
│
├── audio_train.py
├── audio_test_mic.py
├── visual_train.py
├── visual_test.py
├── fusion_test.py
├── gui_app.py
│
├── requirements.txt
└── README.md
```

---

## Kebutuhan Sistem

### Hardware

- Processor Intel Core i3 atau lebih tinggi
- RAM minimal 4 GB
- Webcam
- Mikrofon
- Speaker

### Software

- Windows 10 / Windows 11
- Python 3.10
- Visual Studio Code (Opsional)

---

## Instalasi

### Clone Repository

```bash
git clone https://github.com/afifyusuf055/Sistem-Deteksi-Emosi-Manusia-menggunakan-metode-Audio-Visual-berbasis-MFCC-Random-Forest-dan-CNN.git
```

### Masuk ke Folder Project

```bash
cd Sistem-Deteksi-Emosi-Manusia-menggunakan-metode-Audio-Visual-berbasis-MFCC-Random-Forest-dan-CNN
```

### Install Dependency

```bash
pip install -r requirements.txt
```

---

## Menjalankan Program

### GUI Utama

```bash
python gui_app.py
```

### Training Model Audio

```bash
python audio_train.py
```

### Training Model Visual

```bash
python visual_train.py
```

### Pengujian Audio

```bash
python audio_test_mic.py
```

### Pengujian Visual

```bash
python visual_test.py
```

### Pengujian Fusion

```bash
python fusion_test.py
```

---

## Hasil Penelitian

### Model Audio

Metode:
- MFCC
- Random Forest

Akurasi:
- 70.37%

### Model Visual

Metode:
- CNN
- FER2013 Dataset

Akurasi:
- 77.84%

### Model Fusion

Menggabungkan prediksi audio dan visual secara real-time untuk meningkatkan kestabilan deteksi emosi.

---

## Tampilan Sistem

Fitur yang tersedia pada GUI:

- Mode Visual
- Mode Audio
- Mode Fusion
- Tampilan Confidence
- Bounding Box Wajah
- Pemilihan Mikrofon
- Sound ON/OFF

---

## Documentation

User Manual:

- docs/Buku_Manual_Aplikasi.docx

This document contains:
- Installation guide
- Application usage
- Audio mode
- Visual mode
- Fusion mode
- Troubleshooting

---

## Pengembang

**M. Afif Fuadie Yusuf**

Program Studi Teknik Mekatronika  
Politeknik Negeri Batam

Tugas Akhir 2026

---

## Lisensi

Project ini dikembangkan untuk kebutuhan penelitian dan Tugas Akhir Program Studi Teknik Mekatronika Politeknik Negeri Batam.

Copyright © 2026

M. Afif Fuadie Yusuf
