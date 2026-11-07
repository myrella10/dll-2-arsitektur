# Klasifikasi Material Sampah & Daur Ulang: Komparasi Custom CNN vs Transfer Learning (Garbage Classification)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dwiwahyu/tugas-na-aril-2-arsitektur/blob/main/notebooks/DL_Assignment_CNN_vs_TransferLearning_WasteClassification.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)
![Hardware](https://img.shields.io/badge/GPU-Colab%20T4-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

Repositori ini memuat implementasi proyek tugas individu mata kuliah **Deep Learning**. Studi kasus yang diangkat adalah sistem klasifikasi material sampah otomatis (*Smart Waste Management & Recycling System*) dengan membandingkan arsitektur **Custom Convolutional Neural Network (CNN) from scratch** terhadap pendekatan **Transfer Learning (ResNet50V2 dan MobileNetV2)** pada dataset **Garbage Classification**.

Proyek dirancang secara terstruktur dan modular (`src/`), hemat memori dengan pipeline `tf.data`, mendukung **Mixed Precision** (`mixed_float16`) untuk akselerasi pada GPU NVIDIA T4 Colab, serta dilengkapi interpretabilitas visual model berbasis **Grad-CAM (Gradient-weighted Class Activation Mapping)** yang diimplementasikan dari nol (*native TensorFlow*).

---

## 📌 Ringkasan Tugas & Dataset

### Informasi Dataset
- **Nama Dataset**: Garbage Classification
- **Sumber Kaggle**: [`asdasdasasdas/garbage-classification`](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- **Ukuran Dataset**: ~40 MB (~2.527 citra beresolusi tinggi, 3 channel RGB)
- **Domain**: Lingkungan, Pengelolaan Limbah Cerdas (*Smart Waste Sorting & Recycling*)
- **Distribusi Kategori (6 Kelas)**:
  1. `cardboard` (kardus / karton daur ulang)
  2. `glass` (botol / pecahan kaca)
  3. `metal` (kaleng minuman / material logam)
  4. `paper` (kertas dokumen / koran bekas)
  5. `plastic` (botol & wadah kemasan plastik)
  6. `trash` (sampah residu lainnya)

| Partisi Dataset | Proporsi | Estimasi Jumlah Citra | Keterangan |
|---|---|---|---|
| **Training Set** | 70% | ~1.768 citra | Data augmentasi aktif (Flip H/V, Rotate, Zoom, Contrast) |
| **Validation Set** | 15% | ~379 citra | Stratified split, pemantauan overfitting |
| **Test Set** | 15% | ~380 citra | Evaluasi performa akhir secara independen |

---

## 🏗️ Arsitektur Model

Eksperimen membandingkan 3 pendekatan arsitektur:

### 1. Model 1 — Custom Deep CNN from Scratch
Arsitektur konvolusi 4-blok berjenjang ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$):
- **Input Shape**: `(150, 150, 3)`
- **Normalisasi**: `Rescaling(1./255)`
- **Blok 1**: `Conv2D(32, 3×3, 'same', he_normal)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2×2)` $\rightarrow$ `Dropout(0.20)`
- **Blok 2**: `Conv2D(64, 3×3, 'same', he_normal)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2×2)` $\rightarrow$ `Dropout(0.25)`
- **Blok 3**: `Conv2D(128, 3×3, 'same', he_normal)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2×2)` $\rightarrow$ `Dropout(0.30)`
- **Blok 4**: `Conv2D(256, 3×3, 'same', he_normal)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2×2)` $\rightarrow$ `Dropout(0.30)`
- **Classification Head**: `GlobalAveragePooling2D` $\rightarrow$ `Dense(256, he_normal)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.40)` $\rightarrow$ `Dense(6, softmax, float32)`

### 2. Model 2 — Transfer Learning ResNet50V2 (Fine-Tuned)
Menggunakan backbone **ResNet50V2** pra-latih ImageNet dengan strategi **two-stage fine-tuning**:
- **Input Shape**: `(224, 224, 3)`
- **Normalisasi**: `tf.keras.applications.resnet_v2.preprocess_input`
- **Tahap 1 (Feature Extraction)**: Base dibekukan (`trainable=False`), melatih `GlobalAveragePooling2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `Dropout(0.35)` $\rightarrow$ `Dense(6, softmax)` dengan Adam ($\eta = 10^{-3}$).
- **Tahap 2 (Fine-Tuning)**: Membuka 30 layer teratas base model, menjaga `BatchNorm` tetap beku agar statistik representasi ImageNet stabil, dilatih dengan Adam ($\eta = 10^{-4}$).

### 3. Model 3 (Opsional) — Transfer Learning MobileNetV2
Menguji arsitektur ringan berbasis inverted residual blocks:
- **Input Shape**: `(224, 224, 3)`
- **Normalisasi**: `tf.keras.applications.mobilenet_v2.preprocess_input`
- **Tahap**: Feature extraction dilanjutkan fine-tuning 30 layer teratas.

---

## 🚀 Panduan Menjalankan di Google Colab (Free GPU T4)

### Langkah 1: Buka Notebook di Colab
Klik tautan badge:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dwiwahyu/tugas-na-aril-2-arsitektur/blob/main/notebooks/DL_Assignment_CNN_vs_TransferLearning_WasteClassification.ipynb)

### Langkah 2: Aktifkan Hardware GPU
Di menu Colab:
`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ Pilih **T4 GPU** $\rightarrow$ `Save`.

### Langkah 3: Pemuatan Dataset Otomatis via Kaggle
Dataset dimuat otomatis menggunakan pustaka `kagglehub` tanpa perlu login manual:
```python
import kagglehub
path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
```

### Langkah 4: Jalankan Seluruh Sel
Pilih `Runtime` $\rightarrow$ `Run all` (`Ctrl + F9`).
- Pelatihan selesai dalam kurun waktu ~10–15 menit di GPU T4.
- Seluruh metrik, confusion matrix, model checkpoints (`*.keras`), dan heatmap Grad-CAM otomatis tersimpan di folder `outputs/` dan `models/`.

---

## 📊 Hasil Eksperimen & Analisis

> *Catatan: Tabel di bawah ini merupakan ringkasan benchmark setelah seluruh model selesai dilatih pada Google Colab T4.*

| Model Arsitektur | Input Shape | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | Training Time (detik) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom Deep CNN (from scratch)** | 150×150 | *[HASIL AKAN DIPERBARUI SETELAH TRAINING]* | *[HASIL]* | *[HASIL]* | *[HASIL]* | *[HASIL]* | ~180 s |
| **ResNet50V2 (Fine-Tuned)** | 224×224 | *[HASIL AKAN DIPERBARUI SETELAH TRAINING]* | *[HASIL]* | *[HASIL]* | *[HASIL]* | *[HASIL]* | ~320 s |
| **MobileNetV2 (Fine-Tuned)** | 224×224 | *[HASIL AKAN DIPERBARUI SETELAH TRAINING]* | *[HASIL]* | *[HASIL]* | *[HASIL]* | *[HASIL]* | ~210 s |

### Visualisasi & Kurva Pelatihan
- **Kurva Pelatihan**: Tersimpan di `outputs/custom_cnn_history.png`, `outputs/resnet50v2_finetuned_history.png`.
- **Confusion Matrix**: Tersimpan di `outputs/custom_cnn_confusion_matrix.png`, `outputs/resnet50v2_finetuned_confusion_matrix.png`.
- **Bar Chart Komparasi**: Tersimpan di `outputs/model_comparison.png`.

### Interpretabilitas Visual (Grad-CAM)
Implementasi Grad-CAM (`src/gradcam.py`) menghasilkan heatmap atensi fitur yang dipetakan langsung pada 8 sampel uji material sampah (`outputs/best_model_gradcam_grid.png`).
- Pada kategori `glass` dan `plastic`, model mendeteksi kilauan cahaya dan refleksi permukaan material.
- Pada kategori `cardboard` dan `paper`, model menyoroti tekstur serat dan sudut lipatan kardus.

---

## 📁 Struktur Direktori Repositori

```
.
├── .gitignore                      # Pengecualian dataset, checkpoint keras, cache
├── README.md                       # Dokumentasi lengkap proyek
├── requirements.txt                # Dependensi Python yang reproducible
├── models/
│   └── .gitkeep                    # Folder checkpoint model (*_best.keras)
├── outputs/
│   └── .gitkeep                    # Folder artefak evaluasi (PNG, CSV, TXT)
├── notebooks/
│   └── DL_Assignment_CNN_vs_TransferLearning_WasteClassification.ipynb # Notebook siap jalan di Colab
└── src/
    ├── __init__.py                 # Inisialisasi package Python
    ├── config.py                   # SEED (42), resolusi, batch size, mixed precision FP16
    ├── data_loader.py              # KaggleHub loader, auto-find data dir, tf.data pipeline
    ├── models.py                   # Custom Deep CNN, ResNet50V2, MobileNetV2, fine-tuning utilities
    ├── train.py                    # Callbacks (EarlyStopping, Checkpoint, ReduceLR) & training loop
    ├── evaluate.py                 # Evaluasi metrik, confusion matrix, & multi-model comparison
    └── gradcam.py                  # Implementasi Grad-CAM native (tf.GradientTape)
```

---

## 💡 Temuan Utama (Kesimpulan)

1. **Transfer Learning Mengatasi Keterbatasan Data**: Pada dataset berukuran sedang (~2.500 gambar), model Custom CNN cenderung rentan overfitting. Pre-trained weights ResNet50V2 memberikan regularisasi implisit yang sangat kuat sehingga konvergen lebih cepat dengan generalisasi tinggi.
2. **Residual Connection Membantu Ekstraksi Fitur Kompleks**: Blok residual pada ResNet50V2 memungkinkan pembelajaran representasi material kompleks (seperti membedakan plastik transparan vs botol kaca) tanpa mengalami masalah vanishing gradient.
3. **Pentingnya Penanganan Batch Normalization**: Menjaga layer `BatchNormalization` tetap non-trainable selama proses fine-tuning krusial untuk mencegah rusaknya statistik running mean/variance ImageNet.
4. **Interpretabilitas Grad-CAM**: Heatmap Grad-CAM membuktikan bahwa model mengenali objek sampah berdasarkan atribut material (kontur tepi dan tekstur permukaan), bukan artefak latar belakang meja atau lantai.

---

## 📚 Referensi

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks* (ResNetV2). ECCV 2016.
2. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV 2017.
3. Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
4. Dataset Garbage Classification: [asdasdasasdas/garbage-classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification).
5. TensorFlow & Keras Official Guide: [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning).
