# Prediktor Saham Berbasis Pola (Flask API)

Repositori ini berisi proyek *end-to-end* untuk klasifikasi harga saham. Keunikan proyek ini adalah arsitektur **"satu model per pola"**, di mana model *machine learning* (Random Forest/Logistic Regression) dilatih secara spesifik untuk setiap pola *candlestick* (misal, 'Hammer', 'Doji', 'Marubozu').

Model-model yang sudah terlatih ini kemudian disajikan (di-<em>deploy</em>) sebagai **API** menggunakan **Flask**, sehingga dapat menerima data baru dan memberikan hasil prediksi.

## 🚀 Alur Kerja Proyek (End-to-End)

Proses keseluruhan proyek ini dapat dibagi menjadi dua bagian utama: Pelatihan dan Penerapan.




1.  **Pelatihan Model (Offline)** - Dilakukan di `main_workflow.ipynb`
    * Data mentah (CSV) dimuat dan dibersihkan (`data_emas/`).
    * Data diproses oleh `preprocessor.py` (termasuk *feature engineering* indikator teknikal).
    * Data yang tidak seimbang (imbalanced) diseimbangkan menggunakan **SMOTENC**.
    * Model (RF & LR) dilatih, di-<em>tuning</em> (`GridSearchCV`), dan dievaluasi untuk **setiap pola candlestick secara terpisah**.
    * Model terbaik untuk setiap pola disimpan ke folder `saved_models/` sebagai file `.pkl`.

2.  **Penerapan API** - Dilakukan oleh `app.py`
    * Aplikasi **Flask** (`app.py`) dijalankan.
    * Aplikasi ini memuat **semua model `.pkl`** dari `saved_models/` saat *startup*.
    * Pengguna mengirimkan data baru ke API (misalnya melalui *form* di halaman web dari `templates/`).
    * `app.py` mengidentifikasi pola *candlestick* dari data baru tersebut, memilih model `.pkl` yang sesuai, dan mengembalikan hasil prediksi (harga akan naik/turun).

## 🗂️ Struktur Repositori

```
.
├── 📁 data_emas/           # (Tempat data CSV mentah)
├── 📁 saved_models/        # (Tempat model .pkl disimpan)
├── 📁 templates/           # (Berisi file HTML untuk antarmuka Flask)
│
├── 📜 app.py               # (Aplikasi FASK utama untuk API/Web)
├── 📜 main_workflow        # (Notebook untuk analisis, preprocessing & training)
├── 📜 preprocessor.py      # (Modul.py untuk fungsi preprocessing)
├── 📜 README.md            # (Dokumentasi ini)
└── 📜 requirements.txt     # (Daftar library yang dibutuhkan)
```

## 🛠️ Cara Penggunaan

Ada dua skenario penggunaan untuk repositori ini:

### 1. Menjalankan Aplikasi Web (Flask) dengan Model yang Ada

Ini adalah cara untuk langsung menggunakan aplikasi web dengan model yang sudah Anda latih.

1.  Pastikan folder `saved_models/` Anda sudah berisi file `.pkl` hasil dari pelatihan.
2.  Install semua *library* yang dibutuhkan:
    ```bash
    pip install -r requirements.txt
    ```
3.  Jalankan aplikasi Flask dari terminal:
    ```bash
    python app.py
    ```
4.  Buka browser Anda dan navigasikan ke `http://127.0.0.1:5000` (atau alamat yang muncul di terminal).

### 2. Melatih Ulang Model dari Awal (Opsional)

Jika Anda ingin melatih ulang model dengan data baru atau parameter yang berbeda:

1.  Pastikan semua data mentah Anda ada di folder `data_emas/`.
2.  Buka dan jalankan semua sel di dalam `main_workflow` (Jupyter Notebook).
3.  Proses ini akan memakan waktu, tetapi setelah selesai, folder `saved_models/` Anda akan diperbarui dengan model-model `.pkl` yang baru.
4.  Setelah itu, Anda bisa menjalankan aplikasi Flask seperti pada langkah di atas.

## 💡 Teknologi yang Digunakan

* **Analisis & Model:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTENC)
* **Aplikasi Web/API:** Flask
* **Visualisasi:** Matplotlib, Seaborn
