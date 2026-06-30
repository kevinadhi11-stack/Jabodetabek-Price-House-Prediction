# 🏠 Prediksi Harga Rumah Indonesia

Aplikasi web interaktif untuk memprediksi harga rumah di Indonesia (Jabodetabek) menggunakan Machine Learning, dibangun dengan **Streamlit** dan **Gradient Boosting Regressor**.

## 📁 Struktur Proyek

```
house-price-app/
├── data/
│   └── harga_rumah.csv      # taruh dataset CSV kamu di sini
├── model/                   # otomatis terbuat setelah training
│   ├── pipeline.pkl
│   ├── kota_list.pkl
│   └── metadata.pkl
├── train_model.py           # script training model
├── app.py                   # aplikasi Streamlit
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan dataset
Dataset yang dipakai: [Harga Rumah Jabodetabek](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek), simpan sebagai `data/harga_rumah.csv`

**Penting:** Buka `train_model.py`, lalu sesuaikan `COLUMN_MAP` dengan nama kolom dataset kamu (misal `HARGA`, `LT`, `LB`, dll).

### 3. Latih model
```bash
python train_model.py
```
Ini akan menghasilkan file `model/pipeline.pkl` dan menampilkan metrik evaluasi (MAE, RMSE, R²).

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```
Buka browser di `http://localhost:8501`

## 🌐 Deploy Gratis ke Streamlit Cloud

1. Push folder project ini ke GitHub (termasuk folder `model/` yang sudah berisi `.pkl`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub, pilih repo ini, set main file ke `app.py`
4. Klik Deploy — link bisa langsung dibagikan ke recruiter!

## 🛠️ Tech Stack

- **Model:** Gradient Boosting Regressor (scikit-learn)
- **Preprocessing:** StandardScaler + OneHotEncoder via sklearn Pipeline
- **Visualisasi:** Plotly (gauge chart, bar chart perbandingan kota)
- **Frontend:** Streamlit

## 📊 Fitur Aplikasi

- Input interaktif: kota, luas tanah/bangunan, kamar tidur/mandi, garasi
- Prediksi harga real-time dengan rentang estimasi
- Gauge chart menunjukkan posisi harga relatif terhadap median dataset
- Simulasi perbandingan harga antar kota dengan spek rumah yang sama
- Sidebar info model (R², MAE, RMSE)

## 💡 Insight & Business Value

Model ini mencapai R² sebesar 0.74, yang berarti variasi harga rumah di area Jabodetabek dapat dijelaskan dengan cukup baik dari kombinasi luas tanah, luas bangunan, jumlah kamar, dan lokasi. Fitur lokasi (kota) terbukti menjadi salah satu penentu harga paling signifikan, mencerminkan pola harga properti riil di mana lokasi strategis (misal Jakarta Selatan) secara konsisten lebih mahal dibanding wilayah penyangga.

Aplikasi ini dapat dimanfaatkan oleh agen properti atau calon pembeli untuk mendapatkan estimasi harga wajar sebelum negosiasi, serta membandingkan potensi harga properti di berbagai kota dengan spesifikasi yang sama.

## 📸 Demo

*(Tambahkan screenshot aplikasi dan link live demo Streamlit Cloud di sini setelah deploy)*
