# Fitur Prediksi Banjir 10 Tahun Kedepan

## Deskripsi
Fitur ini memungkinkan pengguna untuk memprediksi potensi banjir untuk 10 tahun ke depan berdasarkan data historis dan model machine learning yang telah dilatih. Prediksi menggunakan tiga skenario iklim yang berbeda untuk memberikan gambaran komprehensif tentang risiko banjir di masa depan.

## Cara Menggunakan

### 1. Akses Halaman Prediksi
- Buka aplikasi web flood prediction
- Klik menu "Prediksi 10 Tahun" di sidebar
- Halaman prediksi akan terbuka dengan form parameter

### 2. Setting Parameter
- **Tahun Mulai**: Pilih tahun awal untuk memulai prediksi (2024-2028)
- **Wilayah**: Pilih wilayah yang ingin diprediksi dari dropdown
- **Skenario Iklim**: Pilih salah satu dari tiga skenario:
  - **Normal**: Kondisi iklim dengan perubahan moderat
  - **Optimis**: Kondisi iklim yang lebih baik dari biasanya
  - **Pesimis**: Kondisi iklim dengan perubahan ekstrem

### 3. Jalankan Prediksi
- Klik tombol "Jalankan Prediksi 10 Tahun"
- Sistem akan memproses data dan menampilkan hasil dalam beberapa detik

## Hasil Prediksi

### Summary Cards
Menampilkan ringkasan utama:
- Rata-rata risiko selama 10 tahun
- Jumlah tahun kritis (risiko tinggi)
- Jumlah tahun aman (risiko rendah)
- Trend risiko secara keseluruhan

### Analisis Trend
- Arah trend risiko (meningkat/menurun)
- Persentase perubahan risiko
- Identifikasi tahun-tahun kritis
- Rekomendasi mitigasi

### Visualisasi
1. **Grafik Trend Risiko**: Menampilkan perubahan risiko dari tahun ke tahun
2. **Distribusi Risiko**: Pie chart yang menunjukkan proporsi setiap level risiko
3. **Peta Lokasi**: Peta interaktif menunjukkan lokasi yang diprediksi

### Data Detail
- **Ringkasan Tahunan**: Informasi risiko untuk setiap tahun
- **Tabel Lengkap**: Data prediksi bulanan selama 10 tahun

## Metodologi Prediksi

### Skenario Iklim
Sistem menggunakan tiga skenario berdasarkan perubahan parameter iklim:

#### Skenario Normal
- Curah hujan: +2% per tahun
- Suhu: +0.05°C per tahun
- Tinggi muka air: +1% per tahun

#### Skenario Optimis
- Curah hujan: -1% per tahun
- Suhu: +0.02°C per tahun
- Tinggi muka air: -0.5% per tahun

#### Skenario Pesimis
- Curah hujan: +5% per tahun
- Suhu: +0.1°C per tahun
- Tinggi muka air: +3% per tahun

### Variasi Musiman
Sistem menambahkan variasi musiman pada prediksi untuk mencerminkan pola cuaca alami di Indonesia, dengan puncak musim hujan memiliki risiko yang lebih tinggi.

### Randomisasi
Untuk membuat prediksi lebih realistis, sistem menambahkan variasi acak kecil pada setiap parameter untuk mensimulasikan ketidakpastian alami dalam prediksi cuaca jangka panjang.

## Level Risiko

- **0 - Aman**: Kondisi normal, risiko banjir sangat rendah
- **1 - Waspada**: Risiko banjir rendah, perlu pemantauan
- **2 - Siaga**: Risiko banjir sedang, persiapan mitigasi diperlukan
- **3 - Awas**: Risiko banjir tinggi, tindakan darurat diperlukan

## Batasan dan Disclaimer

1. **Prediksi Probabilistik**: Hasil prediksi adalah estimasi berdasarkan data historis dan model statistik, bukan kepastian mutlak
2. **Faktor External**: Prediksi tidak memperhitungkan faktor eksternal seperti perubahan infrastruktur, deforestasi, atau kebijakan pemerintah
3. **Akurasi Model**: Akurasi prediksi bergantung pada kualitas data latih dan performa model machine learning
4. **Update Berkala**: Model perlu dilatih ulang secara berkala dengan data terbaru untuk mempertahankan akurasi

## Technical Requirements

### Dependencies
- Flask
- NumPy
- Pandas
- Scikit-learn
- MySQL Connector
- Chart.js (frontend)
- Leaflet (maps)
- Bootstrap (UI)

### Database
Sistem memerlukan akses ke database MySQL dengan tabel:
- `data_banjir`: Data historis untuk training
- `data_uji`: Data untuk testing dan validasi

### Model Training
Pastikan model telah dilatih sebelum menggunakan fitur prediksi 10 tahun. Gunakan endpoint `/train` untuk melatih model dengan data terbaru.

## Troubleshooting

### Error "Model belum dilatih"
- Pastikan telah menjalankan training model melalui menu "Training Model"
- Periksa apakah file `model.pkl` tersedia di direktori aplikasi

### Error Database Connection
- Periksa koneksi database MySQL
- Pastikan kredensial database benar di file `main.py`
- Pastikan tabel `data_banjir` dan `data_uji` tersedia dan berisi data

### Prediksi Tidak Realistis
- Periksa kualitas data training
- Pertimbangkan untuk melatih ulang model dengan data yang lebih banyak
- Sesuaikan parameter skenario iklim jika diperlukan