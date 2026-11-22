# Prediksi Minat Nasabah Terhadap Deposito Berjangka pada Kampanye Telemarketing Bank

Proyek Akhir Praktikum Data Science 2026/2027 — Program Studi Informatika, Fakultas Teknik Industri, Universitas Pembangunan Nasional "Veteran" Yogyakarta.

---

## 1. Ringkasan Proyek

Kampanye pemasaran langsung (*outbound telemarketing*) pada perbankan umumnya memiliki tingkat konversi rendah (sekitar $11{,}70\%$ nasabah berlangganan deposito berjangka), sehingga sebagian besar waktu agen dan biaya operasional habis untuk menghubungi nasabah yang menolak.

Proyek ini menerapkan metode **Klasifikasi Biner Probabilistik** berbasis *Machine Learning* dengan alur **CRISP-DM** untuk memprediksi probabilitas minat nasabah (*propensity score*) **sebelum panggilan dilakukan** (*ex-ante prediction*). Dengan mengurutkan nasabah berdasarkan skor probabilitas tertinggi, tim telemarketing dapat memprioritaskan panggilan secara lebih efisien dan tepat sasaran.

### Ringkasan Hasil Utama (Model Final: `HistGradientBoostingClassifier` Tuned)
* **Dataset:** `bank-full.csv` ($45.211$ baris $\times$ $17$ kolom; $15$ fitur prediktor valid tanpa `duration` $\rightarrow$ $41$ fitur setelah *One-Hot Encoding*).
* **PR-AUC (Test Set):** $0{,}467$ (CI 95%: $0{,}432$–$0{,}500$; target $\ge 0{,}45$; $\approx 4\times$ lebih baik dari baseline acak $0{,}117$).
* **ROC-AUC (Test Set):** $0{,}805$ (CI 95%: $0{,}790$–$0{,}820$; target $\ge 0{,}80$).
* **Performa pada Threshold Optimal OOF ($0{,}18$):** Precision $0{,}444$, Recall $0{,}559$, $F_1$-Score $0{,}495$, $F_2$-Score $0{,}532$.
* **Dampak Bisnis (Lift & Cumulative Gains):**
  * **Desil ke-1 (Top 10% nasabah):** *Response rate* mencapai $51{,}9\%$ (**Lift $4{,}44\times$**), menangkap $44{,}4\%$ dari seluruh calon nasabah deposito.
  * **Top 20% nasabah:** Menangkap $62{,}3\%$ total calon nasabah deposito.
  * **Top 40% nasabah:** Menangkap $78{,}4\%$ total calon nasabah deposito, sehingga bank dapat memangkas hingga $60\%$ volume panggilan sambil mempertahankan hampir $80\%$ potensi akuisisi.

---

## 2. Anggota Kelompok & Kontribusi

> **Catatan:** Silakan sesuaikan kembali rincian kontribusi masing-masing anggota sebelum *final commit*.

| No | Nama | NIM | Kontribusi |
|---|---|---|---|
| 1 | Bagus Samudra | 123240102 | Business Understanding, perumusan metrik bisnis & teknis, penyusunan laporan |
| 2 | Filipus Satrio Dewaki T.S | 123240133 | Data Gathering, Exploratory Data Analysis (EDA), dan analisis kualitas data |
| 3 | Zaky Mubarok Aljauhari | 123240107 | Data Cleaning, Preprocessing, penanganan *imbalance* & pencegahan *data leakage* |
| 4 | Dimas Rizki Ardiansyah | 123240123 | Modeling, Nested Cross-Validation, evaluasi metrik bisnis (Lift/Gains), dan Konklusi |

---

## 3. Sumber Data & Lisensi

* **File yang digunakan:** `bank-full.csv` ($45.211$ baris, $17$ kolom, pemisah `;`).
* **Sumber Dataset:** [Kaggle — Bank Marketing Data Set (Bojan Tunguz)](https://www.kaggle.com/datasets/tunguz/bank-marketing-data-set) yang bersumber dari data historis asli kampanye pemasaran bank di Portugal oleh S. Moro, P. Cortez, dan P. Rita (2014) pada UCI Machine Learning Repository.
* **Lisensi:** CC0 / Public Domain (mirror Kaggle) & CC BY 4.0 (sumber asli UCI).
* **Alasan Pemilihan Dataset:** Data historis nyata (bukan sintetis), memiliki metadata kolom yang jelas, serta mengandung tantangan kualitas data nyata (*imbalanced target* $88{,}3\% : 11{,}7\%$, *missing value* implisit berlabel `'unknown'`, kode khusus `pdays = -1`, *outlier* ekstrem pada `balance` dan `campaign`, serta isu kritis *data leakage* pada kolom `duration`).

---

## 4. Struktur Repository

```text
.
├── README.md                # Dokumentasi proyek, anggota, sumber data, dan cara menjalankan
├── ProjekAkhir.ipynb        # Notebook utama (Business Understanding s.d. Konklusi)
└── bank-full.csv            # Dataset utama yang digunakan dalam eksperimen
```

---

## 5. Panduan Menjalankan Proyek

### Prasyarat
* Python $3.10+$
* Library utama: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `joblib`

### Langkah Eksekusi
1. *Clone* repository ini ke komputer lokal:
   ```bash
   git clone <URL_REPOSITORY_GITHUB_KALIAN>
   cd <NAMA_FOLDER_REPO>
   ```
2. Install seluruh dependensi yang diperlukan:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
   ```
3. Pastikan file `bank-full.csv` berada di direktori yang sama dengan `ProjekAkhir.ipynb`.
4. Buka dan jalankan notebook dari sel pertama hingga terakhir (*Run All*):
   ```bash
   jupyter notebook ProjekAkhir.ipynb
   ```
   *(Atau buka langsung melalui Google Colab / VS Code dengan mengunggah `bank-full.csv` di direktori kerja yang sama).*