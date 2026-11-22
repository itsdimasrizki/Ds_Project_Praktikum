# Prediksi Minat Nasabah Terhadap Deposito Berjangka pada Kampanye Telemarketing Bank

Proyek Akhir Praktikum Data Science 2026/2027 — Program Studi Informatika, Fakultas Teknik Industri, Universitas Pembangunan Nasional "Veteran" Yogyakarta.

---

## 1. Ringkasan Proyek

Kampanye pemasaran langsung (*outbound telemarketing*) pada industri perbankan umumnya menghadapi inefisiensi operasional akibat tingkat konversi yang rendah (hanya $\approx 11{,}70\%$ nasabah bersedia membuka rekening deposito berjangka). Sebagian besar waktu agen telemarketing dan anggaran operasional terbuang untuk menghubungi nasabah yang tidak berminat atau bahkan terganggu oleh panggilan berulang (*customer fatigue*).

Proyek ini menerapkan pendekatan **Klasifikasi Biner Probabilistik** berbasis *Machine Learning* yang terstruktur rapi mengikuti metodologi standar industri **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)**. Model menghasilkan skor kecenderungan (*propensity score*) kontinyu $P(Y=1|X) \in [0, 1]$ **sebelum panggilan telepon dilakukan (*ex-ante prediction*)**, tanpa kebocoran data (*data leakage*). Skor ini memungkinkan perbankan melakukan perankingan kuantil/desil nasabah sehingga tim telemarketing dapat memprioritaskan kontak pada prospek dengan potensi konversi tertinggi.

### Ringkasan Hasil Utama (Model Terbaik: `HistGradientBoostingClassifier` Tuned)
* **Dataset:** `bank-full.csv` ($45.211$ observasi $\times$ $17$ atribut; $15$ fitur prediktor valid setelah eliminasi ex-post feature `duration` $\rightarrow$ $41$ fitur setelah One-Hot Encoding).
* **PR-AUC (Test Set 20% / 9.043 data):** $\mathbf{0{,}462 - 0{,}467}$ (Bootstrap CI 95%: $0{,}432$–$0{,}500$; melampaui target teknis $\ge 0{,}45$; $\approx 4\times$ di atas baseline acak $0{,}117$).
* **ROC-AUC (Test Set):** $\mathbf{0{,}805}$ (Bootstrap CI 95%: $0{,}790$–$0{,}820$; melampaui target teknis $\ge 0{,}80$).
* **Threshold Optimal OOF ($\approx 0{,}18$):** Precision $0{,}444$, Recall $0{,}559$, $F_1$-Score $0{,}495$, $F_2$-Score $0{,}532$ (fokus meminimalkan *False Negative*).
* **Dampak Bisnis (Tabel Desil Lift & Cumulative Gains):**
  * **Desil ke-1 (Top 10% nasabah prioritas):** Menghasilkan *response rate* sebesar **$51{,}9\%$** (**Lift $4{,}44\times - 4{,}47\times$**), menjaring $44{,}4\%$ total nasabah deposito.
  * **Top 20% nasabah:** Menjaring **$62{,}3\%$** total calon nasabah deposito.
  * **Top 40% nasabah:** Menjaring **$77{,}6\% - 78{,}4\%$** total calon nasabah deposito, memungkinkan bank memangkas hingga **$60\%$ volume panggilan telepon** sambil tetap mengamankan hampir $80\%$ potensi pembukaan deposito.

---

## 2. Anggota Kelompok & Kontribusi (Kelompok 44)

| No | Nama | NIM | Peran & Kontribusi Utama |
|---|---|---|---|
| 1 | Bagus Samudra | 123240102 | Business Understanding, perumusan 4 pertanyaan bisnis, target metrik teknis/bisnis, penyusunan laporan |
| 2 | Filipus Satrio Dewaki T.S | 123240133 | Data Gathering & Understanding, Exploratory Data Analysis (EDA), audit missing values & outliers |
| 3 | Zaky Mubarok Aljauhari | 123240107 | Data Cleaning, eliminasi data leakage `duration`, stratified hold-out split, ColumnTransformer |
| 4 | Dimas Rizki Ardiansyah | 123240123 | Modeling (5-Fold CV ImbPipeline), Nested CV Tuning, evaluasi desil/lift, konklusi & tiering bisnis |

---

## 3. Sumber Data & Lisensi

* **Dataset Utama:** `data/raw/bank-full.csv` ($45.211$ baris, $17$ kolom, pemisah `;`).
* **Dataset Pendukung:** Berada pada direktori `data/Dataset/` (mencakup `bank.csv`, `bank-additional.csv`, `bank-additional-full.csv`, serta salinan dataset utama).
* **Sumber & Peneliti Asli:** S. Moro, P. Cortez, dan P. Rita (2014), *"A Data-Driven Approach to Predict the Success of Bank Telemarketing"*, *Decision Support Systems*, Elsevier, Vol. 62, hlm. 22–31. Tersedia di UCI Machine Learning Repository.
* **Kaggle Mirror:** Bojan Tunguz ([Kaggle: `tunguz/bank-marketing-data-set`](https://www.kaggle.com/datasets/tunguz/bank-marketing-data-set)).
* **Lisensi:** CC0: Public Domain (mirror Kaggle) & Creative Commons Attribution 4.0 International / CC BY 4.0 (UCI).
* **Karakteristik Data:** Data riil perbankan dengan ketidakseimbangan kelas ($88{,}30\%$ 'no' vs $11{,}70\%$ 'yes'), nilai terselubung berlabel `'unknown'`, kode khusus `pdays = -1`, nilai ekstrem saldo (*outlier*), dan risiko kebocoran data ex-post pada kolom `duration`.

---

## 4. Struktur Direktori Proyek

```text
ProjekAkhirRevisi/
├── data/
│   ├── raw/
│   │   └── bank-full.csv                 # Dataset utama Bank Marketing (45.211 baris)
│   ├── Dataset/                          # Seluruh dataset pendukung yang sudah ada
│   │   ├── Pendukung/                    # bank.csv, bank-additional.csv, bank-additional-full.csv
│   │   └── Utama/                        # Salinan bank-full.csv
│   └── template_nasabah_baru.csv         # Contoh data input inferensi nasabah baru (15 fitur)
├── models/
│   ├── model_deposito.joblib             # Pipeline model terlatih (Preprocessor + HistGradientBoosting)
│   └── model_config.json                 # Metadata model, threshold optimal (0.18), dan metrik evaluasi
├── docs/
│   └── Laporan Proyek Akhir_PREDIKSI MINAT NASABAH TERHADAP DEPOSITO BERJANGKA PADA KAMPANYE TELEMARKETING BANK.pdf
├── .gitignore
├── ProjekAkhir.ipynb                     # Notebook utama alur CRISP-DM lengkap & teruji
└── README.md                             # Panduan ringkas dan dokumentasi proyek
```

---

## 5. Panduan Menjalankan Proyek

### Prasyarat Sistem
* Python versi 3.10 atau yang lebih baru.
* Library Python utama:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
  ```

### Langkah Eksekusi Notebook
1. Pastikan struktur folder telah sesuai seperti pada bagian **4. Struktur Direktori Proyek**.
2. Buka dan jalankan Jupyter Notebook:
   ```bash
   jupyter notebook ProjekAkhir.ipynb
   ```
3. Pilih opsi **Kernel** $\rightarrow$ **Restart & Run All** untuk mengeksekusi seluruh sel dari Fase 1 hingga Fase 7.
4. Notebook akan secara otomatis memuat dataset dari `data/raw/bank-full.csv`, menjalankan seluruh tahapan preprocessing dan pemodelan, serta menyimpan kembali artefak model ke `models/` dan `data/`.

### Pengujian Inferensi Data Baru
Untuk menguji inferensi skor probabilitas nasabah baru tanpa melatih ulang model:
```python
import joblib, json
import pandas as pd
import numpy as np

# 1. Pemuatan model pipeline dan file konfigurasi
model = joblib.load('models/model_deposito.joblib')
with open('models/model_config.json', 'r') as f:
    config = json.load(f)

# 2. Baca template input data nasabah baru
df_new = pd.read_csv('data/template_nasabah_baru.csv')

# 3. Prediksi probabilitas dan penentuan rekomendasi kontak
df_new['propensity_score'] = np.round(model.predict_proba(df_new)[:, 1], 4)
df_new['rekomendasi_kontak'] = np.where(
    df_new['propensity_score'] >= config['optimal_threshold'],
    'Prioritas Hubungi (Yes)',
    'Lewati (No)'
)
print(df_new[['age', 'job', 'balance', 'housing', 'loan', 'poutcome', 'propensity_score', 'rekomendasi_kontak']])
```
