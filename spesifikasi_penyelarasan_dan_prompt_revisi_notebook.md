# Spesifikasi Penyelarasan & Prompt Revisi `ProjekAkhir.ipynb`

Dokumen ini memetakan seluruh ketidaksesuaian antara **`ProjekAkhir.ipynb`**, **Laporan Proyek Akhir**, dan **Panduan Praktikum Data Science 2026/2027**, sekaligus menyediakan **Master Prompt** yang siap kamu pakai untuk merevisi notebook agar 100% sinkron sebelum *final commit*.

---

## Bagian 1: Audit 8 Masalah Kritis pada `ProjekAkhir.ipynb` Saat Ini

Berdasarkan pemeriksaan penuh seluruh sel di `ProjekAkhir.ipynb` (dari sel 1 hingga sel terakhir), berikut 8 masalah utama yang **wajib diperbaiki**:

1. **Sel 16–29 Belum Selesai Dijalankan (Kosong Outputnya):**
   * Eksekusi berhenti di tengah sel `4.2` (baru mencetak LogReg, Decision Tree, dan KNN). Sel `4.3` sampai Ringkasan Eksekutif tidak memiliki output sama sekali.
2. **Terjadi *Data Leakage* Parah saat Cross-Validation:**
   * Di sel `3.5`, `SMOTE` dijalankan pada keseluruhan `X_train_proc` **sebelum** 5-fold CV di sel `4.2`. Akibatnya sampel sintetis dari baris yang sama masuk ke *train fold* dan *validation fold* sekaligus (terlihat dari KNN yang tiba-tiba mendapat ROC-AUC $0{,}9507$ tanpa `duration` — skor semu karena bocor).
   * **Perbaikan:** Masukkan *resampling* (`SMOTENC`) **di dalam `imblearn.pipeline.Pipeline`** agar hanya diterapkan pada *training fold* di tiap iterasi CV, persis seperti Tabel 3.3.1 & Bab IV Laporan.
3. **Imputasi Modus Merusak Kolom `poutcome` dan `contact`:**
   * Di sel `3.1` dan `3.4`, nilai `'unknown'` diubah ke `NaN` lalu diisi modus (`most_frequent`). Padahal `poutcome` memiliki $36.959$ baris `'unknown'` ($81{,}75\%$), sehingga semuanya berubah keliru menjadi `'failure'`.
   * **Perbaikan:** Pertahankan `'unknown'` sebagai kategori valid dan pertahankan makna `pdays = -1` (sesuai Laporan Bab 3.1).
4. **Jumlah & Rekayasa Fitur Tidak Sinkron dengan Laporan:**
   * Notebook saat ini menambah 7 fitur baru (`was_contacted_before`, `month_num`, `age_group`, `balance_negative`, `log_balance`, `high_campaign`, `prev_success`) tanpa membuang fitur redundan (`month`, `age`, `balance`) sehingga total menjadi $57$ fitur.
   * **Perbaikan:** Gunakan $15$ fitur prediktor asli (drop `duration` saja) + `OneHotEncoder(drop='first')` di dalam `ColumnTransformer` sehingga tepat menghasilkan **$41$ fitur** sesuai Laporan Bab III & Bab VI.
5. **Kandidat Model, Skema Validasi, & Metrik Utama Berbeda dengan Laporan:**
   * Notebook saat ini mengoptimasi `ROC-AUC`, memakai `Dummy`, `LogReg`, `DT`, `KNN`, `RF`, `GB`, memilih `Random Forest` dengan threshold $0{,}5$, dan memakai `SMOTE` biasa.
   * Laporan mengoptimasi **`PR-AUC` (`average_precision`)**, membandingkan **6 kandidat + 1 tuned** (`LogisticRegression` + `class_weight`, `LogisticRegression` + `SMOTENC`, `RandomForest` + `class_weight='balanced_subsample'`, `HistGradientBoosting` tanpa resampling, `HistGradientBoosting` + `class_weight`, `HistGradientBoosting` + `SMOTENC`, dan `HistGradientBoosting` tuned via **Nested CV**), memilih threshold optimal OOF ($\approx 0{,}18$), serta mengevaluasi **Lift & Cumulative Gains per Desil**, **Bootstrap CI 95%**, dan **Kurva Kalibrasi**.
   * **PERINGATAN PENTING PANDUAN (Poin 4 & FAQ):** *"Model yang digunakan harus bersumber dari materi praktikum, model di luar materi praktikum tidak dinilai."* Pastikan dulu apakah `HistGradientBoostingClassifier` masuk materi praktikum kalian. Jika **ya**, ikuti 100% Laporan. Jika **tidak**, lihat opsi penyesuaian di Bagian 3 di bawah.
6. **Ketidakcocokan Sumber Data & Stakeholder di Fase 1 & 2:**
   * Notebook menulis 4 stakeholder (ada CFO & Tim IT), target $F_1 \ge 0{,}60$, dan lisensi CC BY 4.0 saja. Laporan menggunakan 6 stakeholder, target PR-AUC $\ge 0{,}45$, ROC-AUC $\ge 0{,}80$, Lift Desil 1 $\ge 3{,}0\times$, Capture Rate Top 40% ($75\%$–$80\%$), $F_2$-Score $\ge 0{,}55$, serta sumber Kaggle (`tunguz/bank-marketing-data-set`, CC0) yang merujuk ke data asli UCI (Moro et al., 2014).
   * *Catatan untuk file dataset:* Gunakan **hanya `bank-full.csv`** ($45.211$ baris $\times$ $17$ kolom). Jangan campur dengan `bank.csv`, `bank-additional.csv`, atau `bank-additional-full.csv`.
7. **Kurang Penjelasan Markdown Wajib Sesuai Rubrik Panduan:**
   * Belum ada sel Markdown interpretasi temuan **di bawah setiap visual EDA** (Panduan mewajibkan ini!).
   * Belum ada cek duplikat (`df.duplicated().sum()`), countplot `job` vs `y`, grafik `month` (`jan`–`dec`) vs `y`, serta tabel **perbandingan kondisi data sebelum vs sesudah preprocessing**.
   * Belum ada penjelasan eksplisit **Feature Selection** (alasan membuang `duration`, mempertahankan `'unknown'` & `pdays = -1`, dan menggunakan 15 fitur terpilih).
   * Belum ada **Tahap Konklusi** berbasis Markdown yang menjawab pertanyaan Business Understanding, menerjemahkan metrik ke bahasa bisnis, menyatakan keputusan akhir, dan keterbatasan.
8. **Fase Deployment Terlalu Kompleks & Tidak Diminta Panduan:**
   * Panduan menegaskan: *"Tidak perlu deployment, aplikasi web, API, atau dashboard."* Hapus stub `app.py` (FastAPI) yang membuat notebook berantakan. Jika ingin tetap selaras dengan Bab VI Laporan, cukup sisipkan sel opsional ringkas penyimpanan `model_deposito.joblib` & `model_config.json` (atau letakkan sebagai lampiran singkat sebelum Konklusi).

---

## Bagian 2: Spesifikasi Struktur Ideal `ProjekAkhir.ipynb`

Agar nilai rubrik Panduan maksimal (Business Understanding & Data Gathering 15%, EDA 10%, Cleaning & Preprocessing 10%, Modelling 10%, Evaluasi & Konklusi 20%) sekaligus cocok 100% dengan Laporan:

1. **Header & Identitas Kelompok** (Nama + NIM 4 anggota, judul, dataset `bank-full.csv`).
2. **Tahap 1: Business Understanding**
   * Latar belakang masalah (konversi historis $11{,}70\%$, inefisiensi biaya panggilan, kelelahan nasabah).
   * Daftar 6 Stakeholder & kebutuhan bisnis.
   * **Pertanyaan Bisnis yang Ingin Dijawab** (3–4 pertanyaan eksplisit yang nanti dijawab di Konklusi).
   * Metode yang dipilih: **Klasifikasi Biner Probabilistik**.
   * Target Metrik Teknis (PR-AUC $\ge 0{,}45$, ROC-AUC $\ge 0{,}80$, Lift Desil 1 $\ge 3{,}0\times$, Capture Rate Top 40% $75\%$–$80\%$, $F_2$-Score $\ge 0{,}55$) & Metrik Bisnis (Conversion Rate, CPA, Average Calls per Conversion).
3. **Tahap 2: Data Gathering & Understanding**
   * Provenance data (Kaggle `tunguz/bank-marketing-data-set` — CC0 / sumber asli UCI Moro et al., 2014), tanggal akses, deskripsi ke-17 kolom.
   * **Alasan Memilih Dataset** (asli, kredibel, memiliki kondisi data kotor: imbalance, `'unknown'`, `pdays = -1`, outlier ekstrem, leakage `duration`).
   * Penilaian Data Readiness Level (DRL).
   * Load `bank-full.csv` (`sep=';'`).
4. **Tahap 3: Exploratory Data Analysis (EDA)** *(WAJIB: Setiap grafik/tabel diikuti 1 sel Markdown analisis temuan!)*
   * 3.1 Struktur data (`info()`, tipe data) & Statistik Deskriptif numerik (`describe()`) + kategorikal.
   * 3.2 Pemeriksaan kualitas data: `isnull().sum()`, `df.duplicated().sum()` ($=0$), frekuensi `'unknown'` (`job`: 288, `education`: 1.857, `contact`: 13.020, `poutcome`: 36.959), dan kode `pdays = -1` ($36.954$ baris).
   * 3.3 Distribusi Target `y` (Bar chart + Pie chart: `no` $39.922$ [$88{,}30\%$], `yes` $5.289$ [$11{,}70\%$]).
   * 3.4 Deteksi Outlier (IQR + Boxplot untuk `age`, `balance`, `campaign`, serta tinjauan `duration`, `pdays`, `previous`).
   * 3.5 Heatmap Matriks Korelasi Pearson fitur numerik.
   * 3.6 Hubungan Kategorikal vs Target: Countplot + proporsi konversi `job` vs `y` dan `poutcome` vs `y`.
   * 3.7 Pola Kelompok Bulanan: Distribusi `month` (`jan`–`dec`) berdasarkan target `y` (jumlah observasi + proporsi `yes`).
5. **Tahap 4: Data Cleaning, Preprocessing & Feature Selection**
   * 4.1 Pembersihan data (`drop_duplicates()`, pemetaan `y`: `'yes'` $\rightarrow 1$, `'no'` $\rightarrow 0$, keputusan mempertahankan `'unknown'` dan `pdays = -1` serta outlier valid dengan alasan domain).
   * 4.2 **Feature Selection** eksplisit: Tabel alasan membuang (`duration` karena *ex-post data leakage*), mempertahankan $15$ fitur prediktor, dan tidak menambah fitur duplikatif.
   * 4.3 Strategi Pemisahan Data & Anti-Leakage: `train_test_split(test_size=0.2, random_state=42, stratify=y)` $\rightarrow$ Train $36.168$, Test $9.043$.
   * 4.4 Pipeline Preprocessing (`ColumnTransformer` dengan `OneHotEncoder(drop='first', handle_unknown='ignore')` dan `StandardScaler` khusus model linear) $\rightarrow$ menghasilkan $41$ fitur.
   * 4.5 **Tabel Ringkasan Kondisi Data SEBELUM vs SESUDAH Cleaning & Preprocessing**.
6. **Tahap 5: Modeling & Tuning** *(WAJIB: Penjelasan alasan sebelum/sesudah setiap sel kode!)*
   * 5.1 Baseline kuat: `LogisticRegression(class_weight='balanced')` (alasan prinsip parsimoni & perbandingan dengan baseline acak PR-AUC $0{,}117$).
   * 5.2 Perbandingan 6 kandidat model + strategi penanganan imbalance menggunakan `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` di dalam `imblearn.pipeline.Pipeline` (SMOTENC hanya di *training fold*!):
     1. `LogisticRegression` + `class_weight='balanced'`
     2. `LogisticRegression` + `SMOTENC`
     3. `RandomForestClassifier` + `class_weight='balanced_subsample'`
     4. `HistGradientBoostingClassifier` (tanpa resampling)
     5. `HistGradientBoostingClassifier` + `class_weight='balanced'`
     6. `HistGradientBoostingClassifier` + `SMOTENC`
   * 5.3 Grafik bar perbandingan PR-AUC CV antar kandidat + penjelasan temuan kenapa `SMOTENC` menurunkan PR-AUC.
   * 5.4 Nested Cross-Validation (Outer 5-fold, Inner 3-fold) + `RandomizedSearchCV(n_iter=10, scoring='average_precision')` untuk tuning `HistGradientBoostingClassifier` (`learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`), lengkap dengan perbandingan waktu fit & gap train-val.
   * 5.5 Penentuan threshold klasifikasi optimal menggunakan prediksi *Out-Of-Fold* (OOF) pada data latih (memaksimalkan $F_1$-Score $\rightarrow$ threshold $\approx 0{,}18$).
7. **Tahap 6: Evaluation (Data Uji Hold-Out 20%)**
   * 6.1 Evaluasi satu kali pada Test Set ($9.043$ baris): PR-AUC & ROC-AUC + Bootstrap CI 95%, Precision, Recall, $F_1$, $F_2$, Akurasi.
   * 6.2 Confusion Matrix pada threshold OOF ($\approx 0{,}18$) + interpretasi bisnis TN, FP, FN, TP.
   * 6.3 Grafik Kurva Precision-Recall, Kurva ROC, dan **Kurva Kalibrasi** (`CalibrationDisplay`).
   * 6.4 **Tabel Lift & Cumulative Gains per Desil (Desil 1–10)** + grafik Cumulative Gains.
   * 6.5 Tabel Pencapaian Target Metrik (Perbandingan Target Bab I vs Hasil Aktual).
   * 6.6 Interpretasi Fitur (*Permutation Importance* pada data uji) untuk menjawab faktor pendorong minat nasabah.
   * 6.7 *(Opsional ringkas agar selaras Bab VI Laporan)* Penyimpanan `model_deposito.joblib`, `model_config.json`, dan `template_nasabah_baru.csv`.
8. **Tahap 7: Konklusi** *(WAJIB Markdown komprehensif sesuai Panduan!)*
   * Jawaban eksplisit atas setiap pertanyaan di Business Understanding.
   * Terjemahan seluruh metrik teknis ke bahasa bisnis/operasional yang mudah dipahami manajemen.
   * Keputusan & rekomendasi operasional (pembagian prioritas telepon: Top 20% Prioritas Tinggi, 20–40% Prioritas Sedang, sisanya Prioritas Rendah).
   * Keterbatasan jujur (pengaruh tuning tipis, split acak vs temporal, sifat semi-*ex-post* fitur `campaign`/`day`/`month`, perbedaan konteks pasar Portugal vs Indonesia, risiko bias demografis).

---

## Bagian 3: Master Prompt (Siap Copy-Paste) untuk Merevisi `ProjekAkhir.ipynb`

Salin seluruh prompt di dalam kotak di bawah ini dan tempel ke AI coding assistant (VS Code Copilot / Cursor / Google Colab AI / Gemini / Claude) sambil melampirkan file **`ProjekAkhir.ipynb`** dan **`bank-full.csv`**.

> **Tips Penting Sebelum Menjalankan Prompt:**
> Cek materi praktikum kalian terlebih dahulu. Jika **`HistGradientBoostingClassifier` DIAJARKAN** di praktikum (atau dosen mengizinkan keluarga Gradient Boosting), langsung gunakan prompt di bawah apa adanya. Jika **`HistGradientBoostingClassifier` TIDAK diajarkan** di praktikum, ganti kalimat yang ditandai `[OPSIONAL MODEL PRAKTIKUM]` dengan daftar model yang diajarkan di kelas praktikum kalian (misal `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`, `GradientBoostingClassifier`).

```text
Tolong rombak dan bangun ulang file notebook `ProjekAkhir.ipynb` secara lengkap, rapi, dapat dijalankan dari awal sampai akhir tanpa error (reproducible), dan selaras 100% dengan Laporan Proyek Akhir serta Panduan Praktikum Data Science UPN "Veteran" Yogyakarta.

### KONTEKS & DATASET
- Gunakan HANYA file dataset: `bank-full.csv` (delimiter `;`, 45.211 baris x 17 kolom: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, y).
- Jangan mencampur dengan `bank-additional-full.csv` atau membuat fitur makroekonomi fiktif.
- Bahasa penulisan: Bahasa Indonesia yang baku, komunikatif, dan analitis.
- Identitas Kelompok:
  1. Bagus Samudra (123240102)
  2. Filipus Satrio Dewaki T.S (123240133)
  3. Zaky Mubarok Aljauhari (123240107)
  4. Dimas Rizki Ardiansyah (123240123)

### ATURAN WAJIB DARI PANDUAN PRAKTIKUM (JANGAN ADA YANG TERLEWAT)
1. Penjelasan di sel Markdown sangat menentukan nilai. WAJIB berikan sel Markdown penjelasan temuan/insight tepat di BAWAH setiap visualisasi dan tabel EDA.
2. Di tahap Cleaning & Preprocessing, WAJIB tampilkan perbandingan kondisi data SEBELUM dan SESUDAH diproses (jumlah baris, jumlah kolom, tipe data, missing/unknown, contoh baris), serta jelaskan tahap Feature Selection secara eksplisit (alasan membuang, mempertahankan, dan menggunakan fitur).
3. Di tahap Modeling, WAJIB berikan penjelasan alasan di setiap blok kode yang dijalankan dan bandingkan performa seluruh model pembanding.
4. TIDAK PERLU membuat deployment API (hapus kode FastAPI `app.py` dan monitoring drift yang berlebihan). Cukup simpan artefak model ringkas (`model_deposito.joblib`, `model_config.json`, `template_nasabah_baru.csv`) jika diperlukan agar selaras Bab VI Laporan.
5. WAJIB buat bagian "Tahap Konklusi" khusus dalam bentuk Markdown lengkap yang:
   - Menjawab satu per satu pertanyaan di Business Understanding.
   - Menerjemahkan metrik evaluasi ke bahasa bisnis/operasional yang konkret.
   - Menjelaskan keputusan bisnis yang diambil.
   - Menuliskan keterbatasan data & model secara jujur.

### STRUKTUR & SPESIFIKASI TEKNIS YANG HARUS DIIMPLEMENTASIKAN

#### 1. Business Understanding
- Jelaskan latar belakang masalah telemarketing deposito (konversi historis rendah ~11,70%, biaya operasional tinggi, kelelahan nasabah).
- Tuliskan 3-4 Rumusan Pertanyaan Bisnis eksplisit.
- Daftar 6 Stakeholder (VP Marketing/Retail Funding, Manajer Call Center, Agen Telemarketing, Tim Data Science, Divisi Kepatuhan/Risiko, Nasabah).
- Metode yang dipilih: Klasifikasi Biner Probabilistik.
- Target Metrik Keberhasilan Teknis:
  * PR-AUC >= 0,45 (metrik utama; baseline acak = prevalensi positif ~0,117)
  * ROC-AUC >= 0,80 (tanpa kolom `duration`)
  * Lift Desil ke-1 (Top 10%) >= 3,0x
  * Capture Rate (Cumulative Gain) Top 40% populasi = 75% - 80%
  * F2-Score >= 0,55
- Target Metrik Bisnis: Net Conversion Rate >= 25%, penurunan CPA >= 35%, Average Calls per Conversion <= 8.

#### 2. Data Gathering & Understanding
- Tampilkan provenance data (Kaggle: `https://www.kaggle.com/datasets/tunguz/bank-marketing-data-set`, lisensi CC0/Public Domain, merujuk sumber asli UCI Moro et al., 2014).
- Tabel skema lengkap ke-17 kolom beserta tipe data, domain/satuan, deskripsi, dan peran (tandai `duration` sebagai Data Leakage).
- Jelaskan ALASAN memilih dataset ini (data nyata bukan sintetis, kredibel, memiliki karakteristik data kotor: imbalance, nilai `'unknown'`, kode `pdays = -1`, outlier ekstrem, dan risiko leakage).
- Tabel penilaian Data Readiness Level (DRL).
- Load `bank-full.csv` dengan `pd.read_csv('bank-full.csv', sep=';')`.

#### 3. Exploratory Data Analysis (EDA)
Buat kode + visualisasi rapi, dan WAJIB tambahkan sel Markdown interpretasi di bawah SETIAP output:
- Info dataset (`df.info()`) & statistik deskriptif numerik (`df.describe().T`) + kategorikal.
- Cek kualitas data: `df.isnull().sum()`, `df.duplicated().sum()`, jumlah `'unknown'` tiap kolom, dan jumlah `pdays == -1`.
- Visualisasi distribusi kelas target `y` (Bar Chart + Pie Chart).
- Deteksi outlier metode IQR + Boxplot untuk `age`, `balance`, `campaign` (serta ulasan domain untuk `pdays` dan `previous`).
- Heatmap matriks korelasi Pearson untuk kolom numerik.
- Countplot + rasio konversi (`yes` %) berdasarkan `job` dan berdasarkan `poutcome` (tampilkan jumlah observasi sekaligus persentase).
- Visualisasi pola kelompok bulanan `month` berurutan (`jan` sampai `dec`) terhadap target `y` (jumlah observasi + persentase `yes`).

#### 4. Data Cleaning, Preprocessing, & Feature Selection
- Jalankan `drop_duplicates()` (meski hasil = 0, tetap lakukan verifikasi eksplisit).
- Ubah target `y`: `'yes'` -> `1`, `'no'` -> `0`.
- PENTING: JANGAN mengubah `'unknown'` menjadi `NaN` lalu mengimputasi dengan modus (karena 81,75% `poutcome` adalah `'unknown'` yang bermakna belum pernah dihubungi). Pertahankan `'unknown'` sebagai kategori sendiri dan pertahankan nilai `pdays = -1` (atau jelaskan alasannya sesuai domain).
- Feature Selection:
  * Drop kolom `duration` untuk mencegah *ex-post data leakage*.
  * Jelaskan alasan mempertahankan 15 fitur prediktor lainnya (`age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome`) dan tidak menambah fitur baru yang redundan.
- Split data SEBELUM encoding/scaling/resampling: `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
- Gunakan `ColumnTransformer` dengan `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)` pada 9 fitur kategorikal sehingga menghasilkan tepat 41 fitur numerik (tambahkan `StandardScaler()` di dalam pipeline khusus untuk `LogisticRegression`).
- Tampilkan tabel perbandingan kondisi data SEBELUM vs SESUDAH preprocessing (dimensi, missing/unknown, target encoding, jumlah fitur).

#### 5. Modeling & Hyperparameter Tuning (BEBAS DATA LEAKAGE)
- PENTING: JANGAN jalankan SMOTE di luar cross-validation! Semua resampling (`SMOTENC` dari `imblearn.over_sampling` untuk mempertahankan sifat fitur kategorikal) HARUS ditempatkan di dalam `imblearn.pipeline.Pipeline` agar hanya diterapkan pada training fold.
- Tetapkan `random_state=42` dan `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Metrik optimasi utama: `average_precision` (PR-AUC); metrik pendukung: `roc_auc`.
- [OPSIONAL MODEL PRAKTIKUM: Pertahankan daftar model di bawah agar sesuai Laporan. Jika hanya boleh memakai model tertentu dari praktikum, sesuaikan di sini.]
- Bandingkan 6 kandidat model awal pada 5-fold CV (dan catat juga PR-AUC CV, ROC-AUC CV, serta waktu fit):
  1. `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)` (Strong Baseline)
  2. `LogisticRegression(max_iter=1000, random_state=42)` + `SMOTENC` (di dalam Pipeline sebelum OHE)
  3. `RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=42, n_jobs=-1)`
  4. `HistGradientBoostingClassifier(random_state=42)` tanpa resampling
  5. `HistGradientBoostingClassifier(class_weight='balanced', random_state=42)`
  6. `HistGradientBoostingClassifier(random_state=42)` + `SMOTENC`
- Buat tabel perbandingan dan bar chart PR-AUC CV (5-fold) antar kandidat lengkap dengan garis horizontal baseline acak (`y_train.mean()` ~ 0,117). Jelaskan di Markdown mengapa `SMOTENC` justru menurunkan performa pada dataset ini.
- Lakukan Nested Cross-Validation (Outer: `StratifiedKFold(5)`, Inner: `StratifiedKFold(3)`) dengan `RandomizedSearchCV(n_iter=10, scoring='average_precision', random_state=42)` pada `HistGradientBoostingClassifier` tanpa resampling untuk ruang parameter:
  * `learning_rate`: [0.03, 0.05, 0.1]
  * `max_iter`: [100, 200, 300]
  * `max_leaf_nodes`: [15, 31, 63]
  * `min_samples_leaf`: [20, 50, 100]
  * `l2_regularization`: [0.0, 1.0, 10.0, 30.0]
- Bandingkan `HistGradientBoosting` tanpa tuning vs tuned (PR-AUC nested CV mean ± std, gap train vs val, waktu fit).
- Cari threshold klasifikasi optimal HANYA pada prediksi out-of-fold (`cross_val_predict(..., method='predict_proba')`) dari data latih dengan memaksimalkan F1-Score (agar data uji tidak bocor ke pemilihan threshold).

#### 6. Evaluation (Pada Data Uji Hold-Out 20%)
Evaluasi model final SATU KALI pada `X_test, y_test` menggunakan threshold OOF yang didapat:
- Hitung PR-AUC dan ROC-AUC beserta **Confidence Interval 95% menggunakan Bootstrap** (1.000 iterasi pada data uji).
- Hitung Recall, Precision, F1-Score, F2-Score (`fbeta_score(y_test, y_pred, beta=2)`), dan Akurasi.
- Tampilkan Confusion Matrix (tabel angka + heatmap `ConfusionMatrixDisplay`) dan jelaskan arti bisnis TN, FP, FN, TP.
- Plot berdampingan: Kurva Precision-Recall, Kurva ROC, dan Kurva Kalibrasi (`CalibrationDisplay.from_predictions`).
- Hitung dan tampilkan **Tabel Lift & Cumulative Gains per Desil (Desil 1 sampai 10)** + plot Cumulative Gains Chart & Lift Chart.
- Buat Tabel Evaluasi Pencapaian Target (membandingkan Target Bab I vs Hasil Aktual pada Data Uji).
- Hitung dan plot **Top 10 Permutation Importance** pada data uji (metric `average_precision`) agar pemangku kepentingan mengetahui faktor utama pendorong nasabah membuka deposito.
- (Ringkas) Simpan model final (`model_deposito.joblib`), konfigurasi (`model_config.json`), dan template (`template_nasabah_baru.csv`) dengan `joblib` & `json`.

#### 7. Konklusi (Sel Markdown Komprehensif)
- Jawab seluruh pertanyaan dari Business Understanding secara langsung menggunakan angka hasil aktual.
- Terjemahkan metrik teknis ke dampak bisnis nyata (contoh: berapa persen nasabah deposito yang tertangkap jika hanya menelepon Top 20% atau Top 40% nasabah).
- Rumuskan rekomendasi operasional (Tier 1 Top 20%, Tier 2 20%–40%, Tier 3 sisanya, serta pembatasan jumlah panggilan).
- Uraikan keterbatasan proyek secara jujur (tuning hanya naik tipis dibanding default, split acak bukan temporal, potensi semi-leakage dari fitur `campaign`/`day`/`month`, perbedaan pasar Portugal vs Indonesia, dan perlunya audit fairness demografis sebelum produksi).

Pastikan tidak ada error sintaks, seluruh variabel konsisten, dan semua import (`imblearn`, `sklearn`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `joblib`) lengkap di sel paling atas.
```

---

## Bagian 4: Checklist Cepat Sebelum Final Commit GitHub

- [ ] **Verifikasi Aturan Model Praktikum:** Pastikan seluruh model yang dipakai di notebook memang diajarkan di praktikum. Jika `HistGradientBoostingClassifier` tidak diajarkan, segera sesuaikan prompt di atas agar memakai model praktikum (misal `RandomForestClassifier` / `GradientBoostingClassifier` / `LogisticRegression`) dan pastikan anggota kelompok siap menjelaskan alasannya saat tanya jawab.
- [ ] **Run All Cells dari Awal hingga Akhir:** Di Jupyter/Colab, klik **Restart Kernel & Run All Cells** dan pastikan **tidak ada satu pun sel yang error atau kosong outputnya**.
- [ ] **Sinkronisasi Angka Laporan & Notebook:** Setelah notebook selesai dijalankan, periksa angka aktual di output notebook (jumlah duplikat $= 0$, jumlah `'unknown'`, PR-AUC, ROC-AUC, threshold OOF, Lift Desil 1, Capture Rate Top 40%) dan pastikan cocok dengan tabel di Laporan.
- [ ] **Perapian Repository GitHub:** Pastikan di dalam repository minimal sudah ada:
  1. `README.md` (yang sudah diisi pembagian kontribusi anggota kelompok)
  2. `ProjekAkhir.ipynb` (versi final yang seluruh output-nya tersimpan rapi)
  3. `bank-full.csv` (jangan lupa ikut di-commit, bukan file lain yang tidak terpakai)
- [ ] **Siap Tanya Jawab Individu (Bobot 30%):** Pastikan keempat anggota paham kenapa `duration` dibuang (*data leakage*), kenapa metrik utama pakai PR-AUC bukan Accuracy (*imbalanced data* $88{,}3\% : 11{,}7\%$), kenapa `'unknown'` di `poutcome` tidak diimputasi modus, dan kenapa `SMOTENC` harus berada di dalam `Pipeline` saat Cross-Validation.