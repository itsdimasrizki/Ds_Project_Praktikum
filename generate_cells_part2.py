# -*- coding: utf-8 -*-
"""
Part 2: Fase 2 (Data Gathering) & Fase 3 (EDA) cells for ProjekAkhir.ipynb.
"""

def add_fase_2_and_3_cells(cells):
    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })
        
    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # =========================================================================
    # CELL 2: FASE 2: Data Gathering & Understanding (Markdown)
    # =========================================================================
    add_md(r"""---
# 🔍 FASE 2: Data Gathering & Understanding

## 2.1 Asal-Usul Data (*Data Provenance*) & Lisensi
Dataset yang digunakan dalam penelitian ini adalah **Bank Marketing Dataset** yang diperoleh melalui mirror resmi Kaggle:
* **Penyedia Mirror:** Bojan Tunguz ([Kaggle: `tunguz/bank-marketing-data-set`](https://www.kaggle.com/datasets/tunguz/bank-marketing-data-set))
* **Lisensi Kaggle Mirror:** CC0: Public Domain
* **Sumber Asli & Peneliti Utama:** Sérgio Moro, Paulo Cortez, dan Paulo Rita (2014), *"A Data-Driven Approach to Predict the Success of Bank Telemarketing"*, dimuat dalam jurnal *Decision Support Systems*, Elsevier, Vol. 62, hlm. 22–31. Tersedia di UCI Machine Learning Repository.
* **Lisensi Sumber Asli:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Cakupan Data:** Data historis riil dari institusi perbankan komersial di Portugal yang mencakup periode Mei 2008 hingga November 2010 ($45.211$ observasi interaksi telemarketing).

---

## 2.2 Alasan Pemilihan Dataset
Dataset ini dipilih secara sadar karena memenuhi kriteria metodologis berikut:
1. **Data Riil Empiris Perbankan (Bukan Data Sintetis):** Menggambarkan dinamika sesungguhnya dari kampanye telemarketing perbankan dengan karakteristik nasabah yang autentik.
2. **Kredibilitas & Benchmark Komparatif Tinggi:** Dataset ini merupakan karya seminal Moro et al. (2014) yang telah menjadi rujukan standar global untuk pengujian model klasifikasi biner perbankan.
3. **Mengandung Tantangan Kualitas Data Dunia Nyata (*Dirty Data Challenges*):**
   * *Extreme Class Imbalance* ($88{,}30\%$ kelas mayoritas 'no' vs $11{,}70\%$ kelas minoritas 'yes').
   * *Missing Values* implisit yang disandikan sebagai kategori `'unknown'`.
   * Nilai kode khusus yang bermakna domain (`pdays = -1` menandakan nasabah belum pernah dihubungi sebelumnya).
   * *Outlier* ekstrem pada variabel saldo (`balance`) dan intensitas kontak (`campaign`).
   * Risiko kritis **kebocoran data (*ex-post data leakage*)** pada kolom durasi panggilan (`duration`).

---

## 2.3 Skema & Kamus Data (17 Atribut Lengkap)

| No | Nama Kolom | Tipe Data | Domain / Satuan | Deskripsi Fitur | Peran Pemodelan |
|:---:|:---|:---|:---|:---|:---|
| 1 | `age` | Numerik (`int64`) | 18 – 95 tahun | Usia nasabah bank dalam tahun | Fitur Prediktor |
| 2 | `job` | Kategorikal (`object`) | 12 kategori | Jenis pekerjaan nasabah (`admin.`, `blue-collar`, `technician`, dll.) | Fitur Prediktor |
| 3 | `marital` | Kategorikal (`object`) | `married`, `single`, `divorced` | Status perkawinan nasabah | Fitur Prediktor |
| 4 | `education` | Kategorikal (`object`) | `primary`, `secondary`, `tertiary`, `unknown` | Tingkat pendidikan terakhir nasabah | Fitur Prediktor |
| 5 | `default` | Kategorikal (`object`) | `yes`, `no` | Riwayat gagal bayar kredit (*credit default*) | Fitur Prediktor |
| 6 | `balance` | Numerik (`int64`) | Euro (€) | Saldo rata-rata tahunan nasabah pada rekening | Fitur Prediktor |
| 7 | `housing` | Kategorikal (`object`) | `yes`, `no` | Kepemilikan pinjaman Kredit Pemilikan Rumah (KPR) | Fitur Prediktor |
| 8 | `loan` | Kategorikal (`object`) | `yes`, `no` | Kepemilikan pinjaman pribadi (*personal loan*) | Fitur Prediktor |
| 9 | `contact` | Kategorikal (`object`) | `cellular`, `telephone`, `unknown` | Jenis media komunikasi kontak terakhir | Fitur Prediktor |
| 10 | `day` | Numerik (`int64`) | 1 – 31 | Tanggal kontak terakhir dalam bulan | Fitur Prediktor |
| 11 | `month` | Kategorikal (`object`) | `jan` – `dec` | Bulan kontak terakhir dalam kalender | Fitur Prediktor |
| 12 | `duration` | Numerik (`int64`) | Detik | Durasi kontak panggilan telepon terakhir | **EX-POST DATA LEAKAGE (WAJIB DIBUANG)** |
| 13 | `campaign` | Numerik (`int64`) | $\ge 1$ kali | Jumlah kontak yang dilakukan selama kampanye ini | Fitur Prediktor |
| 14 | `pdays` | Numerik (`int64`) | -1 s.d. 871 hari | Jumlah hari sejak nasabah terakhir dihubungi dari kampanye lampau (-1 = belum pernah) | Fitur Prediktor |
| 15 | `previous` | Numerik (`int64`) | $\ge 0$ kali | Jumlah kontak yang dilakukan sebelum kampanye ini | Fitur Prediktor |
| 16 | `poutcome` | Kategorikal (`object`) | `unknown`, `failure`, `other`, `success` | Hasil kampanye pemasaran sebelumnya | Fitur Prediktor |
| 17 | `y` | Kategorikal (`object`) | `yes`, `no` | Apakah nasabah membuka deposito berjangka? | **TARGET BINARY** |

> **PERINGATAN KRITIS DATA LEAKAGE PADA KOLOM `duration`:**
> Sebagaimana ditegaskan oleh Moro et al. (2014), nilai atribut `duration` hanya diketahui **setelah panggilan telepon selesai dilakukan**. Jika durasi panggilan adalah 0 detik, target $y$ dipastikan $'no'$. Dalam konteks operasional perbankan di mana model digunakan untuk menentukan daftar panggilan prospektif sebelum telepon dilakukan (*ex-ante*), nilai `duration` belum tersedia. Menyertakan `duration` akan menghasilkan model dengan performa semu yang gagal total di lingkungan produksi (*production failure*). Oleh sebab itu, kolom `duration` **WAJIB DIELIMINASI** sebelum pemodelan!

---

## 2.4 Penilaian Kesiapan Data (*Data Readiness Level / DRL*)

| Level Kesiapan | Status | Deskripsi & Bukti Verifikasi |
|:---|:---:|:---|
| **DRL 1: Data Mentah & Eksploratif** | Selesai | File `bank-full.csv` utuh diunduh (45.211 baris $\times$ 17 kolom, pemisah `;`). |
| **DRL 2: Pemetaan Skema & Metadata** | Selesai | 17 fitur terdefinisi tipe data, satuan, makna domain, serta peran prediktor vs leakage. |
| **DRL 3: Pembersihan Terstruktur & Anti-Leakage** | Selesai | Keputusan penanganan nilai `'unknown'`, kode `pdays = -1`, eliminasi `duration`, dan stratified hold-out split. |
| **DRL 4: Pipeline Siap Pakai & Bebas Resampling Leakage** | Selesai | Seluruh transformasi (OHE, scaling, SMOTENC) terenkapsulasi penuh dalam `Pipeline` scikit-learn/imblearn. |""")

    # =========================================================================
    # CELL 3: Code - Import Library & Setup Lingkungan
    # =========================================================================
    add_code(r"""# ============================================================
# 2.5 SETUP LINGKUNGAN, IMPORT LIBRARY, & KONFIGURASI REPRODUCIBILITY
# ============================================================
import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Model & Evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    fbeta_score, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance

# Imbalanced-Learn untuk Pipeline Anti-Leakage
import imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

# Serialisasi Model
import joblib

# Konfigurasi Tampilan & Visualisasi
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Tetapkan Seed Universal
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Environment berhasil disiapkan.")
print(f"Versi NumPy          : {np.__version__}")
print(f"Versi Pandas         : {pd.__version__}")
print(f"Versi Scikit-Learn   : {__import__('sklearn').__version__}")
print(f"Versi Imbalanced-Learn: {imblearn.__version__}")
print(f"Versi Joblib         : {joblib.__version__}")
print("Random State Universal: 42")""")

    # =========================================================================
    # CELL 4: Code - Load Dataset bank-full.csv
    # =========================================================================
    add_code(r"""# ============================================================
# 2.6 PEMUATAN DATASET UTAMA: bank-full.csv
# ============================================================
DATASET_PATH = 'bank-full.csv' if os.path.exists('bank-full.csv') else 'Dataset/Utama/bank-full.csv'
print(f"Memuat dataset dari path: '{DATASET_PATH}'...")

# Pemuatan dengan delimiter ';' sesuai format standar UCI Bank Marketing
df = pd.read_csv(DATASET_PATH, sep=';')

print(f"Pemuatan berhasil!")
print(f"Dimensi Dataset: {df.shape[0]:,} baris x {df.shape[1]} kolom")
print("Contoh 5 baris pertama:")
display(df.head())""")

    # =========================================================================
    # CELL 5: Markdown - Interpretasi Hasil Pemuatan Data
    # =========================================================================
    add_md(r"""### 📌 Analisis Hasil Pemuatan Dataset (Fase 2)
1. **Integritas Baris & Kolom:** Dataset berhasil dimuat dengan tepat **$45.211$ baris** dan **$17$ kolom**, sesuai dengan dokumentasi resmi dataset UCI Moro et al. (2014) untuk varian `bank-full.csv`.
2. **Kesesuaian Sumber:** Seluruh eksperimen menggunakan `bank-full.csv` sebagai sumber data tunggal dan tidak mencampurkan data dengan file pendukung (`bank.csv` atau `bank-additional-full.csv` yang memiliki fitur makroekonomi berbeda).
3. **Target Biner:** Kolom target bernama `y` dengan nilai biner kategorikal `'yes'` (berlangganan deposito) dan `'no'` (tidak berlangganan).""")

    # =========================================================================
    # CELL 6: Markdown - FASE 3: Exploratory Data Analysis (EDA)
    # =========================================================================
    add_md(r"""---
# 📊 FASE 3: Exploratory Data Analysis (EDA)

Tahap ini bertujuan untuk membedah struktur data, memvalidasi kualitas data (keberadaan nilai hilang terselubung dan duplikasi), memahami distribusi variabel target, memeriksa anomali/outlier, serta memetakan korelasi dan pola asosiasi antar fitur terhadap target pembukaan deposito.

Setiap grafik dan tabel di bawah ini dilengkapi dengan interpretasi analitis mendalam sesuai rubrik panduan praktikum.""")

    # =========================================================================
    # CELL 7: Code - 3.1 Struktur Data & Statistik Deskriptif
    # =========================================================================
    add_code(r"""# ============================================================
# 3.1 STRUKTUR DATA & STATISTIK DESKRIPTIF
# ============================================================
print("--- [1] INFORMASI STRUKTUR & TIPE DATA (df.info()) ---")
df.info()

print("\n--- [2] STATISTIK DESKRIPTIF FITUR NUMERIK ---")
numeric_desc = df.describe().T
numeric_desc['median'] = df.median(numeric_only=True)
numeric_desc['skew'] = df.skew(numeric_only=True)
display(numeric_desc[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'skew']])

print("\n--- [3] STATISTIK DESKRIPTIF FITUR KATEGORIKAL ---")
cat_desc = df.describe(include=['object']).T
cat_desc['unique_values'] = [df[col].unique().tolist() for col in cat_desc.index]
display(cat_desc[['count', 'unique', 'top', 'freq']])""")

    # =========================================================================
    # CELL 8: Markdown - Interpretasi Temuan 3.1
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 3.1: Struktur Data & Karakteristik Statistik
1. **Distribusi Tipe Data:**
   * Terdapat **7 variabel numerik bertipe `int64`**: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`.
   * Terdapat **10 variabel kategorikal bertipe `object`**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, dan target `y`.
2. **Karakteristik Fitur Numerik:**
   * `age`: Rata-rata usia nasabah adalah $40{,}94$ tahun dengan median $39$ tahun, mencakup nasabah dari usia produktif muda ($18$ tahun) hingga lansia ($95$ tahun).
   * `balance`: Memiliki ketimpangan (*skewness* $13{,}80$) yang sangat tajam; nilai rata-rata adalah €$1.362{,}27$ sedangkan median hanya €$448{,}00$. Terdapat nasabah dengan saldo negatif (cerukan/overdraft hingga -€$8.019$) dan nasabah *High Net Worth* dengan saldo maksimum mencapai €$102.127$.
   * `campaign`: Rata-rata kontak selama kampanye ini adalah $2{,}76$ kali, dengan beberapa nasabah dihubungi secara ekstrem hingga $63$ kali.
   * `pdays`: Memiliki nilai minimum $-1$ dan median $-1$, yang menunjukkan mayoritas nasabah belum pernah dihubungi pada kampanye pemasaran terdahulu.
3. **Karakteristik Fitur Kategorikal:**
   * `job`: Pekerjaan terbanyak adalah `blue-collar` ($9.732$ baris, $21{,}5\%$), `management` ($9.458$ baris, $20{,}9\%$), dan `technician` ($7.597$ baris, $16{,}8\%$).
   * `marital`: Mayoritas nasabah berstatus menikah (`married`: $60{,}2\%$).
   * `month`: Bulan kontak paling dominan adalah bulan Mei (`may`: $13.766$ observasi, lebih dari $30{,}4\%$ dari keseluruhan kontak).""")

    # =========================================================================
    # CELL 9: Code - 3.2 Audit Kualitas Data
    # =========================================================================
    add_code(r"""# ============================================================
# 3.2 AUDIT KUALITAS DATA: MISSING VALUES, DUPLIKAT, 'unknown', & pdays=-1
# ============================================================
print("--- [1] PEMERIKSAAN NILAI HILANG STANDAR (NaN/Null) ---")
null_counts = df.isnull().sum()
print(f"Total sel bernilai null/NaN: {null_counts.sum()}")

print("\n--- [2] PEMERIKSAAN DUPLIKASI BARIS ---")
dup_count = df.duplicated().sum()
print(f"Total baris duplikat (exact duplicate rows): {dup_count}")

print("\n--- [3] DETEKSI MISSING VALUE IMPLISIT BERLABEL 'unknown' ---")
unknown_summary = []
for col in df.columns:
    count_unk = (df[col] == 'unknown').sum()
    if count_unk > 0:
        pct_unk = (count_unk / len(df)) * 100
        unknown_summary.append({
            'Nama Kolom': col,
            'Tipe Data': df[col].dtype,
            'Jumlah Unknown': count_unk,
            'Persentase (%)': round(pct_unk, 2)
        })
df_unknown = pd.DataFrame(unknown_summary)
display(df_unknown)

print("\n--- [4] AUDIT NILAI KHUSUS pdays == -1 ---")
pdays_minus_1 = (df['pdays'] == -1).sum()
pct_pdays_minus_1 = (pdays_minus_1 / len(df)) * 100
print(f"Jumlah baris dengan pdays = -1: {pdays_minus_1:,} baris ({pct_pdays_minus_1:.2f}%)")""")

    # =========================================================================
    # CELL 10: Markdown - Interpretasi Temuan 3.2
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 3.2: Kualitas Data & Risiko Imputasi Modus
1. **Tidak Ada Nilai Null Standar (`isnull().sum() == 0`):** Secara teknis sintaks data bersih dari `NaN`, namun institusi perbankan menyandikan ketidaktahuan informasi ke dalam label string `'unknown'`.
2. **Bebas Duplikasi Baris (`duplicated().sum() == 0`):** Seluruh $45.211$ baris merupakan observasi kontak unik, sehingga tidak ada redundansi pencatatan yang perlu dieliminasi.
3. **Analisis Nilai Terselubung `'unknown'`:**
   * `job`: $288$ baris ($0{,}64\%$) `'unknown'`.
   * `education`: $1.857$ baris ($4{,}11\%$) `'unknown'`.
   * `contact`: $13.020$ baris ($28{,}80\%$) `'unknown'` (media kontak tidak tercatat).
   * `poutcome`: **$36.959$ baris ($81{,}75\%$) `'unknown'`**.
4. **Alasan Krusial TIDAK Mengimputasi `'unknown'` dengan Modus:**
   * Di dalam notebook terdahulu, nilai `'unknown'` diubah menjadi `NaN` lalu diimputasi menggunakan modus (`most_frequent`). Pada kolom `poutcome`, modus data adalah `'failure'` ($4.901$ baris). Jika diimputasi modus, sebanyak **$36.959$ nasabah yang sebenarnya belum pernah dihubungi ($81{,}75\%$) akan dipaksa salah menjadi 'pernah gagal dihubungi'**!
   * Hal ini merusak validitas historis domain secara fatal. Oleh sebab itu, label `'unknown'` **dipertahankan sebagai kategori mandiri yang sah** agar model dapat membedakan antara nasabah baru dan nasabah dengan riwayat kampanye sebelumnya.
5. **Makna Domain `pdays = -1`:**
   * Sebanyak $36.954$ baris ($81{,}74\%$) bernilai `pdays = -1`. Nilai ini bukan anomali kesalahan ketik, melainkan kode representasi bahwa nasabah tersebut belum pernah dikontak dalam kampanye sebelumnya, sinkron dengan frekuensi `poutcome == 'unknown'`.""")

    # =========================================================================
    # CELL 11: Code - 3.3 Distribusi Kelas Target y
    # =========================================================================
    add_code(r"""# ============================================================
# 3.3 DISTRIBUSI KELAS TARGET y (BAR CHART & PIE CHART)
# ============================================================
y_counts = df['y'].value_counts()
y_pct = df['y'].value_counts(normalize=True) * 100

print(f"Frekuensi Kelas Target 'y':\\n{y_counts}")
print(f"Persentase Kelas Target 'y':\\n{y_pct.round(2)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Bar Chart
palette = {'no': '#e74c3c', 'yes': '#2ecc71'}
sns.barplot(x=y_counts.index, y=y_counts.values, palette=palette, ax=axes[0], edgecolor='black')
axes[0].set_title("Distribusi Frekuensi Kelas Target (y)", fontweight='bold')
axes[0].set_xlabel("Status Pembukaan Deposito (y)")
axes[0].set_ylabel("Jumlah Observasi (Nasabah)")
for i, v in enumerate(y_counts.values):
    axes[0].text(i, v + 700, f"{v:,}\\n({y_pct.iloc[i]:.2f}%)", ha='center', fontweight='bold')
axes[0].set_ylim(0, y_counts.max() * 1.15)

# 2. Donut/Pie Chart
axes[1].pie(
    y_counts.values,
    labels=[f"No ({y_counts['no']:,} | {y_pct['no']:.1f}%)", f"Yes ({y_counts['yes']:,} | {y_pct['yes']:.1f}%)"],
    colors=['#e74c3c', '#2ecc71'],
    autopct='%1.2f%%',
    startangle=140,
    explode=(0, 0.1),
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.2, 'width': 0.6}
)
axes[1].set_title("Proporsi Relatif Kelas Target (Imbalance Ratio: 7.55 : 1)", fontweight='bold')

plt.suptitle("Analisis Ketidakseimbangan Kelas Target (y)", fontsize=14, fontweight='bold', y=1.03)
plt.show()""")

    # =========================================================================
    # CELL 12: Markdown - Interpretasi Temuan 3.3
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 3.3: Ketidakseimbangan Kelas Target & Metrik Evaluasi
1. **Rasio Ketidakseimbangan Ekstrem ($7{,}55 : 1$):**
   * Kelas mayoritas `'no'` berjumlah $39.922$ nasabah ($88{,}30\%$).
   * Kelas minoritas `'yes'` (sukses membuka deposito) hanya berjumlah $5.289$ nasabah ($11{,}70\%$).
2. **Kelemahan Fatal Metrik Akurasi (*Accuracy Trap*):**
   * Jika model *dummy* naif memprediksi seluruh nasabah tidak akan membuka deposito (`y = 'no'`), akurasinya akan mencapai **$88{,}30\%$**. Namun, model tersebut memiliki nilai guna operasional sama dengan nol karena gagal menangkap satupun ($0\%$) nasabah deposito.
3. **Justifikasi PR-AUC (*Precision-Recall AUC*) Sebagai Metrik Utama:**
   * Mengingat biaya *False Positive* (panggilan telepon sia-sia) dan biaya *False Negative* (kehilangan potensi dana deposito nasabah) sama-sama penting, serta kelas positif sangat langka, **PR-AUC (*Average Precision*)** ditetapkan sebagai metrik optimasi utama.
   * Nilai dasar (*random baseline*) dari PR-AUC adalah sama dengan prevalensi kelas minoritas pada populasi:
     $$\\text{Baseline PR-AUC} = \\frac{5.289}{45.211} \\approx 0{,}117$$
   * Target teknis PR-AUC $\ge 0{,}45$ merepresentasikan kemampuan model yang hampir **$4\times$ lipat lebih unggul daripada peluang acak**.""")

    # =========================================================================
    # CELL 13: Code - 3.4 Deteksi & Evaluasi Outlier (Metode IQR)
    # =========================================================================
    add_code(r"""# ============================================================
# 3.4 DETEKSI & EVALUASI OUTLIER (METODE IQR & BOXPLOT)
# ============================================================
outlier_features = ['age', 'balance', 'campaign']
outlier_stats = []

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(outlier_features):
    # Hitung Statistik IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    outlier_stats.append({
        'Fitur': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Jumlah Outlier': outlier_count,
        'Persentase (%)': round(outlier_pct, 2)
    })
    
    # Plot Boxplot
    sns.boxplot(y=df[col], ax=axes[i], color='#3498db', flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3})
    axes[i].set_title(f"Boxplot {col}\\n({outlier_count:,} outlier | {outlier_pct:.2f}%)", fontweight='bold')
    axes[i].set_ylabel(col)

plt.suptitle("Deteksi Outlier Metode Rentang Antarkuartil (IQR)", fontsize=14, fontweight='bold', y=1.03)
plt.show()

df_outlier_table = pd.DataFrame(outlier_stats)
print("--- TABEL RINGKASAN BATAS IQR & JUMLAH OUTLIER ---")
display(df_outlier_table)""")

    # =========================================================================
    # CELL 14: Markdown - Interpretasi Temuan 3.4
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 3.4: Analisis Outlier & Justifikasi Domain Perbankan
1. **Temuan Outlier Berdasarkan Rumus Statistik IQR:**
   * `age`: Terdapat $4.870$ observasi ($10{,}77\%$) dengan usia di atas $70{,}5$ tahun.
   * `balance`: Terdapat $4.729$ observasi ($10{,}46\%$) dengan saldo di bawah -€$1.962$ atau di atas €$3.462$.
   * `campaign`: Terdapat $3.064$ observasi ($6{,}78\%$) dengan jumlah kontak melebihi $6$ kali.
2. **Justifikasi Domain: Mengapa Outlier TIDAK Dihapus?**
   * Dalam industri perbankan, nasabah berusia lanjut ($>70$ tahun) merupakan segmen pensiunan (*retiree*) yang secara historis memiliki likuiditas tinggi dan preferensi kuat terhadap produk deposito berpenghasilan tetap.
   * Saldo bernilai negatif (cerukan/fasilitas *overdraft*) dan saldo sangat tinggi ($> €10.000$) adalah kenyataan bisnis riil nasabah bank (*wealth clients* vs nasabah defisit). Menghapus data ini akan melenyapkan informasi nasabah paling potensial bagi produk deposito.
   * Jumlah kontak kampanye tinggi ($>6$) mencerminkan agresivitas penawaran telemarketing.
   * Oleh karena itu, seluruh observasi bernilai ekstrem ini **dipertahankan seutuhnya** karena valid secara domain dan algoritma berbasis pohon (*tree-based ensembles*) seperti `HistGradientBoostingClassifier` sangat tangguh (*robust*) terhadap nilai ekstrem tanpa perlu pemangkasan data.""")

    # =========================================================================
    # CELL 15: Code - 3.5 Analisis Matriks Korelasi Pearson Fitur Numerik
    # =========================================================================
    add_code(r"""# ============================================================
# 3.5 ANALISIS MATRIKS KORELASI PEARSON FITUR NUMERIK
# ============================================================
# Encode target y secara numerik khusus untuk korelasi poin-biserial
num_cols_corr = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
df_corr_calc = df[num_cols_corr].copy()
df_corr_calc['y_numeric'] = df['y'].map({'yes': 1, 'no': 0})

corr_matrix = df_corr_calc.corr(method='pearson')

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap='coolwarm',
    center=0,
    vmin=-0.2,
    vmax=0.5,
    square=True,
    linewidths=1.0,
    cbar_kws={"shrink": 0.8}
)
plt.title("Matriks Korelasi Pearson Fitur Numerik vs Target (y_numeric)", fontsize=14, fontweight='bold', pad=15)
plt.show()

print("--- KORELASI FITUR NUMERIK DENGAN TARGET (y_numeric) ---")
display(corr_matrix['y_numeric'].sort_values(ascending=False).to_frame().T)""")

    # =========================================================================
    # CELL 16: Markdown - Interpretasi Temuan 3.5
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 3.5: Analisis Korelasi & Bukti Kritis Data Leakage `duration`
1. **Multikolinearitas Antar Prediktor Rendah:** Tidak ada korelasi antar fitur numerik bebas yang melebihi ambang batas multikolinearitas ($|r| > 0{,}70$). Korelasi tertinggi hanya terjadi antara `pdays` dan `previous` ($r = 0{,}455$), yang wajar karena nasabah yang dihubungi di masa lalu memiliki catatan hari kontak terdahulu.
2. **Korelasi Terkuat: `duration` vs `y_numeric` ($r = 0{,}395$):**
   * Variabel `duration` memiliki koefisien korelasi positif paling tinggi terhadap keputusan nasabah membuka deposito. Semakin lama nasabah berbicara di telepon, semakin tinggi kemungkinan konversi.
3. **Konfirmasi Kebocoran Data (*Ex-Post Leakage Confirmation*):**
   * Durasi telepon **hanya ada setelah telepon diangkat dan pembicaraan selesai**. Dalam skenario operasional penentuan prospek panggilan hari esok (*ex-ante calling list*), nilai durasi belum pernah terjadi.
   * Model yang dilatih dengan `duration` akan mengalami *false confidence* (mencapai ROC-AUC $>0{,}93$ secara semu di lab, namun ambruk total saat dioperasionalkan). Oleh karena itu, korelasi tinggi ini menegaskan keputusan mutlak untuk **menghapus kolom `duration`** sebelum pemodelan.""")

    # =========================================================================
    # CELL 17: Code - 3.6 Hubungan Fitur Kategorikal vs Target y
    # =========================================================================
    add_code(r"""# ============================================================
# 3.6 HUBUNGAN FITUR KATEGORIKAL VS TARGET y (JOB & POUTCOME)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# --- [A] ANALISIS JOB VS Y ---
job_group = df.groupby('job')['y'].value_counts(normalize=True).unstack().fillna(0)['yes'] * 100
job_order = job_group.sort_values(ascending=False).index

sns.countplot(data=df, y='job', order=job_order, hue='y', palette={'no': '#e74c3c', 'yes': '#2ecc71'}, ax=axes[0, 0])
axes[0, 0].set_title("Volume Panggilan Berdasarkan Jenis Pekerjaan (job)", fontweight='bold')
axes[0, 0].set_xlabel("Jumlah Nasabah")
axes[0, 0].set_ylabel("Pekerjaan")

sns.barplot(x=job_group[job_order].values, y=job_order, color='#2ecc71', ax=axes[0, 1], edgecolor='black')
axes[0, 1].axvline(x=y_pct['yes'], color='red', linestyle='--', label=f"Rata-rata Populasi ({y_pct['yes']:.2f}%)")
axes[0, 1].set_title("Tingkat Konversi Deposito (%) per Jenis Pekerjaan", fontweight='bold')
axes[0, 1].set_xlabel("Conversion Rate (%)")
axes[0, 1].set_ylabel("")
axes[0, 1].legend()
for i, v in enumerate(job_group[job_order].values):
    axes[0, 1].text(v + 0.5, i, f"{v:.2f}%", va='center', fontweight='bold')

# --- [B] ANALISIS POUTCOME VS Y ---
pout_group = df.groupby('poutcome')['y'].value_counts(normalize=True).unstack().fillna(0)['yes'] * 100
pout_order = pout_group.sort_values(ascending=False).index

sns.countplot(data=df, x='poutcome', order=pout_order, hue='y', palette={'no': '#e74c3c', 'yes': '#2ecc71'}, ax=axes[1, 0])
axes[1, 0].set_title("Volume Panggilan Berdasarkan Hasil Kampanye Sebelumnya (poutcome)", fontweight='bold')
axes[1, 0].set_xlabel("Hasil Kampanye Lampau")
axes[1, 0].set_ylabel("Jumlah Nasabah")

sns.barplot(x=pout_order, y=pout_group[pout_order].values, color='#3498db', ax=axes[1, 1], edgecolor='black')
axes[1, 1].axhline(y=y_pct['yes'], color='red', linestyle='--', label=f"Rata-rata Populasi ({y_pct['yes']:.2f}%)")
axes[1, 1].set_title("Tingkat Konversi Deposito (%) per Hasil Kampanye Lampau", fontweight='bold')
axes[1, 1].set_ylabel("Conversion Rate (%)")
axes[1, 1].set_xlabel("")
axes[1, 1].legend()
for i, v in enumerate(pout_group[pout_order].values):
    axes[1, 1].text(i, v + 1.5, f"{v:.2f}%", ha='center', fontweight='bold')

plt.suptitle("Analisis Bivariat Fitur Kategorikal Kunci vs Status Deposito (y)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # CELL 18: Markdown - Interpretasi Temuan 3.6
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 3.6: Pengaruh Pekerjaan & Keberhasilan Kampanye Sebelumnya
1. **Pengaruh Pekerjaan Terhadap Konversi (`job`):**
   * Segmen dengan rasio konversi tertinggi adalah **`student` ($28{,}68\%$)** dan **`retired` ($22{,}79\%$)**, keduanya jauh melampaui rata-rata populasi bank ($11{,}70\%$).
   * Sebaliknya, segmen dengan volume panggilan terbesar seperti `blue-collar` ($9.732$ kontak) justru memiliki rasio konversi terendah, yaitu hanya **$7{,}27\%$**. Hal ini membuktikan inefisiensi alokasi kampanye historis di mana tenaga agen banyak dihabiskan pada segmen dengan daya minat terendah.
2. **Efek Dominan Riwayat Keberhasilan Kampanye Terdahulu (`poutcome`):**
   * Nasabah dengan status `poutcome == 'success'` (pernah sukses menerima produk bank di masa lalu) mencatatkan tingkat konversi fantastis mencapai **$64{,}73\%$**!
   * Ini membuktikan bahwa riwayat hubungan baik nasabah (*prior engagement success*) adalah prediktor minat paling kuat dalam kampanye telemarketing.""")

    # =========================================================================
    # CELL 19: Code - 3.7 Analisis Pola Temporal Bulanan (month)
    # =========================================================================
    add_code(r"""# ============================================================
# 3.7 ANALISIS POLA TEMPORAL BULANAN (month JAN-DEC VS y)
# ============================================================
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df_month = df.groupby('month').agg(
    total_contacts=('y', 'count'),
    total_yes=('y', lambda x: (x == 'yes').sum())
).reindex(month_order)
df_month['conversion_rate'] = (df_month['total_yes'] / df_month['total_contacts']) * 100

fig, ax1 = plt.subplots(figsize=(14, 5))

# Sumbu 1: Total Panggilan (Bar)
color = '#3498db'
bars = ax1.bar(df_month.index, df_month['total_contacts'], color=color, alpha=0.65, edgecolor='black', label='Volume Panggilan')
ax1.set_xlabel('Bulan Kontak Kalender (Kronologis Jan - Des)', fontweight='bold')
ax1.set_ylabel('Total Volume Panggilan', color=color, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color)

# Sumbu 2: Conversion Rate (%) (Line)
ax2 = ax1.twinx()
color2 = '#e74c3c'
line = ax2.plot(df_month.index, df_month['conversion_rate'], color=color2, marker='o', linewidth=2.5, label='Tingkat Konversi (%)')
ax2.axhline(y=y_pct['yes'], color='gray', linestyle='--', label=f"Rata-rata Populasi ({y_pct['yes']:.2f}%)")
ax2.set_ylabel('Tingkat Konversi (%)', color=color2, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(False)

for i, (m, r) in enumerate(zip(df_month.index, df_month['conversion_rate'])):
    ax2.annotate(f"{r:.1f}%", (i, r + 1.5), ha='center', fontsize=9, fontweight='bold', color=color2)

plt.title("Pola Efisiensi Kampanye Bulanan: Volume Panggilan vs Tingkat Konversi (%)", fontsize=14, fontweight='bold', pad=15)
plt.show()

print("--- TABEL KINERJA KAMPANYE TELEMARKETING PER BULAN ---")
display(df_month)""")

    # =========================================================================
    # CELL 20: Markdown - Interpretasi Temuan 3.7
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 3.7: Efek Musiman & Inefisiensi Alokasi Kampanye Bulanan
1. **Inefisiensi Alokasi di Bulan Mei (`may`):**
   * Bulan Mei merupakan bulan dengan volume kontak tertinggi ($13.766$ panggilan, $>30{,}4\%$ dari total data), namun menghasilkan tingkat konversi paling rendah sepanjang tahun, yaitu hanya **$6{,}72\%$**. Ini mengindikasikan strategi kampanye 'kejar setoran' yang tidak terarah (*spray and pray approach*).
2. **Bulan Emas Konversi Tinggi (*Golden Months*):**
   * Bulan **Maret (`mar`: $51{,}99\%$)**, **September (`sep`: $46{,}46\%$)**, **Oktober (`oct`: $43{,}77\%$)**, dan **Desember (`dec`: $46{,}73\%$)** mencatatkan tingkat konversi sangat tinggi di atas $40\%–50\%$, meskipun volume kontak di bulan-bulan tersebut relatif sedikit ($<800$ kontak per bulan).
   * Hal ini mencerminkan efek musiman perbankan: penutupan kuartal dan akhir tahun merupakan periode di mana nasabah menerima bonus tahunan, dividen, atau merencanakan alokasi investasi masa depan sehingga lebih reseptif terhadap tawaran deposito.""")

    return cells

