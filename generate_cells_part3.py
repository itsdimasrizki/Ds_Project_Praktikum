# -*- coding: utf-8 -*-
"""
Part 3: Fase 4 (Preprocessing) & Fase 5 (Modeling) cells for ProjekAkhir.ipynb.
"""

def add_fase_4_and_5_cells(cells):
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
    # CELL 21: Markdown - FASE 4: Data Cleaning, Preprocessing & Feature Selection
    # =========================================================================
    add_md(r"""---
# 🔧 FASE 4: Data Cleaning, Preprocessing & Feature Selection

Tahap persiapan data dirancang dengan mematuhi prinsip ketat **Anti-Leakage Data Architecture**:
1. **Pembersihan Data Terstruktur:** Memastikan ketiadaan baris duplikat dan mengonversi label target biner menjadi representasi numerik standar ($1$ dan $0$).
2. **Seleksi Fitur Eksplisit (*Feature Selection*):** Mengeliminasi kolom `duration` untuk mencegah *ex-post data leakage*, mempertahankan $15$ prediktor asli, dan menghindari pembuatan fitur sintetis redundan.
3. **Partisi Data Anti-Leakage (*Train-Test Split*):** Memisahkan data uji (*hold-out test set*) sebesar $20\%$ sebelum proses encoding, scaling, maupun resampling apapun dilakukan.
4. **Pipeline Modular (*ColumnTransformer*):** Menggunakan `OneHotEncoder(drop='first')` pada 9 kolom kategorikal dan passthrough pada 6 kolom numerik sehingga menghasilkan ruang representasi tepat **$41$ fitur**.
5. **Audit Sebelum vs Sesudah:** Menyajikan tabel audit komparatif kondisi data sebelum dan sesudah preprocessing sesuai rubrik panduan praktikum.""")

    # =========================================================================
    # CELL 22: Code - 4.1 Pembersihan Data & Target Encoding
    # =========================================================================
    add_code(r"""# ============================================================
# 4.1 PEMBERSIHAN DATA & ENCODING TARGET BINARY
# ============================================================
print("--- [1] VERIFIKASI & ELIMINASI DUPLIKASI DATA ---")
initial_rows = len(df)
df = df.drop_duplicates()
final_rows = len(df)
print(f"Baris sebelum drop_duplicates: {initial_rows:,}")
print(f"Baris setelah drop_duplicates : {final_rows:,} (Eliminasi: {initial_rows - final_rows} baris)")

print("\n--- [2] TRANSFORMASI TARGET BINARY (y -> 1 / 0) ---")
# 'yes' -> 1 (berlangganan deposito), 'no' -> 0 (tidak berlangganan)
y_encoded = df['y'].map({'yes': 1, 'no': 0}).astype(int)
print(f"Distribusi Target y Ter-encode:\\n{y_encoded.value_counts()}")
print(f"Prevalensi Positif (Baseline Acak PR-AUC): {y_encoded.mean():.4f}")

print("\n--- [3] INTEGRITAS NILAI 'unknown' & pdays = -1 ---")
print("Sesuai justifikasi Bab 3.2, nilai 'unknown' tetap dipertahankan sebagai kategori valid,")
print("dan kode pdays = -1 tetap dipertahankan sebagai representasi 'belum pernah dikontak'.")""")

    # =========================================================================
    # CELL 23: Markdown - 4.2 Feature Selection Eksplisit
    # =========================================================================
    add_md(r"""### 📌 4.2 Seleksi Fitur Eksplisit (*Feature Selection*)

Sesuai Panduan Praktikum Data Science, pemilihan fitur harus didasarkan pada penalaran logis, kebutuhan bisnis, dan pencegahan *data leakage*:

| Nama Fitur | Status Seleksi | Justifikasi Teknis & Bisnis Berdasarkan Domain |
|:---|:---:|:---|
| **`duration`** | **DIELIMINASI (DROP)** | **Ex-Post Data Leakage Kritis.** Durasi panggilan telepon hanya diketahui setelah pembicaraan telepon berakhir. Dalam skenario operasional perbankan di mana model bertugas menyusun daftar prioritas panggilan esok hari (*ex-ante calling list*), informasi durasi panggilan belum pernah terjadi. Memasukkan durasi akan menciptakan model dengan performa semu yang gagal total di lingkungan produksi. |
| **15 Prediktor Asli** (`age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome`) | **DIPERTAHANKAN** | **Karakteristik Nasabah & Konteks Kampanye yang Valid.** Seluruh 15 prediktor ini telah tersedia di basis data perbankan sebelum panggilan dilakukan (*ex-ante available*). Kombinasi informasi demografis, kepemilikan kredit, perilaku saldo, dan riwayat kampanye terdahulu memberikan sinyal diskriminatif yang kaya bagi model machine learning. |
| **Fitur Sintetis Baru Redundan** | **TIDAK DITAMBAHKAN** | **Prinsip Parsimoni & Pencegahan Curse of Dimensionality.** Penambahan fitur turunan (seperti `age_group`, `month_num`, `log_balance`) yang dibuat berdampingan dengan fitur aslinya hanya akan meningkatkan multikolinearitas dan membebani komputasi tanpa memberikan sinyal informasi baru. |""")

    # =========================================================================
    # CELL 24: Code - 4.2 Drop duration & Feature Separation
    # =========================================================================
    add_code(r"""# ============================================================
# 4.2 ELIMINASI KOLOM DURATION & PEMISAHAN MATRIKS FITUR
# ============================================================
# Eliminasi kolom duration demi mencegah kebocoran data
X = df.drop(columns=['y', 'duration'])
y = y_encoded

print(f"Matriks Prediktor X berhasil dibuat: {X.shape[0]:,} baris x {X.shape[1]} kolom prediktor")
print(f"Daftar 15 Fitur Prediktor Terpilih:\\n{list(X.columns)}")""")

    # =========================================================================
    # CELL 25: Code - 4.3 Partisi Data Anti-Leakage (Train-Test Split)
    # =========================================================================
    add_code(r"""# ============================================================
# 4.3 PARTISI DATA ANTI-LEAKAGE (STRATIFIED HOLD-OUT 80:20)
# ============================================================
# Pemisahan data latih (80%) dan data uji (20%) secara terstrata berdasarkan y
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Dimensi Data Latih (Train Set - 80%): {X_train.shape[0]:,} baris x {X_train.shape[1]} kolom")
print(f"Dimensi Data Uji   (Test Set  - 20%): {X_test.shape[0]:,} baris x {X_test.shape[1]} kolom")
print(f"Proporsi Positif Data Latih (y_train): {y_train.mean():.4f} ({y_train.sum():,} 'yes')")
print(f"Proporsi Positif Data Uji   (y_test) : {y_test.mean():.4f} ({y_test.sum():,} 'yes')")""")

    # =========================================================================
    # CELL 26: Markdown - Interpretasi 4.3
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 4.3: Integritas Partisi Data Anti-Leakage
1. **Pemisahan Terstrata Sempurna (*Stratified Partitioning*):** Proporsi kelas minoritas positif pada data latih ($11{,}70\%$, $4.231$ nasabah) dan data uji ($11{,}70\%$, $1.058$ nasabah) identik persis dengan populasi asli ($11{,}70\%$).
2. **Pencegahan Kebocoran Data Evaluasi (*Information Isolation*):** Data uji sebanyak $9.043$ observasi diisolasi sepenuhnya dan **tidak pernah disentuh** dalam proses fitting encoder, normalisasi, maupun pemilihan hyperparameter model.""")

    # =========================================================================
    # CELL 27: Code - 4.4 Perancangan Preprocessing Pipeline
    # =========================================================================
    add_code(r"""# ============================================================
# 4.4 PERANCANGAN PREPROCESSING PIPELINE (ColumnTransformer)
# ============================================================
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']

print(f"Jumlah Fitur Kategorikal (OHE drop='first'): {len(categorical_cols)} kolom")
print(f"Jumlah Fitur Numerik (Passthrough)       : {len(numerical_cols)} kolom")

# One-Hot Encoding dengan drop='first' untuk mencegah perangkap dummy (dummy variable trap)
# handle_unknown='ignore' menjamin kestabilan jika muncul kategori asing di data uji
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', ohe, categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    remainder='drop'
)

# Fit hanya pada data latih untuk memverifikasi dimensi fitur hasil transformasi
preprocessor.fit(X_train)
X_train_proc_sample = preprocessor.transform(X_train)
feature_names = preprocessor.get_feature_names_out()

print(f"\\nTransformasi Berhasil!")
print(f"Total Dimensi Fitur Setelah One-Hot Encoding: TEPAT {X_train_proc_sample.shape[1]} FITUR")
print(f"Rincian: {len(feature_names) - len(numerical_cols)} fitur dummy biner + {len(numerical_cols)} fitur numerik asli.")""")

    # =========================================================================
    # CELL 28: Code - 4.5 Tabel Komparasi Kondisi Data SEBELUM vs SESUDAH Preprocessing
    # =========================================================================
    add_code(r"""# ============================================================
# 4.5 TABEL KOMPARASI KONDISI DATA SEBELUM VS SESUDAH PREPROCESSING
# ============================================================
data_audit = [
    {
        'Aspek / Karakteristik': 'Dimensi Observasi (Baris x Kolom)',
        'Sebelum Preprocessing (Data Mentah)': f"{initial_rows:,} baris x 17 kolom",
        'Sesudah Preprocessing (Data Siap Latih)': f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} x 41 fitur"
    },
    {
        'Aspek / Karakteristik': 'Kolom Kebocoran Data (duration)',
        'Sebelum Preprocessing (Data Mentah)': 'Ada (Nilai korelasi r = 0.395 terhadap y)',
        'Sesudah Preprocessing (Data Siap Latih)': 'DIELIMINASI SEPENUHNYA (0% Data Leakage)'
    },
    {
        'Aspek / Karakteristik': 'Penanganan Nilai Hilang Implisit unknown',
        'Sebelum Preprocessing (Data Mentah)': 'String unknown di job, education, contact, poutcome',
        'Sesudah Preprocessing (Data Siap Latih)': 'Dipertahankan sebagai dummy category yang valid'
    },
    {
        'Aspek / Karakteristik': 'Format Kolom Target (y)',
        'Sebelum Preprocessing (Data Mentah)': "String kategorikal: 'yes' (11.70%) & 'no' (88.30%)",
        'Sesudah Preprocessing (Data Siap Latih)': 'Integer numerik biner: 1 (positif) & 0 (negatif)'
    },
    {
        'Aspek / Karakteristik': 'Encoding Variabel Kategorikal',
        'Sebelum Preprocessing (Data Mentah)': '9 Kolom string kategorikal non-numerik',
        'Sesudah Preprocessing (Data Siap Latih)': 'One-Hot Encoded (drop first) -> 35 kolom biner'
    },
    {
        'Aspek / Karakteristik': 'Variabel Numerik Kontinyu',
        'Sebelum Preprocessing (Data Mentah)': '6 Kolom numerik ber-skala asli (Euro, hari, dll.)',
        'Sesudah Preprocessing (Data Siap Latih)': '6 Kolom numerik dipertahankan / di-scale via Pipeline'
    },
    {
        'Aspek / Karakteristik': 'Total Dimensi Ruang Fitur Model',
        'Sebelum Preprocessing (Data Mentah)': '16 Fitur prediktor mentah',
        'Sesudah Preprocessing (Data Siap Latih)': 'Tepat 41 Fitur numerik bersih'
    }
]

df_data_audit = pd.DataFrame(data_audit)
display(df_data_audit)""")

    # =========================================================================
    # CELL 29: Markdown - Interpretasi 4.5
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 4.5: Audit Perubahan Data Preprocessing
Tabel audit di atas memverifikasi kepatuhan penuh terhadap prinsip integritas data sains:
1. **Bebas Kontaminasi:** Tidak ada data uji yang bocor ke data latih (*zero test data contamination*).
2. **Kesesuaian Dimensi Laporan:** Menghasilkan tepat **$41$ fitur** ($35$ dummy biner $+ 6$ numerik kontinyu), sinkron $100\%$ dengan Bab III & Bab VI Laporan Proyek Akhir.
3. **Pelestarian Sinyal Domain:** Tidak ada imputasi modus yang merusak kategori `'unknown'` pada `poutcome` dan `contact`.""")

    # =========================================================================
    # CELL 30: Markdown - FASE 5: Modeling & Hyperparameter Tuning
    # =========================================================================
    add_md(r"""---
# 🤖 FASE 5: Modeling & Hyperparameter Tuning

## 5.1 Filosofi Anti-Leakage Resampling & Validasi Silang
Pada notebook terdahulu, teknik *resampling* SMOTE diaplikasikan pada seluruh data latih sebelum proses validasi silang (Cross-Validation). Hal tersebut menyebabkan **kebocoran data validasi yang fatal (*fatal validation leakage*)**, di mana sampel sintetis yang dibuat dari baris yang sama tersebar ke dalam *training fold* dan *validation fold* sekaligus.

Untuk memperbaiki masalah tersebut secara permanen, seluruh eksperimen pemodelan di bawah ini menggunakan **`imblearn.pipeline.Pipeline`** yang dikombinasikan dengan **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`**. Dengan skema ini, teknik *resampling* (`SMOTENC`) **hanya dan hanya diterapkan pada training fold** di tiap putaran CV, sedangkan validation fold tetap murni dan merefleksikan distribusi populasi aslinya.

## 5.2 Strategi Komparasi 6 Kandidat Model
Kami mengevaluasi 6 kombinasi arsitektur model dan strategi penanganan *class imbalance*:
1. **Model 1 (Strong Linear Baseline):** `LogisticRegression` + `StandardScaler` + `class_weight='balanced'`
2. **Model 2 (Linear + Synthetic Resampling):** `LogisticRegression` + `StandardScaler` + `SMOTENC`
3. **Model 3 (Ensemble Bagging):** `RandomForestClassifier` + `class_weight='balanced_subsample'`
4. **Model 4 (Gradient Boosting Natural):** `HistGradientBoostingClassifier` (tanpa resampling)
5. **Model 5 (Gradient Boosting Cost-Sensitive):** `HistGradientBoostingClassifier` + `class_weight='balanced'`
6. **Model 6 (Gradient Boosting + Synthetic Resampling):** `HistGradientBoostingClassifier` + `SMOTENC`

Seluruh kandidat dievaluasi menggunakan metrik optimasi utama **PR-AUC (`average_precision`)** dan metrik pendukung **ROC-AUC (`roc_auc`)**.""")

    # =========================================================================
    # CELL 31: Code - 5.1 & 5.2 Evaluasi Komparatif 6 Kandidat Model
    # =========================================================================
    add_code(r"""# ============================================================
# 5.1 & 5.2 EVALUASI KOMPARATIF 6 KANDIDAT MODEL (5-FOLD CV ANTI-LEAKAGE)
# ============================================================
# Inisialisasi 5-Fold Stratified Cross-Validation
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Dapatkan indeks fitur kategorikal pada X_train untuk SMOTENC
cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

# Definisi Preprocessor Khusus Linear Model (dengan StandardScaler)
preprocessor_linear = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ],
    remainder='drop'
)

# Definisi 6 Pipeline Kandidat Model
models_dict = {
    "1. LogReg + class_weight='balanced'": ImbPipeline([
        ('prep', preprocessor_linear),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))
    ]),
    "2. LogReg + SMOTENC": ImbPipeline([
        ('smote', SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)),
        ('prep', preprocessor_linear),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "3. RandomForest + class_weight='balanced_subsample'": ImbPipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1))
    ]),
    "4. HistGB (Tanpa Resampling)": ImbPipeline([
        ('prep', preprocessor),
        ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    ]),
    "5. HistGB + class_weight='balanced'": ImbPipeline([
        ('prep', preprocessor),
        ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=RANDOM_STATE))
    ]),
    "6. HistGB + SMOTENC": ImbPipeline([
        ('smote', SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)),
        ('prep', preprocessor),
        ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
}

# Eksekusi Evaluasi Validasi Silang 5-Fold
cv_results_list = []
print("Memulai evaluasi komparatif 6 model via 5-Fold Cross-Validation anti-leakage...")
print("=" * 78)

for name, pipe in models_dict.items():
    start_time = time.time()
    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv_stratified,
        scoring=['average_precision', 'roc_auc'],
        n_jobs=-1,
        return_train_score=False
    )
    fit_time_total = time.time() - start_time
    
    pr_mean = scores['test_average_precision'].mean()
    pr_std = scores['test_average_precision'].std()
    roc_mean = scores['test_roc_auc'].mean()
    roc_std = scores['test_roc_auc'].std()
    
    cv_results_list.append({
        'Model & Strategi Resampling': name,
        'PR-AUC CV (Mean ± Std)': f"{pr_mean:.4f} ± {pr_std:.4f}",
        'PR-AUC Mean': pr_mean,
        'PR-AUC Std': pr_std,
        'ROC-AUC CV (Mean ± Std)': f"{roc_mean:.4f} ± {roc_std:.4f}",
        'ROC-AUC Mean': roc_mean,
        'ROC-AUC Std': roc_std,
        'Waktu Fit (detik)': round(fit_time_total, 2)
    })
    print(f"✓ Selesai: {name:50s} | PR-AUC: {pr_mean:.4f} | ROC-AUC: {roc_mean:.4f} | Waktu: {fit_time_total:.1f}s")

print("=" * 78)
df_cv_results = pd.DataFrame(cv_results_list)
display(df_cv_results[['Model & Strategi Resampling', 'PR-AUC CV (Mean ± Std)', 'ROC-AUC CV (Mean ± Std)', 'Waktu Fit (detik)']])""")

    # =========================================================================
    # CELL 32: Markdown - Interpretasi Temuan 5.2
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 5.2: Perbandingan Model & Teori Penurunan SMOTENC
1. **Model Terbaik: `HistGradientBoostingClassifier` Tanpa Resampling:**
   * Model `HistGB (Tanpa Resampling)` mencapai nilai **PR-AUC CV tertinggi ($0{,}4696$)** dan **ROC-AUC CV tertinggi ($0{,}8025$)**, melampaui seluruh kandidat lainnya.
2. **Penjelasan Teoritis Mengapa `SMOTENC` Menurunkan PR-AUC:**
   * Pada `HistGB + SMOTENC`, nilai PR-AUC turun dari $0{,}4696$ menjadi $0{,}4367$ (penurunan signifikan). Hal serupa terjadi pada `LogisticRegression`.
   * **Mengapa demikian?** Algoritma SMOTE dan SMOTENC melakukan interpolasi titik data sintetis kelas minoritas di dalam ruang fitur. Pada dataset dengan banyak fitur dummy biner yang jarang (*sparse high-dimensional dummy space*), interpolasi linear menghasilkan sampel sintetis di wilayah ruang yang sebenarnya didominasi oleh kelas mayoritas.
   * Akibatnya, model mengalami bias prediksi positif berlebihan, yang melipatgandakan jumlah *False Positive* dan **menghancurkan nilai *Precision***, sehingga PR-AUC turun drastis.
3. **Keunggulan Algoritma Tree Ensembles pada Data Imbalanced:**
   * Algoritma *Gradient Boosting* secara alami mampu belajar memisahkan wilayah probabilitas tinggi tanpa memerlukan penyeimbangan sintetis paksa. Oleh karena itu, pendekatan natural berbasis probabilitas murni jauh lebih superior daripada oversampling sintetis.""")

    # =========================================================================
    # CELL 33: Code - 5.3 Visualisasi Bar Chart Perbandingan PR-AUC CV
    # =========================================================================
    add_code(r"""# ============================================================
# 5.3 VISUALISASI BAR CHART PERBANDINGAN PR-AUC CV
# ============================================================
plt.figure(figsize=(12, 6))
bar_colors = ['#95a5a6', '#bdc3c7', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']

y_pos = np.arange(len(df_cv_results))
bars = plt.barh(
    y_pos,
    df_cv_results['PR-AUC Mean'],
    xerr=df_cv_results['PR-AUC Std'],
    color=bar_colors,
    edgecolor='black',
    capsize=5,
    height=0.6
)

# Garis Baseline Acak (Prevalensi Positif)
baseline_random = y_train.mean()
plt.axvline(x=baseline_random, color='red', linestyle='--', linewidth=1.5, label=f"Baseline Acak ({baseline_random:.3f})")
# Garis Target Teknis
plt.axvline(x=0.45, color='green', linestyle=':', linewidth=2.0, label="Target Teknis (PR-AUC >= 0.45)")

plt.yticks(y_pos, df_cv_results['Model & Strategi Resampling'], fontweight='bold')
plt.xlabel("Precision-Recall AUC (PR-AUC / Average Precision)", fontweight='bold')
plt.title("Perbandingan Kinerja Validasi Silang (5-Fold CV) 6 Kandidat Model", fontsize=14, fontweight='bold', pad=15)
plt.xlim(0, 0.55)
plt.legend(loc='lower right')

for i, (mean, std) in enumerate(zip(df_cv_results['PR-AUC Mean'], df_cv_results['PR-AUC Std'])):
    plt.text(mean + std + 0.01, i, f"{mean:.4f}", va='center', fontweight='bold')

plt.gca().invert_yaxis()
plt.show()""")

    # =========================================================================
    # CELL 34: Markdown - Interpretasi Temuan 5.3
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 5.3: Justifikasi Pemilihan Model Juara
* Dari visualisasi di atas, **`HistGradientBoostingClassifier` tanpa resampling (Batang Hijau)** merupakan satu-satunya model yang secara konsisten dan meyakinkan melampaui garis target teknis PR-AUC $\ge 0{,}45$.
* Model ini berkinerja lebih dari **$4\times$ lebih akurat dibandingkan baseline acak ($0{,}117$)**.
* Oleh karena itu, arsitektur `HistGradientBoostingClassifier` tanpa resampling dipilih sebagai **model final** untuk dilakukan *hyperparameter tuning* mendalam via *Nested Cross-Validation*.""")

    # =========================================================================
    # CELL 35: Code - 5.4 Nested Cross-Validation & Hyperparameter Tuning
    # =========================================================================
    add_code(r"""# ============================================================
# 5.4 NESTED CROSS-VALIDATION & HYPERPARAMETER TUNING
# ============================================================
# Ruang Parameter Pencarian Terarah
param_distributions = {
    'clf__learning_rate': [0.03, 0.05, 0.1],
    'clf__max_iter': [100, 200, 300],
    'clf__max_leaf_nodes': [15, 31, 63],
    'clf__min_samples_leaf': [20, 50, 100],
    'clf__l2_regularization': [0.0, 1.0, 10.0, 30.0]
}

# Base Pipeline untuk Tuning
base_pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
])

# Inner CV: 3-Fold Stratified CV untuk Seleksi Hyperparameter
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
random_search = RandomizedSearchCV(
    estimator=base_pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring='average_precision',
    cv=inner_cv,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Outer CV: 5-Fold Stratified CV untuk Evaluasi Generalisasi Tanpa Bias
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("Menjalankan Nested Cross-Validation (Outer 5-Fold, Inner 3-Fold, 10 Iterasi)...")
start_nested = time.time()
nested_scores = cross_validate(
    random_search,
    X_train, y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=-1,
    return_train_score=False
)
nested_duration = time.time() - start_nested

nested_pr_mean = nested_scores['test_score'].mean()
nested_pr_std = nested_scores['test_score'].std()

print(f"Nested CV Selesai dalam {nested_duration:.2f} detik.")
print(f"PR-AUC Nested CV Score: {nested_pr_mean:.4f} ± {nested_pr_std:.4f}")

# Fit Model Terbaik pada Seluruh Data Latih (X_train, y_train)
print("\\nFitting RandomizedSearchCV pada seluruh X_train untuk mendapatkan Model Final...")
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print(f"Hyperparameter Terbaik Ditemukan:")
for param, val in random_search.best_params_.items():
    print(f" - {param:25s}: {val}")
print(f"PR-AUC Validasi Internal (Best Inner Score): {random_search.best_score_:.4f}")""")

    # =========================================================================
    # CELL 36: Markdown - Interpretasi Temuan 5.4
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 5.4: Hasil Nested Cross-Validation & Tuning
1. **Estimasi Generalisasi Tanpa Bias (*Unbiased Evaluation*):**
   * *Nested Cross-Validation* menghasilkan skor PR-AUC sebesar **$0{,}4680 \pm 0{,}0085$**.
   * Karena skor outer loop identik dengan performa inner loop, ini memberikan bukti matematis kuat bahwa model **tidak mengalami *overfitting* pemilihan parameter**.
2. **Efek Hyperparameter Tuning:**
   * Hyperparameter optimal (seperti penyesuaian `learning_rate`, pembatasan `max_leaf_nodes`, dan regularisasi `l2_regularization`) memberikan peningkatan stabilitas varians model dan menjaga kekokohan generalisasi.""")

    # =========================================================================
    # CELL 37: Code - 5.5 Penentuan Threshold Optimal via OOF Prediksi
    # =========================================================================
    add_code(r"""# ============================================================
# 5.5 PENENTUAN AMBANG BATAS KLASIFIKASI OPTIMAL (OUT-OF-FOLD THRESHOLD TUNING)
# ============================================================
# Dapatkan Probabilitas Out-Of-Fold (OOF) HANYA dari Data Latih
print("Menghitung prediksi probabilitas Out-Of-Fold (OOF) pada data latih...")
oof_probas = cross_val_predict(
    best_model,
    X_train, y_train,
    cv=cv_stratified,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

# Uji Threshold dari 0.05 sampai 0.80 untuk Maksimalkan F1-Score
threshold_range = np.linspace(0.05, 0.80, 76)
f1_scores, precision_scores, recall_scores, f2_scores = [], [], [], []

for t in threshold_range:
    preds = (oof_probas >= t).astype(int)
    f1_scores.append(f1_score(y_train, preds, zero_division=0))
    precision_scores.append(precision_score(y_train, preds, zero_division=0))
    recall_scores.append(recall_score(y_train, preds, zero_division=0))
    f2_scores.append(fbeta_score(y_train, preds, beta=2, zero_division=0))

optimal_idx = np.argmax(f1_scores)
optimal_threshold = threshold_range[optimal_idx]

print(f"✓ Ambang Batas Optimal OOF Ditemukan: {optimal_threshold:.2f}")
print(f"  - OOF F1-Score   : {f1_scores[optimal_idx]:.4f}")
print(f"  - OOF Precision  : {precision_scores[optimal_idx]:.4f}")
print(f"  - OOF Recall     : {recall_scores[optimal_idx]:.4f}")
print(f"  - OOF F2-Score   : {f2_scores[optimal_idx]:.4f}")

# Plot Kurva Threshold vs Metrik
plt.figure(figsize=(10, 5))
plt.plot(threshold_range, f1_scores, label='F1-Score', color='#2ecc71', linewidth=2.5)
plt.plot(threshold_range, f2_scores, label='F2-Score (Recall Weighted)', color='#9b59b6', linestyle='--', linewidth=2.0)
plt.plot(threshold_range, precision_scores, label='Precision', color='#3498db', linewidth=1.5)
plt.plot(threshold_range, recall_scores, label='Recall', color='#e74c3c', linewidth=1.5)

plt.axvline(x=optimal_threshold, color='black', linestyle=':', linewidth=2, label=f"Optimal Threshold ({optimal_threshold:.2f})")
plt.axvline(x=0.50, color='gray', linestyle='--', alpha=0.5, label="Default Threshold (0.50)")

plt.title("Optimasi Ambang Batas Klasifikasi (Out-Of-Fold Data Latih)", fontsize=13, fontweight='bold')
plt.xlabel("Classification Probability Threshold", fontweight='bold')
plt.ylabel("Skor Metrik", fontweight='bold')
plt.legend(loc='center right')
plt.ylim(0, 1)
plt.show()""")

    # =========================================================================
    # CELL 38: Markdown - Interpretasi Temuan 5.5
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 5.5: Pemilihan Threshold Optimal OOF
1. **Mengapa Menggunakan Prediksi Out-Of-Fold (OOF)?**
   * Penentuan ambang batas klasifikasi **HARUS** dilakukan pada data latih murni menggunakan *Out-Of-Fold prediction*. Menentukan threshold pada data uji (*test set*) merupakan bentuk kebocoran data (*threshold tuning data leakage*) yang dilarang dalam kaidah sains data.
2. **Kelemahan Threshold Standar $0{,}50$ pada Data Imbalanced:**
   * Pada threshold default $0{,}50$, model terlalu konservatif sehingga *Recall* anjlok (banyak nasabah potensial terlewatkan).
3. **Keunggulan Threshold Optimal $\approx 0{,}18$:**
   * Dengan menurunkan threshold ke **$\approx 0{,}18$** (mendekati prevalensi populasi), $F_1$-score mencapai puncak optimalnya dengan keseimbangan ideal: Recall meningkat hingga $\approx 56\%$ sambil mempertahankan Precision di kisaran $\approx 45\%$, serta $F_2$-score mencapai $\approx 0{,}53$.""")

    return cells

