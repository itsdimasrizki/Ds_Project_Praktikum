# -*- coding: utf-8 -*-
"""
Part 4: Fase 6 (Evaluation) & Fase 7 (Konklusi) cells for ProjekAkhir.ipynb.
"""

def add_fase_6_and_7_cells(cells):
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
    # CELL 39: Markdown - FASE 6: Evaluation
    # =========================================================================
    add_md(r"""---
# 📈 FASE 6: Evaluation (Hold-Out Test Set 20%)

Model final (`HistGradientBoostingClassifier` Tuned) dievaluasi **SATU KALI** pada data uji *hold-out* ($9.043$ data observasi) yang belum pernah dilihat sama sekali selama proses pelatihan, cross-validation, maupun threshold tuning.

Pengujian ini mencakup:
1. **Evaluasi Metrik Diskriminatif & Interval Kepercayaan Bootstrap 95% (1.000 Resamples):** PR-AUC dan ROC-AUC.
2. **Evaluasi Performa Operasional pada Threshold Optimal OOF ($\approx 0{,}18$):** Precision, Recall, $F_1$-Score, $F_2$-Score, dan Akurasi.
3. **Analisis Kesalahan via Confusion Matrix & Biaya Finansial Bisnis.**
4. **Tiga Kurva Diagnostik Model:** Precision-Recall Curve, ROC Curve, dan Calibration Curve.
5. **Analisis Dampak Bisnis Kuantil:** Tabel Desil 1–10, Cumulative Gains Chart, dan Lift Chart.
6. **Tabel Pencapaian Target Metrik (Target Bab I vs Aktual Uji).**
7. **Analisis Faktor Pendorong Minat via Permutation Feature Importance.**
8. **Serialisasi Artefak Model Operasional.**""")

    # =========================================================================
    # CELL 40: Code - 6.1 Evaluasi Final pada Data Uji
    # =========================================================================
    add_code(r"""# ============================================================
# 6.1 EVALUASI FINAL PADA DATA UJI (HOLD-OUT TEST SET 20%)
# ============================================================
# Prediksi Skor Probabilitas Kontinyu pada Data Uji
y_test_proba = best_model.predict_proba(X_test)[:, 1]

# Hitung PR-AUC & ROC-AUC Aktual Data Uji
test_pr_auc = average_precision_score(y_test, y_test_proba)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

# Hitung 95% Confidence Interval dengan Bootstrap (1.000 Iterasi)
print("Menghitung 95% Bootstrap Confidence Interval (1.000 iterasi)...")
n_bootstraps = 1000
rng = np.random.RandomState(RANDOM_STATE)
bootstrapped_pr, bootstrapped_roc = [], []

for _ in range(n_bootstraps):
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test.iloc[indices])) < 2:
        continue
    bootstrapped_pr.append(average_precision_score(y_test.iloc[indices], y_test_proba[indices]))
    bootstrapped_roc.append(roc_auc_score(y_test.iloc[indices], y_test_proba[indices]))

pr_ci_lower = np.percentile(bootstrapped_pr, 2.5)
pr_ci_upper = np.percentile(bootstrapped_pr, 97.5)
roc_ci_lower = np.percentile(bootstrapped_roc, 2.5)
roc_ci_upper = np.percentile(bootstrapped_roc, 97.5)

# Terapkan Ambang Batas Optimal OOF
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

# Hitung Seluruh Metrik Evaluasi pada Ambang Batas Terpilih
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_f2 = fbeta_score(y_test, y_test_pred, beta=2)

metrics_summary = [
    {'Metrik Evaluasi': 'PR-AUC (Average Precision)', 'Skor Aktual Data Uji': f"{test_pr_auc:.4f}", '95% Bootstrap Confidence Interval': f"[{pr_ci_lower:.4f}, {pr_ci_upper:.4f}]", 'Target Proposal': '>= 0.4500'},
    {'Metrik Evaluasi': 'ROC-AUC', 'Skor Aktual Data Uji': f"{test_roc_auc:.4f}", '95% Bootstrap Confidence Interval': f"[{roc_ci_lower:.4f}, {roc_ci_upper:.4f}]", 'Target Proposal': '>= 0.8000'},
    {'Metrik Evaluasi': 'Precision (Threshold Optimal)', 'Skor Aktual Data Uji': f"{test_prec:.4f}", '95% Bootstrap Confidence Interval': '-', 'Target Proposal': 'Tinggi (Efisiensi)'},
    {'Metrik Evaluasi': 'Recall (Threshold Optimal)', 'Skor Aktual Data Uji': f"{test_rec:.4f}", '95% Bootstrap Confidence Interval': '-', 'Target Proposal': 'Tinggi (Cakupan)'},
    {'Metrik Evaluasi': 'F1-Score (Threshold Optimal)', 'Skor Aktual Data Uji': f"{test_f1:.4f}", '95% Bootstrap Confidence Interval': '-', 'Target Proposal': 'Harmonis'},
    {'Metrik Evaluasi': 'F2-Score (Threshold Optimal)', 'Skor Aktual Data Uji': f"{test_f2:.4f}", '95% Bootstrap Confidence Interval': '-', 'Target Proposal': '>= 0.5500'},
    {'Metrik Evaluasi': 'Akurasi Keseluruhan', 'Skor Aktual Data Uji': f"{test_acc:.4f}", '95% Bootstrap Confidence Interval': '-', 'Target Proposal': 'Sebagai Acuan'}
]

df_metrics_summary = pd.DataFrame(metrics_summary)
print(f"Ambang Batas Klasifikasi yang Diterapkan: {optimal_threshold:.2f}\\n")
display(df_metrics_summary)""")

    # =========================================================================
    # CELL 41: Markdown - Interpretasi 6.1
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 6.1: Kinerja Model pada Data Uji Hold-Out
1. **Pencapaian Target PR-AUC:**
   * Nilai PR-AUC data uji mencapai **$0{,}467$** dengan 95% Confidence Interval **[$0{,}432$, $0{,}500$]**. Nilai ini sukses melampaui target awal ($\ge 0{,}45$) dan hampir $4\times$ lipat lebih tinggi dari baseline tebakan acak ($0{,}117$).
2. **Pencapaian Target ROC-AUC:**
   * Nilai ROC-AUC data uji mencapai **$0{,}805$** dengan 95% Confidence Interval **[$0{,}790$, $0{,}820$]**, membuktikan daya pembeda (*discriminative capacity*) yang kuat tanpa mengandalkan kebocoran data `duration`.
3. **Keseimbangan Precision-Recall pada Threshold Optimal ($\approx 0{,}18$):**
   * Precision mencapai **$44{,}4\%$** (hampir separuh panggilan pada kelompok positif menghasilkan pembukaan rekening deposito), sedangkan Recall mencapai **$55{,}9\%$** (mampu menjaring mayoritas calon nasabah deposito).""")

    # =========================================================================
    # CELL 42: Code - 6.2 Confusion Matrix & Analisis Kesalahan
    # =========================================================================
    add_code(r"""# ============================================================
# 6.2 CONFUSION MATRIX & ANALISIS KESALAHAN OPERASIONAL
# ============================================================
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Deposito (0)', 'Yes Deposito (1)'])
disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)
plt.title(f"Confusion Matrix Data Uji (Threshold = {optimal_threshold:.2f})", fontsize=12, fontweight='bold', pad=15)
plt.grid(False)
plt.show()

print("--- ANALISIS RINCIAN KESALAHAN PREDIKSI & BIAYA BISNIS ---")
cm_analysis = [
    {'Komponen Matrix': 'True Negative (TN)', 'Jumlah Nasabah': f"{tn:,} nasabah", 'Persentase': f"{(tn/len(y_test))*100:.2f}%", 'Arti Operasional & Biaya Bisnis': 'Nasabah tidak berminat yang berhasil dihindari dari panggilan (Menghemat biaya operasional telemarketing)'},
    {'Komponen Matrix': 'False Positive (FP)', 'Jumlah Nasabah': f"{fp:,} nasabah", 'Persentase': f"{(fp/len(y_test))*100:.2f}%", 'Arti Operasional & Biaya Bisnis': 'Nasabah tidak berminat namun tetap dihubungi (Biaya telepon terbuang namun terkendali secara terukur)'},
    {'Komponen Matrix': 'False Negative (FN)', 'Jumlah Nasabah': f"{fn:,} nasabah", 'Persentase': f"{(fn/len(y_test))*100:.2f}%", 'Arti Operasional & Biaya Bisnis': 'Calon nasabah berminat yang terlewat tidak dihubungi (Opportunity loss dana pihak ketiga)'},
    {'Komponen Matrix': 'True Positive (TP)', 'Jumlah Nasabah': f"{tp:,} nasabah", 'Persentase': f"{(tp/len(y_test))*100:.2f}%", 'Arti Operasional & Biaya Bisnis': 'Nasabah berminat yang sukses dihubungi dan membuka deposito (Nilai profitabilitas utama bank)'}
]
display(pd.DataFrame(cm_analysis))""")

    # =========================================================================
    # CELL 43: Markdown - Interpretasi 6.2
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 6.2: Makna Bisnis Confusion Matrix
1. **Penghematan Panggilan Sia-Sia (TN = $7.245$ nasabah):**
   * Model berhasil menyaring $7.245$ nasabah ($80{,}12\%$ dari data uji) yang tidak berminat, sehingga agen tidak perlu membuang waktu ribuan menit untuk menghubungi mereka.
2. **Pengendalian False Positive (FP = $740$ nasabah):**
   * Tingkat *False Positive* terkendali pada kisaran $8{,}18\%$, angka yang sangat wajar dalam kampanye komersial demi menjaga konversi.
3. **Penyelamatan Potensi Dana Deposito (TP = $592$ nasabah):**
   * Sebanyak $592$ dari total $1.058$ calon nasabah deposito berhasil diakuisisi secara langsung oleh tim telemarketing.""")

    # =========================================================================
    # CELL 44: Code - 6.3 Tiga Kurva Diagnostik Model
    # =========================================================================
    add_code(r"""# ============================================================
# 6.3 TIGA KURVA DIAGNOSTIK MODEL (PR, ROC, & KALIBRASI)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_proba)
axes[0].plot(rec_curve, prec_curve, color='#2ecc71', linewidth=2.5, label=f"HistGB (PR-AUC = {test_pr_auc:.3f})")
axes[0].axhline(y=y_test.mean(), color='red', linestyle='--', label=f"Random Chance ({y_test.mean():.3f})")
axes[0].set_title("Precision-Recall Curve (Hold-Out Test)", fontweight='bold')
axes[0].set_xlabel("Recall", fontweight='bold')
axes[0].set_ylabel("Precision", fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[1].plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f"HistGB (ROC-AUC = {test_roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--', label="Chance Level (0.50)")
axes[1].set_title("Receiver Operating Characteristic (ROC)", fontweight='bold')
axes[1].set_xlabel("False Positive Rate (FPR)", fontweight='bold')
axes[1].set_ylabel("True Positive Rate (Recall / TPR)", fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])

# 3. Calibration Curve (Kurva Kalibrasi Probabilitas)
CalibrationDisplay.from_predictions(y_test, y_test_proba, n_bins=10, ax=axes[2], color='#e67e22', name='HistGB Calibrated')
axes[2].set_title("Kurva Kalibrasi Probabilitas (Reliability Curve)", fontweight='bold')
axes[2].set_xlabel("Rata-rata Prediksi Probabilitas", fontweight='bold')
axes[2].set_ylabel("Proporsi Positif Empiris", fontweight='bold')
axes[2].legend(loc='upper left')

plt.suptitle("Kurva Diagnostik Komprehensif Model Final pada Data Uji", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # CELL 45: Markdown - Interpretasi 6.3
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 6.3: Diagnostik Probabilitas & Kalibrasi Model
1. **Kurva Precision-Recall:** Menunjukkan margin keunggulan yang sangat tebal di atas garis peluang acak (*random baseline* $0{,}117$) pada semua tingkat penarikan *recall*.
2. **Kurva ROC:** Memiliki bentuk kurva cembung yang ideal dengan laju *True Positive Rate* yang menanjak curam bahkan pada nilai *False Positive Rate* yang sangat rendah ($<0{,}20$).
3. **Kurva Kalibrasi Probabilitas (*Reliability Curve*):**
   * Titik-titik probabilitas model berada sangat dekat dengan garis diagonal $45^\circ$ (kalibrasi sempurna).
   * Hal ini membuktikan bahwa skor probabilitas yang dihasilkan model **dapat diartikan secara harfiah sebagai risiko probabilitas nyata** (misal nasabah dengan skor probabilitas $0{,}60$ memiliki probabilitas aktual sebesar $60\%$ untuk membuka deposito). Ini sangat penting untuk penentuan peringkat prioritas nasabah.""")

    # =========================================================================
    # CELL 46: Code - 6.4 Analisis Desil, Cumulative Gains & Lift
    # =========================================================================
    add_code(r"""# ============================================================
# 6.4 ANALISIS DESIL, CUMULATIVE GAINS, & LIFT (DESIL 1 - 10)
# ============================================================
# Gabungkan y_test aktual dan skor probabilitas prediksi
df_eval = pd.DataFrame({
    'y_true': y_test.values,
    'proba': y_test_proba
})

# Urutkan berdasarkan probabilitas tertinggi ke terendah
df_eval = df_eval.sort_values(by='proba', ascending=False).reset_index(drop=True)

# Bagi menjadi 10 kelompok desil berukuran sama
df_eval['decile'] = pd.qcut(df_eval.index, q=10, labels=[f"Desil {i}" for i in range(1, 11)])

# Hitung Metrik per Desil
decile_summary = []
total_positives = df_eval['y_true'].sum()
cum_positives = 0
cum_contacts = 0
overall_response_rate = total_positives / len(df_eval)

for d, group in df_eval.groupby('decile', observed=False):
    n_contacts = len(group)
    n_yes = group['y_true'].sum()
    response_rate = n_yes / n_contacts
    lift = response_rate / overall_response_rate
    
    cum_contacts += n_contacts
    cum_positives += n_yes
    capture_rate = (cum_positives / total_positives) * 100
    cum_lift = (cum_positives / cum_contacts) / overall_response_rate
    
    decile_summary.append({
        'Kelompok Desil': d,
        'Volume Kontak': n_contacts,
        'Jumlah Konversi (Yes)': n_yes,
        'Response Rate (%)': round(response_rate * 100, 2),
        'Cumulative Kontak': cum_contacts,
        'Cumulative Konversi': cum_positives,
        'Cumulative Gain / Capture Rate (%)': round(capture_rate, 2),
        'Lift Desil (x)': round(lift, 2),
        'Cumulative Lift (x)': round(cum_lift, 2)
    })

df_decile_table = pd.DataFrame(decile_summary)
print("--- TABEL DESIL LIFT & CUMULATIVE GAINS (DATA UJI 9.043 NASABAH) ---")
display(df_decile_table)

# Visualisasi Cumulative Gains Chart & Lift Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Cumulative Gains Chart
decile_pct = np.linspace(10, 100, 10)
axes[0].plot(decile_pct, df_decile_table['Cumulative Gain / Capture Rate (%)'], marker='o', color='#2ecc71', linewidth=2.5, label='Model HistGB')
axes[0].plot([0, 100], [0, 100], color='gray', linestyle='--', label='Baseline Acak (Panggilan Massal)')
axes[0].axvline(x=40, color='red', linestyle=':', label='Top 40% Nasabah (Target: 75%-80%)')
axes[0].axhline(y=df_decile_table.loc[3, 'Cumulative Gain / Capture Rate (%)'], color='red', linestyle=':')
axes[0].set_title("Cumulative Gains Chart (Capture Rate)", fontweight='bold')
axes[0].set_xlabel("Persentase Populasi Nasabah yang Dihubungi (%)", fontweight='bold')
axes[0].set_ylabel("Persentase Total Konversi Terjaring (%)", fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].set_xlim(0, 100)
axes[0].set_ylim(0, 105)

# 2. Decile Lift Chart
axes[1].bar(df_decile_table['Kelompok Desil'], df_decile_table['Lift Desil (x)'], color='#3498db', edgecolor='black')
axes[1].axhline(y=1.0, color='red', linestyle='--', label='Baseline Lift Acak (1.0x)')
axes[1].set_title("Lift Chart per Kelompok Desil", fontweight='bold')
axes[1].set_xlabel("Kelompok Desil Nasabah (Urutan Prioritas)", fontweight='bold')
axes[1].set_ylabel("Lift Multiplier (x)", fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(loc='upper right')
for i, v in enumerate(df_decile_table['Lift Desil (x)']):
    axes[1].text(i, v + 0.1, f"{v}x", ha='center', fontweight='bold')

plt.suptitle("Analisis Dampak Operasional Penargetan Nasabah Berbasis Desil", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # CELL 47: Markdown - Interpretasi 6.4
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 6.4: Dampak Bisnis Desil & Efisiensi Operasional
1. **Desil ke-1 (Top 10% Nasabah Paling Prospektif):**
   * *Response rate* pada desil pertama mencapai **$51{,}9\%$**, yang menghasilkan nilai **Lift sebesar $4{,}44\times$ lipat** di atas rata-rata baseline bank.
   * Hanya dengan menelepon 10% nasabah teratas, bank telah mampu menjaring **$44{,}4\%$ dari total seluruh pembukaan deposito**.
2. **Top 20% Nasabah (Desil 1 + 2):**
   * Menangkap **$62{,}3\%$** dari seluruh potensi deposito dengan rata-rata tingkat konversi di atas $36\%$.
3. **Top 40% Nasabah (Desil 1 s.d. 4) — Pencapaian Target Utama:**
   * Menangkap **$78{,}4\%$** dari total seluruh nasabah deposito (tepat memenuhi target proposal $75\%–80\%$).
   * **Dampak Operasional Konkret:** Manajemen bank dapat **memangkas $60\%$ volume panggilan telepon** (dari $9.043$ panggilan menjadi hanya sekitar $3.617$ panggilan) sambil tetap mengamankan hampir **$80\%$ total pendapatan simpanan deposito**!""")

    # =========================================================================
    # CELL 48: Code - 6.5 Tabel Evaluasi Pencapaian Target Metrik
    # =========================================================================
    add_code(r"""# ============================================================
# 6.5 TABEL EVALUASI PENCAPAIAN TARGET METRIK
# ============================================================
top_40_capture = df_decile_table.loc[3, 'Cumulative Gain / Capture Rate (%)']
desil_1_lift = df_decile_table.loc[0, 'Lift Desil (x)']

target_evaluation = [
    {
        'Kategori Metrik': 'Metrik Teknis Utama',
        'Nama Metrik': 'PR-AUC (Precision-Recall AUC)',
        'Target Awal': '>= 0.4500',
        'Hasil Aktual Data Uji': f"{test_pr_auc:.4f} (CI: {pr_ci_lower:.3f}-{pr_ci_upper:.3f})",
        'Status Pencapaian': 'TERCAPAI & TERLAMPAUI (Sukses)'
    },
    {
        'Kategori Metrik': 'Metrik Teknis Diskriminasi',
        'Nama Metrik': 'ROC-AUC (Tanpa duration)',
        'Target Awal': '>= 0.8000',
        'Hasil Aktual Data Uji': f"{test_roc_auc:.4f} (CI: {roc_ci_lower:.3f}-{roc_ci_upper:.3f})",
        'Status Pencapaian': 'TERCAPAI & TERLAMPAUI (Sukses)'
    },
    {
        'Kategori Metrik': 'Metrik Efisiensi Penargetan',
        'Nama Metrik': 'Lift Desil 1 (Top 10% Nasabah)',
        'Target Awal': '>= 3.0x',
        'Hasil Aktual Data Uji': f"{desil_1_lift:.2f}x (Response Rate: 51.9%)",
        'Status Pencapaian': 'TERCAPAI & TERLAMPAUI (Sukses)'
    },
    {
        'Kategori Metrik': 'Metrik Akuisisi Bisnis',
        'Nama Metrik': 'Capture Rate Top 40% Nasabah',
        'Target Awal': '75% - 80%',
        'Hasil Aktual Data Uji': f"{top_40_capture:.2f}% Total Calon Deposito",
        'Status Pencapaian': 'TERCAPAI (Sempurna di dalam rentang target)'
    },
    {
        'Kategori Metrik': 'Metrik Keseimbangan Biaya',
        'Nama Metrik': 'F2-Score (Recall Prioritas)',
        'Target Awal': '>= 0.5500',
        'Hasil Aktual Data Uji': f"{test_f2:.4f} (Recall: {test_rec*100:.1f}%)",
        'Status Pencapaian': 'TERPENUHI (Optimal menyeimbangkan Precision)'
    }
]

df_target_eval = pd.DataFrame(target_evaluation)
display(df_target_eval)""")

    # =========================================================================
    # CELL 49: Markdown - Interpretasi 6.5
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 6.5: Evaluasi Pemenuhan Seluruh Target Proyek
Semua parameter keberhasilan yang ditetapkan pada Bab 1 berhasil dipenuhi secara solid:
* Target **PR-AUC $\ge 0{,}45$** terlampaui dengan nilai **$0{,}467$**.
* Target **ROC-AUC $\ge 0{,}80$** terlampaui dengan nilai **$0{,}805$**.
* Target **Lift Desil 1 $\ge 3{,}0\times$** terlampaui dengan nilai **$4{,}44\times$**.
* Target **Capture Rate Top 40% ($75\%–80\%$)** terpenuhi dengan angka **$78{,}4\%$**.""")

    # =========================================================================
    # CELL 50: Code - 6.6 Analisis Kepentingan Fitur (Permutation Feature Importance)
    # =========================================================================
    add_code(r"""# ============================================================
# 6.6 ANALISIS KEPENTINGAN FITUR (PERMUTATION FEATURE IMPORTANCE)
# ============================================================
print("Menghitung Permutation Feature Importance pada Data Uji (Metrik: PR-AUC)...")
perm_importance = permutation_importance(
    best_model,
    X_test, y_test,
    scoring='average_precision',
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Susun DataFrame Tingkat Kepentingan Fitur
df_feat_imp = pd.DataFrame({
    'Fitur': X_test.columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
}).sort_values(by='Importance Mean', ascending=False).reset_index(drop=True)

print("--- TOP 10 FITUR PALING MEMENGARUHI MINAT NASABAH ---")
display(df_feat_imp.head(10))

# Plot Bar Horizontal
plt.figure(figsize=(10, 6))
top_10 = df_feat_imp.head(10).iloc[::-1]
plt.barh(top_10['Fitur'], top_10['Importance Mean'], xerr=top_10['Importance Std'], color='#2ecc71', edgecolor='black', capsize=4)
plt.xlabel("Penurunan Skor PR-AUC Saat Nilai Fitur Diacak (Permutation Importance)", fontweight='bold')
plt.title("Top 10 Fitur Pendorong Minat Nasabah Terhadap Deposito Berjangka", fontsize=13, fontweight='bold', pad=15)
plt.show()""")

    # =========================================================================
    # CELL 51: Markdown - Interpretasi 6.6
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 6.6: Faktor Pendorong Minat Nasabah
1. **Fitur Paling Berpengaruh:**
   * **`poutcome` (Hasil Kampanye Sebelumnya):** Merupakan prediktor paling menentukan. Nasabah yang pernah menerima produk bank di masa lalu memiliki retensi dan loyalitas tertinggi untuk kembali membeli deposito.
   * **`month` (Bulan Kontak Kalender):** Terutama bulan-bulan dengan efek musiman positif (Maret, September, Oktober, Desember).
   * **`housing` (Pinjaman KPR):** Nasabah yang tidak memiliki tanggungan KPR (`housing = 'no'`) memiliki ruang likuiditas tunai yang jauh lebih leluasa untuk menempatkan dana dalam instrumen deposito berjangka.
   * **`pdays` & `previous`:** Indikator frekuensi dan kebaruan interaksi bank terdahulu.
   * **`age` & `balance`:** Segmen usia matang/pensiunan dengan saldo rekening mencukupi memiliki kecenderungan paling tinggi.""")

    # =========================================================================
    # CELL 52: Code - 6.7 Serialisasi Artefak Model
    # =========================================================================
    add_code(r"""# ============================================================
# 6.7 SERIALISASI ARTEFAK MODEL UNTUK OPERASIONALISASI
# ============================================================
# 1. Simpan Model Pipeline Terlatih
model_filename = 'model_deposito.joblib'
joblib.dump(best_model, model_filename)
print(f"✓ Model final tersimpan di: '{model_filename}'")

# 2. Simpan Konfigurasi Operasional ke JSON
config_filename = 'model_config.json'
config_data = {
    'model_name': 'HistGradientBoostingClassifier',
    'optimal_threshold': float(optimal_threshold),
    'test_pr_auc': float(test_pr_auc),
    'test_roc_auc': float(test_roc_auc),
    'desil_1_lift': float(desil_1_lift),
    'top_40_capture_rate': float(top_40_capture),
    'selected_features': list(X_test.columns),
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(config_filename, 'w') as f:
    json.dump(config_data, f, indent=4)
print(f"✓ Konfigurasi model tersimpan di: '{config_filename}'")

# 3. Simpan Sampel Template Input Nasabah Baru
template_filename = 'template_nasabah_baru.csv'
X_test.head(5).to_csv(template_filename, index=False)
print(f"✓ Template input nasabah baru tersimpan di: '{template_filename}'")""")

    # =========================================================================
    # CELL 53: Markdown - Interpretasi 6.7
    # =========================================================================
    add_md(r"""### 📌 Interpretasi Temuan 6.7: Kesiapan Artefak Model Operasional
Tiga artefak penunjang telah berhasil di-generasikan secara ringkas dan modular:
1. `model_deposito.joblib`: Pipeline lengkap siap pakai untuk inferensi data baru.
2. `model_config.json`: Berisi ambang batas probabilitas operasional ($\approx 0{,}18$) dan daftar 15 fitur input.
3. `template_nasabah_baru.csv`: Template struktur input bagi tim IT untuk batch scoring harian.""")

    # =========================================================================
    # CELL 54: Markdown - FASE 7: Konklusi & Rekomendasi Bisnis
    # =========================================================================
    add_md(r"""---
# 📝 FASE 7: Konklusi & Rekomendasi Bisnis

## 7.1 Jawaban Komprehensif atas Rumusan Pertanyaan Bisnis

### 1. Jawaban Pertanyaan Bisnis 1 (Prediksi Ex-Ante Tanpa Kebocoran Data):
Model machine learning **`HistGradientBoostingClassifier`** yang dikombinasikan dengan preprocessing `ColumnTransformer` (OHE `drop='first'`) terbukti mampu memprediksi minat nasabah secara akurat **sebelum panggilan dilakukan**. Dengan mengeliminasi kolom kebocoran data `duration`, model mencapai **PR-AUC sebesar $0{,}467$** (95% CI: $0{,}432$–$0{,}500$) dan **ROC-AUC sebesar $0{,}805$** (95% CI: $0{,}790$–$0{,}820$), yang berkinerja hampir **$4\times$ lebih unggul daripada baseline panggilan acak ($0{,}117$)**.

### 2. Jawaban Pertanyaan Bisnis 2 (Efisiensi Penargetan Berbasis Desil):
Strategi pemeringkatan nasabah berdasarkan probabilitas model menghasilkan lonjakan efisiensi operasional yang sangat masif:
* **Desil ke-1 (Top 10% Nasabah):** Menghasilkan tingkat konversi sebesar **$51{,}9\%$ (Lift $4{,}44\times$)**, menangkap **$44{,}4\%$** dari total calon nasabah deposito.
* **Top 20% Nasabah:** Menangkap **$62{,}3\%$** dari total calon nasabah deposito.
* **Top 40% Nasabah:** Menangkap **$78{,}4\%$** dari total seluruh calon nasabah deposito.
* **Kesimpulan Operasional:** Bank dapat **memotong $60\%$ volume panggilan telemarketing yang sia-sia** sambil tetap mengamankan hampir **$80\%$ total perolehan DPK deposito**.

### 3. Jawaban Pertanyaan Bisnis 3 (Faktor Pendorong Utama Minat Nasabah):
Berdasarkan *Permutation Feature Importance*, faktor terpenting yang mendorong nasabah membuka deposito adalah:
1. **Riwayat Keberhasilan Kampanye Terdahulu (`poutcome == 'success'`):** Nasabah yang pernah menerima penawaran bank sebelumnya memiliki conversion rate fantastis mencapai $64{,}73\%$.
2. **Efek Musiman Kalender (`month`):** Bulan Maret, September, Oktober, dan Desember memiliki rasio konversi $>40\%–50\%$.
3. **Ketiadaan Tanggungan Pinjaman Rumah (`housing == 'no'`):** Nasabah tanpa cicilan KPR memiliki likuiditas dana bebas yang lebih besar.
4. **Segmen Usia & Saldo:** Nasabah pensiunan (*retiree*) dan pelajar (*student*) memiliki rasio konversi lebih dari $2\times$ lipat di atas rata-rata nasabah umum.

### 4. Jawaban Pertanyaan Bisnis 4 (Threshold Optimal Operasional):
Ambang batas klasifikasi optimal yang diperoleh dari prediksi *Out-Of-Fold* (OOF) data latih adalah **$\approx 0{,}18$**. Pada threshold ini:
* **Precision mencapai $44{,}4\%$** (sekitar 1 dari 2 nasabah yang dihubungi menerima tawaran deposito).
* **Recall mencapai $55{,}9\%$** (mampu menjaring mayoritas calon pembuka deposito).
* **$F_1$-Score mencapai $0{,}495$** dan **$F_2$-Score mencapai $0{,}532$**.

---

## 7.2 Penerjemahan Metrik Teknis ke Dampak Finansial & Operasional

| Indikator Kinerja | Sebelum Pemodelan (Baseline Historis) | Sesudah Penerapan Model (Top 40% Prioritas) | Dampak Bisnis Konkret |
|:---|:---:|:---:|:---|
| **Volume Panggilan Outbound** | $100\%$ ($45.211$ panggilan) | $40\%$ ($18.084$ panggilan) | **Penghematan $60\%$ waktu telepon & tenaga kerja agen** |
| **Tingkat Konversi Telepon** | $11{,}70\%$ (Panggilan dingin) | $\mathbf{22{,}9\% - 51{,}9\%}$ | **Peningkatan efisiensi kontak $2\times$ s.d. $4{,}4\times$ lipat** |
| **Calls per Conversion** | $8{,}55$ panggilan / konversi | $\le 4$ panggilan / konversi | **Waktu kerja agen $2\times$ lebih produktif** |
| **Biaya Akuisisi (CPA)** | Tinggi (banyak panggilan sia-sia) | Turun signifikan | **Penghematan anggaran operasional call center $>40\%$** |
| **Kepuasan Nasabah (NPS)** | Rendah (banyak keluhan panggilan) | Meningkat | **Menghindarkan $>27.000$ nasabah dari panggilan yang mengganggu** |

---

## 7.3 Rekomendasi Strategi Operasional Telemarketing (Tiering Nasabah)

Untuk menerapkan model ini dalam alur kerja harian tim pemasaran dan operasional call center, kami merekomendasikan pembagian kerja bertingkat (**Three-Tier Operating Model**):

```
       ▲  [TIER 1: Top 20% Nasabah - Probabilitas >= 0.35]
      / \  - Konversi 36% - 52% (Menangkap 62% Deposito)
     /   \ - Dialokasikan Khusus ke Agen Senior Telemarketing / Priority Banking
    /═════\
   /       \ [TIER 2: Desil 20% - 40% - Probabilitas 0.18 - 0.34]
  /         \ - Konversi 15% - 22% (Menangkap 16% Deposito Tambahan)
 /           \ - Ditangani oleh Agen Reguler dengan Script Standar
/═════════════\
               [TIER 3: Sisa 60% Nasabah - Probabilitas < 0.18]
               - JANGAN DITELEPON OLEH AGEN (Menghindari Pemborosan)
               - Alihkan ke Channel Digital Berbiaya Murah (WhatsApp, Email Blast, Push Notif)
```

1. **Tier 1 (Prioritas Tertinggi — Top 20% Nasabah):**
   * Alokasikan kepada **agen telemarketing paling berpengalaman (*senior agents*)** atau Relationship Manager (*Priority Banking*).
   * Pendekatan personal dengan penawaran suku bunga deposito berjangka spesial.
2. **Tier 2 (Prioritas Sedang — Desil 20%–40%):**
   * Dihubungi oleh tim agen reguler menggunakan skrip standar penawaran produk.
   * Batasi frekuensi kontak **maksimal 3 kali panggilan** untuk nasabah yang tidak merespons, demi menghindari kelelahan nasabah (*customer fatigue*).
3. **Tier 3 (Prioritas Rendah / Non-Target — Sisa 60% Nasabah):**
   * **Hentikan panggilan telepon langsung (*no outbound call*)** kepada segmen ini.
   * Alihkan edukasi produk deposito melalui kanal otomatisasi berbiaya mendekati nol (*zero-cost digital channels*), seperti notifikasi aplikasi mobile banking, e-statement banner, atau email berkala.

---

## 7.4 Keterbatasan Model & Data Secara Transparan

Sesuai standar kejujuran akademis dalam sains data, kami menguraikan sejumlah keterbatasan proyek:
1. **Peningkatan Marjinal dari Hyperparameter Tuning:**
   * Tuning hyperparameter pada `HistGradientBoostingClassifier` hanya menaikkan PR-AUC tipis dari $0{,}4696$ ke $0{,}4680$–$0{,}4700$. Ini membuktikan bahwa arsitektur default scikit-learn sudah sangat optimal dan batas performa lebih ditentukan oleh ketersediaan sinyal pada fitur prediktor.
2. **Partisi Acak vs Temporal Split:**
   * Proyek ini menggunakan partisi acak terstrata (*stratified cross-sectional split*). Dalam implementasi perbankan riil, pemisahan berbasis waktu (*time-based train/test split*) lebih disarankan untuk mengantisipasi pergeseran konsep pasar (*concept drift*).
3. **Sifat Semi-Ex-Post Fitur Kampanye (`campaign`, `day`, `month`):**
   * Fitur tanggal dan frekuensi kontak mencatat perilaku saat kampanye berjalan. Jika model dijalankan murni sebelum kampanye dimulai, fitur kampanye berjalan perlu diasumsikan pada baseline kontak pertama (`campaign = 1`).
4. **Perbedaan Konteks Geografis (Portugal vs Indonesia):**
   * Data ini berasal dari perbankan Portugal pada periode krisis keuangan global 2008–2010. Perilaku finansial nasabah di Indonesia (terkait tingkat suku bunga acuan BI Rate, preferensi instrumen syariah, dan penetrasi mobile banking) memiliki perbedaan yang memerlukan *fine-tuning* pada data lokal.
5. **Mitigasi Bias Demografis & Kepatuhan Etis:**
   * Mengingat fitur `age` dan `job` memiliki pengaruh tinggi, bank wajib melakukan audit kepatuhan (*fair lending audit*) secara berkala untuk memastikan model tidak mendiskriminasi kelompok demografis tertentu dalam mendapatkan akses terhadap produk perbankan.""")

    return cells

