# -*- coding: utf-8 -*-
"""
Notebook cell definitions and builder for ProjekAkhir.ipynb.
"""

def get_notebook_cells():
    cells = []
    
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
    # CELL 0: Header & Identitas Kelompok (Markdown)
    # =========================================================================
    add_md(r"""# 🏦 Projek Akhir Praktikum Data Science: Prediksi Minat Nasabah Terhadap Deposito Berjangka pada Kampanye Telemarketing Bank

**Program Studi Informatika — Fakultas Teknik Industri — Universitas Pembangunan Nasional "Veteran" Yogyakarta**  
**Tahun Akademik 2026/2027**

---

### 👥 Identitas Anggota Kelompok (Kelompok 44):
1. **Bagus Samudra** (NIM: `123240102`)
2. **Filipus Satrio Dewaki T.S** (NIM: `123240133`)
3. **Zaky Mubarok Aljauhari** (NIM: `123240107`)
4. **Dimas Rizki Ardiansyah** (NIM: `123240123`)

---

### 📖 Kerangka Kerja Proyek (CRISP-DM Workflow)
Notebook ini mengimplementasikan alur standar industri **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)** secara komprehensif:
1. **Fase 1: Business Understanding** — Latar belakang masalah kampanye telemarketing perbankan, perumusan 4 pertanyaan bisnis, pemetaan 6 pemangku kepentingan (*stakeholders*), metode klasifikasi biner probabilistik, serta target metrik teknis dan bisnis.
2. **Fase 2: Data Gathering & Understanding** — Audit asal-usul data (*provenance* & lisensi Kaggle/UCI), alasan pemilihan dataset, kamus data 17 atribut lengkap (menandai kolom `duration` sebagai kebocoran data *ex-post*), dan penilaian *Data Readiness Level* (DRL).
3. **Fase 3: Exploratory Data Analysis (EDA)** — Audit struktur data, kualitas data (pemeriksaan missing values implisit `'unknown'`, kode `pdays = -1`, dan duplikasi), analisis ketidakseimbangan kelas target ($88{,}30\% : 11{,}70\%$), deteksi *outlier* metode IQR, matriks korelasi Pearson numerik, analisis bivariat kategorikal vs target, serta pola temporal bulanan.
4. **Fase 4: Data Cleaning, Preprocessing & Feature Selection** — Pembersihan data, seleksi fitur eksplisit (eliminasi `duration` demi mencegah *data leakage*, mempertahankan 15 prediktor asli), partisi *stratified hold-out* 80:20 anti-leakage, perancangan pipeline `ColumnTransformer` (OHE `drop='first'` menghasilkan tepat **41 fitur**), serta tabel audit kondisi sebelum vs sesudah.
5. **Fase 5: Modeling & Hyperparameter Tuning** — Filosofi *anti-leakage resampling* di dalam `imblearn.pipeline.Pipeline`, evaluasi komparatif 6 kandidat model via 5-fold *Stratified Cross-Validation*, analisis teoritis kegagalan SMOTE pada data berkategori jarang (*sparse*), optimasi *Nested Cross-Validation* untuk tuning `HistGradientBoostingClassifier`, dan penentuan *threshold* klasifikasi optimal berbasis *Out-Of-Fold* (OOF) data latih ($\approx 0{,}18$).
6. **Fase 6: Evaluation** — Pengujian final pada *hold-out test set* 20% ($9.043$ data): evaluasi PR-AUC dan ROC-AUC dengan 95% Bootstrap Confidence Interval, Confusion Matrix & interpretasi biaya bisnis (TN, FP, FN, TP), kurva PR, ROC, dan Kalibrasi probabilitas, evaluasi dampak bisnis via Tabel Desil Lift & Cumulative Gains (10 desil), verifikasi pemenuhan target metrik, analisis faktor penentu via *Permutation Feature Importance*, dan penyimpanan artefak model ringkas.
7. **Fase 7: Konklusi & Rekomendasi Bisnis** — Menjawab 4 pertanyaan bisnis secara langsung berbasis bukti empiris, menerjemahkan metrik evaluasi ke bahasa operasional manajemen, merumuskan strategi penargetan nasabah bertingkat (*Tier 1–3*), serta menguraikan keterbatasan proyek secara jujur dan transparan.""")

    # =========================================================================
    # CELL 1: FASE 1: Business Understanding (Markdown)
    # =========================================================================
    add_md(r"""---
# 📋 FASE 1: Business Understanding

## 1.1 Latar Belakang Masalah
Dalam industri perbankan komersial, penghimpunan **Dana Pihak Ketiga (DPK)** melalui produk deposito berjangka (*term deposit*) merupakan salah satu pilar stabilitas likuiditas dan profitabilitas utama bank. Salah satu instrumen pemasaran langsung yang paling umum digunakan adalah **kampanye telemarketing outbound** (agen menelepon nasabah secara langsung untuk menawarkan pembukaan rekening deposito).

Namun, pelaksanaan kampanye telemarketing secara massal (*untargeted/cold-calling*) menghadapi 3 kendala operasional yang kritis:
1. **Tingkat Konversi Sangat Rendah:** Rata-rata tingkat keberhasilan historis hanya berkisar **$11{,}70\%$** (hanya $5.289$ dari $45.211$ nasabah yang bersedia membuka deposito). Artinya, hampir **$88{,}30\%$ panggilan berakhir dengan penolakan**.
2. **Inefisiensi Biaya Operasional & Tenaga Kerja:** Agen telemarketing menghabiskan ratusan jam kerja untuk menghubungi nasabah yang tidak memiliki profil atau minat terhadap deposito. Hal ini menyebabkan membengkaknya biaya akuisisi (*Cost per Acquisition / CPA*) dan menurunkan produktivitas agen.
3. **Kelelahan & Resistensi Nasabah (*Customer Fatigue & Brand Irritation*):** Panggilan telepon yang tidak tepat sasaran dan dilakukan secara berulang menimbulkan ketidaknyamanan bagi nasabah, merusak reputasi bank, serta meningkatkan risiko penutupan rekening aktif.

Oleh karena itu, bank memerlukan sistem pemeringkatan probabilitas (*propensity scoring*) yang mampu memperkirakan minat nasabah secara presisi **sebelum panggilan telepon dilakukan (*ex-ante*)**, sehingga agen hanya memprioritaskan nasabah yang berpotensi tinggi untuk membuka deposito.

---

## 1.2 Rumusan Pertanyaan Bisnis
Proyek ini dirancang untuk menjawab 4 pertanyaan bisnis strategis berikut:
1. **Pertanyaan Bisnis 1 (Prediksi Ex-Ante):** Bagaimana cara memprediksi secara akurat kecenderungan (*propensity score*) seorang nasabah untuk membuka deposito berjangka sebelum panggilan telepon dilakukan, tanpa memanfaatkan data interaksi yang belum terjadi?
2. **Pertanyaan Bisnis 2 (Efisiensi Penargetan):** Bagaimana merancang strategi penargetan nasabah berbasis kuantil/desil probabilitas agar tim telemarketing dapat menangkap mayoritas calon nasabah deposito dengan hanya menghubungi sebagian kecil nasabah terbaik?
3. **Pertanyaan Bisnis 3 (Faktor Pendorong Minat):** Fitur atau karakteristik nasabah apa saja yang paling dominan mendorong keputusan pembukaan deposito berjangka, sehingga dapat dijadikan acuan strategi komunikasi tim pemasaran?
4. **Pertanyaan Bisnis 4 (Threshold Optimal):** Berapa nilai ambang batas (*classification threshold*) optimal yang menyeimbangkan antara tingkat penangkapan nasabah potensial (*recall*) dan efisiensi waktu panggilan agen (*precision*)?

---

## 1.3 Pemangku Kepentingan (Stakeholders) & Kebutuhan
Proyek ini mengidentifikasi 6 pemangku kepentingan utama:
1. **VP Marketing / Retail Funding:** Membutuhkan peningkatan *Return on Investment* (ROI) kampanye, penurunan biaya akuisisi (CPA), dan pencapaian target volume DPK perbankan.
2. **Manajer Operasional Call Center:** Memerlukan daftar prioritas kontak (*call priority list*) harian untuk mengoptimalkan kapasitas jam kerja agen dan memangkas waktu terbuang.
3. **Agen Telemarketing:** Membutuhkan daftar prospek (*leads*) yang berkualitas tinggi dengan rasio keberhasilan konversi tinggi agar moral kerja terjaga dan terhindar dari *burnout*.
4. **Tim Data Science & Analitik:** Membutuhkan pipeline machine learning yang bebas kebocoran data (*leak-free*), teruji secara statistik, *reproducible*, dan mudah dipelihara.
5. **Divisi Kepatuhan, Risiko & Regulasi (Compliance & Risk):** Menuntut transparansi model (interpretabilitas fitur) dan memastikan proses penargetan tidak diskriminatif serta mematuhi privasi nasabah.
6. **Nasabah Bank (Customer):** Menginginkan interaksi yang relevan dan personal, terbebas dari panggilan *spam* atau penawaran produk yang tidak sesuai dengan kebutuhan finansial mereka.

---

## 1.4 Pendekatan Metodologi
Metode yang dipilih adalah **Klasifikasi Biner Probabilistik** (*Probabilistic Binary Classification*). Model tidak hanya mengeluarkan keputusan biner (*Yes/No*), melainkan menghitung skor probabilitas kontinyu $P(Y = 1 | X) \in [0, 1]$. Skor probabilitas ini memungkinkan perbankan melakukan perankingan nasabah (*decile ranking / customer tiering*) dari probabilitas tertinggi hingga terendah.

---

## 1.5 Target Metrik Keberhasilan Proyek

### A. Target Metrik Teknis Machine Learning
* **PR-AUC (Precision-Recall AUC / Average Precision) $\ge 0{,}45$ (Metrik Utama):** Mengingat distribusi target sangat tidak seimbang ($11{,}70\%$ positif), PR-AUC adalah metrik objektif paling representatif (baseline model acak = prevalensi positif $\approx 0{,}117$).
* **ROC-AUC $\ge 0{,}80$ (Metrik Sekunder):** Evaluasi daya diskriminasi global model tanpa menggunakan fitur bocor `duration`.
* **Lift Desil ke-1 (Top 10% nasabah) $\ge 3{,}0\times$:** Efektivitas penargetan pada kelompok 10% teratas minimal 3 kali lipat lebih baik dibandingkan panggilan acak.
* **Capture Rate (Cumulative Gain) Top 40% populasi $= 75\% - 80\%$:** Dengan hanya menghubungi 40% nasabah teratas, bank dapat menangkap setidaknya 75%–80% dari seluruh calon nasabah deposito.
* **$F_2$-Score $\ge 0{,}55$ pada Threshold Optimal:** Menaruh bobot lebih tinggi pada *Recall* untuk meminimalkan *False Negative* (nasabah berminat yang terlewat).

### B. Target Metrik Bisnis & Operasional
* **Net Conversion Rate pada Kelompok Target $\ge 25\%$:** Peningkatan tingkat konversi panggilan dari baseline $11{,}70\%$ menjadi minimal $25\%$ pada kelompok sasaran prioritas.
* **Penurunan Biaya Akuisisi (*Cost per Acquisition / CPA*) $\ge 35\%$:** Mengurangi biaya telepon dan kompensasi waktu agen per deposito yang berhasil dibuka.
* **Rata-rata Panggilan per Konversi (*Average Calls per Conversion*) $\le 8$ panggilan:** Memangkas rasio panggilan dari baseline historis $\approx 8{,}55$ panggilan per konversi menjadi $\le 8$ panggilan.""")

    return cells

if __name__ == "__main__":
    c = get_notebook_cells()
    print("Cells generated so far:", len(c))

