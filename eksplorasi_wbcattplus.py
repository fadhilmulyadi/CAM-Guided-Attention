"""
=============================================================================
EKSPLORASI DATASET WBCAtt+  --  Bagian A: Analisis Level CSV (tanpa GPU)
=============================================================================
Tsutsui et al., "WBCAtt+", Medical Image Analysis 2026
https://github.com/apple2373/wbcattplus

Cara pakai:
    pip install pandas numpy scikit-learn matplotlib seaborn scipy
    python eksplorasi_wbcattplus.py

Output: folder ./hasil_eksplorasi/  berisi file CSV + PNG untuk tiap analisis.
Semua analisis di file ini hanya butuh 1 file CSV (~1.3 MB) dan berjalan
di laptop biasa dalam < 1 menit.
=============================================================================
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# KONFIGURASI
# --------------------------------------------------------------------------
CSV_URL = ("https://raw.githubusercontent.com/apple2373/wbcattplus/"
           "main/dataset_txt/pbc_attr_v1_ccrop_all.csv")
CSV_LOCAL = "pbc_attr_v1_ccrop_all.csv"   # kalau sudah ada lokal, dipakai ini
OUT = Path("hasil_eksplorasi")
OUT.mkdir(exist_ok=True)

ATTRS = [
    "cell_size", "cell_shape", "nucleus_shape", "nuclear_cytoplasmic_ratio",
    "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture",
    "cytoplasm_colour", "granule_type", "granule_colour", "granularity",
]

sns.set_theme(style="whitegrid")


def judul(n, teks):
    garis = "=" * 74
    print(f"\n{garis}\n[{n}] {teks}\n{garis}")


def simpan(df, nama):
    path = OUT / f"{nama}.csv"
    df.to_csv(path, index=True)
    print(f"    -> tersimpan: {path}")


# --------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------
if os.path.exists(CSV_LOCAL):
    df = pd.read_csv(CSV_LOCAL)
else:
    print("Mengunduh CSV dari GitHub...")
    df = pd.read_csv(CSV_URL)
    df.to_csv(CSV_LOCAL, index=False)

print(f"Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")


# ==========================================================================
# A1. INTEGRITAS DATA
#     Pertanyaan: apakah datanya bersih? ada duplikat, missing, atau
#     pelanggaran aturan logis antar-atribut?
# ==========================================================================
judul("A1", "INTEGRITAS DATA")

print(f"  Missing values total      : {df.isna().sum().sum()}")
print(f"  Duplikat img_name         : {df.img_name.duplicated().sum()}")
print(f"  Kolom img_name == path    : {(df.img_name == df.path).all()}")
print(f"  Ukuran split              : {dict(df['split'].value_counts())}")

# Aturan logis yang HARUS berlaku secara medis:
#   granularity == 'no'  <=>  granule_type == 'nil'  <=>  granule_colour == 'nil'
aturan = {
    "granularity=no tapi granule_type!=nil":
        ((df.granularity == "no") & (df.granule_type != "nil")).sum(),
    "granularity=yes tapi granule_type=nil":
        ((df.granularity == "yes") & (df.granule_type == "nil")).sum(),
    "granule_type=nil tapi granule_colour!=nil":
        ((df.granule_type == "nil") & (df.granule_colour != "nil")).sum(),
    "granule_colour=nil tapi granule_type!=nil":
        ((df.granule_colour == "nil") & (df.granule_type != "nil")).sum(),
}
print("\n  Pelanggaran aturan logis antar-atribut:")
for k, v in aturan.items():
    tanda = "  <-- CEK INI" if v > 0 else ""
    print(f"    {k:45s}: {v}{tanda}")

pelanggar = df[(df.granularity == "yes") & (df.granule_type == "nil")]
if len(pelanggar):
    print("\n  Baris yang melanggar:")
    print(pelanggar[["img_name", "label", "granularity",
                     "granule_type", "granule_colour"]].to_string(index=False))
    simpan(pelanggar, "A1_baris_tidak_konsisten")


# ==========================================================================
# A2. SUBTIPE TERSEMBUNYI DI NAMA FILE
#     Pertanyaan: apakah nama file menyimpan informasi yang TIDAK ada
#     di kolom label? (ini sering terlewat)
# ==========================================================================
judul("A2", "SUBTIPE TERSEMBUNYI DI NAMA FILE")

df["prefix"] = df.img_name.str.split("_").str[0]
tab = pd.crosstab(df.prefix, df.label)
print(tab.to_string())
simpan(tab, "A2_prefix_vs_label")
print("""
  Catatan: BNE = band neutrophil, SNE = segmented neutrophil.
  Keduanya dilabeli 'Neutrophil' yang sama di kolom `label`, padahal
  rasio band/segmented ("left shift") adalah penanda klinis infeksi.
  Artinya ada label granular GRATIS yang tidak dipakai paper aslinya.
""")

# Seberapa terkait subtipe neutrofil dengan atribut yang dianotasi?
neu = df[df.prefix.isin(["BNE", "SNE"])]
if len(neu):
    t = pd.crosstab(neu.prefix, neu.nucleus_shape, normalize="index") * 100
    print("  Distribusi nucleus_shape per subtipe neutrofil (%):")
    print(t.round(1).to_string())
    simpan(t, "A2_subtipe_neutrofil_vs_nucleus_shape")


# ==========================================================================
# A3. DISTRIBUSI & KETIDAKSEIMBANGAN ATRIBUT
#     Pertanyaan: nilai mana yang langka? seberapa parah imbalance-nya?
# ==========================================================================
judul("A3", "DISTRIBUSI & KETIDAKSEIMBANGAN ATRIBUT")

baris = []
for a in ATTRS:
    vc = df[a].value_counts()
    vc_tr = df[df.split == "train"][a].value_counts()
    baris.append({
        "atribut": a,
        "n_nilai": df[a].nunique(),
        "terbanyak": vc.idxmax(), "n_terbanyak": vc.max(),
        "terlangka": vc.idxmin(), "n_terlangka": vc.min(),
        "n_terlangka_train": vc_tr.min(),
        "rasio_imbalance": round(vc.max() / vc.min(), 2),
    })
ringkas = pd.DataFrame(baris).sort_values("rasio_imbalance", ascending=False)
print(ringkas.to_string(index=False))
simpan(ringkas.set_index("atribut"), "A3_ringkasan_imbalance")

# Rincian tiap nilai
rinci = []
for a in ATTRS:
    vc = df[a].value_counts()
    for v, n in vc.items():
        rinci.append({"atribut": a, "nilai": v, "jumlah": n,
                      "persen": round(100 * n / len(df), 2)})
rinci = pd.DataFrame(rinci)
simpan(rinci.set_index("atribut"), "A3_distribusi_nilai")

fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, a in zip(axes.flat, ATTRS):
    df[a].value_counts().plot(kind="barh", ax=ax, color="#4c72b0")
    ax.set_title(a, fontsize=11)
    ax.set_xlabel("jumlah")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(OUT / "A3_distribusi_atribut.png", dpi=130)
plt.close()
print(f"    -> tersimpan: {OUT}/A3_distribusi_atribut.png")


# ==========================================================================
# A4. KONSISTENSI SPLIT TRAIN/VAL/TEST
#     Pertanyaan: apakah split-nya terstratifikasi? kalau ada drift,
#     evaluasi bisa bias.
# ==========================================================================
judul("A4", "KONSISTENSI SPLIT TRAIN/VAL/TEST")

baris = []
for a in ATTRS:
    ct = pd.crosstab(df[a], df["split"], normalize="columns") * 100
    for v in ct.index:
        baris.append({"atribut": a, "nilai": v,
                      "train%": ct.loc[v, "train"], "val%": ct.loc[v, "val"],
                      "test%": ct.loc[v, "test"]})
drift = pd.DataFrame(baris)
drift["selisih_maks"] = (drift[["train%", "val%", "test%"]].max(axis=1)
                         - drift[["train%", "val%", "test%"]].min(axis=1))
drift = drift.sort_values("selisih_maks", ascending=False)
print(drift.head(10).to_string(index=False, float_format="%.2f"))
print(f"\n  Drift maksimum di seluruh atribut: {drift.selisih_maks.max():.2f} poin persen")
simpan(drift.set_index("atribut"), "A4_drift_split")


# ==========================================================================
# A5. HUBUNGAN ATRIBUT <-> KELAS SEL
#     Pertanyaan: atribut mana yang sebenarnya sudah "membocorkan" kelas?
#     Atribut dengan NMI tinggi = tugas mudah & tidak informatif untuk XAI.
# ==========================================================================
judul("A5", "HUBUNGAN ATRIBUT vs KELAS SEL")

nmi = {a: normalized_mutual_info_score(df[a], df.label) for a in ATTRS}
nmi = pd.Series(nmi).sort_values(ascending=False)
print(nmi.round(3).to_string())
simpan(nmi.to_frame("NMI_dengan_label"), "A5_nmi_atribut_vs_label")

plt.figure(figsize=(8, 5))
nmi.plot(kind="barh", color="#dd8452")
plt.xlabel("Normalized Mutual Information dengan label kelas")
plt.tight_layout()
plt.savefig(OUT / "A5_nmi.png", dpi=130)
plt.close()

# Crosstab tiap atribut per kelas (ini yang dipakai untuk narasi klinis)
with open(OUT / "A5_atribut_per_kelas.txt", "w") as f:
    for a in ATTRS:
        ct = pd.crosstab(df.label, df[a], normalize="index") * 100
        f.write(f"=== {a} (% per kelas) ===\n{ct.round(1).to_string()}\n\n")
print(f"    -> tersimpan: {OUT}/A5_atribut_per_kelas.txt")


# ==========================================================================
# A6. KETERGANTUNGAN ANTAR-ATRIBUT (Cramer's V)
#     Pertanyaan: baseline memperlakukan 11 atribut sebagai 11 head
#     INDEPENDEN. Apakah asumsi itu benar?
# ==========================================================================
judul("A6", "KETERGANTUNGAN ANTAR-ATRIBUT (Cramer's V)")


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan


M = pd.DataFrame(index=ATTRS, columns=ATTRS, dtype=float)
for a in ATTRS:
    for b in ATTRS:
        M.loc[a, b] = 1.0 if a == b else cramers_v(df[a], df[b])
print(M.round(2).to_string())
simpan(M, "A6_cramers_v")

plt.figure(figsize=(11, 9))
sns.heatmap(M.astype(float), annot=True, fmt=".2f", cmap="rocket_r",
            vmin=0, vmax=1, square=True, cbar_kws={"label": "Cramer's V"})
plt.title("Ketergantungan antar-atribut morfologi")
plt.tight_layout()
plt.savefig(OUT / "A6_cramers_v.png", dpi=130)
plt.close()

pasangan = (M.where(np.triu(np.ones(M.shape), 1).astype(bool))
            .stack().sort_values(ascending=False))
print("\n  10 pasangan paling saling bergantung:")
print(pasangan.head(10).round(3).to_string())


# ==========================================================================
# A7. REDUNDANSI: bisakah 1 atribut diprediksi dari 10 atribut lain?
#     Kalau bisa dengan akurasi tinggi, atribut itu redundan --> argumen
#     kuat untuk memodelkan dependensi label (classifier chain / GNN).
# ==========================================================================
judul("A7", "REDUNDANSI ANTAR-ATRIBUT")

hasil = []
for target in ATTRS:
    fitur = [a for a in ATTRS if a != target]
    X = OneHotEncoder(sparse_output=False).fit_transform(df[fitur])
    y = df[target]
    akurasi = cross_val_score(
        DecisionTreeClassifier(max_depth=6, random_state=0), X, y, cv=5
    ).mean()
    baseline = y.value_counts(normalize=True).max()
    hasil.append({"atribut": target,
                  "akurasi_dari_atribut_lain": round(akurasi, 4),
                  "baseline_kelas_mayoritas": round(baseline, 4),
                  "kenaikan": round(akurasi - baseline, 4)})
red = pd.DataFrame(hasil).sort_values("akurasi_dari_atribut_lain",
                                      ascending=False)
print(red.to_string(index=False))
simpan(red.set_index("atribut"), "A7_redundansi")


# ==========================================================================
# A8. PLAFON CONCEPT BOTTLENECK
#     Pertanyaan: kalau model memprediksi 11 atribut dengan SEMPURNA,
#     berapa akurasi maksimum klasifikasi 5 kelas sel? Ini batas atas
#     teoretis untuk Concept Bottleneck Model.
# ==========================================================================
judul("A8", "PLAFON CONCEPT BOTTLENECK (atribut -> kelas sel)")

kombinasi = df[ATTRS].astype(str).agg("|".join, axis=1)
n_unik = kombinasi.nunique()
g = df.assign(k=kombinasi).groupby("k")["label"].nunique()
ambigu = df.assign(k=kombinasi).k.isin(g[g > 1].index)

print(f"  Kombinasi atribut unik       : {n_unik} dari {len(df)} gambar")
print(f"  Kombinasi -> >1 kelas (ambigu): {(g > 1).sum()}")
print(f"  Gambar dalam kombinasi ambigu : {ambigu.sum()} ({100*ambigu.mean():.1f}%)")

# Plafon sempurna = pilih kelas mayoritas di tiap kombinasi
plafon = (df.assign(k=kombinasi).groupby("k")["label"]
          .apply(lambda s: s.value_counts().max()).sum() / len(df))
print(f"  PLAFON akurasi CBM (oracle)   : {100*plafon:.2f}%")

# Akurasi nyata dengan classifier sederhana dari atribut GT
X = OneHotEncoder(sparse_output=False).fit_transform(df[ATTRS])
tr = df.split == "train"
te = df.split == "test"
clf = LogisticRegression(max_iter=2000)
clf.fit(X[tr.values], df.label[tr])
print(f"  Akurasi LogReg(atribut GT)    : {100*clf.score(X[te.values], df.label[te]):.2f}%")

pohon = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X[tr.values], df.label[tr])
print(f"  Akurasi Pohon d=5             : {100*pohon.score(X[te.values], df.label[te]):.2f}%")

pd.DataFrame([{"kombinasi_unik": n_unik, "kombinasi_ambigu": int((g > 1).sum()),
               "gambar_ambigu": int(ambigu.sum()),
               "plafon_cbm": round(plafon, 4)}]).to_csv(
    OUT / "A8_plafon_cbm.csv", index=False)
print(f"    -> tersimpan: {OUT}/A8_plafon_cbm.csv")


# ==========================================================================
# A9. KEMUDAHAN vs KESULITAN: bandingkan dengan performa baseline paper
#     Pertanyaan: atribut mana yang SUDAH selesai (F1 ~99%) dan mana yang
#     MASIH terbuka (F1 ~65%)? Di sinilah celah penelitian berada.
# ==========================================================================
judul("A9", "PERFORMA BASELINE PAPER vs FREKUENSI NILAI")

URL_BASE = ("https://raw.githubusercontent.com/apple2373/wbcattplus/"
            "main/results/tab_baseline_attpred.md")
try:
    raw = pd.read_csv(URL_BASE, sep="|", skiprows=2, engine="python")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.str.contains("^Unnamed")]
    for c in raw.columns:
        raw[c] = raw[c].astype(str).str.strip()
    raw = raw[~raw["Backbone"].str.contains(":---", na=False)]
    raw["F1"] = raw["F-measure"].str.split("±").str[0].astype(float)
    piv = (raw.groupby(["Attribute Name", "Attribute Value"])["F1"]
           .mean().reset_index().sort_values("F1"))
    print("  15 nilai atribut PALING SULIT (rata-rata F1 semua backbone):")
    print(piv.head(15).to_string(index=False, float_format="%.2f"))
    print("\n  5 nilai atribut PALING MUDAH (sudah jenuh):")
    print(piv.tail(5).to_string(index=False, float_format="%.2f"))
    simpan(piv.set_index("Attribute Name"), "A9_kesulitan_per_atribut")

    plt.figure(figsize=(10, 8))
    warna = ["#c44e52" if v < 85 else "#55a868" for v in piv.F1]
    plt.barh(piv["Attribute Name"] + " = " + piv["Attribute Value"],
             piv.F1, color=warna)
    plt.axvline(85, ls="--", c="k", lw=1)
    plt.xlabel("F1 rata-rata baseline (%)")
    plt.title("Merah = masih terbuka (<85), Hijau = sudah jenuh")
    plt.xlim(50, 100)
    plt.tight_layout()
    plt.savefig(OUT / "A9_kesulitan_atribut.png", dpi=130)
    plt.close()
    print(f"    -> tersimpan: {OUT}/A9_kesulitan_atribut.png")
except Exception as e:
    print(f"  Gagal ambil tabel hasil ({e}). Unduh manual dari repo:")
    print("  results/tab_baseline_attpred.md")


print("\n" + "=" * 74)
print(f"SELESAI. Semua output ada di folder: {OUT.resolve()}")
print("=" * 74)
