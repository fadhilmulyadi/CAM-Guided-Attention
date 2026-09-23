"""
=============================================================================
FASE 1 -- EKSPERIMEN INTI SKRIPSI
Indeks Segmentasi Nukleus & Uji Zona Ketidaksepakatan Band vs Segmented
=============================================================================

Skrip ini menjalankan eksperimen yang menentukan apakah skripsi jalan atau
tidak. Kalau hipotesis utama terbukti, seluruh Bab 6 dan 7 punya dasar.

HIPOTESIS UTAMA (H1):
  662 gambar yang bertentangan (dinamai SNE oleh PBC tapi dianotasi
  'unsegmented-band' oleh WBCAtt+) TIDAK tersebar acak, melainkan
  terkonsentrasi di zona tumpang-tindih geometris antara neutrofil band
  yang disepakati dan neutrofil segmented yang disepakati.

  Kalau H1 benar -> ketidaksepakatan itu TERJELASKAN oleh geometri,
                    bukan derau anotasi. Ini temuan utama skripsi.
  Kalau H1 salah -> salah satu sumber label bermasalah. Juga temuan,
                    tapi arah skripsi harus disesuaikan.

DUA BAGIAN:
  Bagian 1 -- hanya butuh hasil_eksplorasi/B2_fitur_mask.csv (sudah ada)
              Jalan dalam hitungan detik. Jalankan ini DULU.
  Bagian 2 -- butuh folder pbcseg_final_v1/ (mask). ~10 menit.
              Menghitung bridge ratio, yaitu kriteria klinis sebenarnya.

    python fase1_indeks_segmentasi.py
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, kruskal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve

# --------------------------------------------------------------------------
FITUR_CSV = Path("hasil_eksplorasi/B2_fitur_mask.csv")
IMG_ROOT = Path("./pbcseg_final_v1")          # untuk Bagian 2
OUT = Path("hasil_fase1")
OUT.mkdir(exist_ok=True)
JALANKAN_BAGIAN_2 = True

sns.set_theme(style="whitegrid")


def judul(n, t):
    print(f"\n{'='*74}\n[{n}] {t}\n{'='*74}")


# ==========================================================================
# BAGIAN 1 -- UJI ZONA KETIDAKSEPAKATAN (dari fitur yang sudah ada)
# ==========================================================================
judul("1.1", "MENYIAPKAN EMPAT KELOMPOK NEUTROFIL")

if not FITUR_CSV.exists():
    raise SystemExit(f"Tidak ada {FITUR_CSV}. Jalankan dulu "
                     "eksplorasi_wbcattplus_mask.py")

df = pd.read_csv(FITUR_CSV)
df["prefix"] = df.img_name.str.split("_").str[0]
neu = df[df.prefix.isin(["BNE", "SNE"])].copy()

SEG = ["segmented-bilobed", "segmented-multilobed"]


def kelompok(r):
    if r.prefix == "BNE" and r.nucleus_shape == "unsegmented-band":
        return "A_sepakat_band"
    if r.prefix == "SNE" and r.nucleus_shape in SEG:
        return "B_sepakat_segmented"
    if r.prefix == "SNE" and r.nucleus_shape == "unsegmented-band":
        return "C_konflik_SNE_dianotasi_band"
    if r.prefix == "BNE" and r.nucleus_shape in SEG:
        return "D_konflik_BNE_dianotasi_segmented"
    return "E_lainnya"


neu["kelompok"] = neu.apply(kelompok, axis=1)
print(neu.kelompok.value_counts().to_string())
neu.to_csv(OUT / "neutrofil_berkelompok.csv", index=False)

FITUR = ["nuc_circularity", "nuc_solidity", "nuc_eccentricity",
         "rasio_NC", "luas_sel", "n_lobus"]
FITUR = [f for f in FITUR if f in neu.columns]


# --------------------------------------------------------------------------
judul("1.2", "SUMBU SEGMENTASI: LDA DILATIH HANYA PADA KELOMPOK SEPAKAT")
# Kunci metodologis: sumbu dibangun TANPA pernah melihat kelompok konflik,
# supaya proyeksi kelompok konflik menjadi uji yang jujur.

latih = neu[neu.kelompok.isin(["A_sepakat_band", "B_sepakat_segmented"])]
X = latih[FITUR].fillna(latih[FITUR].median())
y = (latih.kelompok == "B_sepakat_segmented").astype(int)

lda = LinearDiscriminantAnalysis().fit(X, y)
print(f"  Dilatih pada {len(latih)} sel yang dua sumbernya sepakat.")
print("  Bobot LDA (arah positif = makin bersegmen):")
for f, w in sorted(zip(FITUR, lda.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"    {f:20s} {w:+.4f}")

auc_sepakat = roc_auc_score(y, lda.decision_function(X))
print(f"\n  AUC pada kelompok sepakat saja: {auc_sepakat:.4f}")
print("  (kalau < 0.80, geometri tidak cukup memisahkan; laporkan apa adanya)")

neu["indeks_segmentasi"] = lda.decision_function(
    neu[FITUR].fillna(latih[FITUR].median()))


# --------------------------------------------------------------------------
judul("1.3", "UJI HIPOTESIS UTAMA: DI MANA KELOMPOK KONFLIK BERADA?")

ring = neu.groupby("kelompok")["indeks_segmentasi"].describe()[
    ["count", "mean", "25%", "50%", "75%"]]
print(ring.round(3).to_string())
ring.to_csv(OUT / "H1_posisi_indeks.csv")

mA = neu.loc[neu.kelompok == "A_sepakat_band", "indeks_segmentasi"]
mB = neu.loc[neu.kelompok == "B_sepakat_segmented", "indeks_segmentasi"]
mC = neu.loc[neu.kelompok == "C_konflik_SNE_dianotasi_band", "indeks_segmentasi"]

print(f"\n  Median A (sepakat band)      : {mA.median():+.3f}")
print(f"  Median C (konflik)           : {mC.median():+.3f}")
print(f"  Median B (sepakat segmented) : {mB.median():+.3f}")

di_antara = mA.median() < mC.median() < mB.median()
print(f"\n  >>> Median C berada DI ANTARA A dan B : {di_antara}")
if di_antara:
    posisi = (mC.median() - mA.median()) / (mB.median() - mA.median())
    print(f"  >>> Posisi relatif C pada sumbu A->B  : {posisi:.1%}")
    print("  >>> H1 DIDUKUNG. Ketidaksepakatan terletak di zona antara.")
else:
    print("  >>> H1 TIDAK didukung pada ukuran median.")
    print("  >>> Cek apakah C justru menyatu dengan A (berarti prefix PBC")
    print("      yang keliru) atau tersebar luas (berarti derau anotasi).")

print("\n  Uji statistik (Mann-Whitney U):")
for n1, s1, n2, s2 in [("A", mA, "C", mC), ("C", mC, "B", mB), ("A", mA, "B", mB)]:
    u, p = mannwhitneyu(s1, s2)
    print(f"    {n1} vs {n2}: U={u:.0f}  p={p:.3e}")
h, p = kruskal(mA, mB, mC)
print(f"    Kruskal-Wallis 3 kelompok: H={h:.1f}  p={p:.3e}")

# Berapa persen C jatuh di zona tumpang-tindih A dan B?
lo, hi = np.percentile(mB, 5), np.percentile(mA, 95)
if lo < hi:
    frac = ((mC >= lo) & (mC <= hi)).mean()
    print(f"\n  Zona tumpang-tindih A/B: [{lo:.2f}, {hi:.2f}]")
    print(f"  Proporsi C di dalam zona itu    : {frac:.1%}")
    print(f"  Proporsi A di dalam zona itu    : {((mA>=lo)&(mA<=hi)).mean():.1%}")
    print(f"  Proporsi B di dalam zona itu    : {((mB>=lo)&(mB<=hi)).mean():.1%}")
else:
    print("\n  Distribusi A dan B terpisah bersih, tidak ada zona tumpang-tindih.")

plt.figure(figsize=(11, 6))
urut = ["A_sepakat_band", "C_konflik_SNE_dianotasi_band",
        "D_konflik_BNE_dianotasi_segmented", "B_sepakat_segmented"]
urut = [u for u in urut if u in neu.kelompok.values]
for k in urut:
    sns.kdeplot(neu.loc[neu.kelompok == k, "indeks_segmentasi"],
                label=f"{k} (n={(neu.kelompok==k).sum()})",
                fill=True, alpha=0.25, common_norm=False)
plt.xlabel("Indeks segmentasi nukleus (LDA, negatif = band)")
plt.title("Uji H1: posisi kelompok konflik pada sumbu geometris")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "H1_distribusi_indeks.png", dpi=140)
plt.close()
print(f"\n    -> {OUT}/H1_distribusi_indeks.png")


# ==========================================================================
# BAGIAN 2 -- BRIDGE RATIO: KRITERIA KLINIS YANG SEBENARNYA
# ==========================================================================
# Definisi operasional patolog: sebuah neutrofil disebut BAND jika jembatan
# (isthmus) antar-lobus nukleus lebih lebar dari sepertiga lebar lobus
# terlebar. Kalau jembatan menipis menjadi filamen, disebut SEGMENTED.
#
# Kita hitung ini secara objektif:
#   r_pisah = radius erosi minimum yang memecah nukleus jadi >= 2 komponen
#             (proksi setengah lebar isthmus)
#   r_lobus = nilai maksimum distance transform (setengah lebar lobus tertebal)
#   bridge_ratio = r_pisah / r_lobus     -> ambang klinis di sekitar 1/3
# ==========================================================================
if JALANKAN_BAGIAN_2 and IMG_ROOT.exists():
    judul("2.1", "MENGHITUNG BRIDGE RATIO DARI MASK NUKLEUS")
    from PIL import Image
    from tqdm import tqdm

    def ukur(mask_path):
        m = np.array(Image.open(mask_path))
        nuc = (m == 2)
        if nuc.sum() < 50:
            return None
        dt = ndi.distance_transform_edt(nuc)
        r_lobus = float(dt.max())
        if r_lobus <= 0:
            return None
        # cari radius erosi terkecil yang memecah nukleus
        r_pisah = r_lobus  # default: tidak pernah pecah
        for r in np.arange(0.5, r_lobus, 0.5):
            komponen = ndi.label(dt > r)[1]
            if komponen >= 2:
                r_pisah = float(r)
                break
        return {"r_lobus": r_lobus, "r_pisah": r_pisah,
                "bridge_ratio": r_pisah / r_lobus,
                "luas_nukleus": int(nuc.sum())}

    baris = []
    for nama in tqdm(neu.img_name):
        p = IMG_ROOT / nama.replace(".jpg", "_mask.png")
        if not p.exists():
            continue
        u = ukur(p)
        if u:
            u["img_name"] = nama
            baris.append(u)

    br = pd.DataFrame(baris).merge(
        neu[["img_name", "prefix", "kelompok", "nucleus_shape", "label",
             "split", "indeks_segmentasi"]], on="img_name")
    br.to_csv(OUT / "bridge_ratio.csv", index=False)
    print(f"  Terhitung untuk {len(br)} neutrofil -> {OUT}/bridge_ratio.csv")

    judul("2.2", "APAKAH BRIDGE RATIO MEMISAHKAN BAND DARI SEGMENTED?")
    t = br.groupby("kelompok")["bridge_ratio"].describe()[
        ["count", "mean", "25%", "50%", "75%"]]
    print(t.round(3).to_string())
    t.to_csv(OUT / "bridge_ratio_per_kelompok.csv")

    sep = br[br.kelompok.isin(["A_sepakat_band", "B_sepakat_segmented"])]
    if len(sep) and sep.kelompok.nunique() == 2:
        yy = (sep.kelompok == "A_sepakat_band").astype(int)
        auc = roc_auc_score(yy, sep.bridge_ratio)
        fpr, tpr, thr = roc_curve(yy, sep.bridge_ratio)
        j = np.argmax(tpr - fpr)
        print(f"\n  AUC bridge_ratio (A vs B)     : {auc:.4f}")
        print(f"  Ambang optimal empiris        : {thr[j]:.4f}")
        print(f"  Ambang klinis konvensional    : 0.3333 (sepertiga)")
        print("  >>> Kalau keduanya berdekatan, kamu baru saja MEMVALIDASI")
        print("      kriteria patolog secara kuantitatif dari piksel.")

    plt.figure(figsize=(11, 6))
    for k in urut:
        if k in br.kelompok.values:
            sns.kdeplot(br.loc[br.kelompok == k, "bridge_ratio"],
                        label=f"{k} (n={(br.kelompok==k).sum()})",
                        fill=True, alpha=0.25, common_norm=False)
    plt.axvline(1/3, ls="--", c="k", lw=1.5, label="ambang klinis 1/3")
    plt.xlabel("bridge ratio = lebar isthmus / lebar lobus")
    plt.title("Kriteria klinis band vs segmented, diukur dari piksel")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "H2_bridge_ratio.png", dpi=140)
    plt.close()
    print(f"\n    -> {OUT}/H2_bridge_ratio.png")
else:
    print("\n[2] Bagian 2 dilewati (folder mask tidak ditemukan).")


print("\n" + "=" * 74)
print(f"SELESAI. Output di: {OUT.resolve()}")
print("Kirimkan: H1_posisi_indeks.csv, bridge_ratio_per_kelompok.csv,")
print("          H1_distribusi_indeks.png, H2_bridge_ratio.png")
print("=" * 74)
