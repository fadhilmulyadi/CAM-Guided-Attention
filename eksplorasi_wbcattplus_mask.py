"""
=============================================================================
EKSPLORASI DATASET WBCAtt+  --  Bagian B: Peta Segmentasi + Citra
=============================================================================
Bagian ini menyilangkan ANOTASI PIKSEL dengan ANOTASI ATRIBUT.
Inilah yang tidak bisa dilakukan di dataset WBC lain, dan inilah sumber
ide penelitian paling orisinal.

Persiapan:
  1. Unduh pbcseg_final_v1.tar dari
     https://huggingface.co/datasets/apple2373/wbcattplus/blob/main/pbcseg_final_v1.tar
  2. tar -xf pbcseg_final_v1.tar
  3. Sesuaikan IMG_ROOT di bawah ini.
  4. pip install pillow scikit-image opencv-python tqdm

Format mask: file "<nama>_mask.png", 1 kanal, nilai integer
  0 = background, 1 = sitoplasma/cell, 2 = nukleus,
  3 = trombosit, 4 = sel darah merah/sel lain, 5 = vakuola

Waktu jalan: ~5-15 menit untuk 10.298 gambar di laptop biasa.
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import measure
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# --------------------------------------------------------------------------
IMG_ROOT = Path("./pbcseg_final_v1")          # <-- SESUAIKAN
CSV = "pbc_attr_v1_ccrop_all.csv"
OUT = Path("hasil_eksplorasi")
OUT.mkdir(exist_ok=True)
SAMPEL = None        # None = semua gambar; isi angka (mis. 1000) untuk uji cepat

KELAS_MASK = {0: "background", 1: "sitoplasma", 2: "nukleus",
              3: "trombosit", 4: "sel_lain", 5: "vakuola"}
sns.set_theme(style="whitegrid")


def judul(n, teks):
    print(f"\n{'='*74}\n[{n}] {teks}\n{'='*74}")


df = pd.read_csv(CSV)
if SAMPEL:
    df = df.sample(SAMPEL, random_state=0).reset_index(drop=True)


# ==========================================================================
# B1. INVENTARIS: berapa gambar yang benar-benar punya mask?
# ==========================================================================
judul("B1", "INVENTARIS FILE")

df["ada_gambar"] = df.img_name.apply(lambda f: (IMG_ROOT / f).exists())
df["ada_mask"] = df.img_name.apply(
    lambda f: (IMG_ROOT / f.replace(".jpg", "_mask.png")).exists())
print(f"  Baris di CSV      : {len(df)}")
print(f"  Gambar ditemukan  : {df.ada_gambar.sum()}")
print(f"  Mask ditemukan    : {df.ada_mask.sum()}")
print(f"  Punya keduanya    : {(df.ada_gambar & df.ada_mask).sum()}")
if df.ada_mask.sum() == 0:
    raise SystemExit("Tidak ada mask ditemukan. Cek IMG_ROOT.")

df = df[df.ada_gambar & df.ada_mask].reset_index(drop=True)


# ==========================================================================
# B2. EKSTRAKSI FITUR GEOMETRIS DARI MASK
#     Untuk tiap sel kita hitung ukuran objektif dari piksel, lalu nanti
#     dibandingkan dengan atribut subjektif hasil anotasi patolog.
# ==========================================================================
judul("B2", "EKSTRAKSI FITUR DARI MASK (butuh beberapa menit)")

baris = []
for _, r in tqdm(df.iterrows(), total=len(df)):
    mask = np.array(Image.open(IMG_ROOT / r.img_name.replace(".jpg", "_mask.png")))
    img = np.array(Image.open(IMG_ROOT / r.img_name).convert("RGB"))

    H, W = mask.shape
    total = H * W
    n_cyto = int((mask == 1).sum())
    n_nuc = int((mask == 2).sum())
    n_vac = int((mask == 5).sum())
    n_sel = n_cyto + n_nuc + n_vac          # seluruh badan sel
    if n_sel == 0:
        continue

    # --- bentuk nukleus ---
    nuc_bin = (mask == 2)
    lab = measure.label(nuc_bin)
    props = measure.regionprops(lab)
    props = sorted(props, key=lambda p: p.area, reverse=True)
    n_lobus = sum(1 for p in props if p.area > 0.02 * n_nuc)  # lobus berarti
    if props:
        p0 = props[0]
        solidity = p0.solidity
        eccentricity = p0.eccentricity
        perim = sum(p.perimeter for p in props)
        circularity = 4 * np.pi * n_nuc / (perim ** 2) if perim > 0 else np.nan
    else:
        solidity = eccentricity = circularity = np.nan

    # --- bentuk sel keseluruhan ---
    sel_bin = np.isin(mask, [1, 2, 5])
    lab_s = measure.label(sel_bin)
    ps = measure.regionprops(lab_s)
    ps = sorted(ps, key=lambda p: p.area, reverse=True)
    sel_solidity = ps[0].solidity if ps else np.nan
    sel_ecc = ps[0].eccentricity if ps else np.nan

    # --- warna & tekstur di area sitoplasma ---
    cyto_pix = img[mask == 1]
    if len(cyto_pix) > 10:
        cyto_R, cyto_G, cyto_B = cyto_pix.mean(axis=0)
        cyto_std = cyto_pix.std(axis=0).mean()      # proksi "granularitas"
        gray = cyto_pix.mean(axis=1)
        cyto_entropi = -np.sum(
            (h := np.histogram(gray, bins=32, range=(0, 255))[0] / len(gray))
            [h > 0] * np.log2(h[h > 0]))
    else:
        cyto_R = cyto_G = cyto_B = cyto_std = cyto_entropi = np.nan

    nuc_pix = img[mask == 2]
    nuc_std = nuc_pix.std(axis=1).mean() if len(nuc_pix) > 10 else np.nan
    nuc_gray_std = nuc_pix.mean(axis=1).std() if len(nuc_pix) > 10 else np.nan

    baris.append({
        "img_name": r.img_name,
        "tinggi": H, "lebar": W,
        "piks_sitoplasma": n_cyto, "piks_nukleus": n_nuc, "piks_vakuola": n_vac,
        "piks_trombosit": int((mask == 3).sum()),
        "piks_sel_lain": int((mask == 4).sum()),
        "luas_sel": n_sel,
        "luas_sel_rel": n_sel / total,
        "rasio_NC": n_nuc / n_sel,                 # <-- vs nuclear_cytoplasmic_ratio
        "frak_vakuola": n_vac / n_sel,             # <-- vs cytoplasm_vacuole
        "n_lobus": n_lobus,                        # <-- vs nucleus_shape
        "nuc_solidity": solidity,
        "nuc_eccentricity": eccentricity,
        "nuc_circularity": circularity,
        "sel_solidity": sel_solidity,
        "sel_eccentricity": sel_ecc,
        "cyto_R": cyto_R, "cyto_G": cyto_G, "cyto_B": cyto_B,
        "cyto_std": cyto_std,                      # <-- vs granularity
        "cyto_entropi": cyto_entropi,
        "nuc_std": nuc_std,
        "nuc_gray_std": nuc_gray_std,              # <-- vs chromatin_density
    })

fit = pd.DataFrame(baris).merge(df, on="img_name")
fit.to_csv(OUT / "B2_fitur_mask.csv", index=False)
print(f"  Fitur diekstrak untuk {len(fit)} sel -> {OUT}/B2_fitur_mask.csv")


# ==========================================================================
# B3. KOMPOSISI PIKSEL PER KELAS SEL
# ==========================================================================
judul("B3", "KOMPOSISI PIKSEL PER KELAS SEL")

komposisi = fit.groupby("label")[
    ["piks_sitoplasma", "piks_nukleus", "piks_vakuola",
     "piks_trombosit", "piks_sel_lain"]].sum()
komposisi_pct = komposisi.div(komposisi.sum(axis=1), axis=0) * 100
print(komposisi_pct.round(2).to_string())
komposisi_pct.to_csv(OUT / "B3_komposisi_piksel.csv")

komposisi_pct.plot(kind="barh", stacked=True, figsize=(10, 5), colormap="Set2")
plt.xlabel("% piksel")
plt.title("Komposisi komponen sel per tipe WBC")
plt.tight_layout()
plt.savefig(OUT / "B3_komposisi_piksel.png", dpi=130)
plt.close()


# ==========================================================================
# B4. UJI KONSISTENSI: atribut subjektif vs ukuran objektif dari piksel
#     INI ANALISIS PALING PENTING.
#     Kalau atribut "high N/C ratio" ternyata tidak memisahkan rasio piksel
#     dengan bersih, berarti ada ambiguitas anotasi -> celah penelitian.
# ==========================================================================
judul("B4", "KONSISTENSI ATRIBUT SUBJEKTIF vs UKURAN PIKSEL")

pasangan = [
    ("nuclear_cytoplasmic_ratio", "rasio_NC", "high"),
    ("cell_size", "luas_sel", "big"),
    ("cytoplasm_vacuole", "frak_vakuola", "yes"),
    ("granularity", "cyto_std", "yes"),
    ("chromatin_density", "nuc_gray_std", "densely"),
    ("cell_shape", "sel_solidity", "round"),
]

hasil = []
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, (attr, fitur, positif) in zip(axes.flat, pasangan):
    sub = fit[[attr, fitur]].dropna()
    y = (sub[attr] == positif).astype(int)
    if y.nunique() < 2:
        continue
    auc = roc_auc_score(y, sub[fitur])
    auc = max(auc, 1 - auc)          # arah tidak penting
    # ambang optimal (Youden J)
    fpr, tpr, thr = roc_curve(y, sub[fitur])
    j = np.argmax(tpr - fpr)
    ambang = thr[j]
    akurasi = max((sub[fitur] >= ambang).astype(int).eq(y).mean(),
                  (sub[fitur] < ambang).astype(int).eq(y).mean())
    hasil.append({"atribut": attr, "fitur_piksel": fitur,
                  "AUC": round(auc, 3), "ambang": round(float(ambang), 4),
                  "akurasi_ambang_tunggal": round(akurasi, 3)})
    sns.boxplot(data=sub, x=attr, y=fitur, ax=ax)
    ax.set_title(f"{attr}  (AUC={auc:.3f})")

plt.tight_layout()
plt.savefig(OUT / "B4_konsistensi.png", dpi=130)
plt.close()

kons = pd.DataFrame(hasil)
print(kons.to_string(index=False))
kons.to_csv(OUT / "B4_konsistensi.csv", index=False)
print("""
  Cara membaca:
    AUC ~ 1.00 -> atribut itu sepenuhnya bisa dihitung dari mask.
                  Tidak butuh deep learning, cukup rumus geometri.
    AUC ~ 0.70 -> anotasi patolog memuat pertimbangan di luar geometri
                  (atau memang ambigu). Ini sumber pertanyaan penelitian.
""")


# ==========================================================================
# B5. JUMLAH LOBUS NUKLEUS vs LABEL nucleus_shape
#     nucleus_shape adalah atribut TERSULIT bagi baseline (F1 ~65%).
#     Apakah komponen terhubung di mask bisa menjelaskannya?
# ==========================================================================
judul("B5", "JUMLAH LOBUS vs nucleus_shape")

t = pd.crosstab(fit.nucleus_shape, fit.n_lobus.clip(upper=5),
                normalize="index") * 100
print(t.round(1).to_string())
t.to_csv(OUT / "B5_lobus_vs_bentuk_nukleus.csv")

plt.figure(figsize=(9, 6))
sns.heatmap(t, annot=True, fmt=".1f", cmap="Blues")
plt.xlabel("jumlah komponen nukleus terhubung (>2% luas)")
plt.title("Apakah 'segmented-multilobed' benar punya lebih banyak lobus?")
plt.tight_layout()
plt.savefig(OUT / "B5_lobus.png", dpi=130)
plt.close()

plt.figure(figsize=(11, 5))
sns.boxplot(data=fit, x="nucleus_shape", y="nuc_solidity")
plt.xticks(rotation=20)
plt.title("Solidity nukleus per kategori bentuk")
plt.tight_layout()
plt.savefig(OUT / "B5_solidity.png", dpi=130)
plt.close()


# ==========================================================================
# B6. VARIASI PEWARNAAN (stain) -- risiko bias & domain shift
#     Kalau warna sitoplasma berbeda sistematis antar kelas, model bisa
#     "curang" lewat warna, bukan morfologi.
# ==========================================================================
judul("B6", "VARIASI PEWARNAAN PER KELAS")

warna = fit.groupby("label")[["cyto_R", "cyto_G", "cyto_B"]].agg(["mean", "std"])
print(warna.round(1).to_string())
warna.to_csv(OUT / "B6_warna_per_kelas.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, ch in zip(axes, ["cyto_R", "cyto_G", "cyto_B"]):
    sns.kdeplot(data=fit, x=ch, hue="label", ax=ax, common_norm=False)
    ax.set_title(f"Distribusi {ch} di area sitoplasma")
plt.tight_layout()
plt.savefig(OUT / "B6_warna.png", dpi=130)
plt.close()

t = pd.crosstab(fit.cytoplasm_colour,
                pd.qcut(fit.cyto_B - fit.cyto_R, 4,
                        labels=["paling merah", "2", "3", "paling biru"]),
                normalize="index") * 100
print("\n  Label cytoplasm_colour vs warna terukur (B-R), % per baris:")
print(t.round(1).to_string())
t.to_csv(OUT / "B6_label_warna_vs_terukur.csv")


# ==========================================================================
# B7. CONTOH KASUS TIDAK KONSISTEN
#     Daftar gambar yang label atributnya bertentangan dengan mask-nya.
#     Bahan bagus untuk lampiran skripsi & diskusi dengan pembimbing.
# ==========================================================================
judul("B7", "KANDIDAT ANOTASI BERMASALAH")

anomali = []
# vakuola: dilabeli 'yes' tapi tidak ada piksel vakuola (atau sebaliknya)
a1 = fit[(fit.cytoplasm_vacuole == "yes") & (fit.piks_vakuola == 0)]
a2 = fit[(fit.cytoplasm_vacuole == "no") & (fit.frak_vakuola > 0.02)]
print(f"  cytoplasm_vacuole='yes' tapi 0 piksel vakuola : {len(a1)}")
print(f"  cytoplasm_vacuole='no' tapi >2% piksel vakuola: {len(a2)}")

# N/C ratio: label 'high' tapi rasio piksel rendah
amb = fit[fit.nuclear_cytoplasmic_ratio == "high"].rasio_NC.median()
a3 = fit[(fit.nuclear_cytoplasmic_ratio == "high") & (fit.rasio_NC < 0.4)]
a4 = fit[(fit.nuclear_cytoplasmic_ratio == "low") & (fit.rasio_NC > 0.8)]
print(f"  N/C='high' tapi rasio piksel < 0.40           : {len(a3)}")
print(f"  N/C='low'  tapi rasio piksel > 0.80           : {len(a4)}")

for nama, sub in [("vakuola_yes_tanpa_piksel", a1), ("vakuola_no_ada_piksel", a2),
                  ("nc_high_rasio_rendah", a3), ("nc_low_rasio_tinggi", a4)]:
    if len(sub):
        sub[["img_name", "label", "rasio_NC", "frak_vakuola",
             "cytoplasm_vacuole", "nuclear_cytoplasmic_ratio"]].to_csv(
            OUT / f"B7_{nama}.csv", index=False)
        print(f"    -> tersimpan: {OUT}/B7_{nama}.csv")


print("\n" + "=" * 74)
print(f"SELESAI. Output di: {OUT.resolve()}")
print("Kirim isi folder ini (atau ringkasannya) untuk dibahas arah penelitiannya.")
print("=" * 74)
