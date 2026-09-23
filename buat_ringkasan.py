"""
=============================================================================
BUAT RINGKASAN  --  jalankan SETELAH kedua skrip eksplorasi selesai
=============================================================================
Skrip ini membaca folder hasil_eksplorasi/ dan memadatkan semuanya menjadi
satu file teks: RINGKASAN.txt

File itulah yang kamu kirim untuk dibahas. Ukurannya biasanya < 30 KB.

    python buat_ringkasan.py
=============================================================================
"""

from pathlib import Path
import pandas as pd

OUT = Path("hasil_eksplorasi")
TARGET = Path("RINGKASAN.txt")
baris = []


def tulis(teks=""):
    baris.append(str(teks))


def bagian(judul):
    tulis("\n" + "=" * 74)
    tulis(judul)
    tulis("=" * 74)


def muat(nama, index_col=0, n=None, sort=None, asc=True):
    """Baca CSV hasil eksplorasi kalau ada, kembalikan None kalau tidak."""
    p = OUT / f"{nama}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=index_col)
    if sort and sort in df.columns:
        df = df.sort_values(sort, ascending=asc)
    return df.head(n) if n else df


def cetak(nama, judul, **kw):
    df = muat(nama, **kw)
    tulis(f"\n--- {judul} ---")
    if df is None:
        tulis("  (file tidak ada -- bagian ini belum dijalankan)")
    else:
        tulis(df.to_string(float_format="%.3f"))
    return df


if not OUT.exists():
    raise SystemExit("Folder hasil_eksplorasi/ tidak ditemukan. "
                     "Jalankan dulu eksplorasi_wbcattplus.py")

tulis("RINGKASAN EKSPLORASI DATASET WBCAtt+")
tulis(f"dibuat dari: {OUT.resolve()}")

# ---------------------------------------------------------------- BAGIAN A
bagian("BAGIAN A -- LEVEL CSV")

cetak("A1_baris_tidak_konsisten", "A1. Baris yang melanggar aturan logis")
cetak("A2_prefix_vs_label", "A2. Prefix nama file vs label kelas")
cetak("A2_subtipe_neutrofil_vs_nucleus_shape",
      "A2b. Subtipe neutrofil vs bentuk nukleus (%)")
cetak("A3_ringkasan_imbalance", "A3. Ringkasan imbalance per atribut")
cetak("A3_distribusi_nilai", "A3b. Distribusi tiap nilai atribut")
cetak("A4_drift_split", "A4. Drift split (10 teratas)",
      sort="selisih_maks", asc=False, n=10)
cetak("A5_nmi_atribut_vs_label", "A5. NMI atribut vs kelas sel")
cetak("A6_cramers_v", "A6. Matriks Cramer's V antar-atribut")
cetak("A7_redundansi", "A7. Redundansi (prediksi atribut dari atribut lain)")
cetak("A8_plafon_cbm", "A8. Plafon Concept Bottleneck", index_col=None)
cetak("A9_kesulitan_per_atribut", "A9. Kesulitan per nilai atribut (baseline)",
      sort="F1")

# --- A5 crosstab per kelas (file .txt)
p = OUT / "A5_atribut_per_kelas.txt"
tulis("\n--- A5b. Distribusi atribut per kelas sel ---")
tulis(p.read_text() if p.exists() else "  (file tidak ada)")

# ---------------------------------------------------------------- BAGIAN B
bagian("BAGIAN B -- LEVEL PIKSEL")

cetak("B3_komposisi_piksel", "B3. Komposisi piksel per kelas sel (%)")
cetak("B4_konsistensi", "B4. Konsistensi atribut subjektif vs ukuran piksel",
      index_col=None)
cetak("B5_lobus_vs_bentuk_nukleus", "B5. Jumlah lobus vs nucleus_shape (%)")
cetak("B6_warna_per_kelas", "B6. Statistik warna sitoplasma per kelas")
cetak("B6_label_warna_vs_terukur", "B6b. Label warna vs warna terukur (%)")

# --- B7: cukup jumlahnya + 10 contoh
tulis("\n--- B7. Kandidat anotasi bermasalah ---")
ada_b7 = False
for nama in ["vakuola_yes_tanpa_piksel", "vakuola_no_ada_piksel",
             "nc_high_rasio_rendah", "nc_low_rasio_tinggi"]:
    p = OUT / f"B7_{nama}.csv"
    if p.exists():
        ada_b7 = True
        d = pd.read_csv(p)
        tulis(f"\n  {nama}: {len(d)} gambar")
        tulis(d.head(10).to_string(index=False, float_format="%.3f"))
if not ada_b7:
    tulis("  (belum dijalankan)")

# --- statistik agregat dari B2 (file besar, jangan dikirim mentah)
p = OUT / "B2_fitur_mask.csv"
tulis("\n--- B2. Statistik agregat fitur mask ---")
if p.exists():
    f = pd.read_csv(p)
    kol = ["luas_sel", "luas_sel_rel", "rasio_NC", "frak_vakuola", "n_lobus",
           "nuc_solidity", "nuc_eccentricity", "nuc_circularity",
           "sel_solidity", "cyto_std", "cyto_entropi", "nuc_gray_std",
           "tinggi", "lebar"]
    kol = [c for c in kol if c in f.columns]
    tulis(f"  jumlah sel: {len(f)}")
    tulis(f.set_index("label")[kol].describe().T.to_string(float_format="%.3f"))
    tulis("\n  rata-rata per kelas sel:")
    tulis(f.groupby("label")[kol].mean().to_string(float_format="%.3f"))
    tulis("\n  rata-rata per nilai nucleus_shape:")
    tulis(f.groupby("nucleus_shape")[
        [c for c in ["n_lobus", "nuc_solidity", "nuc_circularity",
                     "nuc_eccentricity"] if c in f.columns]]
        .mean().to_string(float_format="%.3f"))
else:
    tulis("  (belum dijalankan)")

TARGET.write_text("\n".join(baris))
print(f"Selesai. Kirim file ini: {TARGET.resolve()}")
print(f"Ukuran: {TARGET.stat().st_size/1024:.1f} KB")
