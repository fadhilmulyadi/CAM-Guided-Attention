# -*- coding: utf-8 -*-
"""
===========================================================================
GERBANG 0 - Verifikasi konvensi penamaan prefix BNE / SNE pada PBC asli
Proyek skripsi WBCAtt+ - Muhammad Fadhil Mulyadi
===========================================================================

TUJUAN
------
Menutup satu-satunya caveat yang tersisa pada verifikasi prefix:
dokumen Acevedo et al. (2020) tidak pernah menuliskan konvensi penamaan
file secara eksplisit. Skrip ini mencaari bukti konvensi itu di dalam
struktur dataset PBC asli.

LOGIKA VERIFIKASI
-----------------
Bukti tidak dicari di folder neutrofil saja. Kunci argumennya ada di
folder 'ig' (immature granulocytes), yang oleh paper didokumentasikan
berisi tiga subtipe: promielosit, mielosit, metamielosit.

  - Kalau folder 'ig' berisi file berprefix PMY / MY / MMY, maka terbukti
    DI DALAM DATASET YANG SAMA bahwa prefix nama file mengkodekan subtipe
    di bawah level folder, memakai singkatan hematologi standar.
  - Setelah konvensi itu tegak, BNE / SNE di folder neutrofil adalah pola
    yang persis sama, dan band vs segmented adalah satu-satunya pembagian
    standar untuk neutrofil matang.

Argumen berubah dari inferensi menjadi konvensi yang didemonstrasikan.

CARA PAKAI
----------
1. Unduh PBC asli dari:
       https://data.mendeley.com/datasets/snkd93bnjr/1
   (berkas PBC_dataset_normal_DIB.zip, sekitar 2,5 GB)

2. TIDAK PERLU diekstrak. Skrip membaca daftar isi .zip secara langsung.
   Boleh juga diarahkan ke folder hasil ekstrak kalau sudah terlanjur.

3. Sesuaikan blok KONFIG di bawah, lalu:
       python gerbang0_verifikasi_prefix.py

   atau lewat argumen:
       python gerbang0_verifikasi_prefix.py --pbc "D:\\...\\PBC.zip" ^
              --csv "D:\\...\\pbc_attr_v1_ccrop_all.csv"

4. Salin SELURUH keluaran konsol dan kirim balik.
   Keluaran juga tersimpan di hasil_gerbang0/G0_laporan.txt

KEBUTUHAN
---------
Python 3.8+. Tanpa pustaka wajib di luar pustaka standar.
Pillow opsional, hanya untuk memeriksa dimensi citra.
===========================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==========================================================================
# KONFIGURASI - sesuaikan bila tidak memakai argumen CLI
# ==========================================================================

BASIS = r"D:\Muhammad_Fadhil_Mulyadi\Kuliah\Semester_5\Rekayasa Sistem Informasi Cerdas"

KONFIG = {
    # Boleh berupa berkas .zip ATAU folder hasil ekstrak
    "pbc": os.path.join(BASIS, "PBC_dataset_normal_DIB.zip"),
    # CSV atribut WBCAtt+ (opsional, untuk pencocokan nama file)
    "csv": os.path.join(BASIS, "pbc_attr_v1_ccrop_all.csv"),
    "keluar": os.path.join(BASIS, "hasil_gerbang0"),
}


# ==========================================================================
# NILAI HARAPAN
# ==========================================================================

# Tabel 1 Acevedo et al., Data in Brief 30:105474 (2020)
HARAP_FOLDER = {
    "basophil": 1218,
    "eosinophil": 3117,
    "erythroblast": 1551,
    "ig": 2895,
    "lymphocyte": 1214,
    "monocyte": 1420,
    "neutrophil": 3329,
    "platelet": 2348,
}
HARAP_TOTAL = 17092

# Hasil penghitungan kita pada sisi WBCAtt+ (dokumen konteks, Bagian 3)
HARAP_PREFIX_NEUTROFIL = {"BNE": 1633, "SNE": 1646, "NEUTROPHIL": 50}

# Singkatan standar granulosit imatur. Inilah kunci pembuktian konvensi.
PREFIX_IG_STANDAR = {"PMY", "MY", "MMY"}

# Kamus singkatan standar (WBCBench 2026, arXiv:2604.10797)
ARTI_PREFIX = {
    "BNE": "band neutrophil",
    "SNE": "segmented neutrophil",
    "NEUTROPHIL": "neutrophil (tanpa subtipe)",
    "EO": "eosinophil",
    "BA": "basophil",
    "LY": "lymphocyte",
    "MO": "monocyte",
    "MY": "myelocyte",
    "MMY": "metamyelocyte",
    "PMY": "promyelocyte",
    "BL": "blast",
    "ERB": "erythroblast",
    "PLATELET": "platelet",
    "IG": "immature granulocyte",
    "VLY": "variant lymphocyte",
    "PC": "plasma cell",
    "PLY": "prolymphocyte",
}

EKSTENSI_CITRA = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

LIMA_KELAS_WBCATT = {"neutrophil", "eosinophil", "basophil", "lymphocyte", "monocyte"}


# ==========================================================================
# UTILITAS KELUARAN
# ==========================================================================

_penampung = []


def catat(teks=""):
    """Cetak ke konsol sekaligus simpan untuk berkas laporan."""
    print(teks)
    _penampung.append(str(teks))


def garis(char="=", n=75):
    catat(char * n)


def judul(teks):
    catat()
    garis("=")
    catat(teks)
    garis("=")


def subjudul(teks):
    catat()
    catat(teks)
    catat("-" * len(teks))


def cetak_tabel(header, baris, rata_kanan=None):
    """Cetak tabel teks dengan lebar kolom dinamis."""
    if not baris:
        catat("  (kosong)")
        return
    rata_kanan = rata_kanan or set()
    kolom = list(zip(*([header] + [[str(s) for s in b] for b in baris])))
    lebar = [max(len(str(sel)) for sel in k) for k in kolom]

    def fmt(baris_data):
        bagian = []
        for i, sel in enumerate(baris_data):
            s = str(sel)
            bagian.append(s.rjust(lebar[i]) if i in rata_kanan else s.ljust(lebar[i]))
        return "  " + " | ".join(bagian)

    catat(fmt(header))
    catat("  " + "-+-".join("-" * w for w in lebar))
    for b in baris:
        catat(fmt(b))


# ==========================================================================
# PENGUMPULAN DAFTAR BERKAS
# ==========================================================================

def kumpulkan_dari_zip(path_zip):
    """Baca central directory .zip tanpa mengekstrak apa pun."""
    rekaman = []
    with zipfile.ZipFile(path_zip) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            jalur = info.filename.replace("\\", "/")
            bagian = [p for p in jalur.split("/") if p]
            if not bagian:
                continue
            berkas = bagian[-1]
            if berkas.startswith(".") or berkas.startswith("__MACOSX"):
                continue
            if os.path.splitext(berkas)[1].lower() not in EKSTENSI_CITRA:
                continue
            folder = bagian[-2] if len(bagian) >= 2 else "(akar)"
            rekaman.append((folder, berkas, info.filename))
    return rekaman


def kumpulkan_dari_folder(path_folder):
    rekaman = []
    for akar, _, berkas_list in os.walk(path_folder):
        for berkas in berkas_list:
            if berkas.startswith("."):
                continue
            if os.path.splitext(berkas)[1].lower() not in EKSTENSI_CITRA:
                continue
            folder = os.path.basename(akar) or "(akar)"
            rekaman.append((folder, berkas, os.path.join(akar, berkas)))
    return rekaman


def ambil_prefix(nama_berkas):
    batang = os.path.splitext(nama_berkas)[0]
    return (batang.split("_")[0] if "_" in batang else batang).upper()


def batang_kunci(nama_berkas):
    """Normalkan nama berkas agar bisa dipadankan lintas dataset.

    WBCAtt+ : BNE_123456_ccrop.jpg  ->  BNE_123456
    PBC asli: BNE_123456.jpg        ->  BNE_123456
    """
    b = os.path.splitext(str(nama_berkas).strip())[0]
    if b.lower().endswith("_ccrop"):
        b = b[: -len("_ccrop")]
    return b.upper()


def normalkan_folder(nama):
    n = nama.strip().lower()
    if n.endswith("s") and n[:-1] in HARAP_FOLDER:
        return n[:-1]
    return n


# ==========================================================================
# PEMERIKSAAN DIMENSI CITRA (opsional)
# ==========================================================================

def periksa_dimensi(sumber_zip, path_pbc, contoh):
    try:
        from PIL import Image
    except ImportError:
        return None

    hasil = []
    try:
        if sumber_zip:
            with zipfile.ZipFile(path_pbc) as zf:
                for folder, berkas, internal in contoh:
                    with zf.open(internal) as fh:
                        with Image.open(fh) as im:
                            hasil.append((berkas, im.size[0], im.size[1]))
        else:
            for folder, berkas, penuh in contoh:
                with Image.open(penuh) as im:
                    hasil.append((berkas, im.size[0], im.size[1]))
    except Exception as e:
        catat(f"  (gagal membaca dimensi: {e})")
        return None
    return hasil


# ==========================================================================
# PROGRAM UTAMA
# ==========================================================================

def main():
    p = argparse.ArgumentParser(description="Gerbang 0 - verifikasi prefix PBC")
    p.add_argument("--pbc", default=KONFIG["pbc"], help="berkas .zip atau folder PBC")
    p.add_argument("--csv", default=KONFIG["csv"], help="CSV atribut WBCAtt+ (opsional)")
    p.add_argument("--keluar", default=KONFIG["keluar"], help="folder keluaran")
    arg = p.parse_args()

    judul("GERBANG 0 - VERIFIKASI KONVENSI PREFIX BNE / SNE PADA PBC ASLI")
    catat(f"Waktu       : {datetime.now():%Y-%m-%d %H:%M:%S}")
    catat(f"Sumber PBC  : {arg.pbc}")
    catat(f"CSV WBCAtt+ : {arg.csv}")

    if not os.path.exists(arg.pbc):
        catat()
        catat("[GAGAL] Sumber PBC tidak ditemukan.")
        catat("        Unduh dari https://data.mendeley.com/datasets/snkd93bnjr/1")
        catat("        lalu arahkan --pbc ke berkas .zip atau folder hasil ekstrak.")
        simpan(arg.keluar)
        return 1

    # ---------------------------------------------------------------- baca
    subjudul("Membaca daftar berkas")
    sumber_zip = arg.pbc.lower().endswith(".zip")
    if sumber_zip:
        catat("  Mode: membaca central directory .zip (tanpa ekstraksi)")
        rekaman = kumpulkan_dari_zip(arg.pbc)
    else:
        catat("  Mode: menelusuri folder")
        rekaman = kumpulkan_dari_folder(arg.pbc)

    catat(f"  Total berkas citra ditemukan: {len(rekaman):,}")
    if not rekaman:
        catat("[GAGAL] Tidak ada berkas citra. Periksa jalur.")
        simpan(arg.keluar)
        return 1

    # ------------------------------------------------------- struktur data
    per_folder = Counter()
    silang = defaultdict(Counter)          # folder -> prefix -> jumlah
    prefix_ke_folder = defaultdict(set)    # prefix -> {folder}
    kunci_pbc = {}                         # batang -> (folder, prefix)

    for folder, berkas, _ in rekaman:
        f = normalkan_folder(folder)
        pre = ambil_prefix(berkas)
        per_folder[f] += 1
        silang[f][pre] += 1
        prefix_ke_folder[pre].add(f)
        kunci_pbc[batang_kunci(berkas)] = (f, pre)

    # =================================================================== P1
    judul("P1. STRUKTUR FOLDER vs TABEL 1 ACEVEDO ET AL. (2020)")
    baris = []
    total_cocok = True
    for f in sorted(set(list(per_folder.keys()) + list(HARAP_FOLDER.keys()))):
        ada = per_folder.get(f, 0)
        harap = HARAP_FOLDER.get(f)
        if harap is None:
            status = "folder tak dikenal"
        elif ada == harap:
            status = "COCOK"
        elif ada == 0:
            status = "TIDAK ADA"
            total_cocok = False
        else:
            status = f"selisih {ada - harap:+d}"
            total_cocok = False
        baris.append([f, f"{ada:,}", f"{harap:,}" if harap else "-", status])
    cetak_tabel(["folder", "ditemukan", "Tabel 1", "status"], baris, rata_kanan={1, 2})

    total_ada = sum(per_folder.values())
    catat()
    catat(f"  Total ditemukan : {total_ada:,}")
    catat(f"  Total Tabel 1   : {HARAP_TOTAL:,}")
    catat(f"  Selisih         : {total_ada - HARAP_TOTAL:+,}")
    v_struktur = (total_ada == HARAP_TOTAL) and total_cocok
    catat(f"  >>> P1 {'LOLOS' if v_struktur else 'PERLU DIPERIKSA'}")

    # =================================================================== P2
    judul("P2. SILANG LENGKAP FOLDER x PREFIX  [INTI PEMBUKTIAN]")
    catat("Kolom 'arti' memakai singkatan standar WBCBench 2026 (arXiv:2604.10797).")
    catat()
    baris = []
    for f in sorted(silang.keys()):
        for pre, n in silang[f].most_common():
            baris.append([f, pre, f"{n:,}", ARTI_PREFIX.get(pre, "?")])
    cetak_tabel(["folder", "prefix", "jumlah", "arti singkatan standar"], baris,
                rata_kanan={2})

    # ------------------------------------------------- folder multi-prefix
    subjudul("Folder yang memuat lebih dari satu prefix")
    catat("Ini pertanyaan struktural terpenting. Folder dengan banyak prefix")
    catat("membuktikan prefix mengkodekan SUBTIPE di bawah level folder.")
    catat()
    multi = {f: dict(c) for f, c in silang.items() if len(c) > 1}
    if not multi:
        catat("  [GAGAL] Tidak ada folder multi-prefix.")
        catat("          Konvensi subtipe tidak terbukti lewat jalur ini.")
    else:
        for f, c in sorted(multi.items()):
            isi = ", ".join(f"{k}={v:,}" for k, v in sorted(c.items(), key=lambda x: -x[1]))
            catat(f"  {f:<14} ({len(c)} prefix) : {isi}")

    # ---------------------------------------- P2a: folder ig (kunci logika)
    subjudul("P2a. Folder 'ig' - uji kunci konvensi penamaan")
    ig = silang.get("ig", Counter())
    if not ig:
        catat("  [?] Folder 'ig' tidak ditemukan. Uji kunci tidak dapat dijalankan.")
        v_ig = None
    else:
        ditemukan = set(ig.keys())
        cocok_std = ditemukan & PREFIX_IG_STANDAR
        for pre, n in ig.most_common():
            tanda = "<== singkatan standar" if pre in PREFIX_IG_STANDAR else ""
            catat(f"    {pre:<12} {n:>6,}   {ARTI_PREFIX.get(pre, '?'):<28} {tanda}")
        catat()
        v_ig = len(cocok_std) >= 2
        if v_ig:
            catat(f"  >>> P2a LOLOS. Folder 'ig' memuat {len(cocok_std)} singkatan standar")
            catat("      granulosit imatur ({}).".format(", ".join(sorted(cocok_std))))
            catat("      Paper mendokumentasikan folder ini berisi promielosit,")
            catat("      mielosit, dan metamielosit. Prefix nama file memisahkan")
            catat("      ketiganya. KONVENSI 'prefix = subtipe di bawah folder'")
            catat("      TERBUKTI DARI DALAM DATASET ITU SENDIRI.")
        else:
            catat("  >>> P2a TIDAK LOLOS. Prefix folder 'ig' tidak sesuai dugaan.")
            catat("      Konvensi subtipe harus dibuktikan lewat jalur lain.")

    # ---------------------------------------------------- P2b: kebocoran
    subjudul("P2b. Apakah ada prefix yang muncul di lebih dari satu folder?")
    catat("Kebocoran akan merusak klaim 'prefix menentukan identitas sel'.")
    catat()
    bocor = {p: fs for p, fs in prefix_ke_folder.items() if len(fs) > 1}
    if bocor:
        for pre, fs in sorted(bocor.items()):
            catat(f"  [!] {pre} muncul di: {', '.join(sorted(fs))}")
        v_bocor = False
    else:
        catat("  Tidak ada. Setiap prefix hanya milik satu folder.")
        v_bocor = True
    catat(f"  >>> P2b {'LOLOS' if v_bocor else 'PERLU DIPERIKSA'}")

    # =================================================================== P3
    judul("P3. FOLDER NEUTROFIL - PEMERIKSAAN UTAMA")
    neu = silang.get("neutrophil", Counter())
    if not neu:
        catat("  [GAGAL] Folder 'neutrophil' tidak ditemukan.")
        v_neu_ada = v_neu_jml = False
    else:
        baris = []
        for pre, n in neu.most_common():
            harap = HARAP_PREFIX_NEUTROFIL.get(pre)
            if harap is None:
                st = "prefix di luar dugaan"
            elif harap == n:
                st = "COCOK PERSIS"
            else:
                st = f"selisih {n - harap:+d}"
            baris.append([pre, f"{n:,}", f"{harap:,}" if harap else "-",
                          ARTI_PREFIX.get(pre, "?"), st])
        cetak_tabel(["prefix", "di PBC", "di WBCAtt+", "arti", "status"], baris,
                    rata_kanan={1, 2})

        v_neu_ada = {"BNE", "SNE"} <= set(neu.keys())
        v_neu_jml = all(neu.get(k, 0) == v for k, v in HARAP_PREFIX_NEUTROFIL.items())
        jml_neu = sum(neu.values())
        catat()
        catat(f"  Total folder neutrophil : {jml_neu:,}   (Tabel 1: {HARAP_FOLDER['neutrophil']:,})")
        catat(f"  BNE + SNE + NEUTROPHIL  : "
              f"{sum(neu.get(k, 0) for k in HARAP_PREFIX_NEUTROFIL):,}")
        catat()
        catat(f"  >>> P3a prefix BNE dan SNE hadir      : {'LOLOS' if v_neu_ada else 'GAGAL'}")
        catat(f"  >>> P3b jumlah identik dengan WBCAtt+ : {'LOLOS' if v_neu_jml else 'GAGAL'}")

    # =================================================================== P4
    judul("P4. PENCOCOKAN NAMA BERKAS DENGAN WBCAtt+")
    v_padan = v_diagonal = None
    if not os.path.exists(arg.csv):
        catat(f"  [?] CSV tidak ditemukan: {arg.csv}")
        catat("      Pemeriksaan P4 dilewati. Sesuaikan --csv bila perlu.")
    else:
        baris_csv = []
        with open(arg.csv, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                baris_csv.append(r)
        catat(f"  Baris CSV WBCAtt+ : {len(baris_csv):,}")

        cocok, hilang = 0, []
        silang_label = defaultdict(Counter)   # label WBCAtt+ -> folder PBC
        silang_pre = defaultdict(Counter)     # prefix -> label WBCAtt+

        for r in baris_csv:
            nama = r.get("img_name") or r.get("path") or ""
            k = batang_kunci(os.path.basename(nama))
            pre = ambil_prefix(os.path.basename(nama))
            label = (r.get("label") or "?").strip()
            silang_pre[pre][label] += 1
            if k in kunci_pbc:
                cocok += 1
                folder_pbc, _ = kunci_pbc[k]
                silang_label[label][folder_pbc] += 1
            else:
                if len(hilang) < 15:
                    hilang.append(nama)

        catat(f"  Ditemukan di PBC  : {cocok:,}")
        catat(f"  Tidak ditemukan   : {len(baris_csv) - cocok:,}")
        v_padan = (cocok == len(baris_csv))
        if hilang:
            catat("  Contoh yang tidak ditemukan:")
            for h in hilang:
                catat(f"    - {h}")
        catat(f"  >>> P4a padanan 1:1 : {'LOLOS' if v_padan else 'GAGAL'}")

        subjudul("P4b. Silang label WBCAtt+ x folder PBC (harus diagonal)")
        baris = []
        diagonal = True
        for label in sorted(silang_label.keys()):
            for folder, n in silang_label[label].most_common():
                sesuai = label.strip().lower() == folder.strip().lower()
                if not sesuai:
                    diagonal = False
                baris.append([label, folder, f"{n:,}", "ok" if sesuai else "<== TIDAK COCOK"])
        cetak_tabel(["label WBCAtt+", "folder PBC", "jumlah", ""], baris, rata_kanan={2})
        v_diagonal = diagonal
        catat()
        catat(f"  >>> P4b rantai prefix -> folder -> label : "
              f"{'LOLOS' if v_diagonal else 'GAGAL'}")

        subjudul("P4c. Silang prefix x label WBCAtt+")
        baris = []
        for pre in sorted(silang_pre.keys()):
            for label, n in silang_pre[pre].most_common():
                baris.append([pre, ARTI_PREFIX.get(pre, "?"), label, f"{n:,}"])
        cetak_tabel(["prefix", "arti standar", "label WBCAtt+", "jumlah"], baris,
                    rata_kanan={3})
        catat()
        catat("  Catatan: BNE dan SNE sama-sama berlabel 'neutrophil' pada WBCAtt+.")
        catat("  Itu memang diharapkan - WBCAtt+ memakai 5 kelas, bukan subtipe.")
        catat("  Justru itulah sebabnya prefix menjadi sumber label kedua yang")
        catat("  independen terhadap anotasi nucleus_shape.")

    # =================================================================== P5
    judul("P5. DIMENSI CITRA (pemeriksaan versi dataset)")
    contoh = [r for r in rekaman if normalkan_folder(r[0]) == "neutrophil"][:5]
    if not contoh:
        contoh = rekaman[:5]
    dim = periksa_dimensi(sumber_zip, arg.pbc, contoh)
    if dim is None:
        catat("  Pillow tidak tersedia atau gagal dibaca. Dilewati.")
        catat("  (opsional: pip install pillow)")
    else:
        for nama, w, h in dim:
            catat(f"    {nama:<28} {w} x {h}")
        catat()
        catat("  Harapan PBC asli   : 360 x 363")
        catat("  Harapan versi ccrop: 360 x 360")

    # ============================================================== BERKAS
    simpan_silang(arg.keluar, silang)

    # ============================================================== VONIS
    judul("VONIS GERBANG 0")

    def tanda(v):
        return "LOLOS" if v is True else ("GAGAL" if v is False else "DILEWATI")

    baris = [
        ["P1", "Struktur folder cocok Tabel 1 Acevedo", tanda(v_struktur)],
        ["P2a", "Folder 'ig' memakai singkatan subtipe standar", tanda(v_ig)],
        ["P2b", "Tidak ada prefix bocor lintas folder", tanda(v_bocor)],
        ["P3a", "Prefix BNE dan SNE hadir di folder neutrofil", tanda(v_neu_ada)],
        ["P3b", "Jumlah BNE/SNE identik dengan WBCAtt+", tanda(v_neu_jml)],
        ["P4a", "Padanan nama berkas 1:1 dengan WBCAtt+", tanda(v_padan)],
        ["P4b", "Rantai prefix -> folder -> label konsisten", tanda(v_diagonal)],
    ]
    cetak_tabel(["kode", "pemeriksaan", "hasil"], baris)

    inti = [v_neu_ada, v_neu_jml]
    konvensi = v_ig
    pendukung = [v_struktur, v_bocor, v_padan, v_diagonal]

    catat()
    if all(x is True for x in inti) and konvensi is True and \
       all(x is not False for x in pendukung):
        catat("  >>> LOLOS PENUH")
        catat("      PBC asli memakai prefix untuk memisahkan subtipe di dalam")
        catat("      folder, dengan singkatan hematologi standar. Folder neutrofil")
        catat("      terbagi persis menjadi BNE dan SNE dengan jumlah yang sama")
        catat("      dengan penghitungan kita. Caveat dokumen Bagian 3 TERTUTUP.")
        catat("      Kelompok konflik C (662 sel) berdiri di atas dasar yang sah.")
        kode = 0
    elif all(x is True for x in inti):
        catat("  >>> LOLOS SEBAGIAN")
        catat("      Prefix BNE/SNE terkonfirmasi ada dengan jumlah yang cocok,")
        catat("      tetapi argumen konvensi lewat folder 'ig' belum tegak.")
        catat("      Bab 5 aman. Bab 6 perlu caveat tertulis, atau tempuh")
        catat("      konfirmasi surel ke Anna Merino (amerino@clinic.cat).")
        kode = 0
    else:
        catat("  >>> TIDAK LOLOS")
        catat("      Periksa keluaran P2 dan P3 di atas. Bila folder neutrofil")
        catat("      ternyata tidak memakai prefix BNE/SNE, maka kelompok konflik C")
        catat("      kehilangan dasarnya dan Bab 6 harus dirancang ulang.")
        catat("      Bab 5 tetap selamat: kelompok A dan B disaring memakai")
        catat("      nucleus_shape juga, bukan prefix saja.")
        kode = 2

    catat()
    catat("Salin SELURUH keluaran di atas dan kirim balik untuk penentuan")
    catat("langkah berikutnya.")
    garis("=")

    simpan(arg.keluar)
    return kode


def simpan_silang(folder_keluar, silang):
    try:
        os.makedirs(folder_keluar, exist_ok=True)
        jalur = os.path.join(folder_keluar, "G0_silang_folder_prefix.csv")
        with open(jalur, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["folder", "prefix", "jumlah", "arti_standar"])
            for f in sorted(silang.keys()):
                for pre, n in sorted(silang[f].items(), key=lambda x: -x[1]):
                    w.writerow([f, pre, n, ARTI_PREFIX.get(pre, "")])
        catat()
        catat(f"  Tabel silang tersimpan: {jalur}")
    except Exception as e:
        catat(f"  (gagal menyimpan tabel silang: {e})")


def simpan(folder_keluar):
    try:
        os.makedirs(folder_keluar, exist_ok=True)
        jalur = os.path.join(folder_keluar, "G0_laporan.txt")
        with open(jalur, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_penampung))
        print()
        print(f"Laporan lengkap tersimpan: {jalur}")
    except Exception as e:
        print(f"(gagal menyimpan laporan: {e})")


if __name__ == "__main__":
    sys.exit(main())
