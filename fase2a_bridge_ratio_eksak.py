# -*- coding: utf-8 -*-
"""
===========================================================================
FASE 2A - Pengukuran bridge ratio EKSAK via analisis persistensi topologis
Proyek skripsi WBCAtt+ - Muhammad Fadhil Mulyadi
===========================================================================

MASALAH YANG DIPERBAIKI
-----------------------
Fase 1 mencari r_pisah dengan loop erosi berlangkah 0,5 piksel. Itu
memberi bias diskretisasi sistematis ke atas, tidak punya penanganan
untuk nukleus yang tidak pernah pecah, dan tidak bisa membedakan lobus
sejati dari tonjolan derau di tepi mask.

RUMUSAN EKSAK
-------------
Erosi dengan disk radius r menyisakan piksel dengan DT > r. Jadi hasil
erosi adalah SUPERLEVEL SET dari distance transform. Pertanyaan
"radius erosi minimum yang memecah nukleus" identik dengan
"pada nilai DT berapa superlevel set terpisah menjadi dua komponen".

Jawabannya adalah nilai SADDLE tertinggi pada DT, dan itu dapat dihitung
persis tanpa satu pun operasi erosi:

  1. Hitung EDT float pada mask nukleus.
  2. Urutkan piksel menurun berdasarkan nilai DT.
  3. Masukkan satu per satu ke struktur union-find.
  4. Piksel pertama yang MENYATUKAN dua komponen berbeda adalah saddle.
     Nilai DT di titik itu adalah r_pisah eksak.

Setiap penggabungan juga memberi PERSISTENCE lobus yang mati
(= puncak lobus dikurangi nilai saddle). Itu memberi filter derau yang
berprinsip: lobus dengan persistence di bawah tau adalah tonjolan tepi,
bukan lobus anatomis.

Tidak ada langkah 0,5 piksel. Tidak ada bias diskretisasi.
Tidak ada interpolasi.

KELUARAN
--------
  hasil_fase2a/bridge_ratio_v2.csv      <- data inti baru
  hasil_fase2a/kejadian_merge.csv       <- daftar saddle+persistence per sel
  hasil_fase2a/sensitivitas_tau.csv     <- sapuan tau
  hasil_fase2a/F2A_laporan.txt          <- salinan keluaran konsol
  hasil_fase2a/*.png                    <- grafik (bila matplotlib ada)

CARA PAKAI
----------
  Uji cepat dulu (300 sel, ~30 detik):
      python fase2a_bridge_ratio_eksak.py --sampel 300

  Bila lancar, jalankan penuh:
      python fase2a_bridge_ratio_eksak.py

  Hanya neutrofil (lebih cepat, tanpa kontrol lintas kelas):
      python fase2a_bridge_ratio_eksak.py --hanya-neutrofil

KEBUTUHAN
---------
  numpy, scipy, pillow    (wajib)
  matplotlib              (opsional, untuk grafik)
===========================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==========================================================================
# KONFIGURASI
# ==========================================================================

BASIS = r"D:\Muhammad_Fadhil_Mulyadi\Kuliah\Semester_5\Rekayasa Sistem Informasi Cerdas"

KONFIG = {
    "mask":   os.path.join(BASIS, "pbcseg_final_v1"),
    "csv":    os.path.join(BASIS, "pbc_attr_v1_ccrop_all.csv"),
    "v1":     os.path.join(BASIS, "hasil_fase1", "bridge_ratio.csv"),
    "keluar": os.path.join(BASIS, "hasil_fase2a"),
}

NILAI_NUKLEUS = 2          # kode piksel nukleus pada peta segmentasi
TAU_BAKU = 1.0             # persistence threshold baku, satuan piksel
TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
TAU_SIMPAN_KEJADIAN = 0.25 # kejadian di bawah ini tidak disimpan ke CSV
AMBANG_KLINIS = 1.0 / 3.0
N_BOOTSTRAP = 2000
SEED = 20260917

BENTUK_SEGMENTED = {"segmented-bilobed", "segmented-multilobed"}
BENTUK_BAND = {"unsegmented-band"}


# ==========================================================================
# UTILITAS KELUARAN
# ==========================================================================

_penampung = []


def catat(teks=""):
    print(teks)
    _penampung.append(str(teks))


def garis(ch="=", n=75):
    catat(ch * n)


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
    if not baris:
        catat("  (kosong)")
        return
    rata_kanan = rata_kanan or set()
    data = [list(map(str, header))] + [[str(s) for s in b] for b in baris]
    lebar = [max(len(r[i]) for r in data) for i in range(len(header))]

    def fmt(r):
        return "  " + " | ".join(
            r[i].rjust(lebar[i]) if i in rata_kanan else r[i].ljust(lebar[i])
            for i in range(len(header))
        )

    catat(fmt(data[0]))
    catat("  " + "-+-".join("-" * w for w in lebar))
    for r in data[1:]:
        catat(fmt(r))


def ringkas(x):
    """Statistik ringkas satu vektor."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan,
                    q25=np.nan, med=np.nan, q75=np.nan, max=np.nan)
    return dict(
        n=x.size, mean=x.mean(), std=x.std(ddof=1) if x.size > 1 else 0.0,
        min=x.min(), q25=np.percentile(x, 25), med=np.median(x),
        q75=np.percentile(x, 75), max=x.max(),
    )


def f(v, d=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


# ==========================================================================
# INTI: ANALISIS PERSISTENSI PADA DISTANCE TRANSFORM
# ==========================================================================

def persistensi_dt(dt, conn8=True):
    """Bangun pohon penggabungan superlevel set dari distance transform.

    Mengembalikan:
        r_lobus  : nilai DT maksimum (radius lobus terlebar)
        kejadian : daftar (saddle, persistence) untuk setiap penggabungan
        bertahan : puncak setiap komponen yang tidak pernah bergabung
    """
    H, W = dt.shape
    datar = dt.ravel()
    idx = np.flatnonzero(datar > 0)
    if idx.size == 0:
        return None

    # urutan menurun; stable agar hasil deterministik saat nilai seri
    urut = idx[np.argsort(-datar[idx], kind="stable")]

    off = (-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1) if conn8 else (-W, -1, 1, W)

    induk = np.full(H * W, -1, dtype=np.int64)
    puncak = np.zeros(H * W, dtype=np.float64)
    kejadian = []

    def cari(x):
        r = x
        while induk[r] != r:
            r = induk[r]
        while induk[x] != r:      # pemadatan jalur
            nx = induk[x]
            induk[x] = r
            x = nx
        return r

    for p in urut:
        v = datar[p]

        akar = []
        for d in off:
            q = p + d
            if induk[q] >= 0:
                r = cari(q)
                if r not in akar:
                    akar.append(r)

        induk[p] = p
        puncak[p] = v

        if not akar:
            continue                      # lobus baru lahir

        utama = akar[0]
        for r in akar[1:]:
            if puncak[r] > puncak[utama]:
                utama = r

        for r in akar:
            if r == utama:
                continue
            # r adalah lobus yang mati pada ketinggian v
            kejadian.append((float(v), float(puncak[r] - v)))
            induk[r] = utama

        induk[p] = utama

    akar_akhir = set()
    for p in urut:
        akar_akhir.add(cari(p))
    bertahan = [float(puncak[r]) for r in akar_akhir]

    return float(datar.max()), kejadian, bertahan


def turunkan(r_lobus, kejadian, bertahan, tau):
    """Turunkan r_pisah, bridge_ratio, dan jumlah lobus untuk satu nilai tau.

    Aturan lengkap:
      - Bila >=2 komponen bertahan dengan puncak >= tau, nukleus SUDAH
        terpisah pada r=0            -> r_pisah = 0,  bridge_ratio = 0
      - Bila ada penggabungan signifikan, r_pisah = saddle tertinggi
      - Bila tidak ada sama sekali, nukleus TIDAK PERNAH PECAH
                                     -> bridge_ratio = 1,0 (ditandai)
    """
    sig_gabung = [(s, p) for (s, p) in kejadian if p >= tau]
    sig_tahan = [pk for pk in bertahan if pk >= tau]

    n_lobus = len(sig_gabung) + len(sig_tahan)

    if len(sig_tahan) >= 2:
        return 0.0, 0.0, n_lobus, "terpisah_di_nol"

    if sig_gabung:
        r_pisah = max(s for s, _ in sig_gabung)
        br = r_pisah / r_lobus if r_lobus > 0 else np.nan
        return r_pisah, br, n_lobus, "normal"

    return np.nan, 1.0, max(n_lobus, 1), "tak_pernah_pecah"


# ==========================================================================
# PEMROSESAN SATU SEL
# ==========================================================================

def proses_satu(tugas):
    """Dijalankan per sel. Argumen dan hasil sengaja sederhana agar
    aman untuk multiprocessing di Windows (spawn)."""
    kunci, jalur_mask, isi_lubang, conn8 = tugas

    from PIL import Image
    from scipy.ndimage import distance_transform_edt, binary_fill_holes, label

    try:
        with Image.open(jalur_mask) as im:
            m = np.array(im)
    except Exception as e:
        return {"kunci": kunci, "galat": f"baca:{e}"}

    if m.ndim == 3:
        m = m[..., 0]

    nuk = (m == NILAI_NUKLEUS)
    luas_mentah = int(nuk.sum())
    if luas_mentah == 0:
        return {"kunci": kunci, "galat": "nukleus_kosong"}

    komponen_mentah = int(label(nuk)[1])

    lubang = 0
    if isi_lubang:
        terisi = binary_fill_holes(nuk)
        lubang = int(terisi.sum() - nuk.sum())
        nuk = terisi

    ys, xs = np.nonzero(nuk)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    sub = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=bool)
    sub[1:1 + (y1 - y0), 1:1 + (x1 - x0)] = nuk[y0:y1, x0:x1]

    dt = distance_transform_edt(sub)

    hasil = persistensi_dt(dt, conn8=conn8)
    if hasil is None:
        return {"kunci": kunci, "galat": "dt_kosong"}

    r_lobus, kejadian, bertahan = hasil

    keluaran = {
        "kunci": kunci,
        "galat": "",
        "r_lobus": r_lobus,
        "luas_nukleus": int(nuk.sum()),
        "luas_nukleus_mentah": luas_mentah,
        "piks_lubang": lubang,
        "komponen_mentah": komponen_mentah,
        "n_kejadian": len(kejadian),
        "kejadian": [(s, p) for (s, p) in kejadian if p >= TAU_SIMPAN_KEJADIAN],
        "bertahan": bertahan,
    }

    for tau in TAU_GRID:
        rp, br, nl, st = turunkan(r_lobus, kejadian, bertahan, tau)
        kunci_tau = f"{tau:g}".replace(".", "p")
        keluaran[f"rp_{kunci_tau}"] = rp
        keluaran[f"br_{kunci_tau}"] = br
        keluaran[f"nl_{kunci_tau}"] = nl
        keluaran[f"st_{kunci_tau}"] = st

    return keluaran


# ==========================================================================
# METRIK
# ==========================================================================

def auc_peringkat(pos, neg):
    """AUC via statistik peringkat Mann-Whitney, tahan nilai seri."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    gabung = np.concatenate([pos, neg])
    r = rankdata(gabung)
    return (r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def ambang_youden(band, seg):
    """Ambang yang memaksimalkan TPR - FPR, dengan 'band' = kelas positif
    dan aturan keputusan band bila nilai >= ambang."""
    band = np.asarray(band, float); band = band[np.isfinite(band)]
    seg = np.asarray(seg, float);  seg = seg[np.isfinite(seg)]
    if band.size == 0 or seg.size == 0:
        return np.nan, np.nan
    kand = np.unique(np.concatenate([band, seg]))
    if kand.size > 4000:
        kand = np.quantile(kand, np.linspace(0, 1, 4000))
    bs = np.sort(band)
    ss = np.sort(seg)
    tpr = 1.0 - np.searchsorted(bs, kand, side="left") / bs.size
    fpr = 1.0 - np.searchsorted(ss, kand, side="left") / ss.size
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(kand[i]), float(j[i])


def akurasi_pada(band, seg, ambang):
    band = np.asarray(band, float); band = band[np.isfinite(band)]
    seg = np.asarray(seg, float);  seg = seg[np.isfinite(seg)]
    benar = int((band >= ambang).sum() + (seg < ambang).sum())
    return benar / (band.size + seg.size)


def entropi_biner(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ==========================================================================
# PROGRAM UTAMA
# ==========================================================================

def main():
    p = argparse.ArgumentParser(description="Fase 2A - bridge ratio eksak")
    p.add_argument("--mask", default=KONFIG["mask"])
    p.add_argument("--csv", default=KONFIG["csv"])
    p.add_argument("--v1", default=KONFIG["v1"])
    p.add_argument("--keluar", default=KONFIG["keluar"])
    p.add_argument("--tau", type=float, default=TAU_BAKU)
    p.add_argument("--sampel", type=int, default=0, help="uji cepat N sel")
    p.add_argument("--hanya-neutrofil", action="store_true")
    p.add_argument("--tanpa-isi-lubang", action="store_true")
    p.add_argument("--conn4", action="store_true", help="konektivitas 4 (baku 8)")
    p.add_argument("--pekerja", type=int, default=0, help="0 = otomatis")
    arg = p.parse_args()

    isi_lubang = not arg.tanpa_isi_lubang
    conn8 = not arg.conn4
    os.makedirs(arg.keluar, exist_ok=True)

    judul("FASE 2A - BRIDGE RATIO EKSAK VIA PERSISTENSI TOPOLOGIS")
    catat(f"Waktu          : {datetime.now():%Y-%m-%d %H:%M:%S}")
    catat(f"Folder mask    : {arg.mask}")
    catat(f"CSV WBCAtt+    : {arg.csv}")
    catat(f"tau baku       : {arg.tau} piksel")
    catat(f"Konektivitas   : {'8' if conn8 else '4'}")
    catat(f"Isi lubang     : {'ya' if isi_lubang else 'tidak'}")

    # ------------------------------------------------------------ indeks mask
    subjudul("Mengindeks berkas mask")
    if not os.path.isdir(arg.mask):
        catat(f"[GAGAL] Folder mask tidak ditemukan: {arg.mask}")
        simpan(arg.keluar)
        return 1

    indeks = {}
    for akar, _, berkas in os.walk(arg.mask):
        for b in berkas:
            if not b.lower().endswith(".png"):
                continue
            batang = os.path.splitext(b)[0]
            if batang.lower().endswith("_mask"):
                batang = batang[:-5]
            indeks.setdefault(batang, os.path.join(akar, b))
            if batang.lower().endswith("_ccrop"):
                indeks.setdefault(batang[:-6], os.path.join(akar, b))
    catat(f"  Mask terindeks: {len(indeks):,}")

    # ------------------------------------------------------------- baca CSV
    subjudul("Membaca CSV atribut")
    if not os.path.exists(arg.csv):
        catat(f"[GAGAL] CSV tidak ditemukan: {arg.csv}")
        simpan(arg.keluar)
        return 1

    with open(arg.csv, "r", encoding="utf-8-sig", newline="") as fh:
        baris_csv = list(csv.DictReader(fh))
    catat(f"  Baris: {len(baris_csv):,}")

    for r in baris_csv:
        nama = os.path.basename(r.get("img_name") or r.get("path") or "")
        batang = os.path.splitext(nama)[0]
        r["_batang"] = batang
        r["_prefix"] = (batang.split("_")[0] if "_" in batang else batang).upper()

    def kelompokkan(r):
        pre, bentuk = r["_prefix"], (r.get("nucleus_shape") or "").strip()
        if pre == "NEUTROPHIL":
            return "F_tak_bersubtipe"
        if pre == "BNE":
            if bentuk in BENTUK_BAND:
                return "A_sepakat_band"
            if bentuk in BENTUK_SEGMENTED:
                return "D_konflik_BNE_segmented"
            return "E_lainnya"
        if pre == "SNE":
            if bentuk in BENTUK_SEGMENTED:
                return "B_sepakat_segmented"
            if bentuk in BENTUK_BAND:
                return "C_konflik_SNE_band"
            return "E_lainnya"
        return "Z_non_neutrofil"

    for r in baris_csv:
        r["_kelompok"] = kelompokkan(r)

    subjudul("Kelompok (replikasi definisi Fase 1)")
    hit = Counter(r["_kelompok"] for r in baris_csv)
    harap = {"A_sepakat_band": 1555, "B_sepakat_segmented": 976,
             "C_konflik_SNE_band": 662, "D_konflik_BNE_segmented": 48,
             "E_lainnya": 38, "F_tak_bersubtipe": 50}
    baris = []
    ok_kel = True
    for k in sorted(hit):
        h = harap.get(k)
        st = "-" if h is None else ("COCOK" if h == hit[k] else f"selisih {hit[k]-h:+d}")
        if h is not None and h != hit[k]:
            ok_kel = False
        baris.append([k, f"{hit[k]:,}", f"{h:,}" if h else "-", st])
    cetak_tabel(["kelompok", "n", "Fase 1", "status"], baris, rata_kanan={1, 2})
    catat(f"  >>> Replikasi kelompok: {'LOLOS' if ok_kel else 'PERIKSA'}")

    # ------------------------------------------------------- pilih tugas
    target = baris_csv
    if arg.hanya_neutrofil:
        target = [r for r in target if r["_kelompok"] != "Z_non_neutrofil"]
    if arg.sampel:
        rng = np.random.default_rng(SEED)
        pilih = rng.choice(len(target), size=min(arg.sampel, len(target)),
                           replace=False)
        target = [target[i] for i in sorted(pilih)]

    tugas, tanpa_mask = [], 0
    for r in target:
        jalur = indeks.get(r["_batang"])
        if jalur is None:
            tanpa_mask += 1
            continue
        tugas.append((r["_batang"], jalur, isi_lubang, conn8))

    catat()
    catat(f"  Sel akan diproses : {len(tugas):,}")
    catat(f"  Tanpa mask        : {tanpa_mask:,}")
    if not tugas:
        catat("[GAGAL] Tidak ada sel yang bisa diproses.")
        simpan(arg.keluar)
        return 1

    # ---------------------------------------------------------- eksekusi
    subjudul("Menghitung")
    n_pekerja = arg.pekerja or max(1, (os.cpu_count() or 2) - 1)
    catat(f"  Pekerja: {n_pekerja}")
    t0 = datetime.now()
    hasil = []

    if n_pekerja > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_pekerja) as ex:
            for i, h in enumerate(ex.map(proses_satu, tugas, chunksize=32), 1):
                hasil.append(h)
                if i % 1000 == 0 or i == len(tugas):
                    lewat = (datetime.now() - t0).total_seconds()
                    print(f"    {i:,}/{len(tugas):,}  ({lewat:.0f} s)", flush=True)
    else:
        for i, t in enumerate(tugas, 1):
            hasil.append(proses_satu(t))
            if i % 500 == 0 or i == len(tugas):
                lewat = (datetime.now() - t0).total_seconds()
                print(f"    {i:,}/{len(tugas):,}  ({lewat:.0f} s)", flush=True)

    durasi = (datetime.now() - t0).total_seconds()
    catat(f"  Selesai dalam {durasi:.1f} detik "
          f"({durasi / max(1, len(tugas)) * 1000:.1f} ms per sel)")

    galat = [h for h in hasil if h.get("galat")]
    if galat:
        catat(f"  Galat: {len(galat):,}")
        for g in galat[:5]:
            catat(f"    - {g['kunci']}: {g['galat']}")
    hasil = {h["kunci"]: h for h in hasil if not h.get("galat")}
    catat(f"  Berhasil: {len(hasil):,}")

    # ------------------------------------------------------ gabung + simpan
    kunci_tau = f"{arg.tau:g}".replace(".", "p")
    if f"br_{kunci_tau}" not in next(iter(hasil.values())):
        catat(f"[GAGAL] tau={arg.tau} tidak ada di TAU_GRID. Pilih salah satu: {TAU_GRID}")
        simpan(arg.keluar)
        return 1

    v1 = {}
    if os.path.exists(arg.v1):
        with open(arg.v1, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                nm = os.path.basename(r.get("img_name", ""))
                v1[os.path.splitext(nm)[0]] = r
        catat(f"  Data Fase 1 dimuat: {len(v1):,} baris")
    else:
        catat(f"  (Fase 1 tidak ditemukan di {arg.v1}, perbandingan v1-v2 dilewati)")

    gabung = []
    for r in target:
        h = hasil.get(r["_batang"])
        if h is None:
            continue
        b = {
            "img_name": r.get("img_name", ""),
            "prefix": r["_prefix"],
            "kelompok": r["_kelompok"],
            "label": r.get("label", ""),
            "nucleus_shape": r.get("nucleus_shape", ""),
            "split": r.get("split", ""),
            "r_lobus": h["r_lobus"],
            "r_pisah_v2": h[f"rp_{kunci_tau}"],
            "bridge_ratio_v2": h[f"br_{kunci_tau}"],
            "n_lobus_persisten": h[f"nl_{kunci_tau}"],
            "status_topologi": h[f"st_{kunci_tau}"],
            "luas_nukleus": h["luas_nukleus"],
            "piks_lubang": h["piks_lubang"],
            "komponen_mentah": h["komponen_mentah"],
            "n_kejadian_total": h["n_kejadian"],
        }
        vr = v1.get(r["_batang"])
        if vr:
            for k_lama, k_baru in (("r_pisah", "r_pisah_v1"),
                                   ("bridge_ratio", "bridge_ratio_v1"),
                                   ("r_lobus", "r_lobus_v1"),
                                   ("indeks_segmentasi", "indeks_segmentasi")):
                try:
                    b[k_baru] = float(vr.get(k_lama, ""))
                except (TypeError, ValueError):
                    b[k_baru] = np.nan
        gabung.append(b)

    jalur_utama = os.path.join(arg.keluar, "bridge_ratio_v2.csv")
    kolom = list(gabung[0].keys())
    with open(jalur_utama, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=kolom)
        w.writeheader()
        for b in gabung:
            w.writerow(b)
    catat(f"  Tersimpan: {jalur_utama}  ({len(gabung):,} baris)")

    jalur_kej = os.path.join(arg.keluar, "kejadian_merge.csv")
    n_kej = 0
    with open(jalur_kej, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img_name", "saddle", "persistence"])
        for r in target:
            h = hasil.get(r["_batang"])
            if not h:
                continue
            for s, pers in h["kejadian"]:
                w.writerow([r.get("img_name", ""), f"{s:.6f}", f"{pers:.6f}"])
                n_kej += 1
    catat(f"  Tersimpan: {jalur_kej}  ({n_kej:,} kejadian)")

    # ---------------------------------------------------------- pintasan
    def ambil(kelompok, kolom_nama="bridge_ratio_v2"):
        return np.array([b[kolom_nama] for b in gabung
                         if b["kelompok"] == kelompok], dtype=float)

    A = ambil("A_sepakat_band")
    B = ambil("B_sepakat_segmented")
    C = ambil("C_konflik_SNE_band")
    D = ambil("D_konflik_BNE_segmented")
    E = ambil("E_lainnya")
    F = ambil("F_tak_bersubtipe")

    # ================================================================== D1
    judul("D1. DIAGNOSTIK PENGUKURAN")
    neu = [b for b in gabung if b["kelompok"] != "Z_non_neutrofil"]
    baris = []
    for nama, vals in (("r_lobus", [b["r_lobus"] for b in neu]),
                       ("r_pisah_v2", [b["r_pisah_v2"] for b in neu]),
                       ("bridge_ratio_v2", [b["bridge_ratio_v2"] for b in neu]),
                       ("luas_nukleus", [b["luas_nukleus"] for b in neu])):
        s = ringkas(vals)
        baris.append([nama, s["n"], f(s["mean"]), f(s["std"]), f(s["min"]),
                      f(s["q25"]), f(s["med"]), f(s["q75"]), f(s["max"])])
    cetak_tabel(["fitur", "n", "mean", "std", "min", "q25", "med", "q75", "max"],
                baris, rata_kanan=set(range(1, 9)))

    subjudul("Kasus tepi yang metode erosi lama tidak tangani")
    st = Counter(b["status_topologi"] for b in neu)
    for k, v in st.most_common():
        catat(f"    {k:<20} {v:>6,}  ({v / max(1, len(neu)) * 100:5.2f}%)")
    n_lubang = sum(1 for b in neu if b["piks_lubang"] > 0)
    n_multi = sum(1 for b in neu if b["komponen_mentah"] > 1)
    catat()
    catat(f"    nukleus dengan lubang terisi : {n_lubang:,}")
    catat(f"    mask sudah >1 komponen       : {n_multi:,}")

    # ================================================================== D2
    if any("bridge_ratio_v1" in b for b in gabung):
        judul("D2. PERBANDINGAN v1 (erosi) vs v2 (persistensi)")
        pas = [b for b in gabung
               if np.isfinite(b.get("bridge_ratio_v1", np.nan))
               and np.isfinite(b["bridge_ratio_v2"])]
        if pas:
            a1 = np.array([b["bridge_ratio_v1"] for b in pas])
            a2 = np.array([b["bridge_ratio_v2"] for b in pas])
            d = a2 - a1
            from scipy.stats import spearmanr, pearsonr
            catat(f"  Sel terbandingkan : {len(pas):,}")
            catat(f"  Pearson  r        : {f(pearsonr(a1, a2)[0])}")
            catat(f"  Spearman rho      : {f(spearmanr(a1, a2).statistic)}")
            catat(f"  Selisih (v2 - v1) : mean {f(d.mean())}  median {f(np.median(d))}"
                  f"  sd {f(d.std(ddof=1))}")
            pindah = int(((a1 >= AMBANG_KLINIS) != (a2 >= AMBANG_KLINIS)).sum())
            catat(f"  Berpindah sisi ambang 1/3 : {pindah:,} "
                  f"({pindah / len(pas) * 100:.2f}%)")
            catat()
            catat("  Bias diskretisasi v1 diperkirakan positif. Selisih median")
            catat("  negatif berarti dugaan itu benar dan v1 memang over-estimate.")

    # ================================================================== D3
    judul("D3. SEBARAN bridge_ratio_v2 PER KELOMPOK")
    baris = []
    for nama, v in (("A_sepakat_band", A), ("B_sepakat_segmented", B),
                    ("C_konflik_SNE_band", C), ("D_konflik_BNE_segmented", D),
                    ("E_lainnya", E), ("F_tak_bersubtipe", F)):
        s = ringkas(v)
        baris.append([nama, s["n"], f(s["mean"]), f(s["q25"]), f(s["med"]),
                      f(s["q75"])])
    cetak_tabel(["kelompok", "n", "mean", "q25", "median", "q75"], baris,
                rata_kanan=set(range(1, 6)))

    # ================================================================== D4
    judul("D4. VALIDASI AMBANG SEPERTIGA [TEMUAN UTAMA]")
    auc = auc_peringkat(A, B)
    catat(f"  AUC bridge_ratio_v2 (A vs B) : {f(auc)}")
    catat(f"    pembanding Fase 1 (v1)     : 0.9227")

    amb_y, j = ambang_youden(A, B)
    catat(f"  Ambang optimal Youden        : {f(amb_y)}   (J = {f(j)})")

    rng = np.random.default_rng(SEED)
    Ac = A[np.isfinite(A)]
    Bc = B[np.isfinite(B)]
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sa = Ac[rng.integers(0, Ac.size, Ac.size)]
        sb = Bc[rng.integers(0, Bc.size, Bc.size)]
        boot[i] = ambang_youden(sa, sb)[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    catat(f"  Bootstrap {N_BOOTSTRAP}x median      : {f(np.median(boot))}")
    catat(f"  IK 95% bootstrap             : [{f(lo)} - {f(hi)}]")
    catat(f"  Ambang klinis konvensional   : {f(AMBANG_KLINIS)}  (sepertiga)")

    dalam = lo <= AMBANG_KLINIS <= hi
    catat(f"  >>> 1/3 berada di dalam IK 95% : {'YA' if dalam else 'TIDAK'}")

    akur = akurasi_pada(A, B, AMBANG_KLINIS)
    catat(f"  Akurasi memakai 1/3 tanpa fitting : {akur * 100:.2f}%"
          f"   (Fase 1: 90.04%)")

    subjudul("Generalisasi lintas split (bukan hanya stabilitas sampling)")
    def subset(kel, spl):
        return np.array([b["bridge_ratio_v2"] for b in gabung
                         if b["kelompok"] == kel and b["split"] == spl], float)
    At, Bt = subset("A_sepakat_band", "train"), subset("B_sepakat_segmented", "train")
    Ae, Be = subset("A_sepakat_band", "test"), subset("B_sepakat_segmented", "test")
    if At.size and Bt.size and Ae.size and Be.size:
        amb_tr, _ = ambang_youden(At, Bt)
        catat(f"  Ambang dari TRAIN saja   : {f(amb_tr)}  (n={At.size}+{Bt.size})")
        catat(f"  Akurasi di TEST          : {akurasi_pada(Ae, Be, amb_tr) * 100:.2f}%"
              f"   (n={Ae.size}+{Be.size})")
        catat(f"  Akurasi di TEST pakai 1/3: {akurasi_pada(Ae, Be, AMBANG_KLINIS) * 100:.2f}%")
    else:
        catat("  (kolom split tidak lengkap)")

    # ================================================================== D5
    judul("D5. POSISI KELOMPOK KONFLIK DAN KUANTIFIKASI AMBIGUITAS")
    mA, mB = np.median(A[np.isfinite(A)]), np.median(B[np.isfinite(B)])
    for nama, v in (("C_konflik", C), ("D_konflik", D), ("F_tak_bersubtipe", F)):
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            continue
        pos = (np.median(vv) - mA) / (mB - mA) * 100 if mB != mA else np.nan
        catat(f"  Posisi {nama:<18} pada sumbu A->B : {pos:6.1f}%")
    catat("  (0% = identik dengan A band, 100% = identik dengan B segmented)")

    subjudul("Entropi keputusan geometris pada ambang 1/3")
    baris = []
    for nama, v in (("A_sepakat_band", A), ("B_sepakat_segmented", B),
                    ("C_konflik_SNE_band", C), ("D_konflik_BNE_segmented", D),
                    ("E_lainnya", E), ("F_tak_bersubtipe", F)):
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            continue
        pb = float((vv >= AMBANG_KLINIS).mean())
        baris.append([nama, vv.size, f"{int(round(pb * vv.size)):,}",
                      f"{pb:.3f}", f"{entropi_biner(pb):.3f}"])
    cetak_tabel(["kelompok", "n", "-> band", "p(band)", "entropi (bit)"],
                baris, rata_kanan={1, 2, 3, 4})
    catat("  1,000 bit = setara lempar koin = ambiguitas maksimum")

    subjudul("Kepadatan di zona perbatasan")
    baris = []
    for pita in ((0.28, 0.38), (0.23, 0.43)):
        r = [f"[{pita[0]:.2f} - {pita[1]:.2f}]"]
        for v in (A, B, C):
            vv = v[np.isfinite(v)]
            r.append(f"{((vv >= pita[0]) & (vv <= pita[1])).mean() * 100:.1f}%"
                     if vv.size else "-")
        baris.append(r)
    cetak_tabel(["pita bridge_ratio", "A", "B", "C"], baris, rata_kanan={1, 2, 3})

    # ================================================================== D6
    judul("D6. SENSITIVITAS TERHADAP tau [UJI KEKOKOHAN]")
    catat("Bila ambang optimal stabil di seluruh rentang tau yang wajar,")
    catat("klaim validasi 1/3 tidak bergantung pada pilihan parameter.")
    catat()
    baris = []
    jalur_sens = os.path.join(arg.keluar, "sensitivitas_tau.csv")
    with open(jalur_sens, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "auc_A_vs_B", "ambang_youden", "akurasi_1_3",
                    "n_tak_pernah_pecah", "n_terpisah_di_nol", "posisi_C_persen"])
        for tau in TAU_GRID:
            kt = f"{tau:g}".replace(".", "p")
            va, vb, vc = {}, {}, {}
            tak, nol = 0, 0
            for r in target:
                h = hasil.get(r["_batang"])
                if not h:
                    continue
                kel = r["_kelompok"]
                if kel == "Z_non_neutrofil":
                    continue
                br = h[f"br_{kt}"]
                stt = h[f"st_{kt}"]
                tak += (stt == "tak_pernah_pecah")
                nol += (stt == "terpisah_di_nol")
                if kel == "A_sepakat_band":
                    va[r["_batang"]] = br
                elif kel == "B_sepakat_segmented":
                    vb[r["_batang"]] = br
                elif kel == "C_konflik_SNE_band":
                    vc[r["_batang"]] = br
            aa = np.array(list(va.values()), float)
            bb = np.array(list(vb.values()), float)
            cc = np.array(list(vc.values()), float)
            u = auc_peringkat(aa, bb)
            th, _ = ambang_youden(aa, bb)
            ak = akurasi_pada(aa, bb, AMBANG_KLINIS)
            mmA, mmB = np.median(aa), np.median(bb)
            pos = ((np.median(cc) - mmA) / (mmB - mmA) * 100) if mmB != mmA else np.nan
            w.writerow([tau, f"{u:.4f}", f"{th:.4f}", f"{ak:.4f}", tak, nol,
                        f"{pos:.1f}"])
            tanda = "  <== baku" if abs(tau - arg.tau) < 1e-9 else ""
            baris.append([f"{tau:g}", f(u), f(th), f"{ak * 100:.2f}%",
                          f"{tak:,}", f"{nol:,}", f"{pos:.1f}%" + tanda])
    cetak_tabel(["tau", "AUC", "ambang Youden", "akurasi 1/3",
                 "tak pecah", "pecah di 0", "posisi C"],
                baris, rata_kanan={1, 2, 3, 4, 5, 6})
    catat(f"  Tersimpan: {jalur_sens}")

    # ================================================================== D7
    judul("D7. UJI PREDIKSI - 50 SEL BERPREFIX 'NEUTROPHIL_'")
    catat("Prediksi dibuat SEBELUM data dilihat: sel-sel ini adalah kasus")
    catat("yang anotator PBC menolak menentukan subtipenya. Bila zona ambigu")
    catat("di sekitar 1/3 itu nyata, mereka seharusnya menumpuk di sana.")
    catat("Sumber label ketiga, sepenuhnya independen.")
    catat()
    Fc = F[np.isfinite(F)]
    if Fc.size:
        s = ringkas(Fc)
        catat(f"  n = {s['n']},  median = {f(s['med'])},  "
              f"IQR = [{f(s['q25'])} - {f(s['q75'])}]")
        catat(f"  Di pita [0,23 - 0,43] : {((Fc >= .23) & (Fc <= .43)).mean() * 100:.1f}%"
              f"   (pembanding A dan B di tabel D5)")
        pb = float((Fc >= AMBANG_KLINIS).mean())
        catat(f"  p(band) = {pb:.3f},  entropi = {entropi_biner(pb):.3f} bit")
        catat()
        catat("  Catatan: n=50, jadi ini indikasi, bukan bukti kuat sendirian.")
    else:
        catat("  (tidak ada sel F dalam sampel ini)")

    # ================================================================== D8
    if not arg.hanya_neutrofil:
        judul("D8. KONTROL LINTAS KELAS SEL")
        catat("Limfosit berinti bulat dan konveks. Secara matematis nukleusnya")
        catat("TIDAK BOLEH pecah. Bila v2 benar, mereka harus menumpuk di 1,0.")
        catat("Fase 1 melaporkan NOL sel yang tidak pernah pecah - itu gejala")
        catat("bahwa metode erosi lama menangkap pecahan akibat derau tepi.")
        catat()
        baris = []
        for lab in sorted({b["label"] for b in gabung if b["label"]}):
            v = np.array([b["bridge_ratio_v2"] for b in gabung
                          if b["label"] == lab], float)
            stt = [b["status_topologi"] for b in gabung if b["label"] == lab]
            nl = np.array([b["n_lobus_persisten"] for b in gabung
                           if b["label"] == lab], float)
            vv = v[np.isfinite(v)]
            tak = sum(1 for x in stt if x == "tak_pernah_pecah")
            baris.append([lab, vv.size, f(np.median(vv)),
                          f"{tak:,} ({tak / max(1, len(stt)) * 100:.1f}%)",
                          f"{nl.mean():.2f}"])
        cetak_tabel(["kelas", "n", "median br", "tak pernah pecah",
                     "rata n_lobus"], baris, rata_kanan={1, 2, 3, 4})

    # ================================================================== D9
    judul("D9. PENGHITUNGAN LOBUS BERBASIS PERSISTENSI vs T2")
    catat("Dokumen T2: penghitungan lobus via komponen terhubung GAGAL.")
    catat("81,4% sel 'segmented-multilobed' hanya menghasilkan 1 komponen.")
    catat("Persistensi menghitung lobus yang tersambung filamen kromatin tipis")
    catat("sebagai puncak terpisah, jadi seharusnya jauh lebih baik.")
    catat()
    urutan = ["unsegmented-round", "unsegmented-indented", "irregular",
              "unsegmented-band", "segmented-bilobed", "segmented-multilobed"]
    baris = []
    for bentuk in urutan:
        sel = [b for b in gabung if b["nucleus_shape"] == bentuk]
        if not sel:
            continue
        nl = np.array([b["n_lobus_persisten"] for b in sel], float)
        br = np.array([b["bridge_ratio_v2"] for b in sel], float)
        p1 = float((nl <= 1).mean())
        baris.append([bentuk, len(sel), f"{nl.mean():.3f}",
                      f"{np.median(nl):.1f}", f"{p1 * 100:.1f}%",
                      f(np.median(br[np.isfinite(br)]))])
    cetak_tabel(["nucleus_shape", "n", "mean n_lobus", "median",
                 "hanya 1 lobus", "median br"], baris,
                rata_kanan={1, 2, 3, 4, 5})
    mult = [b for b in gabung if b["nucleus_shape"] == "segmented-multilobed"]
    if mult:
        nl = np.array([b["n_lobus_persisten"] for b in mult], float)
        catat()
        catat(f"  segmented-multilobed dengan hanya 1 lobus : "
              f"{(nl <= 1).mean() * 100:.1f}%    (T2 metode lama: 81,4%)")

    # ================================================================= D10
    punya_lda = any(np.isfinite(b.get("indeks_segmentasi", np.nan)) for b in gabung)
    if punya_lda:
        judul("D10. TRIANGULASI DENGAN INDEKS LDA FASE 1")
        from scipy.stats import spearmanr
        pas = [b for b in gabung
               if np.isfinite(b.get("indeks_segmentasi", np.nan))
               and np.isfinite(b["bridge_ratio_v2"])]
        if pas:
            x = np.array([b["bridge_ratio_v2"] for b in pas])
            y = np.array([b["indeks_segmentasi"] for b in pas])
            catat(f"  Spearman(bridge_ratio_v2, indeks_LDA) = "
                  f"{f(spearmanr(x, y).statistic)}   (n={len(pas):,})")
            catat(f"  Pembanding Fase 1 (v1)                = -0.7403")
            catat()
            catat("  Dua ukuran geometris independen. Korelasi negatif kuat")
            catat("  berarti keduanya mengukur sumbu segmentasi yang sama.")

    # ------------------------------------------------------------- grafik
    buat_grafik(arg.keluar, gabung, A, B, C, D, F)

    # ------------------------------------------------------------- vonis
    judul("RINGKASAN UNTUK KEPUTUSAN LANJUT")
    catat(f"  AUC v2 (A vs B)          : {f(auc)}        [v1: 0.9227]")
    catat(f"  Ambang bootstrap median  : {f(np.median(boot))}   [v1: 0.3333]")
    catat(f"  IK 95%                   : [{f(lo)} - {f(hi)}]  [v1: 0.3192-0.3671]")
    catat(f"  Akurasi pada 1/3         : {akur * 100:.2f}%      [v1: 90.04%]")
    catat(f"  1/3 di dalam IK 95%      : {'YA' if dalam else 'TIDAK'}")
    catat()
    catat("  Yang perlu diperiksa dari keluaran di atas:")
    catat("    1. Apakah ambang Youden stabil lintas tau (tabel D6)?")
    catat("    2. Apakah limfosit menumpuk di 'tak pernah pecah' (tabel D8)?")
    catat("    3. Apakah penghitungan lobus mengalahkan 81,4% di T2 (D9)?")
    catat("    4. Apakah 50 sel tanpa subtipe jatuh di zona ambigu (D7)?")
    catat()
    catat("Kirim seluruh keluaran ini untuk penentuan Fase 2B.")
    garis("=")

    simpan(arg.keluar)
    return 0


# ==========================================================================
# GRAFIK
# ==========================================================================

def buat_grafik(keluar, gabung, A, B, C, D, F):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        catat()
        catat("  (matplotlib tidak tersedia, grafik dilewati)")
        return

    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for nama, v, w in (("A sepakat band", A, 2.0),
                           ("B sepakat segmented", B, 2.0),
                           ("C konflik SNE->band", C, 2.4),
                           ("D konflik BNE->segmented", D, 1.4),
                           ("F tanpa subtipe", F, 1.4)):
            vv = v[np.isfinite(v)]
            if vv.size < 5:
                continue
            ax.hist(vv, bins=60, range=(0, 1), density=True, histtype="step",
                    linewidth=w, label=f"{nama} (n={vv.size})")
        ax.axvline(1 / 3, color="k", linestyle="--", linewidth=1.5,
                   label="ambang klinis 1/3")
        ax.set_xlabel("bridge ratio v2 (persistensi eksak)")
        ax.set_ylabel("kepadatan")
        ax.set_title("Sebaran bridge ratio per kelompok neutrofil")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(keluar, "F2A_sebaran_bridge_ratio.png"), dpi=150)
        plt.close(fig)

        pas = [b for b in gabung
               if np.isfinite(b.get("bridge_ratio_v1", np.nan))
               and np.isfinite(b["bridge_ratio_v2"])]
        if pas:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter([b["bridge_ratio_v1"] for b in pas],
                       [b["bridge_ratio_v2"] for b in pas], s=3, alpha=0.25)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.axvline(1 / 3, color="r", linewidth=0.8, alpha=0.6)
            ax.axhline(1 / 3, color="r", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("bridge ratio v1 (erosi, langkah 0,5 px)")
            ax.set_ylabel("bridge ratio v2 (persistensi eksak)")
            ax.set_title(f"v1 vs v2  (n={len(pas):,})")
            fig.tight_layout()
            fig.savefig(os.path.join(keluar, "F2A_v1_vs_v2.png"), dpi=150)
            plt.close(fig)

        catat()
        catat(f"  Grafik tersimpan di {keluar}")
    except Exception as e:
        catat(f"  (gagal membuat grafik: {e})")


def simpan(keluar):
    try:
        os.makedirs(keluar, exist_ok=True)
        jalur = os.path.join(keluar, "F2A_laporan.txt")
        with open(jalur, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_penampung))
        print()
        print(f"Laporan lengkap tersimpan: {jalur}")
    except Exception as e:
        print(f"(gagal menyimpan laporan: {e})")


if __name__ == "__main__":
    sys.exit(main())
