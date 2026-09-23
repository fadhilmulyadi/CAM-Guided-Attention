# -*- coding: utf-8 -*-
"""
===========================================================================
FASE 2B - Kalibrasi parameter, invariansi ambang, dan audit bias band
Proyek skripsi WBCAtt+ - Muhammad Fadhil Mulyadi
===========================================================================

TIGA MASALAH YANG DISELESAIKAN SKRIP INI
----------------------------------------

1. UJI SANITAS SEBELUMNYA TERLALU LONGGAR
   Nilai harapan yang saya tulis di 2A-rev meleset satu piksel:
   fungsi batang(setengah=4) menghasilkan 9 baris, sehingga piksel
   tengah berjarak 5 ke latar, bukan 4. Jawaban benar 5/20,02 = 0,2498
   dan skrip menjawab 0,2497 - eksak. Toleransi 0,06 menutupi
   ketidakcocokan itu. Sekarang toleransi 0,005 pada rasio dan
   0,02 piksel pada r_pisah, dengan harapan yang sudah dibetulkan,
   ditambah dua kasus baru (sudah terpisah di nol, dan tonjolan derau).

2. PARAMETER DIPILIH MEMAKAI KRITERIA YANG SALAH
   2A-rev memilih (tau, rho) yang memaksimalkan AUC pada A vs B, lalu
   memvalidasi ambang pada A vs B juga. Permukaan AUC ternyata datar
   (0,880-0,895), jadi argmax mengejar derau dan mendarat di tau=0,1
   yang merusak penghitungan lobus: bilobed jadi 6,52 lobus dan
   limfosit yang tidak pernah pecah turun dari 96,3% ke 60,6%.

   Skrip ini mengkalibrasi (tau, rho) dari KECOCOKAN JUMLAH LOBUS
   terhadap label nucleus_shape, DAN HANYA PADA SEL NON-NEUTROFIL.
   Sel basofil, eosinofil, limfosit, monosit tidak pernah masuk
   kelompok A maupun B, jadi tidak ada satu pun sel yang dipakai
   untuk menyetel parameter sekaligus untuk menguji ambang.
   Sirkularitasnya hilang sepenuhnya.

3. ARBITRASE GEOMETRIS CONDONG KE BAND
   Pada ambang 1/3: A benar 94,6% tetapi B benar hanya 80,4%. Galatnya
   asimetris, sehingga kedua kelompok konflik sama-sama tertarik ke
   band (~80%). Hipotesis: wbcsegmentor menjembatani filamen kromatin
   selebar 1-2 piksel, sehingga leher terukur lebih tebal dari aslinya.
   Bila benar, r_pisah sel B yang salah klasifikasi akan menumpuk di
   kisaran 2-4 piksel, bukan tersebar. Skrip ini mengujinya dan
   membuat montase visual untuk diperiksa mata.

KELUARAN
--------
  hasil_fase2b/bridge_ratio_final.csv
  hasil_fase2b/kalibrasi_lobus.csv
  hasil_fase2b/invariansi_ambang.csv
  hasil_fase2b/audit_ekor_B.csv
  hasil_fase2b/montase_*.png
  hasil_fase2b/F2B_laporan.txt
  hasil_fase2b/cache_persistensi.pkl.gz   (agar sapuan berikutnya instan)

CARA PAKAI
----------
  python fase2b_kalibrasi_dan_audit.py --sampel 300    (uji cepat)
  python fase2b_kalibrasi_dan_audit.py                 (penuh)

  Jalanan kedua dan seterusnya memakai cache, jadi hanya beberapa detik.
  Paksa hitung ulang dengan --abaikan-cache
===========================================================================
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import pickle
import sys
from collections import Counter
from datetime import datetime

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


BASIS = r"D:\Muhammad_Fadhil_Mulyadi\Kuliah\Semester_5\Rekayasa Sistem Informasi Cerdas"

KONFIG = {
    "mask":   os.path.join(BASIS, "pbcseg_final_v1"),
    "csv":    os.path.join(BASIS, "pbc_attr_v1_ccrop_all.csv"),
    "v1":     os.path.join(BASIS, "hasil_fase1", "bridge_ratio.csv"),
    "keluar": os.path.join(BASIS, "hasil_fase2b"),
}

NILAI_NUKLEUS = 2
PRUNE = 0.05
TAU_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
RHO_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
AMBANG_KLINIS = 1.0 / 3.0
N_BOOTSTRAP = 2000
SEED = 20260917

BENTUK_SEGMENTED = {"segmented-bilobed", "segmented-multilobed"}
BENTUK_BAND = {"unsegmented-band"}

# Jangkar kalibrasi: berapa lobus yang SEHARUSNYA dihitung untuk tiap
# label bentuk nukleus. 'irregular' dan 'unsegmented-band' sengaja
# dikecualikan karena jumlah lobusnya memang tidak tunggal secara definisi.
HARAP_LOBUS = {
    "unsegmented-round": 1,
    "unsegmented-indented": 1,
    "segmented-bilobed": 2,
    "segmented-multilobed": 3,     # ditafsirkan sebagai ">= 3"
}

_penampung = []


def catat(t=""):
    print(t)
    _penampung.append(str(t))


def garis(ch="=", n=75):
    catat(ch * n)


def judul(t):
    catat(); garis("="); catat(t); garis("=")


def subjudul(t):
    catat(); catat(t); catat("-" * len(t))


def cetak_tabel(header, baris, rk=None):
    if not baris:
        catat("  (kosong)"); return
    rk = rk or set()
    data = [list(map(str, header))] + [[str(s) for s in b] for b in baris]
    w = [max(len(r[i]) for r in data) for i in range(len(header))]
    def fmt(r):
        return "  " + " | ".join(r[i].rjust(w[i]) if i in rk else r[i].ljust(w[i])
                                 for i in range(len(header)))
    catat(fmt(data[0]))
    catat("  " + "-+-".join("-" * x for x in w))
    for r in data[1:]:
        catat(fmt(r))


def f(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{d}f}"


# ==========================================================================
# INTI PERSISTENSI
# ==========================================================================

def persistensi_dt(dt, conn8=True):
    """Pohon penggabungan superlevel set dari distance transform.

    kejadian: (saddle, persistence_mati, puncak_mati, puncak_utama, pos_datar)
    """
    H, W = dt.shape
    datar = dt.ravel()
    idx = np.flatnonzero(datar > 0)
    if idx.size == 0:
        return None
    urut = idx[np.argsort(-datar[idx], kind="stable")]
    off = (-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1) if conn8 else (-W, -1, 1, W)

    induk = np.full(H * W, -1, dtype=np.int64)
    puncak = np.zeros(H * W, dtype=np.float64)
    kejadian = []

    def cari(x):
        r = x
        while induk[r] != r:
            r = induk[r]
        while induk[x] != r:
            nx = induk[x]; induk[x] = r; x = nx
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
            continue
        utama = akar[0]
        for r in akar[1:]:
            if puncak[r] > puncak[utama]:
                utama = r
        for r in akar:
            if r == utama:
                continue
            kejadian.append((float(v), float(puncak[r] - v), float(puncak[r]),
                             float(puncak[utama]), int(p)))
            induk[r] = utama
        induk[p] = utama

    akar_akhir = set()
    for p in urut:
        akar_akhir.add(cari(p))
    return float(datar.max()), kejadian, [float(puncak[r]) for r in akar_akhir]


def turunkan(r_lobus, kejadian, bertahan, tau, rho=0.0, aturan="min"):
    """r_pisah, bridge_ratio, n_lobus, status, posisi_saddle_terpilih."""
    amb = rho * r_lobus
    sig_g = [(s, pos) for (s, pers, pkm, _, pos) in kejadian
             if pers >= tau and pkm >= amb]
    sig_b = [pk for pk in bertahan if pk >= tau and pk >= amb]
    n_lobus = len(sig_g) + len(sig_b)

    if len(sig_b) >= 2:
        return 0.0, 0.0, n_lobus, "terpisah_di_nol", -1
    if sig_g:
        if aturan == "min":
            s, pos = min(sig_g, key=lambda x: x[0])
        else:
            s, pos = max(sig_g, key=lambda x: x[0])
        return s, (s / r_lobus if r_lobus > 0 else np.nan), n_lobus, "normal", pos
    return np.nan, 1.0, max(n_lobus, 1), "tak_pernah_pecah", -1


# ==========================================================================
# UJI SANITAS - DIPERKETAT
# ==========================================================================

def uji_sanitas():
    from scipy.ndimage import distance_transform_edt

    def cakram(k, cy, cx, r):
        yy, xx = np.ogrid[:k.shape[0], :k.shape[1]]
        k[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True

    def batang(k, cy, x0, x1, setengah):
        """Menghasilkan 2*setengah+1 baris. Piksel tengah berjarak
        (setengah+1) ke latar terdekat. Itu nilai r_pisah yang benar."""
        k[cy - setengah:cy + setengah + 1, x0:x1] = True

    kasus = []

    # 1. Cakram konveks -> superlevel set selalu terhubung
    k = np.zeros((120, 120), bool); cakram(k, 60, 60, 30)
    kasus.append(("cakram konveks", k, "tak_pernah_pecah", None, 1))

    # 2. Halter, leher 9 baris -> r_pisah = 5,0 eksak
    k = np.zeros((120, 200), bool)
    cakram(k, 60, 50, 20); cakram(k, 60, 150, 20); batang(k, 60, 50, 150, 4)
    kasus.append(("halter, leher 9 baris", k, "normal", 5.0, 2))

    # 3. Rantai leher 7 baris (r=4) dan 17 baris (r=9)
    #    Aturan min HARUS memilih 4,0. Aturan max akan memilih 9,0.
    k = np.zeros((140, 300), bool)
    cakram(k, 70, 50, 22); cakram(k, 70, 150, 22); cakram(k, 70, 250, 22)
    batang(k, 70, 50, 150, 3); batang(k, 70, 150, 250, 8)
    kasus.append(("rantai leher 7 dan 17 baris", k, "normal", 4.0, 3))

    # 4. Dua cakram terpisah total -> sudah pecah di r=0
    k = np.zeros((120, 220), bool)
    cakram(k, 60, 50, 20); cakram(k, 60, 170, 20)
    kasus.append(("dua cakram terpisah", k, "terpisah_di_nol", 0.0, 2))

    # 5. Cakram besar dengan tonjolan kecil -> filter persistence harus
    #    menolak tonjolan itu sebagai lobus pada tau = 1,0
    k = np.zeros((140, 140), bool)
    cakram(k, 70, 70, 30); cakram(k, 70, 103, 6)
    kasus.append(("cakram + tonjolan kecil", k, "tak_pernah_pecah", None, 1))

    baris = []
    lolos = True
    for nama, k, st_harap, rp_harap, nl_harap in kasus:
        pad = np.zeros((k.shape[0] + 2, k.shape[1] + 2), bool)
        pad[1:-1, 1:-1] = k
        dt = distance_transform_edt(pad)
        r_lobus, kej, bert = persistensi_dt(dt)
        rp, br, nl, st, _ = turunkan(r_lobus, kej, bert, 1.0, 0.0, "min")
        rpx, brx, _, _, _ = turunkan(r_lobus, kej, bert, 1.0, 0.0, "max")

        ok = (st == st_harap) and (nl == nl_harap)
        if rp_harap is not None:
            ok = ok and np.isfinite(rp) and abs(rp - rp_harap) < 0.02
            if r_lobus > 0:
                ok = ok and abs(br - rp_harap / r_lobus) < 0.005
        lolos = lolos and ok
        baris.append([nama, f(r_lobus, 3), st, f(rp, 4),
                      "-" if rp_harap is None else f(rp_harap, 4),
                      f(rpx, 4), nl, nl_harap, "OK" if ok else "GAGAL"])

    cetak_tabel(["kasus sintetis", "r_lobus", "status", "r_pisah(min)",
                 "harapan", "r_pisah(max)", "n_lob", "harap", ""],
                baris, rk={1, 3, 4, 5, 6, 7})
    catat()
    catat("  Toleransi: 0,02 piksel pada r_pisah dan 0,005 pada rasio.")
    catat("  Kasus 3 membedakan min dari max: 4,0 benar, 9,0 salah.")
    catat("  Kasus 5 menguji filter persistence menolak tonjolan derau.")
    catat(f"  >>> UJI SANITAS: {'LOLOS' if lolos else 'GAGAL'}")
    return lolos


# ==========================================================================
# PEMROSESAN SATU SEL
# ==========================================================================

def proses_satu(tugas):
    kunci, jalur, isi_lubang, conn8 = tugas
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, binary_fill_holes, label
    try:
        with Image.open(jalur) as im:
            m = np.array(im)
    except Exception as e:
        return {"kunci": kunci, "galat": f"baca:{e}"}
    if m.ndim == 3:
        m = m[..., 0]
    nuk = (m == NILAI_NUKLEUS)
    if not nuk.any():
        return {"kunci": kunci, "galat": "nukleus_kosong"}
    komp = int(label(nuk)[1])
    lub = 0
    if isi_lubang:
        t = binary_fill_holes(nuk)
        lub = int(t.sum() - nuk.sum())
        nuk = t
    ys, xs = np.nonzero(nuk)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sub = np.zeros((y1 - y0 + 2, x1 - x0 + 2), bool)
    sub[1:1 + (y1 - y0), 1:1 + (x1 - x0)] = nuk[y0:y1, x0:x1]
    dt = distance_transform_edt(sub)
    h = persistensi_dt(dt, conn8=conn8)
    if h is None:
        return {"kunci": kunci, "galat": "dt_kosong"}
    r_lobus, kej, bert = h
    return {
        "kunci": kunci, "galat": "", "r_lobus": r_lobus,
        "luas_nukleus": int(nuk.sum()), "piks_lubang": lub,
        "komponen_mentah": komp,
        "kejadian": [t for t in kej if t[1] >= PRUNE],
        "bertahan": bert,
        "crop": (y0, x0, sub.shape[0], sub.shape[1]),
    }


# ==========================================================================
# METRIK
# ==========================================================================

def auc_peringkat(pos, neg):
    from scipy.stats import rankdata
    pos = np.asarray(pos, float); pos = pos[np.isfinite(pos)]
    neg = np.asarray(neg, float); neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def ambang_youden(band, seg):
    band = np.asarray(band, float); band = band[np.isfinite(band)]
    seg = np.asarray(seg, float); seg = seg[np.isfinite(seg)]
    if band.size == 0 or seg.size == 0:
        return np.nan, np.nan
    kand = np.unique(np.concatenate([band, seg]))
    if kand.size > 4000:
        kand = np.quantile(kand, np.linspace(0, 1, 4000))
    bs, ss = np.sort(band), np.sort(seg)
    tpr = 1.0 - np.searchsorted(bs, kand, "left") / bs.size
    fpr = 1.0 - np.searchsorted(ss, kand, "left") / ss.size
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(kand[i]), float(j[i])


def akurasi_pada(band, seg, amb):
    band = np.asarray(band, float); band = band[np.isfinite(band)]
    seg = np.asarray(seg, float); seg = seg[np.isfinite(seg)]
    if band.size + seg.size == 0:
        return np.nan
    return int((band >= amb).sum() + (seg < amb).sum()) / (band.size + seg.size)


def entropi_biner(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ==========================================================================
# UTAMA
# ==========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", default=KONFIG["mask"])
    p.add_argument("--csv", default=KONFIG["csv"])
    p.add_argument("--v1", default=KONFIG["v1"])
    p.add_argument("--keluar", default=KONFIG["keluar"])
    p.add_argument("--sampel", type=int, default=0)
    p.add_argument("--tanpa-isi-lubang", action="store_true")
    p.add_argument("--conn4", action="store_true")
    p.add_argument("--pekerja", type=int, default=0)
    p.add_argument("--abaikan-cache", action="store_true")
    p.add_argument("--n-montase", type=int, default=24)
    arg = p.parse_args()

    isi_lubang = not arg.tanpa_isi_lubang
    conn8 = not arg.conn4
    os.makedirs(arg.keluar, exist_ok=True)

    judul("FASE 2B - KALIBRASI, INVARIANSI AMBANG, DAN AUDIT BIAS BAND")
    catat(f"Waktu        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    catat(f"Konektivitas : {'8' if conn8 else '4'}   Isi lubang: {'ya' if isi_lubang else 'tidak'}")

    judul("S0. UJI SANITAS DIPERKETAT (toleransi 0,005)")
    if not uji_sanitas():
        catat()
        catat("[BERHENTI] Uji sanitas gagal.")
        simpan(arg.keluar); return 1

    # ------------------------------------------------------------- data
    subjudul("Mengindeks mask dan membaca CSV")
    indeks = {}
    for akar, _, berkas in os.walk(arg.mask):
        for b in berkas:
            if not b.lower().endswith(".png"):
                continue
            st = os.path.splitext(b)[0]
            if st.lower().endswith("_mask"):
                st = st[:-5]
            indeks.setdefault(st, os.path.join(akar, b))
    catat(f"  Mask terindeks: {len(indeks):,}")

    with open(arg.csv, "r", encoding="utf-8-sig", newline="") as fh:
        baris_csv = list(csv.DictReader(fh))
    for r in baris_csv:
        nama = os.path.basename(r.get("img_name") or r.get("path") or "")
        st = os.path.splitext(nama)[0]
        r["_batang"] = st
        r["_prefix"] = (st.split("_")[0] if "_" in st else st).upper()
        pre, bt = r["_prefix"], (r.get("nucleus_shape") or "").strip()
        if pre == "NEUTROPHIL":
            g = "F_tak_bersubtipe"
        elif pre == "BNE":
            g = ("A_sepakat_band" if bt in BENTUK_BAND else
                 "D_konflik_BNE_segmented" if bt in BENTUK_SEGMENTED else "E_lainnya")
        elif pre == "SNE":
            g = ("B_sepakat_segmented" if bt in BENTUK_SEGMENTED else
                 "C_konflik_SNE_band" if bt in BENTUK_BAND else "E_lainnya")
        else:
            g = "Z_non_neutrofil"
        r["_kelompok"] = g
    catat(f"  Baris CSV     : {len(baris_csv):,}")

    hit = Counter(r["_kelompok"] for r in baris_csv)
    harap = {"A_sepakat_band": 1555, "B_sepakat_segmented": 976,
             "C_konflik_SNE_band": 662, "D_konflik_BNE_segmented": 48,
             "E_lainnya": 38, "F_tak_bersubtipe": 50}
    catat(f"  Replikasi kelompok Fase 1: "
          f"{'LOLOS' if all(hit.get(k) == v for k, v in harap.items()) else 'PERIKSA'}")

    target = baris_csv
    if arg.sampel:
        rng0 = np.random.default_rng(SEED)
        pil = rng0.choice(len(target), min(arg.sampel, len(target)), replace=False)
        target = [target[i] for i in sorted(pil)]

    # ----------------------------------------------------------- cache
    tanda_cache = f"conn{8 if conn8 else 4}_lub{int(isi_lubang)}_pr{PRUNE:g}"
    jalur_cache = os.path.join(arg.keluar, f"cache_persistensi_{tanda_cache}.pkl.gz")
    H = None
    if os.path.exists(jalur_cache) and not arg.abaikan_cache:
        try:
            with gzip.open(jalur_cache, "rb") as fh:
                H = pickle.load(fh)
            catat(f"  Cache dimuat: {len(H):,} sel dari {jalur_cache}")
            if any(r["_batang"] not in H for r in target):
                catat("  Cache tidak lengkap untuk target ini, menghitung ulang.")
                H = None
        except Exception as e:
            catat(f"  Cache gagal dibaca ({e}), menghitung ulang.")
            H = None

    if H is None:
        subjudul("Menghitung persistensi")
        tugas = [(r["_batang"], indeks[r["_batang"]], isi_lubang, conn8)
                 for r in target if r["_batang"] in indeks]
        npk = arg.pekerja or max(1, (os.cpu_count() or 2) - 1)
        catat(f"  Sel: {len(tugas):,}   Pekerja: {npk}")
        t0 = datetime.now()
        hasil = []
        if npk > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=npk) as ex:
                for i, h in enumerate(ex.map(proses_satu, tugas, chunksize=32), 1):
                    hasil.append(h)
                    if i % 2000 == 0 or i == len(tugas):
                        print(f"    {i:,}/{len(tugas):,}", flush=True)
        else:
            for i, t in enumerate(tugas, 1):
                hasil.append(proses_satu(t))
                if i % 1000 == 0 or i == len(tugas):
                    print(f"    {i:,}/{len(tugas):,}", flush=True)
        catat(f"  Selesai {(datetime.now()-t0).total_seconds():.1f} detik")
        H = {h["kunci"]: h for h in hasil if not h.get("galat")}
        try:
            with gzip.open(jalur_cache, "wb", compresslevel=6) as fh:
                pickle.dump(H, fh, protocol=4)
            catat(f"  Cache disimpan: {jalur_cache}")
        except Exception as e:
            catat(f"  (cache gagal disimpan: {e})")

    catat(f"  Sel tersedia: {len(H):,}")

    baris_ada = [r for r in target if r["_batang"] in H]
    non_neu = [r for r in baris_ada if r["_kelompok"] == "Z_non_neutrofil"]
    neu = [r for r in baris_ada if r["_kelompok"] != "Z_non_neutrofil"]
    catat(f"  Non-neutrofil (kalibrasi): {len(non_neu):,}")
    catat(f"  Neutrofil (evaluasi)     : {len(neu):,}")

    def hitung(r, tau, rho, aturan="min"):
        h = H[r["_batang"]]
        return turunkan(h["r_lobus"], h["kejadian"], h["bertahan"], tau, rho, aturan)

    def vek(kel, tau, rho, split=None, indeks_out=1):
        out = []
        for r in baris_ada:
            if r["_kelompok"] != kel:
                continue
            if split and r.get("split") != split:
                continue
            out.append(hitung(r, tau, rho)[indeks_out])
        return np.array(out, float)

    # ================================================================== P1
    judul("P1. KALIBRASI PARAMETER DARI JUMLAH LOBUS - SEL NON-NEUTROFIL SAJA")
    catat("Jangkar: label nucleus_shape pada sel basofil, eosinofil, limfosit,")
    catat("monosit. Tidak satu pun sel ini masuk kelompok A atau B, jadi")
    catat("parameter tidak pernah disetel memakai data yang menguji ambang.")
    catat()
    catat("Harapan jumlah lobus: round 1 | indented 1 | bilobed 2 | multilobed >=3")
    catat("('irregular' dan 'unsegmented-band' dikecualikan: jumlah lobusnya")
    catat("memang tidak tunggal secara definisi.)")
    catat()

    kal = [r for r in non_neu
           if (r.get("nucleus_shape") or "").strip() in HARAP_LOBUS]
    catat(f"  Sel kalibrasi: {len(kal):,}")
    limfo = [r for r in non_neu if (r.get("label") or "").strip() == "Lymphocyte"]

    jalur_kal = os.path.join(arg.keluar, "kalibrasi_lobus.csv")
    hasil_kal = []
    with open(jalur_kal, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "rho", "skor_lobus", "limfosit_tak_pecah",
                    "mean_round", "mean_bilobed", "mean_multilobed"])
        for tau in TAU_GRID:
            for rho in RHO_GRID:
                cocok = 0
                per_bentuk = {k: [] for k in HARAP_LOBUS}
                for r in kal:
                    bt = (r.get("nucleus_shape") or "").strip()
                    nl = hitung(r, tau, rho)[2]
                    per_bentuk[bt].append(nl)
                    hrp = HARAP_LOBUS[bt]
                    cocok += (nl >= 3) if hrp == 3 else (nl == hrp)
                skor = cocok / max(1, len(kal))
                tp = np.mean([hitung(r, tau, rho)[3] == "tak_pernah_pecah"
                              for r in limfo]) if limfo else np.nan
                mr = np.mean(per_bentuk["unsegmented-round"]) if per_bentuk["unsegmented-round"] else np.nan
                mb = np.mean(per_bentuk["segmented-bilobed"]) if per_bentuk["segmented-bilobed"] else np.nan
                mm = np.mean(per_bentuk["segmented-multilobed"]) if per_bentuk["segmented-multilobed"] else np.nan
                w.writerow([tau, rho, f"{skor:.4f}", f"{tp:.4f}",
                            f"{mr:.3f}", f"{mb:.3f}", f"{mm:.3f}"])
                hasil_kal.append((tau, rho, skor, tp, mr, mb, mm))

    subjudul("Skor kecocokan jumlah lobus (baris tau, kolom rho)")
    baris = []
    for tau in TAU_GRID:
        r_ = [f"{tau:g}"]
        for rho in RHO_GRID:
            s = next(x[2] for x in hasil_kal if x[0] == tau and x[1] == rho)
            r_.append(f"{s:.3f}")
        baris.append(r_)
    cetak_tabel(["tau \\ rho"] + [f"{x:g}" for x in RHO_GRID], baris,
                rk=set(range(1, len(RHO_GRID) + 1)))

    terbaik = max(hasil_kal, key=lambda x: (x[2], x[3]))
    TAU, RHO = terbaik[0], terbaik[1]
    catat()
    catat(f"  >>> Terpilih: tau = {TAU:g}, rho = {RHO:g}")
    catat(f"      skor kecocokan lobus     : {terbaik[2]:.4f}")
    catat(f"      limfosit tak pernah pecah: {terbaik[3]*100:.1f}%")
    catat(f"      mean lobus round / bilobed / multilobed: "
          f"{terbaik[4]:.2f} / {terbaik[5]:.2f} / {terbaik[6]:.2f}")
    catat("      (harapan: 1,00 / 2,00 / >=3,00)")

    subjudul("Pembanding: parameter yang dipilih AUC di 2A-rev")
    lama = next((x for x in hasil_kal if x[0] == 0.1 and x[1] == 0.3), None)
    if lama:
        catat(f"  tau=0,1 rho=0,3 -> skor {lama[2]:.4f}, limfosit {lama[3]*100:.1f}%, "
              f"lobus {lama[4]:.2f}/{lama[5]:.2f}/{lama[6]:.2f}")
        catat("  Itulah yang ditolak kriteria baru.")

    # ================================================================== P2
    judul("P2. INVARIANSI AMBANG TERHADAP PARAMETER [HASIL KUNCI]")
    catat("Lobus palsu muncul di punggungan tebal, jadi bergabung pada saddle")
    catat("TINGGI. Operator 'min' mengabaikannya. Bridge ratio karena itu")
    catat("kebal terhadap tau secara struktural, sementara jumlah lobus tidak.")
    catat()
    jalur_inv = os.path.join(arg.keluar, "invariansi_ambang.csv")
    baris = []
    semua_amb = []
    with open(jalur_inv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "rho", "ambang_youden", "auc", "akurasi_1_3"])
        for tau in TAU_GRID:
            r_ = [f"{tau:g}"]
            for rho in RHO_GRID:
                A_ = vek("A_sepakat_band", tau, rho)
                B_ = vek("B_sepakat_segmented", tau, rho)
                th = ambang_youden(A_, B_)[0]
                u = auc_peringkat(A_, B_)
                ak = akurasi_pada(A_, B_, AMBANG_KLINIS)
                w.writerow([tau, rho, f"{th:.4f}", f"{u:.4f}", f"{ak:.4f}"])
                r_.append(f"{th:.4f}")
                if np.isfinite(th):
                    semua_amb.append(th)
            baris.append(r_)
    cetak_tabel(["tau \\ rho"] + [f"{x:g}" for x in RHO_GRID], baris,
                rk=set(range(1, len(RHO_GRID) + 1)))
    sa = np.array(semua_amb)
    catat()
    catat(f"  Ambang Youden di {sa.size} kombinasi parameter:")
    catat(f"    min {f(sa.min())}   maks {f(sa.max())}   "
          f"rentang {f(sa.max()-sa.min())}   sd {f(sa.std(ddof=1))}")
    catat(f"    nilai unik: {len(np.unique(np.round(sa, 4)))}")

    # ================================================================== P3
    judul(f"P3. HASIL PADA PARAMETER TERKALIBRASI (tau={TAU:g}, rho={RHO:g})")
    A = vek("A_sepakat_band", TAU, RHO)
    B = vek("B_sepakat_segmented", TAU, RHO)
    C = vek("C_konflik_SNE_band", TAU, RHO)
    D = vek("D_konflik_BNE_segmented", TAU, RHO)
    E = vek("E_lainnya", TAU, RHO)
    Fg = vek("F_tak_bersubtipe", TAU, RHO)

    auc = auc_peringkat(A, B)
    amb_y, j = ambang_youden(A, B)
    catat(f"  AUC (A vs B)            : {f(auc)}     [v1 0.9227 | 2A-rev 0.8933]")
    catat(f"  Ambang optimal Youden   : {f(amb_y)}   (J = {f(j)})")

    rng = np.random.default_rng(SEED)
    Ac, Bc = A[np.isfinite(A)], B[np.isfinite(B)]
    boot = np.array([ambang_youden(Ac[rng.integers(0, Ac.size, Ac.size)],
                                   Bc[rng.integers(0, Bc.size, Bc.size)])[0]
                     for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dalam = bool(lo <= AMBANG_KLINIS <= hi)
    catat(f"  Bootstrap {N_BOOTSTRAP}x median : {f(np.median(boot))}   [v1 0.3333]")
    catat(f"  IK 95%                  : [{f(lo)} - {f(hi)}]")
    catat(f"  >>> 1/3 di dalam IK 95% : {'YA' if dalam else 'TIDAK'}")
    catat(f"  Akurasi pada 1/3        : {akurasi_pada(A, B, AMBANG_KLINIS)*100:.2f}%"
          f"   [v1 90.04%]")

    subjudul("Generalisasi train -> test")
    At = vek("A_sepakat_band", TAU, RHO, "train")
    Bt = vek("B_sepakat_segmented", TAU, RHO, "train")
    Ae = vek("A_sepakat_band", TAU, RHO, "test")
    Be = vek("B_sepakat_segmented", TAU, RHO, "test")
    if At.size and Ae.size:
        amb_tr = ambang_youden(At, Bt)[0]
        catat(f"  Ambang dari TRAIN    : {f(amb_tr)}   (n={At.size}+{Bt.size})")
        catat(f"  AUC di TEST          : {f(auc_peringkat(Ae, Be))}   (n={Ae.size}+{Be.size})")
        catat(f"  Akurasi TEST @ambang : {akurasi_pada(Ae, Be, amb_tr)*100:.2f}%")
        catat(f"  Akurasi TEST @ 1/3   : {akurasi_pada(Ae, Be, AMBANG_KLINIS)*100:.2f}%")

    subjudul("Matriks konfusi pada ambang 1/3 (A dan B saja)")
    ab = int((Ac >= AMBANG_KLINIS).sum()); asg = Ac.size - ab
    bb = int((Bc >= AMBANG_KLINIS).sum()); bsg = Bc.size - bb
    cetak_tabel(["sumber", "-> band", "-> segmented", "n", "benar"],
                [["A (band)", f"{ab:,}", f"{asg:,}", f"{Ac.size:,}",
                  f"{ab/Ac.size*100:.1f}%"],
                 ["B (segmented)", f"{bb:,}", f"{bsg:,}", f"{Bc.size:,}",
                  f"{bsg/Bc.size*100:.1f}%"]], rk={1, 2, 3, 4})
    catat()
    catat(f"  Sensitivitas band      : {ab/Ac.size*100:.2f}%")
    catat(f"  Spesifisitas band      : {bsg/Bc.size*100:.2f}%")
    catat(f"  ASIMETRI GALAT         : band salah {asg/Ac.size*100:.2f}% vs "
          f"segmented salah {bb/Bc.size*100:.2f}%")

    # ================================================================== P4
    judul("P4. AUDIT EKOR B - APAKAH MASK MENJEMBATANI FILAMEN?")
    catat("Hipotesis: wbcsegmentor mengisi filamen kromatin selebar 1-2 piksel,")
    catat("sehingga leher terukur lebih tebal dari aslinya. Bila benar, r_pisah")
    catat("sel B yang salah klasifikasi akan MENUMPUK di kisaran sempit")
    catat("(sekitar 2-4 piksel), bukan tersebar merata.")
    catat()

    rekB = []
    for r in baris_ada:
        if r["_kelompok"] != "B_sepakat_segmented":
            continue
        rp, br, nl, st, pos = hitung(r, TAU, RHO)
        rekB.append((r, rp, br, nl, st))
    salah = [x for x in rekB if np.isfinite(x[2]) and x[2] >= AMBANG_KLINIS]
    benar = [x for x in rekB if np.isfinite(x[2]) and x[2] < AMBANG_KLINIS]

    rekA = []
    for r in baris_ada:
        if r["_kelompok"] != "A_sepakat_band":
            continue
        rp, br, nl, st, pos = hitung(r, TAU, RHO)
        rekA.append((r, rp, br, nl, st))
    A_band = [x for x in rekA if np.isfinite(x[2]) and x[2] >= AMBANG_KLINIS]

    def stat_rp(rek):
        v = np.array([x[1] for x in rek if np.isfinite(x[1])], float)
        if v.size == 0:
            return ["-"] * 6
        return [v.size, f(v.mean(), 2), f(np.percentile(v, 25), 2),
                f(np.median(v), 2), f(np.percentile(v, 75), 2),
                f"{((v >= 2) & (v <= 4)).mean()*100:.1f}%"]

    cetak_tabel(["kelompok", "n", "mean rp", "q25", "median", "q75",
                 "rp dalam 2-4 px"],
                [["B salah -> band"] + stat_rp(salah),
                 ["B benar -> segmented"] + stat_rp(benar),
                 ["A benar -> band"] + stat_rp(A_band)],
                rk={1, 2, 3, 4, 5, 6})
    catat()
    catat("  Bila 'B salah' menumpuk di 2-4 px jauh lebih rapat daripada")
    catat("  'A benar', hipotesis penjembatanan mask didukung.")

    subjudul("Apakah lobusnya terdeteksi tapi lehernya terukur tebal?")
    for nama, rek in (("B salah -> band", salah), ("B benar -> segmented", benar)):
        if not rek:
            continue
        nl = np.array([x[3] for x in rek], float)
        rl = np.array([x[1] / (H[x[0]["_batang"]]["r_lobus"]) for x in rek
                       if np.isfinite(x[1])], float)
        luas = np.array([H[x[0]["_batang"]]["luas_nukleus"] for x in rek], float)
        rlob = np.array([H[x[0]["_batang"]]["r_lobus"] for x in rek], float)
        catat(f"  {nama:<24} n={len(rek):>4}  mean n_lobus {nl.mean():.2f}  "
              f">=2 lobus {(nl >= 2).mean()*100:5.1f}%  "
              f"mean r_lobus {rlob.mean():.2f}  mean luas {luas.mean():,.0f}")
    catat()
    catat("  Bila 'B salah' tetap punya >=2 lobus, berarti lobusnya MEMANG")
    catat("  terdeteksi dan hanya lehernya yang terukur terlalu tebal.")
    catat("  Itu menunjuk resolusi mask, bukan kegagalan metode.")

    subjudul("Rincian ekor B menurut nucleus_shape")
    baris = []
    for bt in ("segmented-bilobed", "segmented-multilobed"):
        tot = [x for x in rekB if (x[0].get("nucleus_shape") or "").strip() == bt]
        sal = [x for x in salah if (x[0].get("nucleus_shape") or "").strip() == bt]
        if not tot:
            continue
        baris.append([bt, len(tot), len(sal), f"{len(sal)/len(tot)*100:.1f}%"])
    cetak_tabel(["nucleus_shape", "n di B", "salah -> band", "persen"],
                baris, rk={1, 2, 3})

    jalur_audit = os.path.join(arg.keluar, "audit_ekor_B.csv")
    with open(jalur_audit, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img_name", "nucleus_shape", "split", "r_lobus", "r_pisah",
                    "bridge_ratio", "n_lobus", "status", "luas_nukleus",
                    "komponen_mentah", "piks_lubang"])
        for (r, rp, br, nl, st) in sorted(salah, key=lambda x: -x[2]):
            h = H[r["_batang"]]
            w.writerow([r.get("img_name", ""), r.get("nucleus_shape", ""),
                        r.get("split", ""), f"{h['r_lobus']:.4f}",
                        "" if not np.isfinite(rp) else f"{rp:.4f}",
                        f"{br:.4f}", nl, st, h["luas_nukleus"],
                        h["komponen_mentah"], h["piks_lubang"]])
    catat()
    catat(f"  Daftar lengkap {len(salah):,} sel: {jalur_audit}")

    # ================================================================== P5
    judul("P5. KONTROL, LOBUS, DAN KELOMPOK KONFLIK")
    subjudul("Kontrol lintas kelas sel")
    baris = []
    for lab in sorted({r.get("label", "") for r in baris_ada if r.get("label")}):
        sub = [r for r in baris_ada if r.get("label") == lab]
        brs = np.array([hitung(r, TAU, RHO)[1] for r in sub], float)
        nls = np.array([hitung(r, TAU, RHO)[2] for r in sub], float)
        tp = np.mean([hitung(r, TAU, RHO)[3] == "tak_pernah_pecah" for r in sub])
        baris.append([lab, len(sub), f(np.median(brs[np.isfinite(brs)])),
                      f"{tp*100:.1f}%", f"{nls.mean():.2f}"])
    cetak_tabel(["kelas", "n", "median br", "tak pernah pecah", "rata n_lobus"],
                baris, rk={1, 2, 3, 4})

    subjudul("Jumlah lobus per nucleus_shape (T2 metode lama: 81,4% gagal)")
    baris = []
    for bt in ["unsegmented-round", "unsegmented-indented", "irregular",
               "unsegmented-band", "segmented-bilobed", "segmented-multilobed"]:
        sub = [r for r in baris_ada if (r.get("nucleus_shape") or "").strip() == bt]
        if not sub:
            continue
        nls = np.array([hitung(r, TAU, RHO)[2] for r in sub], float)
        brs = np.array([hitung(r, TAU, RHO)[1] for r in sub], float)
        baris.append([bt, len(sub), f"{nls.mean():.3f}", f"{np.median(nls):.0f}",
                      f"{(nls <= 1).mean()*100:.1f}%",
                      f(np.median(brs[np.isfinite(brs)]))])
    cetak_tabel(["nucleus_shape", "n", "mean n_lobus", "median",
                 "hanya 1 lobus", "median br"], baris, rk={1, 2, 3, 4, 5})

    subjudul("Kelompok konflik")
    baris = []
    for nama, v in (("A_sepakat_band", A), ("B_sepakat_segmented", B),
                    ("C_konflik_SNE_band", C), ("D_konflik_BNE_segmented", D),
                    ("E_lainnya", E), ("F_tak_bersubtipe", Fg)):
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            continue
        pb = float((vv >= AMBANG_KLINIS).mean())
        baris.append([nama, vv.size, f(np.percentile(vv, 25)),
                      f(np.median(vv)), f(np.percentile(vv, 75)),
                      f"{pb:.3f}", f"{entropi_biner(pb):.3f}"])
    cetak_tabel(["kelompok", "n", "q25", "median", "q75", "p(band)", "entropi"],
                baris, rk=set(range(1, 7)))

    mA, mB = np.median(A[np.isfinite(A)]), np.median(B[np.isfinite(B)])
    catat()
    catat("  Posisi pada sumbu A->B (0% = band murni, 100% = segmented murni):")
    for nama, v in (("C_konflik", C), ("D_konflik", D), ("F_tak_bersubtipe", Fg)):
        vv = v[np.isfinite(v)]
        if vv.size and mB != mA:
            catat(f"    {nama:<18} {(np.median(vv)-mA)/(mB-mA)*100:6.1f}%")
    catat("  Pembanding Fase 1 pada bridge ratio: C = 33,9%")
    catat("  (dokumen konteks menulis 54,2%, tetapi itu diukur pada indeks LDA,")
    catat("   bukan bridge ratio - dua sumbu berbeda)")

    subjudul("Kepadatan di zona perbatasan")
    baris = []
    for pita in ((0.28, 0.38), (0.23, 0.43)):
        r_ = [f"[{pita[0]:.2f} - {pita[1]:.2f}]"]
        for v in (A, B, C, D):
            vv = v[np.isfinite(v)]
            r_.append(f"{((vv >= pita[0]) & (vv <= pita[1])).mean()*100:.1f}%"
                      if vv.size else "-")
        baris.append(r_)
    cetak_tabel(["pita", "A", "B", "C", "D"], baris, rk={1, 2, 3, 4})

    # ================================================================== P6
    judul("P6. TRIANGULASI DENGAN FASE 1")
    v1 = {}
    if os.path.exists(arg.v1):
        with open(arg.v1, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                v1[os.path.splitext(os.path.basename(r.get("img_name", "")))[0]] = r
    if v1:
        from scipy.stats import spearmanr, pearsonr
        a1, a2, li, lb = [], [], [], []
        for r in neu:
            vr = v1.get(r["_batang"])
            if not vr:
                continue
            br = hitung(r, TAU, RHO)[1]
            try:
                b1 = float(vr.get("bridge_ratio", ""))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(br) and np.isfinite(b1)):
                continue
            a1.append(b1); a2.append(br)
            try:
                ix = float(vr.get("indeks_segmentasi", ""))
                if np.isfinite(ix):
                    li.append(ix); lb.append(br)
            except (TypeError, ValueError):
                pass
        a1, a2 = np.array(a1), np.array(a2)
        catat(f"  n terbandingkan         : {a1.size:,}")
        catat(f"  Pearson  (v1, final)    : {f(pearsonr(a1, a2)[0])}")
        catat(f"  Spearman (v1, final)    : {f(spearmanr(a1, a2).statistic)}")
        pindah = int(((a1 >= AMBANG_KLINIS) != (a2 >= AMBANG_KLINIS)).sum())
        catat(f"  Berpindah sisi 1/3      : {pindah:,} ({pindah/a1.size*100:.2f}%)")
        if li:
            catat(f"  Spearman(br, indeks LDA): "
                  f"{f(spearmanr(np.array(lb), np.array(li)).statistic)}"
                  f"   [v1: -0.7403]")

    # ---------------------------------------------------------- simpan
    jalur = os.path.join(arg.keluar, "bridge_ratio_final.csv")
    with open(jalur, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img_name", "prefix", "kelompok", "label", "nucleus_shape",
                    "split", "r_lobus", "r_pisah", "bridge_ratio", "n_lobus",
                    "status_topologi", "luas_nukleus", "piks_lubang",
                    "komponen_mentah", "tau", "rho"])
        for r in baris_ada:
            h = H[r["_batang"]]
            rp, br, nl, st, _ = hitung(r, TAU, RHO)
            w.writerow([r.get("img_name", ""), r["_prefix"], r["_kelompok"],
                        r.get("label", ""), r.get("nucleus_shape", ""),
                        r.get("split", ""), f"{h['r_lobus']:.6f}",
                        "" if not np.isfinite(rp) else f"{rp:.6f}",
                        f"{br:.6f}", nl, st, h["luas_nukleus"],
                        h["piks_lubang"], h["komponen_mentah"],
                        f"{TAU:g}", f"{RHO:g}"])
    catat()
    catat(f"  Tersimpan: {jalur}")

    montase(arg, H, salah, benar, A_band, TAU, RHO, isi_lubang, indeks)
    grafik(arg.keluar, A, B, C, D, Fg, TAU, RHO)

    # ---------------------------------------------------------- ringkasan
    judul("RINGKASAN")
    catat(f"  Parameter terkalibrasi  : tau={TAU:g}, rho={RHO:g}")
    catat(f"    dipilih dari kecocokan lobus pada {len(kal):,} sel NON-neutrofil")
    catat(f"  AUC (A vs B)            : {f(auc)}")
    catat(f"  Ambang bootstrap        : {f(np.median(boot))}")
    catat(f"  IK 95%                  : [{f(lo)} - {f(hi)}]")
    catat(f"  1/3 di dalam IK         : {'YA' if dalam else 'TIDAK'}")
    catat(f"  Akurasi @1/3            : {akurasi_pada(A, B, AMBANG_KLINIS)*100:.2f}%")
    catat(f"  Rentang ambang di {sa.size} kombinasi parameter: {f(sa.max()-sa.min())}")
    catat(f"  Asimetri galat          : A salah {asg/Ac.size*100:.1f}% vs "
          f"B salah {bb/Bc.size*100:.1f}%")
    catat()
    catat("Kirim seluruh keluaran ini, dan periksa montase secara visual.")
    garis("=")
    simpan(arg.keluar)
    return 0


# ==========================================================================
# MONTASE VISUAL
# ==========================================================================

def montase(arg, H, salah, benar, A_band, TAU, RHO, isi_lubang, indeks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        from scipy.ndimage import binary_fill_holes
    except Exception:
        catat("  (montase dilewati: pustaka tidak tersedia)")
        return

    def gambar(rek, nama_berkas, judul_gbr):
        n = min(arg.n_montase, len(rek))
        if n == 0:
            return
        kol = 6
        bar = int(np.ceil(n / kol))
        fig, axs = plt.subplots(bar, kol, figsize=(kol * 2.1, bar * 2.3))
        axs = np.atleast_1d(axs).ravel()
        for ax in axs:
            ax.axis("off")
        for i, (r, rp, br, nl, st) in enumerate(rek[:n]):
            h = H[r["_batang"]]
            jalur = indeks.get(r["_batang"])
            if not jalur:
                continue
            try:
                with Image.open(jalur) as im:
                    m = np.array(im)
                if m.ndim == 3:
                    m = m[..., 0]
                nuk = (m == NILAI_NUKLEUS)
                if isi_lubang:
                    nuk = binary_fill_holes(nuk)
                ys, xs = np.nonzero(nuk)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                pot = nuk[y0:y1, x0:x1]
                axs[i].imshow(pot, cmap="gray_r", interpolation="nearest")
                axs[i].set_title(f"br={br:.2f} rp={rp:.1f}\nlobus={nl}",
                                 fontsize=7)
            except Exception:
                continue
        fig.suptitle(judul_gbr, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        j = os.path.join(arg.keluar, nama_berkas)
        fig.savefig(j, dpi=130)
        plt.close(fig)
        catat(f"  Montase: {j}")

    catat()
    salah_urut = sorted(salah, key=lambda x: -x[2])
    benar_urut = sorted(benar, key=lambda x: x[2])
    gambar(salah_urut, "montase_B_salah_disebut_band.png",
           f"B (SNE+segmented) yang geometri sebut BAND - br tertinggi "
           f"(tau={TAU:g}, rho={RHO:g})")
    gambar(benar_urut, "montase_B_benar_segmented.png",
           "B (SNE+segmented) yang geometri sebut SEGMENTED - br terendah")
    gambar(sorted(A_band, key=lambda x: -x[2]), "montase_A_benar_band.png",
           "A (BNE+band) yang geometri sebut BAND - pembanding")


def grafik(keluar, A, B, C, D, Fg, TAU, RHO):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for nama, v, w in (("A sepakat band", A, 2.0),
                           ("B sepakat segmented", B, 2.0),
                           ("C konflik SNE->band", C, 2.4),
                           ("D konflik BNE->segmented", D, 1.3),
                           ("F tanpa subtipe", Fg, 1.3)):
            vv = v[np.isfinite(v)]
            if vv.size < 5:
                continue
            ax.hist(vv, bins=60, range=(0, 1), density=True, histtype="step",
                    linewidth=w, label=f"{nama} (n={vv.size})")
        ax.axvline(1 / 3, color="k", ls="--", lw=1.5, label="ambang klinis 1/3")
        ax.set_xlabel(f"bridge ratio (tau={TAU:g}, rho={RHO:g}, terkalibrasi lobus)")
        ax.set_ylabel("kepadatan")
        ax.set_title("Sebaran bridge ratio pada parameter terkalibrasi")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(keluar, "F2B_sebaran.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass


def simpan(keluar):
    try:
        os.makedirs(keluar, exist_ok=True)
        j = os.path.join(keluar, "F2B_laporan.txt")
        with open(j, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_penampung))
        print()
        print(f"Laporan tersimpan: {j}")
    except Exception as e:
        print(f"(gagal menyimpan: {e})")


if __name__ == "__main__":
    sys.exit(main())
