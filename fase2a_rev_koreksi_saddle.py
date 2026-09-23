# -*- coding: utf-8 -*-
"""
===========================================================================
FASE 2A-REV - Koreksi aturan pemilihan saddle
Proyek skripsi WBCAtt+ - Muhammad Fadhil Mulyadi
===========================================================================

KOREKSI TERHADAP JALANAN SEBELUMNYA
-----------------------------------
Skrip 2A sebelumnya memakai r_pisah = saddle TERTINGGI. Itu salah.

Superlevel set {DT > r} terpecah begitu leher TERSEMPIT hilang.
Untuk nukleus berantai L1 -(leher 2)- L2 -(leher 5)- L3:
    r = 1,9  -> leher 2 masih ada  -> 1 komponen
    r = 2,1  -> leher 2 hilang     -> 2 komponen
Jadi radius erosi minimum yang memecah = 2 = saddle TERENDAH.

Aturan 'min' juga yang benar secara klinis: kriteria band vs segmented
menanyakan ada tidaknya SATU filamen tipis. Satu leher sempit sudah
cukup. Dua alasan independen menunjuk operator yang sama.

TAMBAHAN PADA REVISI INI
------------------------
1. Kedua aturan (min dan max) dihitung berdampingan sebagai ablasi,
   supaya koreksinya terdokumentasi, bukan sekadar diperbaiki diam-diam.
2. Filter lobus RELATIF (rho): sebuah lobus dianggap nyata bila
   puncaknya >= rho * r_lobus. Filter absolut tau saja tidak cukup,
   karena ukuran nukleus bervariasi.
3. Pemilihan parameter (tau, rho) HANYA di split train, lalu dievaluasi
   di test. Menyetel parameter untuk memaksimalkan AUC lalu melaporkan
   AUC yang sama adalah overfitting.
4. Uji sanitas otomatis pada nukleus sintetis dengan jawaban yang sudah
   diketahui, dijalankan sebelum data asli disentuh.

CARA PAKAI
----------
  python fase2a_rev_koreksi_saddle.py --sampel 300     (uji cepat)
  python fase2a_rev_koreksi_saddle.py                  (penuh, ~4 menit)
===========================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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
    "keluar": os.path.join(BASIS, "hasil_fase2a_rev"),
}

NILAI_NUKLEUS = 2
PRUNE = 0.05               # kejadian di bawah ini dibuang di pekerja
TAU_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
RHO_GRID = [0.0, 0.10, 0.20, 0.30, 0.40]
AMBANG_KLINIS = 1.0 / 3.0
N_BOOTSTRAP = 2000
SEED = 20260917

BENTUK_SEGMENTED = {"segmented-bilobed", "segmented-multilobed"}
BENTUK_BAND = {"unsegmented-band"}

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


def ringkas(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, q25=np.nan, med=np.nan, q75=np.nan)
    return dict(n=x.size, mean=x.mean(), q25=np.percentile(x, 25),
                med=np.median(x), q75=np.percentile(x, 75))


# ==========================================================================
# INTI
# ==========================================================================

def persistensi_dt(dt, conn8=True):
    """Pohon penggabungan superlevel set.

    kejadian: (saddle, persistence_yang_mati, puncak_yang_mati, puncak_utama)
    bertahan: puncak komponen yang tidak pernah bergabung
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
            kejadian.append((float(v), float(puncak[r] - v),
                             float(puncak[r]), float(puncak[utama])))
            induk[r] = utama
        induk[p] = utama

    akar_akhir = set()
    for p in urut:
        akar_akhir.add(cari(p))
    bertahan = [float(puncak[r]) for r in akar_akhir]
    return float(datar.max()), kejadian, bertahan


def turunkan(r_lobus, kejadian, bertahan, tau, rho=0.0, aturan="min"):
    """r_pisah, bridge_ratio, jumlah lobus, status.

    Sebuah lobus dianggap nyata bila persistence >= tau DAN puncak >= rho*r_lobus.

    Aturan 'min' (BENAR): r_pisah = saddle TERENDAH di antara penggabungan
    signifikan. Itulah radius erosi minimum yang memecah nukleus, dan itulah
    leher tersempit yang ditanyakan kriteria klinis.

    Aturan 'max' (SALAH, disimpan untuk ablasi): saddle tertinggi.
    """
    amb_p = rho * r_lobus
    sig_g = [(s, pk_mati) for (s, pers, pk_mati, _) in kejadian
             if pers >= tau and pk_mati >= amb_p]
    sig_b = [pk for pk in bertahan if pk >= tau and pk >= amb_p]
    n_lobus = len(sig_g) + len(sig_b)

    if len(sig_b) >= 2:
        return 0.0, 0.0, n_lobus, "terpisah_di_nol"
    if sig_g:
        s = [x[0] for x in sig_g]
        r_pisah = min(s) if aturan == "min" else max(s)
        br = r_pisah / r_lobus if r_lobus > 0 else np.nan
        return r_pisah, br, n_lobus, "normal"
    return np.nan, 1.0, max(n_lobus, 1), "tak_pernah_pecah"


# ==========================================================================
# UJI SANITAS PADA NUKLEUS SINTETIS
# ==========================================================================

def uji_sanitas():
    """Bentuk dengan jawaban yang sudah diketahui. Dijalankan sebelum
    data asli disentuh. Kalau gagal di sini, jangan percaya hasil apa pun."""
    from scipy.ndimage import distance_transform_edt

    def cakram(kanvas, cy, cx, r):
        yy, xx = np.ogrid[:kanvas.shape[0], :kanvas.shape[1]]
        kanvas[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True

    def batang(kanvas, cy, x0, x1, setengah):
        kanvas[cy - setengah:cy + setengah + 1, x0:x1] = True

    kasus = []

    # 1. Cakram tunggal konveks -> tidak boleh pernah pecah
    k = np.zeros((120, 120), bool); cakram(k, 60, 60, 30)
    kasus.append(("cakram konveks", k, "tak_pernah_pecah", None))

    # 2. Halter: dua cakram r=20 dihubungkan leher setengah-lebar 4
    k = np.zeros((120, 200), bool)
    cakram(k, 60, 50, 20); cakram(k, 60, 150, 20); batang(k, 60, 50, 150, 4)
    kasus.append(("halter leher 4 / lobus 20", k, "normal", 4.0 / 20.0))

    # 3. Rantai tiga lobus, leher 3 dan 8 -> harus memilih 3, bukan 8
    k = np.zeros((140, 300), bool)
    cakram(k, 70, 50, 22); cakram(k, 70, 150, 22); cakram(k, 70, 250, 22)
    batang(k, 70, 50, 150, 3)     # leher sempit
    batang(k, 70, 150, 250, 8)    # leher lebar
    kasus.append(("rantai leher 3 dan 8 / lobus 22", k, "normal", 3.0 / 22.0))

    baris = []
    lolos = True
    for nama, k, st_harap, br_harap in kasus:
        pad = np.zeros((k.shape[0] + 2, k.shape[1] + 2), bool)
        pad[1:-1, 1:-1] = k
        dt = distance_transform_edt(pad)
        r_lobus, kej, bert = persistensi_dt(dt)
        _, br_min, nl, st = turunkan(r_lobus, kej, bert, tau=1.0, rho=0.0,
                                     aturan="min")
        _, br_max, _, _ = turunkan(r_lobus, kej, bert, tau=1.0, rho=0.0,
                                   aturan="max")
        ok = (st == st_harap)
        if br_harap is not None:
            ok = ok and abs(br_min - br_harap) < 0.06
        lolos = lolos and ok
        baris.append([nama, f(r_lobus, 2), st,
                      f(br_min, 4), f(br_harap, 4) if br_harap else "-",
                      f(br_max, 4), nl, "OK" if ok else "GAGAL"])

    cetak_tabel(["kasus sintetis", "r_lobus", "status", "br(min)",
                 "harapan", "br(max)", "n_lobus", ""], baris,
                rk={1, 3, 4, 5, 6})
    catat()
    catat("  Kasus 3 adalah uji pembeda. Jawaban benar 3/22 = 0,1364.")
    catat("  Aturan 'max' akan menjawab 8/22 = 0,3636 - itulah bug jalanan lalu.")
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
        "komponen_mentah": komp, "n_kejadian_total": len(kej),
        "kejadian": [t for t in kej if t[1] >= PRUNE],
        "bertahan": bert,
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
    p.add_argument("--hanya-neutrofil", action="store_true")
    p.add_argument("--tanpa-isi-lubang", action="store_true")
    p.add_argument("--conn4", action="store_true")
    p.add_argument("--pekerja", type=int, default=0)
    arg = p.parse_args()

    isi_lubang = not arg.tanpa_isi_lubang
    conn8 = not arg.conn4
    os.makedirs(arg.keluar, exist_ok=True)

    judul("FASE 2A-REV - KOREKSI ATURAN SADDLE (min, bukan max)")
    catat(f"Waktu        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    catat(f"Konektivitas : {'8' if conn8 else '4'}   Isi lubang: {'ya' if isi_lubang else 'tidak'}")

    judul("S0. UJI SANITAS PADA NUKLEUS SINTETIS")
    catat("Dijalankan sebelum data asli disentuh.")
    catat()
    if not uji_sanitas():
        catat()
        catat("[BERHENTI] Uji sanitas gagal. Tidak ada gunanya melanjutkan.")
        simpan(arg.keluar)
        return 1

    # ------------------------------------------------------------- indeks
    subjudul("Mengindeks mask dan membaca CSV")
    if not os.path.isdir(arg.mask):
        catat(f"[GAGAL] Folder mask tidak ada: {arg.mask}")
        simpan(arg.keluar); return 1
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
    catat(f"  Baris CSV     : {len(baris_csv):,}")

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

    hit = Counter(r["_kelompok"] for r in baris_csv)
    harap = {"A_sepakat_band": 1555, "B_sepakat_segmented": 976,
             "C_konflik_SNE_band": 662, "D_konflik_BNE_segmented": 48,
             "E_lainnya": 38, "F_tak_bersubtipe": 50}
    ok_kel = all(hit.get(k) == v for k, v in harap.items())
    catat(f"  Replikasi kelompok Fase 1: {'LOLOS' if ok_kel else 'PERIKSA'}")

    target = baris_csv
    if arg.hanya_neutrofil:
        target = [r for r in target if r["_kelompok"] != "Z_non_neutrofil"]
    if arg.sampel:
        rng0 = np.random.default_rng(SEED)
        pil = rng0.choice(len(target), min(arg.sampel, len(target)), replace=False)
        target = [target[i] for i in sorted(pil)]

    tugas = [(r["_batang"], indeks[r["_batang"]], isi_lubang, conn8)
             for r in target if r["_batang"] in indeks]
    catat(f"  Sel diproses  : {len(tugas):,}   (tanpa mask: {len(target)-len(tugas):,})")

    # ---------------------------------------------------------- eksekusi
    subjudul("Menghitung")
    npk = arg.pekerja or max(1, (os.cpu_count() or 2) - 1)
    catat(f"  Pekerja: {npk}")
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
    catat(f"  Berhasil: {len(H):,}")

    baris_neu = [r for r in target
                 if r["_kelompok"] != "Z_non_neutrofil" and r["_batang"] in H]

    def vek(kel, tau, rho, aturan, split=None):
        out = []
        for r in baris_neu:
            if r["_kelompok"] != kel:
                continue
            if split and r.get("split") != split:
                continue
            h = H[r["_batang"]]
            out.append(turunkan(h["r_lobus"], h["kejadian"], h["bertahan"],
                                tau, rho, aturan)[1])
        return np.array(out, float)

    # ================================================================== K0
    judul("K0. MEKANISME BUG - JUMLAH PENGGABUNGAN SIGNIFIKAN PER KELOMPOK")
    catat("Aturan 'max' hanya menyimpang bila sebuah sel punya LEBIH DARI SATU")
    catat("saddle. Kelompok dengan banyak saddle adalah yang paling rusak.")
    catat()
    baris = []
    for kel in ("A_sepakat_band", "B_sepakat_segmented", "C_konflik_SNE_band",
                "D_konflik_BNE_segmented", "F_tak_bersubtipe"):
        ns = []
        for r in baris_neu:
            if r["_kelompok"] != kel:
                continue
            h = H[r["_batang"]]
            amb = 0.0
            ns.append(sum(1 for (s, pers, pkm, _) in h["kejadian"]
                          if pers >= 1.0 and pkm >= amb))
        if not ns:
            continue
        ns = np.array(ns, float)
        baris.append([kel, len(ns), f"{ns.mean():.2f}", f"{np.median(ns):.0f}",
                      f"{(ns >= 2).mean()*100:.1f}%"])
    cetak_tabel(["kelompok", "n", "mean saddle", "median", ">=2 saddle"],
                baris, rk={1, 2, 3, 4})

    # ================================================================== K1
    judul("K1. ABLASI ATURAN SADDLE  [KOREKSI TERDOKUMENTASI]")
    baris = []
    for tau in TAU_GRID:
        r_ = [f"{tau:g}"]
        for aturan in ("min", "max"):
            A_ = vek("A_sepakat_band", tau, 0.0, aturan)
            B_ = vek("B_sepakat_segmented", tau, 0.0, aturan)
            r_ += [f(auc_peringkat(A_, B_)), f(ambang_youden(A_, B_)[0]),
                   f"{akurasi_pada(A_, B_, AMBANG_KLINIS)*100:.1f}%"]
        baris.append(r_)
    cetak_tabel(["tau", "AUC min", "ambang min", "akur1/3 min",
                 "AUC max", "ambang max", "akur1/3 max"], baris,
                rk=set(range(1, 7)))
    catat("  Kolom 'max' mereproduksi jalanan yang salah sebagai pembanding.")

    # ================================================================== K2
    judul("K2. PEMILIHAN PARAMETER DI SPLIT TRAIN SAJA")
    catat("AUC dihitung hanya pada A dan B di split train. Test tidak disentuh.")
    catat()
    baris = []
    terbaik = (None, None, -1)
    for tau in TAU_GRID:
        r_ = [f"{tau:g}"]
        for rho in RHO_GRID:
            At = vek("A_sepakat_band", tau, rho, "min", "train")
            Bt = vek("B_sepakat_segmented", tau, rho, "min", "train")
            u = auc_peringkat(At, Bt)
            r_.append(f(u, 3))
            if np.isfinite(u) and u > terbaik[2]:
                terbaik = (tau, rho, u)
        baris.append(r_)
    cetak_tabel(["tau \\ rho"] + [f"{x:g}" for x in RHO_GRID], baris,
                rk=set(range(1, len(RHO_GRID) + 1)))
    TAU, RHO, auc_tr = terbaik
    catat()
    catat(f"  >>> Terbaik di TRAIN: tau = {TAU:g}, rho = {RHO:g}  (AUC train {f(auc_tr)})")

    # ================================================================== K3
    judul(f"K3. HASIL PADA PARAMETER TERPILIH (tau={TAU:g}, rho={RHO:g})")
    A = vek("A_sepakat_band", TAU, RHO, "min")
    B = vek("B_sepakat_segmented", TAU, RHO, "min")
    C = vek("C_konflik_SNE_band", TAU, RHO, "min")
    D = vek("D_konflik_BNE_segmented", TAU, RHO, "min")
    E = vek("E_lainnya", TAU, RHO, "min")
    F_ = vek("F_tak_bersubtipe", TAU, RHO, "min")

    auc = auc_peringkat(A, B)
    amb_y, j = ambang_youden(A, B)
    catat(f"  AUC (A vs B, seluruh split) : {f(auc)}     [Fase 1 v1: 0.9227]")
    catat(f"  Ambang optimal Youden       : {f(amb_y)}   (J = {f(j)})")

    rng = np.random.default_rng(SEED)
    Ac, Bc = A[np.isfinite(A)], B[np.isfinite(B)]
    boot = np.array([ambang_youden(Ac[rng.integers(0, Ac.size, Ac.size)],
                                   Bc[rng.integers(0, Bc.size, Bc.size)])[0]
                     for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dalam = bool(lo <= AMBANG_KLINIS <= hi)
    catat(f"  Bootstrap {N_BOOTSTRAP}x median     : {f(np.median(boot))}   [v1: 0.3333]")
    catat(f"  IK 95% bootstrap            : [{f(lo)} - {f(hi)}]   [v1: 0.3192-0.3671]")
    catat(f"  Ambang klinis 1/3           : {f(AMBANG_KLINIS)}")
    catat(f"  >>> 1/3 di dalam IK 95%     : {'YA' if dalam else 'TIDAK'}")
    catat(f"  Akurasi pada 1/3            : {akurasi_pada(A, B, AMBANG_KLINIS)*100:.2f}%"
          f"   [v1: 90.04%]")

    subjudul("Generalisasi train -> test")
    At = vek("A_sepakat_band", TAU, RHO, "min", "train")
    Bt = vek("B_sepakat_segmented", TAU, RHO, "min", "train")
    Ae = vek("A_sepakat_band", TAU, RHO, "min", "test")
    Be = vek("B_sepakat_segmented", TAU, RHO, "min", "test")
    if At.size and Ae.size:
        amb_tr = ambang_youden(At, Bt)[0]
        catat(f"  Ambang dari TRAIN     : {f(amb_tr)}   (n={At.size}+{Bt.size})")
        catat(f"  AUC di TEST           : {f(auc_peringkat(Ae, Be))}   (n={Ae.size}+{Be.size})")
        catat(f"  Akurasi TEST @ambang  : {akurasi_pada(Ae, Be, amb_tr)*100:.2f}%")
        catat(f"  Akurasi TEST @ 1/3    : {akurasi_pada(Ae, Be, AMBANG_KLINIS)*100:.2f}%")

    # ================================================================== K4
    judul("K4. SEBARAN DAN AMBIGUITAS PER KELOMPOK")
    baris = []
    for nama, v in (("A_sepakat_band", A), ("B_sepakat_segmented", B),
                    ("C_konflik_SNE_band", C), ("D_konflik_BNE_segmented", D),
                    ("E_lainnya", E), ("F_tak_bersubtipe", F_)):
        s = ringkas(v)
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            continue
        pb = float((vv >= AMBANG_KLINIS).mean())
        baris.append([nama, s["n"], f(s["q25"]), f(s["med"]), f(s["q75"]),
                      f"{pb:.3f}", f"{entropi_biner(pb):.3f}"])
    cetak_tabel(["kelompok", "n", "q25", "median", "q75", "p(band)", "entropi"],
                baris, rk=set(range(1, 7)))
    catat("  1,000 bit = ambiguitas maksimum")

    mA, mB = np.median(A[np.isfinite(A)]), np.median(B[np.isfinite(B)])
    catat()
    for nama, v in (("C_konflik", C), ("D_konflik", D), ("F_tak_bersubtipe", F_)):
        vv = v[np.isfinite(v)]
        if vv.size and mB != mA:
            catat(f"  Posisi {nama:<18} pada sumbu A->B : "
                  f"{(np.median(vv)-mA)/(mB-mA)*100:6.1f}%    [v1 untuk C: 54,2% LDA]")

    subjudul("Kepadatan di zona perbatasan")
    baris = []
    for pita in ((0.28, 0.38), (0.23, 0.43)):
        r_ = [f"[{pita[0]:.2f} - {pita[1]:.2f}]"]
        for v in (A, B, C):
            vv = v[np.isfinite(v)]
            r_.append(f"{((vv >= pita[0]) & (vv <= pita[1])).mean()*100:.1f}%"
                      if vv.size else "-")
        baris.append(r_)
    cetak_tabel(["pita", "A", "B", "C"], baris, rk={1, 2, 3})

    # ================================================================== K5
    judul("K5. KONTROL LINTAS KELAS DAN LOBUS vs T2")
    baris = []
    for lab in sorted({r.get("label", "") for r in target if r.get("label")}):
        brs, nls, sts = [], [], []
        for r in target:
            if r.get("label") != lab or r["_batang"] not in H:
                continue
            h = H[r["_batang"]]
            _, br, nl, st = turunkan(h["r_lobus"], h["kejadian"], h["bertahan"],
                                     TAU, RHO, "min")
            brs.append(br); nls.append(nl); sts.append(st)
        if not brs:
            continue
        v = np.array(brs, float)
        tak = sum(1 for s in sts if s == "tak_pernah_pecah")
        baris.append([lab, len(brs), f(np.median(v[np.isfinite(v)])),
                      f"{tak/len(sts)*100:.1f}%", f"{np.mean(nls):.2f}"])
    cetak_tabel(["kelas", "n", "median br", "tak pernah pecah", "rata n_lobus"],
                baris, rk={1, 2, 3, 4})

    subjudul("Jumlah lobus per nucleus_shape (pembanding T2: 81,4% gagal)")
    baris = []
    for bentuk in ["unsegmented-round", "unsegmented-indented", "irregular",
                   "unsegmented-band", "segmented-bilobed", "segmented-multilobed"]:
        nls, brs = [], []
        for r in target:
            if r.get("nucleus_shape") != bentuk or r["_batang"] not in H:
                continue
            h = H[r["_batang"]]
            _, br, nl, _ = turunkan(h["r_lobus"], h["kejadian"], h["bertahan"],
                                    TAU, RHO, "min")
            nls.append(nl); brs.append(br)
        if not nls:
            continue
        nls = np.array(nls, float); brs = np.array(brs, float)
        baris.append([bentuk, len(nls), f"{nls.mean():.3f}",
                      f"{(nls <= 1).mean()*100:.1f}%",
                      f(np.median(brs[np.isfinite(brs)]))])
    cetak_tabel(["nucleus_shape", "n", "mean n_lobus", "hanya 1 lobus",
                 "median br"], baris, rk={1, 2, 3, 4})

    # ================================================================== K6
    judul("K6. PERBANDINGAN DENGAN FASE 1 DAN TRIANGULASI")
    v1 = {}
    if os.path.exists(arg.v1):
        with open(arg.v1, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                v1[os.path.splitext(os.path.basename(r.get("img_name", "")))[0]] = r
    if v1:
        from scipy.stats import spearmanr, pearsonr
        a1, a2, li, lb = [], [], [], []
        for r in baris_neu:
            vr = v1.get(r["_batang"])
            if not vr:
                continue
            h = H[r["_batang"]]
            br = turunkan(h["r_lobus"], h["kejadian"], h["bertahan"],
                          TAU, RHO, "min")[1]
            try:
                b1 = float(vr.get("bridge_ratio", ""))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(br) and np.isfinite(b1)):
                continue
            a1.append(b1); a2.append(br)
            try:
                idx = float(vr.get("indeks_segmentasi", ""))
                if np.isfinite(idx):
                    li.append(idx); lb.append(br)
            except (TypeError, ValueError):
                pass
        a1, a2 = np.array(a1), np.array(a2)
        catat(f"  n terbandingkan        : {a1.size:,}")
        catat(f"  Pearson  (v1, v2-rev)  : {f(pearsonr(a1, a2)[0])}"
              f"    [jalanan max: 0.4938]")
        catat(f"  Spearman (v1, v2-rev)  : {f(spearmanr(a1, a2).statistic)}"
              f"    [jalanan max: 0.4658]")
        d = a2 - a1
        catat(f"  Selisih v2-rev minus v1: mean {f(d.mean())}  median {f(np.median(d))}")
        pindah = int(((a1 >= AMBANG_KLINIS) != (a2 >= AMBANG_KLINIS)).sum())
        catat(f"  Berpindah sisi 1/3     : {pindah:,} ({pindah/a1.size*100:.2f}%)"
              f"   [jalanan max: 29.40%]")
        if li:
            catat(f"  Spearman(br, indeks LDA): "
                  f"{f(spearmanr(np.array(lb), np.array(li)).statistic)}"
                  f"   [v1: -0.7403, jalanan max: -0.2964]")

    # --------------------------------------------------------------- simpan
    jalur = os.path.join(arg.keluar, "bridge_ratio_v2rev.csv")
    with open(jalur, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img_name", "prefix", "kelompok", "label", "nucleus_shape",
                    "split", "r_lobus", "r_pisah_min", "bridge_ratio_min",
                    "bridge_ratio_max", "n_lobus", "status_topologi",
                    "luas_nukleus", "piks_lubang", "komponen_mentah",
                    "n_saddle_signifikan"])
        for r in target:
            h = H.get(r["_batang"])
            if not h:
                continue
            rp, br, nl, st = turunkan(h["r_lobus"], h["kejadian"], h["bertahan"],
                                      TAU, RHO, "min")
            brx = turunkan(h["r_lobus"], h["kejadian"], h["bertahan"],
                           TAU, RHO, "max")[1]
            ns = sum(1 for (s, pers, pkm, _) in h["kejadian"]
                     if pers >= TAU and pkm >= RHO * h["r_lobus"])
            w.writerow([r.get("img_name", ""), r["_prefix"], r["_kelompok"],
                        r.get("label", ""), r.get("nucleus_shape", ""),
                        r.get("split", ""), f"{h['r_lobus']:.6f}",
                        "" if not np.isfinite(rp) else f"{rp:.6f}",
                        f"{br:.6f}", f"{brx:.6f}", nl, st,
                        h["luas_nukleus"], h["piks_lubang"],
                        h["komponen_mentah"], ns])
    catat()
    catat(f"  Tersimpan: {jalur}")

    jk = os.path.join(arg.keluar, "kejadian_merge_v2rev.csv")
    n = 0
    with open(jk, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img_name", "saddle", "persistence", "puncak_mati",
                    "puncak_utama"])
        for r in target:
            h = H.get(r["_batang"])
            if not h:
                continue
            for (s, pers, pkm, pku) in h["kejadian"]:
                w.writerow([r.get("img_name", ""), f"{s:.6f}", f"{pers:.6f}",
                            f"{pkm:.6f}", f"{pku:.6f}"])
                n += 1
    catat(f"  Tersimpan: {jk}  ({n:,} kejadian)")

    grafik(arg.keluar, A, B, C, D, F_, TAU, RHO)

    # --------------------------------------------------------------- vonis
    judul("RINGKASAN")
    catat(f"  Parameter terpilih di train : tau={TAU:g}, rho={RHO:g}")
    catat(f"  AUC (A vs B)                : {f(auc)}     [v1 0.9227 | bug-max 0.6665]")
    catat(f"  Ambang bootstrap            : {f(np.median(boot))}   [v1 0.3333]")
    catat(f"  IK 95%                      : [{f(lo)} - {f(hi)}]")
    catat(f"  1/3 di dalam IK             : {'YA' if dalam else 'TIDAK'}")
    catat(f"  Akurasi @1/3                : {akurasi_pada(A, B, AMBANG_KLINIS)*100:.2f}%"
          f"   [v1 90.04%]")
    catat()
    catat("Kirim seluruh keluaran ini.")
    garis("=")
    simpan(arg.keluar)
    return 0


def grafik(keluar, A, B, C, D, F_, TAU, RHO):
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
                           ("F tanpa subtipe", F_, 1.3)):
            vv = v[np.isfinite(v)]
            if vv.size < 5:
                continue
            ax.hist(vv, bins=60, range=(0, 1), density=True, histtype="step",
                    linewidth=w, label=f"{nama} (n={vv.size})")
        ax.axvline(1 / 3, color="k", ls="--", lw=1.5, label="ambang klinis 1/3")
        ax.set_xlabel(f"bridge ratio (saddle min, tau={TAU:g}, rho={RHO:g})")
        ax.set_ylabel("kepadatan")
        ax.set_title("Sebaran bridge ratio setelah koreksi aturan saddle")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(keluar, "F2Arev_sebaran.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass


def simpan(keluar):
    try:
        os.makedirs(keluar, exist_ok=True)
        j = os.path.join(keluar, "F2Arev_laporan.txt")
        with open(j, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_penampung))
        print()
        print(f"Laporan tersimpan: {j}")
    except Exception as e:
        print(f"(gagal menyimpan: {e})")


if __name__ == "__main__":
    sys.exit(main())
