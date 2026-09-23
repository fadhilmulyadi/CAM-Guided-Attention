#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fase2c_kalibrasi_dua_sisi.py
============================================================================
FASE 2C -- Perbaikan kalibrasi, pembersihan artefak, dan uji kejujuran
           terhadap klaim invariansi ambang.

Menjawab M1-M8 dari dokumen migrasi v2.

  M4  praproses baru: nukleus dibatasi ke komponen sel target
  M2  skor kalibrasi dua sisi (macro-recall), bukan agregat mentah
  M1  grid diperluas: tau sampai 10, rho sampai 0.7
  M3  kalibrasi TERPISAH untuk jumlah lobus dan untuk bridge ratio
  M5  plateau kurva Youden + UJI PERGESERAN-KONSTAN (lihat catatan di bawah)
  M6  ablasi praproses untuk menjelaskan selisih AUC v1 (0.9227) vs v2 (~0.88)
  M7  montase 130 sel B berleher lebar
  M8  montase distratifikasi per status + penanda posisi saddle

CATATAN PENTING TENTANG M5
--------------------------
Temuan Fase 2B "ambang 0.3588 identik di 60 kombinasi" diduga hampir
tautologis. Menaikkan tau/rho hanya memindahkan sel ke status
'tak_pernah_pecah' yang nilainya br=1.0 -- di ATAS setiap kandidat ambang
< 1. Maka TPR(t) dan FPR(t) bergeser oleh KONSTANTA untuk semua t < 1, J(t)
ikut bergeser konstan, dan argmax tidak mungkin berpindah.

Tugas 'ambang' menguji ini secara langsung:
  - std dari [J_kombinasi(t) - J_referensi(t)] di seluruh t
    -> mendekati 0 berarti pergeseran konstan, invariansi TIDAK informatif
  - berapa persen sel ber-status 'normal' yang nilai br-nya benar-benar sama
  - lebar plateau: interval ambang yang optimal persis, dan yang
    J-nya dalam 0.001 dari maksimum

PEMAKAIAN
---------
  # uji sanitas dulu, wajib, 5 detik, tanpa menyentuh data asli
  python fase2c_kalibrasi_dua_sisi.py --tugas sanitas

  # uji cepat pipeline penuh pada 400 sel
  python fase2c_kalibrasi_dua_sisi.py --sampel 400

  # jalanan penuh
  python fase2c_kalibrasi_dua_sisi.py --pekerja 7

  # setelah kalibrasi terlihat, pin parameternya lalu jalankan ulang analisis
  python fase2c_kalibrasi_dua_sisi.py --tau-lobus 2 --rho-lobus 0.2 \
         --tau-br 1 --rho-br 0 --tugas ambang,audit,laporan

TUGAS: sanitas, cache, kalibrasi, ambang, ablasi, audit, laporan, semua
============================================================================
"""

import argparse
import gzip
import math
import os
import pickle
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:  # pragma: no cover
    roc_auc_score = None
    roc_curve = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

STRUKTUR8 = np.ones((3, 3), dtype=bool)
STRUKTUR4 = ndi.generate_binary_structure(2, 1)

NILAI_SITOPLASMA, NILAI_NUKLEUS, NILAI_VAKUOLA = 1, 2, 5

TAU_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
            2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

BENTUK_HARAPAN = {
    "unsegmented-round":    1,
    "unsegmented-indented": 1,
    "segmented-bilobed":    2,
    "segmented-multilobed": 3,   # arti: >= 3
}

URUT_BENTUK = ["unsegmented-round", "unsegmented-indented", "irregular",
               "unsegmented-band", "segmented-bilobed", "segmented-multilobed"]

URUT_KELOMPOK = ["A_sepakat_band", "B_sepakat_segmented", "C_konflik_SNE_band",
                 "D_konflik_BNE_segmented", "E_lainnya", "F_tak_bersubtipe"]

JUMLAH_KELOMPOK_HARAPAN = {
    "A_sepakat_band": 1555, "B_sepakat_segmented": 976,
    "C_konflik_SNE_band": 662, "D_konflik_BNE_segmented": 48,
    "E_lainnya": 38, "F_tak_bersubtipe": 50,
}

SEPERTIGA = 1.0 / 3.0


# ===========================================================================
# BAGIAN 1 -- PRAPROSES DAN PERSISTENSI (level pekerja)
# ===========================================================================

def baca_mask(path):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def siapkan_nukleus(m, batasi_sel=True, isi_lubang=True, min_frak_komp=0.02,
                    konektivitas=8):
    """
    Praproses M4.

    1. nukleus mentah = (mask == 2)
    2. bila batasi_sel: ambil komponen terhubung dari {sitoplasma, nukleus,
       vakuola} yang memuat pusat citra (fallback: terbesar), lalu iris
       nukleus dengan komponen itu. Ini membuang inti sel tetangga yang
       masuk bingkai -- penyebab artefak 'terpisah_di_nol'.
    3. buang komponen nukleus yang luasnya < min_frak_komp x total
    4. isi lubang

    Kembalikan (nukleus_bool | None, diagnostik).
    """
    st = STRUKTUR8 if konektivitas == 8 else STRUKTUR4
    d = dict(piks_nuk_mentah=0, komponen_mentah=0, komponen_setelah_sel=0,
             komponen_akhir=0, piks_buang_luar_sel=0, piks_buang_kecil=0,
             piks_lubang=0, piks_nuk_akhir=0, pusat_fallback=0,
             pusat_beda_terbesar=0, n_komponen_sel=0)

    nuk = (m == NILAI_NUKLEUS)
    d["piks_nuk_mentah"] = int(nuk.sum())
    if d["piks_nuk_mentah"] == 0:
        return None, d
    d["komponen_mentah"] = int(ndi.label(nuk, structure=st)[1])

    if batasi_sel:
        sel = (m == NILAI_SITOPLASMA) | (m == NILAI_NUKLEUS) | (m == NILAI_VAKUOLA)
        labs, ns = ndi.label(sel, structure=st)
        d["n_komponen_sel"] = int(ns)
        if ns >= 1:
            cy, cx = m.shape[0] // 2, m.shape[1] // 2
            id_pusat = int(labs[cy, cx])
            ukuran = np.bincount(labs.ravel(), minlength=ns + 1)
            ukuran[0] = 0
            id_besar = int(ukuran.argmax())
            if id_pusat == 0:
                y0 = max(0, cy - 20); y1 = min(m.shape[0], cy + 21)
                x0 = max(0, cx - 20); x1 = min(m.shape[1], cx + 21)
                jd = labs[y0:y1, x0:x1]
                nz = jd[jd > 0]
                if nz.size:
                    id_pusat = int(np.bincount(nz).argmax())
                    d["pusat_fallback"] = 1
                else:
                    id_pusat = id_besar
                    d["pusat_fallback"] = 2
            if id_pusat != id_besar:
                d["pusat_beda_terbesar"] = 1
            nuk_baru = nuk & (labs == id_pusat)
            if nuk_baru.any():
                d["piks_buang_luar_sel"] = int(nuk.sum() - nuk_baru.sum())
                nuk = nuk_baru
        d["komponen_setelah_sel"] = int(ndi.label(nuk, structure=st)[1])
    else:
        d["komponen_setelah_sel"] = d["komponen_mentah"]

    if min_frak_komp > 0:
        labn, nn = ndi.label(nuk, structure=st)
        if nn > 1:
            uk = np.bincount(labn.ravel(), minlength=nn + 1)
            uk[0] = 0
            total = uk.sum()
            simpan = uk >= (min_frak_komp * total)
            simpan[0] = False
            if not simpan.any():
                simpan[int(uk.argmax())] = True
            nuk_baru = simpan[labn]
            d["piks_buang_kecil"] = int(nuk.sum() - nuk_baru.sum())
            nuk = nuk_baru

    d["komponen_akhir"] = int(ndi.label(nuk, structure=st)[1])

    if isi_lubang:
        terisi = ndi.binary_fill_holes(nuk)
        d["piks_lubang"] = int(terisi.sum() - nuk.sum())
        nuk = terisi

    d["piks_nuk_akhir"] = int(nuk.sum())
    return nuk, d


def pohon_merge(dt, konektivitas=8, prune=0.05):
    """
    Pohon penggabungan superlevel-set dari distance transform, via union-find
    pada daftar sisi terurut menurun. Bobot sisi = min(dt[a], dt[b]) = nilai
    level saat kedua piksel sudah hadir.

    Kembalikan (kejadian, bertahan):
      kejadian : list (saddle, persistence, puncak_mati, puncak_utama, y, x)
      bertahan : list puncak komponen yang tidak pernah bergabung
    """
    h, w = dt.shape
    dtf = dt.ravel()
    piks = np.flatnonzero(dtf > 0)
    if piks.size == 0:
        return [], []

    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    pas = [(idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])]
    if konektivitas == 8:
        pas += [(idx[:-1, :-1], idx[1:, 1:]), (idx[:-1, 1:], idx[1:, :-1])]

    A = np.concatenate([a.ravel() for a, _ in pas])
    B = np.concatenate([b.ravel() for _, b in pas])
    va = dtf[A]
    vb = dtf[B]
    ok = (va > 0) & (vb > 0)
    if not ok.any():
        return [], [float(dtf[p]) for p in piks.tolist()]

    A = A[ok]; B = B[ok]; va = va[ok]; vb = vb[ok]
    wg = np.minimum(va, vb)
    o = np.argsort(-wg, kind="stable")
    Al = A[o].tolist(); Bl = B[o].tolist()
    Wl = wg[o].tolist(); Val = va[o].tolist(); Vbl = vb[o].tolist()

    par = list(range(h * w))
    puncak = dtf.tolist()
    kej = []

    for i in range(len(Wl)):
        a0 = Al[i]; b0 = Bl[i]
        a = a0
        while par[a] != a:
            a = par[a]
        ra = a
        a = a0
        while par[a] != ra:
            par[a], a = ra, par[a]
        b = b0
        while par[b] != b:
            b = par[b]
        rb = b
        b = b0
        while par[b] != rb:
            par[b], b = rb, par[b]
        if ra == rb:
            continue
        pa = puncak[ra]; pb = puncak[rb]
        if pa >= pb:
            hidup, mati, p_mati, p_utama = ra, rb, pb, pa
        else:
            hidup, mati, p_mati, p_utama = rb, ra, pa, pb
        s = Wl[i]
        pers = p_mati - s
        if pers >= prune:
            p = a0 if Val[i] <= Vbl[i] else b0
            kej.append((s, pers, p_mati, p_utama, p // w, p % w))
        par[mati] = hidup

    akar = set()
    for p in piks.tolist():
        r = p
        while par[r] != r:
            r = par[r]
        akar.add(r)
    bertahan = sorted((float(puncak[r]) for r in akar), reverse=True)
    return kej, bertahan


def proses_satu_sel(tugas):
    """Pekerja multiprocessing. tugas = (nama, path_str, cfg_tuple)."""
    nama, path, cfg = tugas
    (batasi_sel, isi_lubang, pad, min_frak, konek, prune) = cfg
    try:
        m = baca_mask(path)
        nuk, d = siapkan_nukleus(m, batasi_sel=batasi_sel, isi_lubang=isi_lubang,
                                 min_frak_komp=min_frak, konektivitas=konek)
        if nuk is None or not nuk.any():
            return nama, None
        ys, xs = np.nonzero(nuk)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        potong = nuk[y0:y1, x0:x1]
        if pad:
            potong = np.pad(potong, 1)
            base_y, base_x = y0 - 1, x0 - 1
        else:
            base_y, base_x = y0, x0
        dt = ndi.distance_transform_edt(potong)
        r_lobus = float(dt.max())
        dt2 = np.pad(dt, 1)                 # selalu aman untuk traversal
        oy, ox = base_y - 1, base_x - 1
        kej, bertahan = pohon_merge(dt2, konektivitas=konek, prune=prune)

        if kej:
            K = np.array([[k[0], k[1], k[2], k[3]] for k in kej], dtype=np.float32)
            P = np.array([[k[4] + oy, k[5] + ox] for k in kej], dtype=np.int16)
        else:
            K = np.zeros((0, 4), dtype=np.float32)
            P = np.zeros((0, 2), dtype=np.int16)

        entri = dict(
            r_lobus=r_lobus,
            luas_nukleus=int(d["piks_nuk_akhir"]),
            piks_nuk_mentah=int(d["piks_nuk_mentah"]),
            komponen_mentah=int(d["komponen_mentah"]),
            komponen_setelah_sel=int(d["komponen_setelah_sel"]),
            komponen_akhir=int(d["komponen_akhir"]),
            piks_buang_luar_sel=int(d["piks_buang_luar_sel"]),
            piks_buang_kecil=int(d["piks_buang_kecil"]),
            piks_lubang=int(d["piks_lubang"]),
            pusat_fallback=int(d["pusat_fallback"]),
            pusat_beda_terbesar=int(d["pusat_beda_terbesar"]),
            n_komponen_sel=int(d["n_komponen_sel"]),
            kejadian=K,
            pos=P,
            bertahan=np.array(bertahan, dtype=np.float32),
            bbox=(y0, x0, y1, x1),
        )
        return nama, entri
    except Exception as e:  # pragma: no cover
        return nama, {"galat": f"{type(e).__name__}: {e}"}


# ===========================================================================
# BAGIAN 2 -- DERIVASI TERVEKTORISASI
# ===========================================================================

class Turunan:
    """
    Meratakan cache jadi array datar sehingga seluruh grid (tau, rho) bisa
    diturunkan dengan operasi numpy, bukan loop per sel.
    """

    def __init__(self, cache, nama_urut):
        self.nama = list(nama_urut)
        N = len(self.nama)
        self.N = N
        self.rl = np.zeros(N, dtype=np.float64)
        self.luas = np.zeros(N, dtype=np.int64)
        self.komp_mentah = np.zeros(N, dtype=np.int32)
        self.komp_akhir = np.zeros(N, dtype=np.int32)

        sad, pers, pmati = [], [], []
        off = [0]
        bts, boff = [], [0]
        for i, nm in enumerate(self.nama):
            e = cache[nm]
            self.rl[i] = e["r_lobus"]
            self.luas[i] = e["luas_nukleus"]
            self.komp_mentah[i] = e["komponen_mentah"]
            self.komp_akhir[i] = e["komponen_akhir"]
            K = e["kejadian"]
            if K.shape[0]:
                sad.append(K[:, 0]); pers.append(K[:, 1]); pmati.append(K[:, 2])
            off.append(off[-1] + int(K.shape[0]))
            Bt = e["bertahan"]
            bts.append(Bt)
            boff.append(boff[-1] + int(Bt.size))

        self.sad = np.concatenate(sad).astype(np.float64) if sad else np.zeros(0)
        self.pers = np.concatenate(pers).astype(np.float64) if pers else np.zeros(0)
        self.pmati = np.concatenate(pmati).astype(np.float64) if pmati else np.zeros(0)
        self.off = np.array(off, dtype=np.int64)
        cnt = np.diff(self.off)
        self.idx_ada = np.flatnonzero(cnt > 0)
        self.start = self.off[:-1][cnt > 0]
        self.rl_ev = np.repeat(self.rl, cnt)

        self.bt = np.concatenate(bts).astype(np.float64) if bts else np.zeros(0)
        self.boff = np.array(boff, dtype=np.int64)
        bcnt = np.diff(self.boff)
        self.bidx_ada = np.flatnonzero(bcnt > 0)
        self.bstart = self.boff[:-1][bcnt > 0]
        self.rl_bt = np.repeat(self.rl, bcnt)

    def hitung(self, tau, rho, aturan="min", diskret=0.0):
        N = self.N
        nkej = np.zeros(N, dtype=np.int64)
        smin = np.full(N, np.inf)
        smax = np.full(N, -np.inf)
        if self.sad.size:
            mev = (self.pers >= tau) & (self.pmati >= rho * self.rl_ev)
            nkej[self.idx_ada] = np.add.reduceat(mev.astype(np.int64), self.start)
            smin[self.idx_ada] = np.minimum.reduceat(
                np.where(mev, self.sad, np.inf), self.start)
            smax[self.idx_ada] = np.maximum.reduceat(
                np.where(mev, self.sad, -np.inf), self.start)

        nbt = np.zeros(N, dtype=np.int64)
        if self.bt.size:
            mbt = (self.bt >= tau) & (self.bt >= rho * self.rl_bt)
            nbt[self.bidx_ada] = np.add.reduceat(mbt.astype(np.int64), self.bstart)

        n_lobus = np.maximum(nkej + nbt, 1)
        terpisah = nbt >= 2
        normal = (~terpisah) & (nkej > 0)
        takpecah = (~terpisah) & (nkej == 0)

        rp = np.full(N, np.nan)
        br = np.full(N, np.nan)
        sd = smin if aturan == "min" else smax

        rp[terpisah] = 0.0
        br[terpisah] = 0.0
        rp[normal] = sd[normal]
        if diskret > 0:
            rp[normal] = np.ceil(rp[normal] / diskret) * diskret
        br[normal] = rp[normal] / self.rl[normal]
        rp[takpecah] = np.nan
        br[takpecah] = 1.0
        br = np.clip(br, 0.0, 1.0)

        status = np.empty(N, dtype=object)
        status[terpisah] = "terpisah_di_nol"
        status[normal] = "normal"
        status[takpecah] = "tak_pernah_pecah"

        return dict(img_name=np.array(self.nama, dtype=object),
                    r_lobus=self.rl.copy(), r_pisah=rp, bridge_ratio=br,
                    n_lobus=n_lobus, status_topologi=status,
                    luas_nukleus=self.luas.copy(),
                    komponen_mentah=self.komp_mentah.copy(),
                    komponen_akhir=self.komp_akhir.copy())

    def frame(self, tau, rho, aturan="min", diskret=0.0):
        return pd.DataFrame(self.hitung(tau, rho, aturan, diskret))


# ===========================================================================
# BAGIAN 3 -- UJI SANITAS SINTETIS (dijalankan SEBELUM data asli)
# ===========================================================================

def _kanvas(h=160, w=220):
    return np.zeros((h, w), dtype=np.uint8)


def _cakram(m, cy, cx, r, nilai=NILAI_NUKLEUS):
    Y, X = np.ogrid[:m.shape[0], :m.shape[1]]
    m[(Y - cy) ** 2 + (X - cx) ** 2 <= r * r] = nilai


def _batang(m, cy, x0, x1, setengah, nilai=NILAI_NUKLEUS):
    """Batang setinggi (2*setengah+1) baris -> piksel tengah berjarak
    (setengah+1) ke latar."""
    m[cy - setengah:cy + setengah + 1, x0:x1] = nilai


def _bungkus_sitoplasma(m, tebal=6):
    """Selubungi setiap nukleus dengan sitoplasma agar komponen sel terbentuk."""
    nuk = (m == NILAI_NUKLEUS)
    lebar = ndi.binary_dilation(nuk, structure=STRUKTUR8, iterations=tebal)
    m[lebar & ~nuk] = NILAI_SITOPLASMA
    return m


def kasus_sanitas():
    kasus = []

    # 1. cakram konveks
    m = _kanvas(); _cakram(m, 80, 110, 30); _bungkus_sitoplasma(m)
    kasus.append(("cakram_konveks", m, dict(status="tak_pernah_pecah", n_lobus=1),
                  dict(tau=1.0, rho=0.0)))

    # 2. halter, leher 9 baris -> r_pisah = 5.0
    m = _kanvas()
    _cakram(m, 80, 60, 20); _cakram(m, 80, 160, 20); _batang(m, 80, 60, 161, 4)
    _bungkus_sitoplasma(m)
    kasus.append(("halter_leher9", m,
                  dict(status="normal", n_lobus=2, rp_min=5.0, rp_max=5.0),
                  dict(tau=1.0, rho=0.0)))

    # 3. rantai, leher 7 dan 17 baris -> min 4.0, max 9.0
    m = _kanvas(h=170, w=300)
    _cakram(m, 85, 50, 20); _cakram(m, 85, 150, 20); _cakram(m, 85, 250, 20)
    _batang(m, 85, 50, 151, 3)
    _batang(m, 85, 150, 251, 8)
    _bungkus_sitoplasma(m)
    kasus.append(("rantai_leher7_17", m,
                  dict(status="normal", n_lobus=3, rp_min=4.0, rp_max=9.0),
                  dict(tau=1.0, rho=0.0)))

    # 4. dua cakram terpisah DI DALAM satu sel
    m = _kanvas()
    _cakram(m, 80, 70, 20); _cakram(m, 80, 150, 20)
    m[70:91, 70:151] = np.where(m[70:91, 70:151] == 0, NILAI_SITOPLASMA,
                                m[70:91, 70:151])
    _bungkus_sitoplasma(m)
    kasus.append(("dua_cakram_terpisah", m,
                  dict(status="terpisah_di_nol", n_lobus=2, rp_min=0.0),
                  dict(tau=1.0, rho=0.0)))

    # 5. cakram + tonjolan kecil (harus ditolak filter persistensi)
    m = _kanvas(); _cakram(m, 80, 110, 30); _cakram(m, 48, 110, 4)
    _bungkus_sitoplasma(m)
    kasus.append(("cakram_plus_tonjolan", m,
                  dict(status="tak_pernah_pecah", n_lobus=1),
                  dict(tau=1.0, rho=0.0)))

    # 6. BARU (M4): inti sel tetangga di luar komponen sel -> harus dibuang
    m = _kanvas(h=200, w=260)
    _cakram(m, 100, 130, 28); _bungkus_sitoplasma(m)
    _cakram(m, 30, 35, 16)          # tetangga, tanpa sitoplasma tersambung
    kasus.append(("tetangga_di_luar_sel", m,
                  dict(status="tak_pernah_pecah", n_lobus=1,
                       komponen_mentah=2, komponen_akhir=1),
                  dict(tau=1.0, rho=0.0)))

    # 7a. BARU (M4): serpihan DI BAWAH ambang 2% -> dibuang
    #     cakram r=28 luas 2453 px, cakram r=3 luas 29 px -> 29/2482 = 1.17%
    m = _kanvas(h=200, w=260)
    _cakram(m, 100, 130, 28)
    _cakram(m, 100, 175, 3)
    _bungkus_sitoplasma(m, tebal=14)
    kasus.append(("serpihan_di_bawah_2persen", m,
                  dict(status="tak_pernah_pecah", n_lobus=1,
                       komponen_mentah=2, komponen_akhir=1),
                  dict(tau=1.0, rho=0.0)))

    # 7b. BARU (M4): serpihan DI ATAS ambang 2% -> dipertahankan
    #     cakram r=6 luas 113 px -> 113/2566 = 4.4%
    m = _kanvas(h=200, w=260)
    _cakram(m, 100, 130, 28)
    _cakram(m, 100, 175, 6)
    _bungkus_sitoplasma(m, tebal=14)
    kasus.append(("serpihan_di_atas_2persen", m,
                  dict(status="terpisah_di_nol", n_lobus=2,
                       komponen_mentah=2, komponen_akhir=2),
                  dict(tau=1.0, rho=0.0)))

    # 8. BARU: dua lobus sejati terpisah, dua-duanya besar -> keduanya bertahan
    m = _kanvas(h=200, w=300)
    _cakram(m, 100, 90, 22); _cakram(m, 100, 200, 22)
    m[78:123, 90:201] = np.where(m[78:123, 90:201] == 0, NILAI_SITOPLASMA,
                                 m[78:123, 90:201])
    _bungkus_sitoplasma(m)
    kasus.append(("dua_lobus_besar_terpisah", m,
                  dict(status="terpisah_di_nol", n_lobus=2, komponen_akhir=2),
                  dict(tau=1.0, rho=0.0)))
    return kasus


def _proses_array(m, batasi_sel=True, isi_lubang=True, pad=True,
                  min_frak=0.02, konek=8, prune=0.05):
    nuk, d = siapkan_nukleus(m, batasi_sel, isi_lubang, min_frak, konek)
    if nuk is None or not nuk.any():
        return None, d
    ys, xs = np.nonzero(nuk)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    potong = nuk[y0:y1, x0:x1]
    if pad:
        potong = np.pad(potong, 1)
    dt = ndi.distance_transform_edt(potong)
    kej, bertahan = pohon_merge(np.pad(dt, 1), konek, prune)
    K = (np.array([[k[0], k[1], k[2], k[3]] for k in kej], dtype=np.float32)
         if kej else np.zeros((0, 4), dtype=np.float32))
    e = dict(r_lobus=float(dt.max()), luas_nukleus=int(nuk.sum()),
             komponen_mentah=d["komponen_mentah"], komponen_akhir=d["komponen_akhir"],
             kejadian=K, pos=np.zeros((K.shape[0], 2), np.int16),
             bertahan=np.array(bertahan, dtype=np.float32))
    return e, d


def jalankan_sanitas(out, tol_rp=0.02, tol_rasio=0.005):
    baris = []
    lolos_semua = True
    print("\n=== UJI SANITAS SINTETIS (toleransi r_pisah %.3f px) ===" % tol_rp)
    for nama, m, harap, prm in kasus_sanitas():
        e, d = _proses_array(m)
        T = Turunan({nama: e}, [nama])
        hmin = T.hitung(prm["tau"], prm["rho"], "min")
        hmax = T.hitung(prm["tau"], prm["rho"], "max")
        got = dict(status=hmin["status_topologi"][0],
                   n_lobus=int(hmin["n_lobus"][0]),
                   rp_min=float(hmin["r_pisah"][0]),
                   rp_max=float(hmax["r_pisah"][0]),
                   komponen_mentah=int(e["komponen_mentah"]),
                   komponen_akhir=int(e["komponen_akhir"]),
                   r_lobus=float(e["r_lobus"]))
        ok = True
        pesan = []
        for k, v in harap.items():
            g = got.get(k)
            if k in ("rp_min", "rp_max"):
                if g is None or (isinstance(g, float) and math.isnan(g)):
                    ok = False; pesan.append("%s=NaN" % k)
                elif abs(g - v) > tol_rp:
                    ok = False; pesan.append("%s %.4f != %.4f" % (k, g, v))
            else:
                if g != v:
                    ok = False; pesan.append("%s %r != %r" % (k, g, v))
        lolos_semua &= ok
        print("  [%s] %-26s r_lobus=%7.3f status=%-17s rp_min=%s rp_max=%s "
              "lobus=%d komp %d->%d %s"
              % ("OK " if ok else "GAGAL", nama, got["r_lobus"], got["status"],
                 ("%.4f" % got["rp_min"]) if not math.isnan(got["rp_min"]) else "  nan",
                 ("%.4f" % got["rp_max"]) if not math.isnan(got["rp_max"]) else "  nan",
                 got["n_lobus"], got["komponen_mentah"], got["komponen_akhir"],
                 ("<- " + "; ".join(pesan)) if pesan else ""))
        baris.append(dict(kasus=nama, lolos=ok, catatan="; ".join(pesan), **got))

    # pemeriksaan mekanisme bug r_lobus v1 (hipotesis M6).
    # Bentuk yang MENGISI penuh bounding box-nya tidak punya piksel latar di
    # dalam potongan, sehingga EDT tanpa padding menggelembung parah. Inilah
    # kandidat penjelasan r_lobus = 70.38 px pada nukleus seluas ~5.900 px.
    m = _kanvas(h=200, w=300)
    m[70:131, 90:211] = NILAI_NUKLEUS      # persegi panjang 61 x 121
    _bungkus_sitoplasma(m)
    e_pad, _ = _proses_array(m, pad=True)
    e_nop, _ = _proses_array(m, pad=False)
    rasio = e_nop["r_lobus"] / e_pad["r_lobus"]
    ok_pad = rasio > 3.0
    lolos_semua &= ok_pad
    print("  [%s] %-26s dengan pad %.4f, tanpa pad %.4f, rasio %.2f "
          "(harapan > 3, membuktikan mekanisme bug v1)"
          % ("OK " if ok_pad else "GAGAL", "pad_vs_tanpa_pad",
             e_pad["r_lobus"], e_nop["r_lobus"], rasio))
    baris.append(dict(kasus="pad_vs_tanpa_pad", lolos=ok_pad,
                      catatan="rasio %.3f" % rasio,
                      r_lobus=e_nop["r_lobus"], status="-", n_lobus=-1,
                      rp_min=e_pad["r_lobus"], rp_max=e_nop["r_lobus"],
                      komponen_mentah=-1, komponen_akhir=-1))

    df = pd.DataFrame(baris)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "C0_uji_sanitas.csv", index=False)
    print("  -> %s" % ("SELURUH UJI LOLOS" if lolos_semua else
                       "ADA UJI GAGAL -- JANGAN LANJUT KE DATA ASLI"))
    return lolos_semua, df


# ===========================================================================
# BAGIAN 4 -- DATA, CACHE, KELOMPOK
# ===========================================================================

def indeks_mask(dirmask):
    peta = {}
    n = 0
    for p in Path(dirmask).rglob("*.png"):
        s = p.stem
        if s.endswith("_mask"):
            s = s[:-5]
        if s not in peta:
            peta[s] = p
            n += 1
        if s.endswith("_ccrop"):
            peta.setdefault(s[:-6], p)
    return peta, n


def cari_mask(peta, img_name):
    s = Path(str(img_name)).stem
    if s in peta:
        return peta[s]
    if s.endswith("_ccrop") and s[:-6] in peta:
        return peta[s[:-6]]
    if (s + "_ccrop") in peta:
        return peta[s + "_ccrop"]
    return None


def nama_cache(cfg, tandai=""):
    b, l, p, mf, k, pr = cfg
    return ("cache_p2c%s_sel%d_lub%d_pad%d_mf%g_conn%d_pr%g.pkl.gz"
            % (tandai, int(b), int(l), int(p), mf, k, pr))


def bangun_cache(df, peta, cfg, out, pekerja, tandai="", paksa=False):
    out.mkdir(parents=True, exist_ok=True)
    fc = out / nama_cache(cfg, tandai)
    if fc.exists() and not paksa:
        print("  cache ditemukan, memuat: %s" % fc.name)
        with gzip.open(fc, "rb") as f:
            cache = pickle.load(f)
        hilang = [n for n in df["img_name"] if n not in cache]
        if not hilang:
            return cache, fc
        print("  cache tidak lengkap (%d sel hilang), membangun ulang" % len(hilang))

    tugas = []
    tidak_ada = []
    for nm in df["img_name"]:
        p = cari_mask(peta, nm)
        if p is None:
            tidak_ada.append(nm)
        else:
            tugas.append((nm, str(p), cfg))
    if tidak_ada:
        print("  PERINGATAN: %d mask tidak ditemukan (contoh: %s)"
              % (len(tidak_ada), tidak_ada[:3]))

    print("  menghitung persistensi untuk %d sel dengan %d pekerja..."
          % (len(tugas), pekerja))
    t0 = time.time()
    cache = {}
    galat = 0
    if pekerja <= 1:
        for i, t in enumerate(tugas):
            nm, e = proses_satu_sel(t)
            if e is None or "galat" in e:
                galat += 1
            else:
                cache[nm] = e
            if (i + 1) % 500 == 0:
                print("    %d/%d (%.1f dtk)" % (i + 1, len(tugas), time.time() - t0))
    else:
        with ProcessPoolExecutor(max_workers=pekerja) as ex:
            for i, (nm, e) in enumerate(ex.map(proses_satu_sel, tugas, chunksize=24)):
                if e is None or "galat" in e:
                    galat += 1
                else:
                    cache[nm] = e
                if (i + 1) % 500 == 0:
                    print("    %d/%d (%.1f dtk)" % (i + 1, len(tugas), time.time() - t0))
    dt = time.time() - t0
    print("  selesai: %d sel, %d galat, %.1f dtk (%.1f ms/sel)"
          % (len(cache), galat, dt, 1000 * dt / max(len(tugas), 1)))
    with gzip.open(fc, "wb", compresslevel=5) as f:
        pickle.dump(cache, f, protocol=4)
    print("  cache disimpan: %s" % fc.name)
    return cache, fc


def beri_kelompok(df):
    pref = df["img_name"].astype(str).str.split("_").str[0].str.upper()
    ns = df["nucleus_shape"].astype(str)
    neu = df["label"].astype(str) == "Neutrophil"
    seg = ns.isin(["segmented-bilobed", "segmented-multilobed"])
    band = ns == "unsegmented-band"
    kel = pd.Series("bukan_neutrofil", index=df.index, dtype=object)
    kel[neu & (pref == "BNE") & band] = "A_sepakat_band"
    kel[neu & (pref == "SNE") & seg] = "B_sepakat_segmented"
    kel[neu & (pref == "SNE") & band] = "C_konflik_SNE_band"
    kel[neu & (pref == "BNE") & seg] = "D_konflik_BNE_segmented"
    kel[neu & pref.isin(["BNE", "SNE"]) & ~(band | seg)] = "E_lainnya"
    kel[neu & (pref == "NEUTROPHIL")] = "F_tak_bersubtipe"
    return pref, kel


def verifikasi_kelompok(df, penuh=True):
    hit = df["kelompok"].value_counts().to_dict()
    print("\n=== REPLIKASI DEFINISI KELOMPOK (Fase 1 4.1) ===")
    ok = True
    for k in URUT_KELOMPOK:
        g = int(hit.get(k, 0))
        h = JUMLAH_KELOMPOK_HARAPAN[k]
        if not penuh:
            tanda = "smpl"
        else:
            tanda = "OK " if g == h else "BEDA"
            if g != h:
                ok = False
        print("  [%s] %-26s %5d (harapan penuh %5d)" % (tanda, k, g, h))
    if penuh:
        print("  -> %s" % ("replikasi LOLOS" if ok else "REPLIKASI GAGAL, periksa data"))
    else:
        print("  -> mode sampel, replikasi tidak diperiksa")
    return ok


# ===========================================================================
# BAGIAN 5 -- TUGAS KALIBRASI (M1, M2, M3)
# ===========================================================================

def skor_kalibrasi(sub):
    """
    sub: DataFrame sel non-neutrofil dengan kolom nucleus_shape, n_lobus,
         status_topologi, label.
    Kembalikan dict berisi skor mentah, makro-3, makro-4, dan skor bridge ratio.
    """
    kal = sub[sub["nucleus_shape"].isin(BENTUK_HARAPAN)].copy()
    kal["target"] = kal["nucleus_shape"].map(BENTUK_HARAPAN)
    benar = np.where(kal["target"] == 3,
                     kal["n_lobus"] >= 3,
                     kal["n_lobus"] == kal["target"])
    kal["benar"] = benar

    r = {}
    r["n_kalibrasi"] = int(len(kal))
    r["skor_mentah"] = float(kal["benar"].mean()) if len(kal) else np.nan

    rec_t = kal.groupby("target")["benar"].mean()
    r["rec_t1"] = float(rec_t.get(1, np.nan))
    r["rec_t2"] = float(rec_t.get(2, np.nan))
    r["rec_t3"] = float(rec_t.get(3, np.nan))
    r["skor_makro3"] = float(np.nanmean([r["rec_t1"], r["rec_t2"], r["rec_t3"]]))

    rec_b = kal.groupby("nucleus_shape")["benar"].mean()
    vals = [float(rec_b.get(b, np.nan)) for b in BENTUK_HARAPAN]
    r["skor_makro4"] = float(np.nanmean(vals))
    for b in BENTUK_HARAPAN:
        r["rec_" + b.replace("unsegmented-", "u_").replace("segmented-", "s_")] = \
            float(rec_b.get(b, np.nan))
        m = kal[kal["nucleus_shape"] == b]["n_lobus"]
        r["lobus_" + b.replace("unsegmented-", "u_").replace("segmented-", "s_")] = \
            float(m.mean()) if len(m) else np.nan

    # --- skor untuk BRIDGE RATIO: kriteria status, bukan jumlah lobus
    pos = sub[sub["nucleus_shape"].isin(["segmented-bilobed", "segmented-multilobed"])]
    neg = sub[sub["nucleus_shape"] == "unsegmented-round"]
    rp = float((pos["status_topologi"] != "tak_pernah_pecah").mean()) if len(pos) else np.nan
    rn = float((neg["status_topologi"] == "tak_pernah_pecah").mean()) if len(neg) else np.nan
    r["rec_harus_pecah"] = rp
    r["rec_harus_utuh"] = rn
    r["skor_br"] = float(np.nanmean([rp, rn]))

    lim = sub[sub["label"] == "Lymphocyte"]
    r["limf_tak_pecah"] = float((lim["status_topologi"] == "tak_pernah_pecah").mean()) \
        if len(lim) else np.nan
    r["limf_lobus"] = float(lim["n_lobus"].mean()) if len(lim) else np.nan
    return r


def tugas_kalibrasi(T, meta, out, aturan="min"):
    print("\n=== KALIBRASI DUA SISI PADA SEL NON-NEUTROFIL (M1, M2, M3) ===")
    idx_non = meta["label"].astype(str) != "Neutrophil"
    print("  sel non-neutrofil: %d" % int(idx_non.sum()))

    baris = []
    for tau in TAU_GRID:
        for rho in RHO_GRID:
            h = T.hitung(tau, rho, aturan)
            d = pd.DataFrame({
                "nucleus_shape": meta["nucleus_shape"].values,
                "label": meta["label"].values,
                "n_lobus": h["n_lobus"],
                "status_topologi": h["status_topologi"],
            })[idx_non.values]
            r = skor_kalibrasi(d)
            r["tau"] = tau
            r["rho"] = rho
            baris.append(r)
    dfk = pd.DataFrame(baris)
    kol = ["tau", "rho"] + [c for c in dfk.columns if c not in ("tau", "rho")]
    dfk = dfk[kol]
    dfk.to_csv(out / "C2_kalibrasi_grid.csv", index=False)

    def pilih_terbaik(kol):
        """Argmax dengan tie-break KONSERVATIF: bila skor seri, ambil tau dan
        rho terkecil. Penyaringan agresif adalah moda kegagalan yang sudah
        terbukti (Fase 2B), jadi seri tidak boleh dimenangkan olehnya."""
        s = dfk.sort_values([kol, "tau", "rho"],
                            ascending=[False, True, True])
        return s.iloc[0]

    best_l = pilih_terbaik("skor_makro3")
    best_m = pilih_terbaik("skor_mentah")
    best_b = pilih_terbaik("skor_br")

    def dipojok(r):
        """Hanya batas ATAS yang bermasalah; tau=0 dan rho=0 memang batas
        alami (tidak ada penyaringan sama sekali)."""
        return (r["tau"] == TAU_GRID[-1]) or (r["rho"] == RHO_GRID[-1])

    def datar(kol, n=10):
        v = dfk.nlargest(n, kol)[kol]
        return float(v.max() - v.min())

    print("\n  -- optimum menurut SKOR MENTAH (kriteria Fase 2B, berat sebelah)")
    print("     tau=%g rho=%g  mentah=%.4f makro3=%.4f"
          % (best_m["tau"], best_m["rho"], best_m["skor_mentah"], best_m["skor_makro3"]))
    print("\n  -- optimum JUMLAH LOBUS menurut SKOR MAKRO-3 (kriteria baru)")
    print("     tau=%g rho=%g  makro3=%.4f (rec t1=%.3f t2=%.3f t3=%.3f)  mentah=%.4f"
          % (best_l["tau"], best_l["rho"], best_l["skor_makro3"],
             best_l["rec_t1"], best_l["rec_t2"], best_l["rec_t3"], best_l["skor_mentah"]))
    print("     limfosit tak pernah pecah %.3f, rata lobus limfosit %.3f"
          % (best_l["limf_tak_pecah"], best_l["limf_lobus"]))
    print("     di batas atas grid: %s"
          % ("YA -- perluas grid sebelum memakai angka ini" if dipojok(best_l) else "tidak"))
    print("     rentang skor 10 kombinasi teratas: %.4f%s"
          % (datar("skor_makro3"),
             "  <- PERMUKAAN DATAR, argmax mengejar derau"
             if datar("skor_makro3") < 0.01 else ""))
    print("\n  -- optimum BRIDGE RATIO menurut SKOR STATUS (harus pecah vs harus utuh)")
    print("     tau=%g rho=%g  skor_br=%.4f (harus pecah %.3f, harus utuh %.3f)"
          % (best_b["tau"], best_b["rho"], best_b["skor_br"],
             best_b["rec_harus_pecah"], best_b["rec_harus_utuh"]))
    print("     di batas atas grid: %s"
          % ("YA -- perluas grid sebelum memakai angka ini" if dipojok(best_b) else "tidak"))
    print("     rentang skor 10 kombinasi teratas: %.4f%s"
          % (datar("skor_br"),
             "  <- PERMUKAAN DATAR, argmax mengejar derau"
             if datar("skor_br") < 0.01 else ""))
    print("\n  -- sepuluh kombinasi teratas menurut skor makro-3")
    kolom = ["tau", "rho", "skor_makro3", "skor_mentah", "rec_t1", "rec_t2",
             "rec_t3", "skor_br", "limf_tak_pecah"]
    print(dfk.nlargest(10, "skor_makro3")[kolom].to_string(
        index=False, float_format=lambda v: "%.4f" % v))

    # pembanding eksplisit dengan parameter Fase 2B dan 2A-rev
    print("\n  -- pembanding parameter lama")
    for tau, rho, ket in [(4.0, 0.5, "Fase 2B"), (0.1, 0.3, "Fase 2A-rev"), (1.0, 0.0, "Fase 2A")]:
        s = dfk[(dfk["tau"] == tau) & (dfk["rho"] == rho)]
        if len(s):
            s = s.iloc[0]
            print("     %-12s tau=%g rho=%g: mentah=%.4f makro3=%.4f skor_br=%.4f limf=%.3f"
                  % (ket, tau, rho, s["skor_mentah"], s["skor_makro3"],
                     s["skor_br"], s["limf_tak_pecah"]))

    _peta_panas(dfk, out)
    return dfk, (float(best_l["tau"]), float(best_l["rho"])), \
        (float(best_b["tau"]), float(best_b["rho"]))


def _peta_panas(dfk, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, kol, jud in zip(axes,
                            ["skor_mentah", "skor_makro3", "skor_br"],
                            ["Skor mentah (kriteria 2B)",
                             "Skor makro-3 (jumlah lobus)",
                             "Skor status (bridge ratio)"]):
        piv = dfk.pivot(index="tau", columns="rho", values=kol)
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([("%g" % c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([("%g" % i) for i in piv.index], fontsize=7)
        ax.set_xlabel("rho"); ax.set_ylabel("tau")
        ax.set_title(jud, fontsize=10)
        j, i = np.unravel_index(np.nanargmax(piv.values), piv.values.shape)
        ax.plot(i, j, "r*", markersize=14)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "F2C_kalibrasi_peta_panas.png", dpi=140)
    plt.close(fig)


# ===========================================================================
# BAGIAN 6 -- TUGAS AMBANG, PLATEAU, DAN UJI PERGESERAN-KONSTAN (M5)
# ===========================================================================

def _j_pada(br_a, br_b, t):
    return (br_a >= t).mean() - (br_b >= t).mean()


def analisis_ambang(br_a, br_b, eps=1e-3):
    """
    Positif = band (kelompok A). Prediksi band bila bridge_ratio >= t.
    Kembalikan dict: AUC, ambang argmax, interval optimal PERSIS, plateau eps.
    """
    y = np.r_[np.ones(len(br_a)), np.zeros(len(br_b))]
    s = np.r_[br_a, br_b]
    hasil = {}
    hasil["n_A"] = len(br_a); hasil["n_B"] = len(br_b)
    if roc_auc_score is None or len(br_a) < 2 or len(br_b) < 2:
        hasil["auc"] = np.nan
        return hasil
    hasil["auc"] = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)
    J = tpr - fpr
    # sklearn < 1.3 memakai max(skor)+1 sebagai ambang pertama, >= 1.3 memakai inf
    fin = np.isfinite(thr) & (thr <= 1.0 + 1e-12)
    if not fin.any():
        fin = np.isfinite(thr)
    Jf, thrf = J[fin], thr[fin]
    k = int(np.argmax(Jf))
    hasil["J_max"] = float(Jf[k])
    hasil["ambang"] = float(thrf[k])

    # interval optimal PERSIS: semua t di [thr_k, nilai data berikutnya) setara
    nilai = np.unique(s)
    lebih = nilai[nilai > hasil["ambang"]]
    hasil["ambang_atas"] = float(lebih.min()) if lebih.size else 1.0
    hasil["lebar_persis"] = hasil["ambang_atas"] - hasil["ambang"]
    hasil["sepertiga_di_interval_persis"] = bool(
        hasil["ambang"] <= SEPERTIGA < hasil["ambang_atas"])

    ok = Jf >= (hasil["J_max"] - eps)
    hasil["plateau_lo"] = float(thrf[ok].min())
    hasil["plateau_hi"] = float(thrf[ok].max())
    hasil["lebar_plateau_eps"] = hasil["plateau_hi"] - hasil["plateau_lo"]
    hasil["sepertiga_di_plateau_eps"] = bool(
        hasil["plateau_lo"] <= SEPERTIGA <= hasil["plateau_hi"])

    hasil["J_sepertiga"] = float(_j_pada(br_a, br_b, SEPERTIGA))
    hasil["defisit_J_sepertiga"] = hasil["J_max"] - hasil["J_sepertiga"]
    pred = s >= SEPERTIGA
    hasil["akurasi_sepertiga"] = float(((pred == (y == 1)).mean()))
    pred2 = s >= hasil["ambang"]
    hasil["akurasi_ambang"] = float(((pred2 == (y == 1)).mean()))
    hasil["sens_band"] = float((br_a >= SEPERTIGA).mean())
    hasil["spes_band"] = float((br_b < SEPERTIGA).mean())
    return hasil


def bootstrap_ambang(br_a, br_b, n=2000, rng=0):
    if roc_curve is None:
        return np.nan, np.nan, np.nan
    r = np.random.default_rng(rng)
    ambs = []
    ia = np.arange(len(br_a)); ib = np.arange(len(br_b))
    for _ in range(n):
        a = br_a[r.choice(ia, len(ia), replace=True)]
        b = br_b[r.choice(ib, len(ib), replace=True)]
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        s = np.r_[a, b]
        fpr, tpr, thr = roc_curve(y, s)
        J = tpr - fpr
        fin = np.isfinite(thr) & (thr <= 1.0 + 1e-12)
        if not fin.any():
            fin = np.isfinite(thr)
        ambs.append(thr[fin][int(np.argmax(J[fin]))])
    ambs = np.array(ambs)
    return float(np.median(ambs)), float(np.percentile(ambs, 2.5)), \
        float(np.percentile(ambs, 97.5))


def tugas_ambang(T, meta, out, ref=(1.0, 0.0), pilih=(1.0, 0.0), aturan="min",
                 n_boot=2000):
    print("\n=== AMBANG, PLATEAU, DAN UJI PERGESERAN-KONSTAN (M5) ===")
    kel = meta["kelompok"].values
    split = meta["split"].astype(str).values
    isA = kel == "A_sepakat_band"
    isB = kel == "B_sepakat_segmented"

    tgrid = np.linspace(0.005, 0.995, 199)
    hr = T.hitung(ref[0], ref[1], aturan)
    br_ref = hr["bridge_ratio"]
    st_ref = hr["status_topologi"]
    J_ref = np.array([_j_pada(br_ref[isA], br_ref[isB], t) for t in tgrid])

    baris = []
    for tau in TAU_GRID:
        for rho in RHO_GRID:
            h = T.hitung(tau, rho, aturan)
            br = h["bridge_ratio"]
            st = h["status_topologi"]
            a = analisis_ambang(br[isA], br[isB])
            a["tau"] = tau; a["rho"] = rho
            a["n_takpecah_A"] = int((st[isA] == "tak_pernah_pecah").sum())
            a["n_takpecah_B"] = int((st[isB] == "tak_pernah_pecah").sum())
            a["n_terpisah0"] = int((st == "terpisah_di_nol").sum())
            # uji pergeseran konstan
            J = np.array([_j_pada(br[isA], br[isB], t) for t in tgrid])
            d = J - J_ref
            a["geser_mean"] = float(d.mean())
            a["geser_sd"] = float(d.std())
            a["geser_rentang"] = float(d.max() - d.min())
            # berapa banyak sel yang nilai br-nya tetap sama persis
            sama = np.isclose(br, br_ref, atol=1e-9, equal_nan=True)
            a["frak_br_identik"] = float(sama.mean())
            tetap_normal = (st == "normal") & (st_ref == "normal")
            a["frak_br_identik_normal"] = float(
                sama[tetap_normal].mean()) if tetap_normal.any() else np.nan
            baris.append(a)
    dfa = pd.DataFrame(baris)
    kol = ["tau", "rho"] + [c for c in dfa.columns if c not in ("tau", "rho")]
    dfa = dfa[kol]
    dfa.to_csv(out / "C3_ambang_grid.csv", index=False)

    nu = dfa["ambang"].nunique()
    print("  ambang Youden di %d kombinasi: %d nilai unik, rentang %.4f, sd %.4f"
          % (len(dfa), nu, dfa["ambang"].max() - dfa["ambang"].min(),
             dfa["ambang"].std()))
    print("  lebar interval optimal PERSIS: median %.4f, maks %.4f"
          % (dfa["lebar_persis"].median(), dfa["lebar_persis"].max()))
    print("  lebar plateau (J dalam 0.001): median %.4f, maks %.4f"
          % (dfa["lebar_plateau_eps"].median(), dfa["lebar_plateau_eps"].max()))
    print("  1/3 di dalam interval optimal PERSIS: %d/%d kombinasi"
          % (int(dfa["sepertiga_di_interval_persis"].sum()), len(dfa)))
    print("  1/3 di dalam plateau eps          : %d/%d kombinasi"
          % (int(dfa["sepertiga_di_plateau_eps"].sum()), len(dfa)))
    print("\n  UJI PERGESERAN-KONSTAN terhadap referensi tau=%g rho=%g:"
          % (ref[0], ref[1]))
    print("    sd[J_komb(t) - J_ref(t)] : median %.5f, maks %.5f"
          % (dfa["geser_sd"].median(), dfa["geser_sd"].max()))
    print("    rentang selisih          : median %.5f, maks %.5f"
          % (dfa["geser_rentang"].median(), dfa["geser_rentang"].max()))
    print("    fraksi sel dengan br identik: median %.4f (sel yang tetap 'normal': %.4f)"
          % (dfa["frak_br_identik"].median(), dfa["frak_br_identik_normal"].median()))
    if dfa["geser_sd"].max() < 0.01:
        print("    -> PERGESERAN PRAKTIS KONSTAN. Invariansi ambang adalah")
        print("       konsekuensi struktural (sel yang berubah semuanya ke br=1.0),")
        print("       BUKAN bukti kekokohan independen. Turunkan klaimnya.")
    else:
        print("    -> pergeseran TIDAK konstan; invariansi ambang lebih berarti.")

    # --- analisis rinci pada parameter terpilih
    h = T.hitung(pilih[0], pilih[1], aturan)
    br = h["bridge_ratio"]; st = h["status_topologi"]
    a = analisis_ambang(br[isA], br[isB])
    med, lo, hi = bootstrap_ambang(br[isA], br[isB], n=n_boot)
    print("\n  -- parameter terpilih tau=%g rho=%g" % pilih)
    print("     AUC %.4f | Youden %.4f (J=%.4f) | interval optimal persis [%.4f, %.4f)"
          % (a["auc"], a["ambang"], a["J_max"], a["ambang"], a["ambang_atas"]))
    print("     plateau eps [%.4f, %.4f] lebar %.4f"
          % (a["plateau_lo"], a["plateau_hi"], a["lebar_plateau_eps"]))
    print("     bootstrap median %.4f, IK95 [%.4f, %.4f], 1/3 di dalam IK: %s"
          % (med, lo, hi, "YA" if lo <= SEPERTIGA <= hi else "TIDAK"))
    print("     J(1/3)=%.4f, defisit terhadap J_maks = %.4f"
          % (a["J_sepertiga"], a["defisit_J_sepertiga"]))
    print("     akurasi @1/3 %.4f | @ambang %.4f | sens band %.4f | spes band %.4f"
          % (a["akurasi_sepertiga"], a["akurasi_ambang"], a["sens_band"], a["spes_band"]))

    # sensitivitas: buang sel tak_pernah_pecah (mereka menumpuk di br=1.0)
    m = st != "tak_pernah_pecah"
    aa = analisis_ambang(br[isA & m], br[isB & m])
    print("     [sensitivitas] tanpa sel 'tak_pernah_pecah' (A n=%d, B n=%d): "
          "AUC %.4f, ambang %.4f, akurasi@1/3 %.4f"
          % (aa["n_A"], aa["n_B"], aa["auc"], aa["ambang"], aa["akurasi_sepertiga"]))

    # generalisasi train -> test
    trA = isA & (split == "train"); trB = isB & (split == "train")
    teA = isA & (split == "test"); teB = isB & (split == "test")
    gen = {}
    if trA.sum() and trB.sum() and teA.sum() and teB.sum():
        at = analisis_ambang(br[trA], br[trB])
        y_te = np.r_[np.ones(int(teA.sum())), np.zeros(int(teB.sum()))]
        s_te = np.r_[br[teA], br[teB]]
        akur_amb = float(((s_te >= at["ambang"]) == (y_te == 1)).mean())
        akur_13 = float(((s_te >= SEPERTIGA) == (y_te == 1)).mean())
        auc_te = float(roc_auc_score(y_te, s_te)) if roc_auc_score else np.nan
        gen = dict(ambang_train=at["ambang"], n_train=int(trA.sum() + trB.sum()),
                   auc_test=auc_te, n_test=int(teA.sum() + teB.sum()),
                   akurasi_test_ambang=akur_amb, akurasi_test_sepertiga=akur_13)
        print("     train->test: ambang train %.4f (n=%d) | AUC test %.4f (n=%d) | "
              "akurasi test @ambang %.4f, @1/3 %.4f"
              % (at["ambang"], gen["n_train"], auc_te, gen["n_test"],
                 akur_amb, akur_13))

    _plot_youden(br, isA, isB, a, out, pilih)
    _plot_sebaran(br, kel, out, pilih)

    rinci = dict(a)
    rinci.update(dict(tau=pilih[0], rho=pilih[1], boot_median=med,
                      boot_lo=lo, boot_hi=hi,
                      sepertiga_di_IK=bool(lo <= SEPERTIGA <= hi),
                      auc_tanpa_takpecah=aa["auc"],
                      ambang_tanpa_takpecah=aa["ambang"],
                      akurasi13_tanpa_takpecah=aa["akurasi_sepertiga"]))
    rinci.update(gen)
    pd.DataFrame([rinci]).to_csv(out / "C4_ambang_terpilih.csv", index=False)
    return dfa, rinci


def _plot_youden(br, isA, isB, a, out, pilih):
    t = np.linspace(0.0, 1.0, 1001)
    J = np.array([_j_pada(br[isA], br[isB], x) for x in t])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, J, lw=1.6, color="#2b5d8a")
    ax.axvspan(a["plateau_lo"], a["plateau_hi"], color="#ffd27f", alpha=0.5,
               label="plateau J dalam 0.001")
    ax.axvline(a["ambang"], color="#c0392b", lw=1.2, label="Youden %.4f" % a["ambang"])
    ax.axvline(SEPERTIGA, color="#27795b", lw=1.2, ls="--", label="1/3")
    ax.axhline(a["J_max"], color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("ambang bridge ratio")
    ax.set_ylabel("Youden J = TPR - FPR")
    ax.set_title("Kurva Youden, tau=%g rho=%g (J_maks=%.4f, J(1/3)=%.4f)"
                 % (pilih[0], pilih[1], a["J_max"], a["J_sepertiga"]), fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "F2C_kurva_youden.png", dpi=140)
    plt.close(fig)
    pd.DataFrame({"ambang": t, "J": J}).to_csv(out / "C4_kurva_youden.csv", index=False)


def _plot_sebaran(br, kel, out, pilih):
    fig, ax = plt.subplots(figsize=(9, 5))
    warna = {"A_sepakat_band": "#2b5d8a", "B_sepakat_segmented": "#c0392b",
             "C_konflik_SNE_band": "#e08a1e", "D_konflik_BNE_segmented": "#7b4fa8",
             "F_tak_bersubtipe": "#2f8f5b"}
    tepi = np.linspace(0, 1, 51)
    for k, c in warna.items():
        v = br[kel == k]
        if len(v) < 5:
            continue
        ax.hist(v, bins=tepi, density=True, histtype="step", lw=1.8,
                color=c, label="%s (n=%d)" % (k, len(v)))
    ax.axvline(SEPERTIGA, color="k", ls="--", lw=1.2, label="1/3")
    ax.set_xlabel("bridge ratio"); ax.set_ylabel("kerapatan")
    ax.set_title("Sebaran bridge ratio, tau=%g rho=%g" % pilih, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "F2C_sebaran.png", dpi=140)
    plt.close(fig)


# ===========================================================================
# BAGIAN 7 -- TUGAS ABLASI PRAPROSES (M6)
# ===========================================================================

VARIAN = [
    # nama,           batasi, lubang, pad, min_frak, pakai_tau_terpilih, diskret
    ("final",              1, 1, 1, 0.02, True,  0.0),
    ("tanpa_batas_sel",    0, 1, 1, 0.02, True,  0.0),
    ("tanpa_isi_lubang",   1, 0, 1, 0.02, True,  0.0),
    ("tanpa_pad",          1, 1, 0, 0.02, True,  0.0),
    ("tanpa_filter",       1, 1, 1, 0.02, False, 0.0),
    ("diskret_0p5",        1, 1, 1, 0.02, True,  0.5),
    ("emulasi_v1",         0, 0, 0, 0.0,  False, 0.5),
]


def tugas_ablasi(df_neu, peta, out, pekerja, pilih_br, aturan="min",
                 dir_fase1=None, prune=0.05, paksa=False):
    print("\n=== ABLASI PRAPROSES: MENJELASKAN AUC v1 0.9227 vs v2 ~0.88 (M6) ===")
    isA = (df_neu["kelompok"] == "A_sepakat_band").values
    isB = (df_neu["kelompok"] == "B_sepakat_segmented").values

    v1 = None
    if dir_fase1 is not None:
        p = Path(dir_fase1) / "bridge_ratio.csv"
        if p.exists():
            v1 = pd.read_csv(p)[["img_name", "bridge_ratio"]].rename(
                columns={"bridge_ratio": "br_v1"})
            print("  pembanding v1 dimuat: %d baris" % len(v1))
        else:
            print("  (hasil_fase1/bridge_ratio.csv tidak ditemukan, korelasi v1 dilewati)")

    baris = []
    for nama, bs, lb, pd_, mf, pakai, disk in VARIAN:
        cfg = (bool(bs), bool(lb), bool(pd_), float(mf), 8, prune)
        cache, _ = bangun_cache(df_neu, peta, cfg, out, pekerja,
                                tandai="_neu_" + nama, paksa=paksa)
        nm = [n for n in df_neu["img_name"] if n in cache]
        T = Turunan(cache, nm)
        tau, rho = (pilih_br if pakai else (0.0, 0.0))
        h = T.hitung(tau, rho, aturan, diskret=disk)
        d = pd.DataFrame({"img_name": h["img_name"], "br": h["bridge_ratio"],
                          "rl": h["r_lobus"], "st": h["status_topologi"],
                          "rp": h["r_pisah"]})
        d = df_neu.merge(d, on="img_name", how="inner")
        a = analisis_ambang(d.loc[d["kelompok"] == "A_sepakat_band", "br"].values,
                            d.loc[d["kelompok"] == "B_sepakat_segmented", "br"].values)
        r = dict(varian=nama, tau=tau, rho=rho, diskret=disk,
                 batasi_sel=bs, isi_lubang=lb, pad=pd_, min_frak=mf,
                 n=len(d), auc=a["auc"], ambang=a["ambang"],
                 akurasi_sepertiga=a["akurasi_sepertiga"],
                 r_lobus_maks=float(d["rl"].max()),
                 r_lobus_mean=float(d["rl"].mean()),
                 frak_takpecah=float((d["st"] == "tak_pernah_pecah").mean()),
                 frak_terpisah0=float((d["st"] == "terpisah_di_nol").mean()))
        if v1 is not None:
            g = d.merge(v1, on="img_name", how="inner")
            if len(g) > 10:
                r["pearson_v1"] = float(g["br"].corr(g["br_v1"]))
                r["spearman_v1"] = float(g["br"].corr(g["br_v1"], method="spearman"))
                r["pindah_sisi_13"] = int((((g["br"] >= SEPERTIGA).astype(int) !=
                                            (g["br_v1"] >= SEPERTIGA).astype(int))).sum())
                r["n_banding_v1"] = int(len(g))
        baris.append(r)
        print("  %-18s AUC %.4f | ambang %.4f | akur@1/3 %.4f | r_lobus maks %.2f | "
              "tak pecah %.3f%s"
              % (nama, r["auc"], r["ambang"], r["akurasi_sepertiga"],
                 r["r_lobus_maks"], r["frak_takpecah"],
                 (" | Pearson v1 %.3f" % r["pearson_v1"]) if "pearson_v1" in r else ""))

    dfab = pd.DataFrame(baris)
    dfab.to_csv(out / "C5_ablasi_praproses.csv", index=False)
    base = dfab.loc[dfab["varian"] == "final", "auc"].iloc[0]
    print("\n  selisih AUC terhadap varian 'final' (%.4f):" % base)
    for _, r in dfab.iterrows():
        if r["varian"] == "final":
            continue
        print("    %-18s %+.4f" % (r["varian"], r["auc"] - base))
    print("  target yang harus dijelaskan: v1 = 0.9227")
    return dfab


# ===========================================================================
# BAGIAN 8 -- AUDIT EKOR B DAN MONTASE (M7, M8)
# ===========================================================================

def _potret(m, e, bbox_margin=12):
    y0, x0, y1, x1 = e["bbox"]
    h, w = m.shape
    y0 = max(0, y0 - bbox_margin); x0 = max(0, x0 - bbox_margin)
    y1 = min(h, y1 + bbox_margin); x1 = min(w, x1 + bbox_margin)
    sub = m[y0:y1, x0:x1]
    rgb = np.zeros(sub.shape + (3,), dtype=np.uint8)
    rgb[sub == 0] = (18, 18, 24)
    rgb[sub == NILAI_SITOPLASMA] = (120, 140, 175)
    rgb[sub == NILAI_NUKLEUS] = (235, 235, 245)
    rgb[sub == 3] = (90, 70, 70)
    rgb[sub == 4] = (55, 45, 55)
    rgb[sub == NILAI_VAKUOLA] = (170, 70, 70)
    return rgb, (y0, x0)


def buat_montase(sub, cache, peta, out, judul, nama_file, tau, rho,
                 aturan="min", n=24):
    if len(sub) == 0:
        print("  (montase %s dilewati: tidak ada sel)" % nama_file)
        return
    sub = sub.head(n)
    k = int(math.ceil(len(sub) / 6.0))
    fig, axes = plt.subplots(k, 6, figsize=(16, 2.9 * k))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, (_, r) in enumerate(sub.iterrows()):
        e = cache.get(r["img_name"])
        p = cari_mask(peta, r["img_name"])
        if e is None or p is None:
            continue
        m = baca_mask(p)
        rgb, (oy, ox) = _potret(m, e)
        ax = axes[i]
        ax.imshow(rgb)
        K = e["kejadian"]; P = e["pos"]
        if K.shape[0] and r["status_topologi"] == "normal":
            rl = e["r_lobus"]
            sel = (K[:, 1] >= tau) & (K[:, 2] >= rho * rl)
            if sel.any():
                j = int(np.argmin(K[sel, 0]) if aturan == "min"
                        else np.argmax(K[sel, 0]))
                yy, xx = P[sel][j]
                ax.plot(xx - ox, yy - oy, "+", color="#ff2d2d", ms=13, mew=2.2)
        ax.set_title("br=%.2f rp=%s lob=%d\n%s"
                     % (r["bridge_ratio"],
                        ("%.1f" % r["r_pisah"]) if not pd.isna(r["r_pisah"]) else "nan",
                        int(r["n_lobus"]), r["status_topologi"]), fontsize=7)
        ax.axis("off")
    fig.suptitle(judul, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / nama_file, dpi=125)
    plt.close(fig)
    print("  montase disimpan: %s (%d panel)" % (nama_file, len(sub)))


def tugas_audit(dfin, cache, peta, out, pilih, aturan="min"):
    print("\n=== AUDIT EKOR B DAN MONTASE (M7, M8) ===")
    tau, rho = pilih
    d = dfin.copy()
    d["pred_band"] = d["bridge_ratio"] >= SEPERTIGA

    A = d[d["kelompok"] == "A_sepakat_band"]
    B = d[d["kelompok"] == "B_sepakat_segmented"]
    print("  matriks konfusi pada ambang 1/3")
    print("    A (band)      -> band %d, segmented %d  (benar %.2f%%)"
          % (int(A["pred_band"].sum()), int((~A["pred_band"]).sum()),
             100 * A["pred_band"].mean()))
    print("    B (segmented) -> band %d, segmented %d  (benar %.2f%%)"
          % (int(B["pred_band"].sum()), int((~B["pred_band"]).sum()),
             100 * (~B["pred_band"]).mean()))

    salah = B[B["pred_band"]].copy()
    tp = salah[salah["status_topologi"] == "tak_pernah_pecah"]
    nm = salah[salah["status_topologi"] == "normal"]
    print("\n  ekor B (%d sel salah): tak_pernah_pecah %d (%.1f%%), "
          "normal/leher lebar %d (%.1f%%), terpisah_di_nol %d"
          % (len(salah), len(tp), 100 * len(tp) / max(len(salah), 1),
             len(nm), 100 * len(nm) / max(len(salah), 1),
             int((salah["status_topologi"] == "terpisah_di_nol").sum())))
    if len(nm):
        print("    leher lebar: r_pisah median %.2f px, q25 %.2f, q75 %.2f, "
              "r_lobus mean %.2f"
              % (nm["r_pisah"].median(), nm["r_pisah"].quantile(.25),
                 nm["r_pisah"].quantile(.75), nm["r_lobus"].mean()))
    salah.to_csv(out / "C6_ekor_B.csv", index=False)

    print("\n  status artefak terpisah_di_nol setelah praproses M4:")
    neu = d[d["label"] == "Neutrophil"]
    print("    neutrofil terpisah_di_nol: %d (%.2f%%)  [Fase 2B: 157 / 4.72%%]"
          % (int((neu["status_topologi"] == "terpisah_di_nol").sum()),
             100 * (neu["status_topologi"] == "terpisah_di_nol").mean()))
    print("    sel dengan komponen nukleus mentah > 1: %d  [Fase 2B: 170]"
          % int((d["komponen_mentah"] > 1).sum()))
    print("    sel dengan komponen nukleus AKHIR > 1  : %d"
          % int((d["komponen_akhir"] > 1).sum()))

    # M7: 130 sel B berleher lebar, distratifikasi bukan hanya ujung
    if len(nm):
        s = nm.sort_values("r_pisah")
        ambil = s.iloc[np.linspace(0, len(s) - 1, min(24, len(s))).astype(int)]
        buat_montase(ambil, cache, peta, out,
                     "M7: sel B disebut band, status NORMAL (leher terukur lebar)",
                     "montase_M7_B_leher_lebar.png", tau, rho, aturan)
    # M8: stratifikasi per status
    for st, judul, berkas in [
        ("tak_pernah_pecah", "M8: sel B disebut band, status TAK_PERNAH_PECAH",
         "montase_M8_B_takpernahpecah.png"),
    ]:
        s = salah[salah["status_topologi"] == st]
        if len(s):
            s = s.sort_values("r_lobus", ascending=False)
            ambil = s.iloc[np.linspace(0, len(s) - 1, min(24, len(s))).astype(int)]
            buat_montase(ambil, cache, peta, out, judul, berkas, tau, rho, aturan)

    s = d[(d["kelompok"] == "B_sepakat_segmented") & (~d["pred_band"])]
    if len(s):
        ambil = s.sample(min(24, len(s)), random_state=0)
        buat_montase(ambil, cache, peta, out, "Pembanding: sel B benar segmented",
                     "montase_M8_B_benar.png", tau, rho, aturan)
    s = d[(d["kelompok"] == "A_sepakat_band") & (d["pred_band"])]
    if len(s):
        ambil = s.sample(min(24, len(s)), random_state=0)
        buat_montase(ambil, cache, peta, out, "Pembanding: sel A benar band",
                     "montase_M8_A_benar.png", tau, rho, aturan)
    return salah


# ===========================================================================
# BAGIAN 9 -- TABEL RINGKAS DAN LAPORAN
# ===========================================================================

def df_ke_md(df, ndesimal=4):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else ("%.*f" % (ndesimal, v)))
        else:
            d[c] = d[c].astype(str)
    kol = list(d.columns)
    out = ["| " + " | ".join(kol) + " |",
           "|" + "|".join(["---"] * len(kol)) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(r[c] for c in kol) + " |")
    return "\n".join(out)


def tabel_kelompok(dfin):
    b = []
    for k in URUT_KELOMPOK:
        s = dfin[dfin["kelompok"] == k]
        if not len(s):
            continue
        p = float((s["bridge_ratio"] >= SEPERTIGA).mean())
        ent = 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        b.append(dict(kelompok=k, n=len(s),
                      q25=s["bridge_ratio"].quantile(.25),
                      median=s["bridge_ratio"].median(),
                      q75=s["bridge_ratio"].quantile(.75),
                      p_band=p, entropi=ent,
                      zona_28_38=float(s["bridge_ratio"].between(0.28, 0.38).mean()),
                      zona_23_43=float(s["bridge_ratio"].between(0.23, 0.43).mean()),
                      tak_pecah=float((s["status_topologi"] == "tak_pernah_pecah").mean()),
                      mean_lobus=float(s["n_lobus"].mean())))
    t = pd.DataFrame(b)
    mA = t.loc[t["kelompok"] == "A_sepakat_band", "median"]
    mB = t.loc[t["kelompok"] == "B_sepakat_segmented", "median"]
    if len(mA) and len(mB) and abs(float(mB.iloc[0]) - float(mA.iloc[0])) > 1e-9:
        t["posisi_AB"] = (t["median"] - float(mA.iloc[0])) / \
                         (float(mB.iloc[0]) - float(mA.iloc[0]))
    return t


def tabel_bentuk(dfin):
    b = []
    for s_ in URUT_BENTUK:
        s = dfin[dfin["nucleus_shape"] == s_]
        if not len(s):
            continue
        b.append(dict(nucleus_shape=s_, n=len(s),
                      mean_lobus=float(s["n_lobus"].mean()),
                      median_lobus=float(s["n_lobus"].median()),
                      hanya_1_lobus=float((s["n_lobus"] == 1).mean()),
                      median_br=float(s["bridge_ratio"].median())))
    return pd.DataFrame(b)


def tabel_kelas(dfin):
    b = []
    for k in ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]:
        s = dfin[dfin["label"] == k]
        if not len(s):
            continue
        b.append(dict(kelas=k, n=len(s), median_br=float(s["bridge_ratio"].median()),
                      tak_pernah_pecah=float((s["status_topologi"] == "tak_pernah_pecah").mean()),
                      mean_lobus=float(s["n_lobus"].mean()),
                      terpisah_di_nol=float((s["status_topologi"] == "terpisah_di_nol").mean())))
    return pd.DataFrame(b)


def tulis_laporan(out, ctx):
    L = []
    A = L.append
    A("# LAPORAN FASE 2C\n")
    A("Dihasilkan otomatis oleh `fase2c_kalibrasi_dua_sisi.py` pada %s.\n"
      % time.strftime("%Y-%m-%d %H:%M"))
    A("Aturan saddle: **%s**. Angka memakai titik desimal.\n" % ctx.get("aturan", "min"))
    A("\n## 0. Konfigurasi\n")
    for k in ["n_sel", "praproses", "tau_lobus", "rho_lobus", "tau_br", "rho_br",
              "sanitas_lolos", "replikasi_kelompok"]:
        if k in ctx:
            A("- **%s**: %s" % (k, ctx[k]))
    if "diag" in ctx:
        A("\n## 1. Diagnostik praproses M4 (pembatasan ke komponen sel target)\n")
        A(df_ke_md(ctx["diag"]))
    if "kal" in ctx:
        A("\n## 2. Kalibrasi dua sisi (M1, M2, M3)\n")
        A("Grid: tau %s x rho %s = %d kombinasi. Dikalibrasi HANYA pada sel "
          "non-neutrofil, sehingga tidak ada sel yang dipakai untuk menyetel "
          "parameter sekaligus untuk menguji ambang.\n"
          % (TAU_GRID, RHO_GRID, len(TAU_GRID) * len(RHO_GRID)))
        A("\n### 2.1 Sepuluh kombinasi teratas menurut skor makro-3 (jumlah lobus)\n")
        A(df_ke_md(ctx["kal"].nlargest(10, "skor_makro3")[
            ["tau", "rho", "skor_makro3", "skor_mentah", "rec_t1", "rec_t2",
             "rec_t3", "limf_tak_pecah", "limf_lobus"]]))
        A("\n### 2.2 Sepuluh kombinasi teratas menurut skor status (bridge ratio)\n")
        A(df_ke_md(ctx["kal"].nlargest(10, "skor_br")[
            ["tau", "rho", "skor_br", "rec_harus_pecah", "rec_harus_utuh",
             "skor_makro3", "limf_tak_pecah"]]))
        A("\n### 2.3 Pembanding parameter lama\n")
        s = ctx["kal"]
        sub = s[((s["tau"] == 4.0) & (s["rho"] == 0.5)) |
                ((s["tau"] == 0.1) & (s["rho"] == 0.3)) |
                ((s["tau"] == 1.0) & (s["rho"] == 0.0))]
        A(df_ke_md(sub[["tau", "rho", "skor_mentah", "skor_makro3", "skor_br",
                        "rec_t1", "rec_t2", "rec_t3", "limf_tak_pecah"]]))
    if "amb" in ctx:
        A("\n## 3. Ambang, plateau, dan uji pergeseran-konstan (M5)\n")
        a = ctx["amb"]
        A("- nilai ambang unik di %d kombinasi: **%d**, rentang %.4f, sd %.4f"
          % (len(a), a["ambang"].nunique(),
             a["ambang"].max() - a["ambang"].min(), a["ambang"].std()))
        A("- lebar interval optimal PERSIS: median %.4f, maks %.4f"
          % (a["lebar_persis"].median(), a["lebar_persis"].max()))
        A("- lebar plateau (J dalam 0.001): median %.4f, maks %.4f"
          % (a["lebar_plateau_eps"].median(), a["lebar_plateau_eps"].max()))
        A("- 1/3 di interval optimal persis: %d/%d kombinasi"
          % (int(a["sepertiga_di_interval_persis"].sum()), len(a)))
        A("- **uji pergeseran-konstan**: sd[J_komb(t) - J_ref(t)] median %.5f, "
          "maks %.5f; rentang selisih maks %.5f"
          % (a["geser_sd"].median(), a["geser_sd"].max(), a["geser_rentang"].max()))
        A("- fraksi sel dengan bridge ratio identik terhadap referensi: median %.4f"
          % a["frak_br_identik"].median())
        if a["geser_sd"].max() < 0.01:
            A("\n> **Kesimpulan M5.** Pergeseran J praktis konstan di seluruh grid. "
              "Invariansi ambang adalah konsekuensi struktural dari fakta bahwa "
              "sel yang berubah status semuanya mendarat di br = 1.0, di atas "
              "setiap kandidat ambang. Klaim di Bagian 14 dokumen migrasi harus "
              "diturunkan dari 'ambang kebal terhadap parameter' menjadi "
              "'parameter tidak mengubah urutan sel yang terukur'.\n")
        else:
            A("\n> **Kesimpulan M5.** Pergeseran J tidak konstan; invariansi ambang "
              "membawa informasi nyata.\n")
    if "rinci" in ctx:
        A("\n### 3.1 Parameter terpilih\n")
        A(df_ke_md(pd.DataFrame([ctx["rinci"]]).T.reset_index().rename(
            columns={"index": "metrik", 0: "nilai"}), ndesimal=4))
    if "kel" in ctx:
        A("\n## 4. Kelompok neutrofil pada parameter terpilih\n")
        A(df_ke_md(ctx["kel"]))
    if "bentuk" in ctx:
        A("\n## 5. Jumlah lobus per nucleus_shape\n")
        A(df_ke_md(ctx["bentuk"]))
    if "kelas" in ctx:
        A("\n## 6. Kontrol lintas kelas sel\n")
        A(df_ke_md(ctx["kelas"]))
    if "abl" in ctx:
        A("\n## 7. Ablasi praproses (M6)\n")
        A("Target yang dijelaskan: AUC v1 = 0.9227 versus v2 sekitar 0.88.\n")
        A(df_ke_md(ctx["abl"]))
    if "ekor" in ctx:
        A("\n## 8. Audit ekor B (M7)\n")
        e = ctx["ekor"]
        A("- sel B salah klasifikasi: **%d**" % len(e))
        vc = e["status_topologi"].value_counts()
        for k, v in vc.items():
            A("  - %s: %d (%.1f%%)" % (k, v, 100 * v / max(len(e), 1)))
    A("\n## 9. Yang harus dijawab berikutnya\n")
    A("- Apakah optimum kalibrasi masih di pojok grid? Bila ya, perluas lagi.")
    A("- Apakah plateau Youden cukup lebar sehingga 0.3588 harus ditulis "
      "sebagai interval, bukan titik?")
    A("- Apakah ablasi sudah menutup selisih AUC v1 vs v2 secara kuantitatif?")
    A("- Fase 2D (analisis deferral) dan Fase 2E (sumbu pematangan ig).\n")
    (out / "LAPORAN_FASE2C.md").write_text("\n".join(L), encoding="utf-8")
    print("\n  laporan ditulis: %s" % (out / "LAPORAN_FASE2C.md"))


# ===========================================================================
# BAGIAN 10 -- MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fase 2C: kalibrasi dua sisi, pembersihan artefak, plateau Youden.")
    ap.add_argument("--dir-kerja", default=".",
                    help="direktori kerja (tempat CSV dan folder hasil)")
    ap.add_argument("--csv", default=None,
                    help="path CSV atribut (default: <dir-kerja>/pbc_attr_v1_ccrop_all.csv)")
    ap.add_argument("--dir-mask", default=None,
                    help="folder mask (default: <dir-kerja>/pbcseg_final_v1)")
    ap.add_argument("--dir-fase1", default=None,
                    help="folder hasil Fase 1 untuk pembanding v1 "
                         "(default: <dir-kerja>/hasil_fase1)")
    ap.add_argument("--out", default=None,
                    help="folder keluaran (default: <dir-kerja>/hasil_fase2c)")
    ap.add_argument("--pekerja", type=int, default=7)
    ap.add_argument("--sampel", type=int, default=0,
                    help="pakai N sel saja (terstratifikasi) untuk uji cepat")
    ap.add_argument("--tugas", default="semua",
                    help="sanitas,cache,kalibrasi,ambang,ablasi,audit,laporan,semua")
    ap.add_argument("--aturan", default="min", choices=["min", "max"])
    ap.add_argument("--min-frak-komp", type=float, default=0.02)
    ap.add_argument("--prune", type=float, default=0.05)
    ap.add_argument("--tanpa-batas-sel", action="store_true",
                    help="matikan praproses M4 (untuk membandingkan dengan 2B)")
    ap.add_argument("--tau-lobus", type=float, default=None)
    ap.add_argument("--rho-lobus", type=float, default=None)
    ap.add_argument("--tau-br", type=float, default=None)
    ap.add_argument("--rho-br", type=float, default=None)
    ap.add_argument("--ref-tau", type=float, default=1.0,
                    help="kombinasi referensi untuk uji pergeseran-konstan")
    ap.add_argument("--ref-rho", type=float, default=0.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--paksa-cache", action="store_true")
    args = ap.parse_args()

    dk = Path(args.dir_kerja)
    out = Path(args.out) if args.out else dk / "hasil_fase2c"
    out.mkdir(parents=True, exist_ok=True)
    tugas = set(t.strip() for t in args.tugas.split(","))
    semua = "semua" in tugas

    def mau(t):
        return semua or t in tugas

    print("=" * 76)
    print("FASE 2C -- kalibrasi dua sisi, pembersihan artefak, uji kejujuran ambang")
    print("=" * 76)
    print("keluaran: %s" % out.resolve())

    # ---- 1. SANITAS (selalu, lesson #1 Bagian 16)
    lolos, _ = jalankan_sanitas(out)
    if not lolos:
        print("\nHENTI: uji sanitas gagal. Perbaiki implementasi sebelum data asli.")
        sys.exit(2)
    if tugas == {"sanitas"}:
        return

    # ---- 2. DATA
    csvp = Path(args.csv) if args.csv else dk / "pbc_attr_v1_ccrop_all.csv"
    if not csvp.exists():
        kand = list(dk.rglob("pbc_attr_v1_ccrop_all.csv"))
        if kand:
            csvp = kand[0]
        else:
            print("HENTI: CSV atribut tidak ditemukan. Pakai --csv.")
            sys.exit(2)
    df = pd.read_csv(csvp)
    print("\nCSV: %s (%d baris)" % (csvp, len(df)))

    dm = Path(args.dir_mask) if args.dir_mask else dk / "pbcseg_final_v1"
    if not dm.exists():
        print("HENTI: folder mask tidak ditemukan. Pakai --dir-mask.")
        sys.exit(2)
    peta, n_unik = indeks_mask(dm)
    print("mask terindeks: %d berkas unik (alias _ccrop tidak dihitung ganda)" % n_unik)

    pref, kel = beri_kelompok(df)
    df["prefix"] = pref
    df["kelompok"] = kel

    penuh = args.sampel <= 0
    if not penuh:
        # stratifikasi per (label, kelompok) supaya A/B/C/D/E/F tetap terwakili
        kunci = df["label"].astype(str) + "|" + df["kelompok"].astype(str)
        n_grup = kunci.nunique()
        n_per = max(4, args.sampel // max(n_grup, 1))
        pilih = []
        for _, s in df.groupby(kunci, sort=False):
            pilih.extend(s.sample(min(len(s), n_per), random_state=0).index.tolist())
        df = df.loc[pilih].reset_index(drop=True)
        print("MODE SAMPEL: %d sel dari %d grup (label x kelompok)" % (len(df), n_grup))
    rep = verifikasi_kelompok(df, penuh=penuh)

    cfg = (not args.tanpa_batas_sel, True, True, args.min_frak_komp, 8, args.prune)
    print("\npraproses: batasi_komponen_sel=%s, isi_lubang=True, pad=True, "
          "min_frak_komp=%g, conn=8, prune=%g"
          % (cfg[0], args.min_frak_komp, args.prune))

    # ---- 3. CACHE
    cache, fc = bangun_cache(df, peta, cfg, out, args.pekerja, paksa=args.paksa_cache)
    nama = [n for n in df["img_name"] if n in cache]
    meta = df.set_index("img_name").loc[nama].reset_index()
    T = Turunan(cache, nama)
    print("cache siap: %d sel" % len(nama))

    diag = pd.DataFrame([{
        "sel_total": len(nama),
        "komponen_nukleus_mentah_>1": int(sum(cache[n]["komponen_mentah"] > 1 for n in nama)),
        "komponen_nukleus_akhir_>1": int(sum(cache[n]["komponen_akhir"] > 1 for n in nama)),
        "sel_kehilangan_piksel_luar_sel": int(sum(cache[n]["piks_buang_luar_sel"] > 0 for n in nama)),
        "piksel_dibuang_luar_sel_total": int(sum(cache[n]["piks_buang_luar_sel"] for n in nama)),
        "sel_kehilangan_serpihan_kecil": int(sum(cache[n]["piks_buang_kecil"] > 0 for n in nama)),
        "sel_dengan_lubang": int(sum(cache[n]["piks_lubang"] > 0 for n in nama)),
        "pusat_fallback": int(sum(cache[n]["pusat_fallback"] > 0 for n in nama)),
        "pusat_beda_terbesar": int(sum(cache[n]["pusat_beda_terbesar"] > 0 for n in nama)),
        "r_lobus_maks": float(max(cache[n]["r_lobus"] for n in nama)),
    }])
    print("\n=== DIAGNOSTIK PRAPROSES M4 ===")
    for k, v in diag.iloc[0].items():
        print("  %-34s %s" % (k, ("%.4f" % v) if k == "r_lobus_maks" else "%d" % int(v)))
    print("  (Fase 2B sebagai pembanding: 170 mask >1 komponen, 157 neutrofil")
    print("   terpisah_di_nol, 578 nukleus berlubang, r_lobus maks 51.87)")
    diag.to_csv(out / "C1_diagnostik_praproses.csv", index=False)

    ctx = dict(n_sel=len(nama), aturan=args.aturan,
               praproses=("batasi_sel=%s, min_frak=%g, prune=%g"
                          % (cfg[0], args.min_frak_komp, args.prune)),
               sanitas_lolos=lolos, replikasi_kelompok=rep, diag=diag)

    # ---- 4. KALIBRASI
    pil_l = (args.tau_lobus, args.rho_lobus)
    pil_b = (args.tau_br, args.rho_br)
    if mau("kalibrasi"):
        dfk, bl, bb = tugas_kalibrasi(T, meta, out, aturan=args.aturan)
        ctx["kal"] = dfk
        if pil_l[0] is None:
            pil_l = bl
        if pil_b[0] is None:
            pil_b = bb
    if pil_l[0] is None:
        pil_l = (1.0, 0.0)
    if pil_b[0] is None:
        pil_b = (1.0, 0.0)
    pil_l = (float(pil_l[0]), float(pil_l[1]))
    pil_b = (float(pil_b[0]), float(pil_b[1]))
    ctx.update(tau_lobus=pil_l[0], rho_lobus=pil_l[1],
               tau_br=pil_b[0], rho_br=pil_b[1])
    print("\nPARAMETER YANG DIPAKAI: lobus (tau=%g, rho=%g) | bridge ratio (tau=%g, rho=%g)"
          % (pil_l[0], pil_l[1], pil_b[0], pil_b[1]))

    # ---- 5. DATA INTI PADA PARAMETER TERPILIH
    h = T.frame(pil_b[0], pil_b[1], args.aturan)
    hl = T.frame(pil_l[0], pil_l[1], args.aturan)
    h["n_lobus_kal"] = hl["n_lobus"].values
    dfin = meta.merge(h, on="img_name", how="left")
    dfin["tau_br"] = pil_b[0]; dfin["rho_br"] = pil_b[1]
    dfin["tau_lobus"] = pil_l[0]; dfin["rho_lobus"] = pil_l[1]
    dfin.to_csv(out / "bridge_ratio_2c.csv", index=False)
    print("data inti disimpan: bridge_ratio_2c.csv (%d baris)" % len(dfin))

    dfl = meta.merge(hl, on="img_name", how="left")
    ctx["bentuk"] = tabel_bentuk(dfl)
    ctx["kelas"] = tabel_kelas(dfl)
    print("\n=== JUMLAH LOBUS PER nucleus_shape (parameter lobus) ===")
    print(ctx["bentuk"].to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\n=== KONTROL LINTAS KELAS (parameter lobus) ===")
    print(ctx["kelas"].to_string(index=False, float_format=lambda v: "%.4f" % v))

    # ---- 6. AMBANG
    if mau("ambang"):
        dfa, rinci = tugas_ambang(T, meta, out, ref=(args.ref_tau, args.ref_rho),
                                  pilih=pil_b, aturan=args.aturan, n_boot=args.boot)
        ctx["amb"] = dfa
        ctx["rinci"] = rinci

    ctx["kel"] = tabel_kelompok(dfin)
    print("\n=== KELOMPOK NEUTROFIL (parameter bridge ratio) ===")
    print(ctx["kel"].to_string(index=False, float_format=lambda v: "%.4f" % v))

    # ---- 7. AUDIT DAN MONTASE
    if mau("audit"):
        ctx["ekor"] = tugas_audit(dfin, cache, peta, out, pil_b, args.aturan)

    # ---- 8. ABLASI
    if mau("ablasi"):
        d1 = Path(args.dir_fase1) if args.dir_fase1 else dk / "hasil_fase1"
        neu = df[df["label"] == "Neutrophil"].copy()
        ctx["abl"] = tugas_ablasi(neu, peta, out, args.pekerja, pil_b,
                                  aturan=args.aturan, dir_fase1=d1,
                                  prune=args.prune, paksa=args.paksa_cache)

    # ---- 9. LAPORAN
    if mau("laporan"):
        tulis_laporan(out, ctx)

    print("\nSELESAI. Kirimkan LAPORAN_FASE2C.md beserta C1-C6*.csv untuk analisis.")


if __name__ == "__main__":
    main()
