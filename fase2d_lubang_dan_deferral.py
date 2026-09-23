#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fase2d_lubang_dan_deferral.py
============================================================================
FASE 2D -- Aturan pengisian lubang, dan analisis deferral.

LATAR BELAKANG
--------------
Ablasi Fase 2C menemukan bahwa MEMATIKAN `binary_fill_holes` menaikkan AUC
A-vs-B dari 0.9072 ke 0.9664 dan memangkas `tak_pernah_pecah` dari 5.44%
ke 1.86%. Dugaan mekanismenya: `binary_fill_holes` pada (mask == 2) mengisi
SETIAP region tertutup, termasuk region yang label aslinya SITOPLASMA.
Nukleus segmented yang lobusnya tersusun melingkar mengurung sitoplasma di
tengah; pengisian menyambung lobus-lobus itu lewat massa tengah yang lebar,
leher sejatinya hilang, dan sel divonis band.

ATURAN MAIN YANG DIPEGANG SKRIP INI
-----------------------------------
Keputusan tentang aturan lubang TIDAK BOLEH diambil dari AUC A-vs-B, karena
itu data pengujian (pelajaran 3, Bagian 16 dokumen migrasi). Maka:

  KRITERIA PRIMER   : sensus mekanistik. Dari apa lubang itu tersusun? Bila
                      mayoritas pikselnya berlabel sitoplasma/vakuola, maka
                      mengisinya salah secara faktual, apa pun efeknya.
  KRITERIA SEKUNDER : skor kalibrasi pada sel NON-NEUTROFIL (independen dari
                      kelompok A/B yang dipakai menguji ambang).
  KONSEKUENSI       : AUC A-vs-B dilaporkan sesudahnya, dengan nama kolom
                      berakhiran _KONSEKUENSI, dan tidak pernah dipakai
                      untuk memilih.

TIGA ATURAN LUBANG
------------------
  semua    : isi semua region tertutup           (perilaku Fase 2A-2C)
  tidak    : jangan isi apa pun                  (perilaku v1)
  selektif : isi hanya bila region KECIL (luas <= --lubang-maks-px) ATAU
             bukan jaringan (fraksi piksel sitoplasma+vakuola < --lubang-frak-sel)

YANG DIKERJAKAN
---------------
  T1  sensus lubang per sel dan per region, lengkap dengan komposisi label
  T2  ablasi tiga aturan lubang, dinilai pada kriteria independen,
      plus perbandingan BERPASANGAN: galat ekor B mana yang diperbaiki
  T3  kalibrasi dua sisi ulang pada aturan terpilih
  T4  ambang, plateau, interval optimal persis, bootstrap, train->test
  T5  audit ekor B + diagnostik JARAK SADDLE KE PIKSEL LUBANG TERISI
      (uji langsung: apakah leher lebar itu buatan pengisian?)
  T6  analisis deferral / selective prediction + pengayaan kelompok C
  T7  laporan markdown

PEMAKAIAN
---------
  python fase2d_lubang_dan_deferral.py --tugas sanitas        # wajib, ~10 dtk
  python fase2d_lubang_dan_deferral.py --sampel 400 --pekerja 7
  python fase2d_lubang_dan_deferral.py --pekerja 7            # ~20 menit
  python fase2d_lubang_dan_deferral.py --tugas deferral,laporan   # dari cache

TUGAS: sanitas, sensus, ablasi, kalibrasi, ambang, audit, deferral, laporan, semua
============================================================================
"""

import argparse
import gzip
import math
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
except Exception:                                      # pragma: no cover
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

LATAR, SITOPLASMA, NUKLEUS, TROMBOSIT, SEL_LAIN, VAKUOLA = 0, 1, 2, 3, 4, 5
NAMA_NILAI = {0: "latar", 1: "sitoplasma", 2: "nukleus", 3: "trombosit",
              4: "sel_lain", 5: "vakuola"}
# Region tertutup yang isinya ini dianggap JARINGAN NYATA, bukan artefak.
NILAI_JARINGAN = (SITOPLASMA, VAKUOLA)

ATURAN_LUBANG = ("semua", "tidak", "selektif")

TAU_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
            2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

BENTUK_HARAPAN = {"unsegmented-round": 1, "unsegmented-indented": 1,
                  "segmented-bilobed": 2, "segmented-multilobed": 3}
URUT_BENTUK = ["unsegmented-round", "unsegmented-indented", "irregular",
               "unsegmented-band", "segmented-bilobed", "segmented-multilobed"]
URUT_KELOMPOK = ["A_sepakat_band", "B_sepakat_segmented", "C_konflik_SNE_band",
                 "D_konflik_BNE_segmented", "E_lainnya", "F_tak_bersubtipe"]
JUMLAH_KELOMPOK_HARAPAN = {"A_sepakat_band": 1555, "B_sepakat_segmented": 976,
                           "C_konflik_SNE_band": 662,
                           "D_konflik_BNE_segmented": 48,
                           "E_lainnya": 38, "F_tak_bersubtipe": 50}
SEPERTIGA = 1.0 / 3.0
JARAK_JAUH = 999.0

# Pembanding dari fase sebelumnya (dicetak berdampingan, bukan dipakai menyetel)
REF_2C = dict(auc=0.9072334088872488, akurasi13=0.9043856183326748,
              ambang=0.3587657577822075, frak_takpecah=0.05437068188645239,
              ekor_B=195, takpecah_neu=0.1334, terpisah0_neu=0.0228,
              posisi_C=0.3214)


# ===========================================================================
# BAGIAN 1 -- PRAPROSES: SENSUS DAN ATURAN LUBANG
# ===========================================================================

def baca_mask(path):
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def sensus_lubang(m, nuk, aturan="selektif", maks_px=15, frak_sel=0.5):
    """
    Cari semua region tertutup di dalam nukleus, catat komposisi labelnya,
    lalu putuskan mana yang diisi menurut `aturan`.

    Kembalikan (nukleus_setelah_isi, piksel_yang_diisi, info).
    info["region"] = list (luas, nilai_dominan, frak_jaringan, diisi).
    """
    info = dict(n_region=0, piks_total=0, n_diisi=0, piks_diisi=0,
                n_dibiarkan=0, piks_dibiarkan=0, luas_maks=0,
                komposisi=np.zeros(6, dtype=np.int64), region=[])
    terisi_penuh = ndi.binary_fill_holes(nuk)
    lubang = terisi_penuh & ~nuk
    if not lubang.any():
        return nuk, np.zeros_like(nuk), info

    lab, n = ndi.label(lubang, structure=STRUKTUR4)
    info["n_region"] = int(n)

    # --- komposisi per region, tervektorisasi (tanpa loop mask per region)
    idx = lab[lubang].astype(np.int64)                     # id region per piksel
    val = np.clip(m[lubang].astype(np.int64), 0, 5)        # label asli per piksel
    luas = np.bincount(idx, minlength=n + 1)
    komp = np.bincount(idx * 6 + val, minlength=(n + 1) * 6).reshape(n + 1, 6)
    info["komposisi"] = komp[1:].sum(axis=0)
    info["piks_total"] = int(luas[1:].sum())
    info["luas_maks"] = int(luas[1:].max())

    jar = komp[:, list(NILAI_JARINGAN)].sum(axis=1)
    frak_j = np.zeros(n + 1)
    np.divide(jar, np.maximum(luas, 1), out=frak_j, casting="unsafe")
    dom = komp.argmax(axis=1)

    if aturan == "semua":
        diisi = np.ones(n + 1, dtype=bool)
    elif aturan == "tidak":
        diisi = np.zeros(n + 1, dtype=bool)
    else:                                                   # selektif
        diisi = (luas <= maks_px) | (frak_j < frak_sel)
    diisi[0] = False

    for i in range(1, n + 1):
        info["region"].append((int(luas[i]), int(dom[i]), round(float(frak_j[i]), 4),
                               int(diisi[i])))
    info["n_diisi"] = int(diisi[1:].sum())
    info["n_dibiarkan"] = int(n - info["n_diisi"])
    info["piks_diisi"] = int(luas[1:][diisi[1:]].sum()) if info["n_diisi"] else 0
    info["piks_dibiarkan"] = info["piks_total"] - info["piks_diisi"]

    isi = diisi[lab]
    return (nuk | isi), isi, info


def siapkan_nukleus(m, batasi_sel=True, aturan_lubang="selektif",
                    lubang_maks_px=15, lubang_frak_sel=0.5,
                    min_frak_komp=0.02, konektivitas=8):
    """
    Praproses lengkap: batasi ke komponen sel target (M4 Fase 2C), buang
    serpihan kecil, lalu terapkan aturan lubang (baru di Fase 2D).
    """
    st = STRUKTUR8 if konektivitas == 8 else STRUKTUR4
    d = dict(piks_nuk_mentah=0, komponen_mentah=0, komponen_akhir=0,
             piks_buang_luar_sel=0, piks_buang_kecil=0, piks_nuk_akhir=0,
             pusat_fallback=0, pusat_beda_terbesar=0)

    nuk = (m == NUKLEUS)
    d["piks_nuk_mentah"] = int(nuk.sum())
    if d["piks_nuk_mentah"] == 0:
        return None, None, d, None
    d["komponen_mentah"] = int(ndi.label(nuk, structure=st)[1])

    if batasi_sel:
        selmask = np.isin(m, (SITOPLASMA, NUKLEUS, VAKUOLA))
        labs, ns = ndi.label(selmask, structure=st)
        if ns >= 1:
            cy, cx = m.shape[0] // 2, m.shape[1] // 2
            id_pusat = int(labs[cy, cx])
            uk = np.bincount(labs.ravel(), minlength=ns + 1)
            uk[0] = 0
            id_besar = int(uk.argmax())
            if id_pusat == 0:
                y0, y1 = max(0, cy - 20), min(m.shape[0], cy + 21)
                x0, x1 = max(0, cx - 20), min(m.shape[1], cx + 21)
                nz = labs[y0:y1, x0:x1]
                nz = nz[nz > 0]
                if nz.size:
                    id_pusat = int(np.bincount(nz).argmax())
                    d["pusat_fallback"] = 1
                else:
                    id_pusat = id_besar
                    d["pusat_fallback"] = 2
            if id_pusat != id_besar:
                d["pusat_beda_terbesar"] = 1
            baru = nuk & (labs == id_pusat)
            if baru.any():
                d["piks_buang_luar_sel"] = int(nuk.sum() - baru.sum())
                nuk = baru

    if min_frak_komp > 0:
        labn, nn = ndi.label(nuk, structure=st)
        if nn > 1:
            uk = np.bincount(labn.ravel(), minlength=nn + 1)
            uk[0] = 0
            simpan = uk >= (min_frak_komp * uk.sum())
            simpan[0] = False
            if not simpan.any():
                simpan[int(uk.argmax())] = True
            baru = simpan[labn]
            d["piks_buang_kecil"] = int(nuk.sum() - baru.sum())
            nuk = baru

    d["komponen_akhir"] = int(ndi.label(nuk, structure=st)[1])
    nuk, isi, info = sensus_lubang(m, nuk, aturan_lubang,
                                   lubang_maks_px, lubang_frak_sel)
    d["piks_nuk_akhir"] = int(nuk.sum())
    return nuk, isi, d, info


# ---------------------------------------------------------------------------
# Pohon penggabungan superlevel-set
# ---------------------------------------------------------------------------

def pohon_merge(dt, jarak_lubang, konektivitas=8, prune=0.05):
    """
    Union-find pada daftar sisi terurut menurun. Bobot sisi = min(dt[a],dt[b]).
    Setiap kejadian penggabungan mencatat juga JARAK piksel saddle ke piksel
    lubang terisi terdekat, supaya bisa diuji apakah leher itu buatan.

    Kembalikan (kejadian, bertahan):
      kejadian : (saddle, persistensi, puncak_mati, puncak_utama, y, x, jarak_lubang)
      bertahan : puncak komponen yang tidak pernah bergabung
    """
    h, w = dt.shape
    dtf = dt.ravel()
    jlf = jarak_lubang.ravel()
    piks = np.flatnonzero(dtf > 0)
    if piks.size == 0:
        return [], []

    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    pas = [(idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])]
    if konektivitas == 8:
        pas += [(idx[:-1, :-1], idx[1:, 1:]), (idx[:-1, 1:], idx[1:, :-1])]
    A = np.concatenate([a.ravel() for a, _ in pas])
    B = np.concatenate([b.ravel() for _, b in pas])
    va, vb = dtf[A], dtf[B]
    ok = (va > 0) & (vb > 0)
    if not ok.any():
        return [], [float(dtf[p]) for p in piks.tolist()]
    A, B, va, vb = A[ok], B[ok], va[ok], vb[ok]
    wg = np.minimum(va, vb)
    o = np.argsort(-wg, kind="stable")
    Al, Bl = A[o].tolist(), B[o].tolist()
    Wl, Val, Vbl = wg[o].tolist(), va[o].tolist(), vb[o].tolist()

    par = list(range(h * w))
    puncak = dtf.tolist()
    kej = []
    for i in range(len(Wl)):
        a0, b0 = Al[i], Bl[i]
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
        pa, pb = puncak[ra], puncak[rb]
        if pa >= pb:
            hidup, mati, p_mati, p_utama = ra, rb, pb, pa
        else:
            hidup, mati, p_mati, p_utama = rb, ra, pa, pb
        s = Wl[i]
        pers = p_mati - s
        if pers >= prune:
            p = a0 if Val[i] <= Vbl[i] else b0
            kej.append((s, pers, p_mati, p_utama, p // w, p % w, jlf[p]))
        par[mati] = hidup

    akar = set()
    for p in piks.tolist():
        r = p
        while par[r] != r:
            r = par[r]
        akar.add(r)
    return kej, sorted((float(puncak[r]) for r in akar), reverse=True)


def _persistensi_dari_nukleus(nuk, isi, pad=True, konektivitas=8, prune=0.05):
    ys, xs = np.nonzero(nuk)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    potong = nuk[y0:y1, x0:x1]
    pot_isi = (isi[y0:y1, x0:x1] if isi is not None
               else np.zeros_like(potong))
    if pad:
        potong = np.pad(potong, 1)
        pot_isi = np.pad(pot_isi, 1)
        by, bx = y0 - 1, x0 - 1
    else:
        by, bx = y0, x0
    dt = ndi.distance_transform_edt(potong)
    r_lobus = float(dt.max())
    dt2 = np.pad(dt, 1)                       # selalu aman untuk traversal
    isi2 = np.pad(pot_isi, 1)
    jl = (ndi.distance_transform_edt(~isi2) if isi2.any()
          else np.full(dt2.shape, JARAK_JAUH))
    kej, bertahan = pohon_merge(dt2, jl, konektivitas, prune)
    return r_lobus, kej, bertahan, (by - 1, bx - 1), (y0, x0, y1, x1)


def proses_satu_sel(tugas):
    """Pekerja multiprocessing. tugas = (nama, path, cfg)."""
    nama, path, cfg = tugas
    (batasi, aturan, lmaks, lfrak, min_frak, konek, prune, pad) = cfg
    try:
        m = baca_mask(path)
        nuk, isi, d, info = siapkan_nukleus(m, batasi, aturan, lmaks, lfrak,
                                            min_frak, konek)
        if nuk is None or not nuk.any():
            return nama, None
        r_lobus, kej, bertahan, off, bbox = _persistensi_dari_nukleus(
            nuk, isi, pad, konek, prune)
        if kej:
            K = np.array([[k[0], k[1], k[2], k[3], k[6]] for k in kej],
                         dtype=np.float32)
            P = np.array([[k[4] + off[0], k[5] + off[1]] for k in kej],
                         dtype=np.int16)
        else:
            K = np.zeros((0, 5), dtype=np.float32)
            P = np.zeros((0, 2), dtype=np.int16)
        e = dict(r_lobus=r_lobus, luas_nukleus=int(d["piks_nuk_akhir"]),
                 piks_nuk_mentah=int(d["piks_nuk_mentah"]),
                 komponen_mentah=int(d["komponen_mentah"]),
                 komponen_akhir=int(d["komponen_akhir"]),
                 piks_buang_luar_sel=int(d["piks_buang_luar_sel"]),
                 piks_buang_kecil=int(d["piks_buang_kecil"]),
                 pusat_fallback=int(d["pusat_fallback"]),
                 pusat_beda_terbesar=int(d["pusat_beda_terbesar"]),
                 lub_n=int(info["n_region"]), lub_piks=int(info["piks_total"]),
                 lub_diisi_n=int(info["n_diisi"]),
                 lub_diisi_piks=int(info["piks_diisi"]),
                 lub_dibiarkan_n=int(info["n_dibiarkan"]),
                 lub_dibiarkan_piks=int(info["piks_dibiarkan"]),
                 lub_luas_maks=int(info["luas_maks"]),
                 lub_komposisi=np.asarray(info["komposisi"], dtype=np.int64),
                 lub_region=info["region"],
                 kejadian=K, pos=P,
                 bertahan=np.array(bertahan, dtype=np.float32), bbox=bbox)
        return nama, e
    except Exception as ex:                                # pragma: no cover
        return nama, {"galat": "%s: %s" % (type(ex).__name__, ex)}


# ===========================================================================
# BAGIAN 2 -- DERIVASI TERVEKTORISASI
# ===========================================================================

class Turunan:
    """Ratakan cache jadi array datar; seluruh grid (tau,rho) lewat numpy."""

    def __init__(self, cache, nama_urut):
        self.nama = list(nama_urut)
        N = len(self.nama)
        self.N = N
        self.rl = np.zeros(N)
        self.luas = np.zeros(N, dtype=np.int64)
        sad, per, pma, jlu = [], [], [], []
        off = [0]
        bts, boff = [], [0]
        for i, nm in enumerate(self.nama):
            e = cache[nm]
            self.rl[i] = e["r_lobus"]
            self.luas[i] = e["luas_nukleus"]
            K = e["kejadian"]
            if K.shape[0]:
                sad.append(K[:, 0]); per.append(K[:, 1])
                pma.append(K[:, 2]); jlu.append(K[:, 4])
            off.append(off[-1] + int(K.shape[0]))
            Bt = e["bertahan"]
            bts.append(Bt)
            boff.append(boff[-1] + int(Bt.size))
        cat = lambda L: (np.concatenate(L).astype(np.float64) if L else np.zeros(0))
        self.sad, self.per = cat(sad), cat(per)
        self.pma, self.jlu = cat(pma), cat(jlu)
        self.off = np.array(off, dtype=np.int64)
        self.cnt = np.diff(self.off)
        self.idx_ada = np.flatnonzero(self.cnt > 0)
        self.start = self.off[:-1][self.cnt > 0]
        self.rl_ev = np.repeat(self.rl, self.cnt)
        self.bt = cat(bts)
        self.boff = np.array(boff, dtype=np.int64)
        bcnt = np.diff(self.boff)
        self.bidx = np.flatnonzero(bcnt > 0)
        self.bstart = self.boff[:-1][bcnt > 0]
        self.rl_bt = np.repeat(self.rl, bcnt)

    def hitung(self, tau, rho, aturan="min"):
        N = self.N
        nkej = np.zeros(N, dtype=np.int64)
        smin = np.full(N, np.inf)
        smax = np.full(N, -np.inf)
        jsel = np.full(N, np.nan)
        if self.sad.size:
            mev = (self.per >= tau) & (self.pma >= rho * self.rl_ev)
            nkej[self.idx_ada] = np.add.reduceat(mev.astype(np.int64), self.start)
            smin[self.idx_ada] = np.minimum.reduceat(
                np.where(mev, self.sad, np.inf), self.start)
            smax[self.idx_ada] = np.maximum.reduceat(
                np.where(mev, self.sad, -np.inf), self.start)
            # jarak-ke-lubang pada saddle TERPILIH (patokan: aturan 'min')
            ref = np.repeat(smin, self.cnt)
            terpilih = mev & (self.sad <= ref + 1e-12)
            jj = np.where(terpilih, self.jlu, np.inf)
            jm = np.full(N, np.inf)
            jm[self.idx_ada] = np.minimum.reduceat(jj, self.start)
            jsel = np.where(np.isfinite(jm), jm, np.nan)

        nbt = np.zeros(N, dtype=np.int64)
        if self.bt.size:
            mbt = (self.bt >= tau) & (self.bt >= rho * self.rl_bt)
            nbt[self.bidx] = np.add.reduceat(mbt.astype(np.int64), self.bstart)

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
        br[normal] = sd[normal] / self.rl[normal]
        br[takpecah] = 1.0
        br = np.clip(br, 0.0, 1.0)

        st = np.empty(N, dtype=object)
        st[terpisah] = "terpisah_di_nol"
        st[normal] = "normal"
        st[takpecah] = "tak_pernah_pecah"
        jsel = np.where(normal, jsel, np.nan)

        return dict(img_name=np.array(self.nama, dtype=object),
                    r_lobus=self.rl.copy(), r_pisah=rp, bridge_ratio=br,
                    n_lobus=n_lobus, status_topologi=st,
                    luas_nukleus=self.luas.copy(),
                    jarak_saddle_ke_lubang=jsel)

    def frame(self, tau, rho, aturan="min"):
        return pd.DataFrame(self.hitung(tau, rho, aturan))


# ===========================================================================
# BAGIAN 3 -- PENILAI KALIBRASI (numpy, tanpa DataFrame di dalam grid)
# ===========================================================================

class Penilai:
    """
    Skor kalibrasi pada sel NON-NEUTROFIL saja. Semua mask dihitung sekali
    di konstruktor supaya evaluasi 128 kombinasi murah.
    """

    def __init__(self, meta):
        ns = meta["nucleus_shape"].astype(str).values
        lab = meta["label"].astype(str).values
        self.non = lab != "Neutrophil"
        self.target = np.zeros(len(ns), dtype=np.int64)
        for b, t in BENTUK_HARAPAN.items():
            self.target[(ns == b) & self.non] = t
        self.m_kal = self.target > 0
        self.m_t = {t: (self.target == t) for t in (1, 2, 3)}
        self.m_pecah = self.non & np.isin(
            ns, ["segmented-bilobed", "segmented-multilobed"])
        self.m_utuh = self.non & (ns == "unsegmented-round")
        self.m_limf = lab == "Lymphocyte"
        self.m_eo = lab == "Eosinophil"
        self.n_non = int(self.non.sum())

    def skor(self, h):
        nl = h["n_lobus"]
        st = h["status_topologi"]
        takpecah = (st == "tak_pernah_pecah")
        benar = np.where(self.target == 3, nl >= 3, nl == self.target)
        r = {}
        r["n_kalibrasi"] = int(self.m_kal.sum())
        r["skor_mentah"] = float(benar[self.m_kal].mean()) if self.m_kal.any() else np.nan
        for t in (1, 2, 3):
            mt = self.m_t[t]
            r["rec_t%d" % t] = float(benar[mt].mean()) if mt.any() else np.nan
        r["skor_makro3"] = float(np.nanmean([r["rec_t1"], r["rec_t2"], r["rec_t3"]]))
        r["rec_harus_pecah"] = (float((~takpecah)[self.m_pecah].mean())
                                if self.m_pecah.any() else np.nan)
        r["rec_harus_utuh"] = (float(takpecah[self.m_utuh].mean())
                               if self.m_utuh.any() else np.nan)
        r["skor_br"] = float(np.nanmean([r["rec_harus_pecah"], r["rec_harus_utuh"]]))
        r["limf_tak_pecah"] = (float(takpecah[self.m_limf].mean())
                               if self.m_limf.any() else np.nan)
        r["limf_lobus"] = float(nl[self.m_limf].mean()) if self.m_limf.any() else np.nan
        r["eo_lobus"] = float(nl[self.m_eo].mean()) if self.m_eo.any() else np.nan
        return r


def grid_kalibrasi(T, penilai, aturan="min"):
    baris = []
    for tau in TAU_GRID:
        for rho in RHO_GRID:
            r = penilai.skor(T.hitung(tau, rho, aturan))
            r["tau"], r["rho"] = tau, rho
            baris.append(r)
    dfk = pd.DataFrame(baris)
    return dfk[["tau", "rho"] + [c for c in dfk.columns if c not in ("tau", "rho")]]


def pilih_terbaik(dfk, kol):
    """Argmax dengan tie-break KONSERVATIF: skor seri -> tau dan rho terkecil.
    Penyaringan agresif adalah moda kegagalan yang sudah terbukti (Fase 2B)."""
    return dfk.sort_values([kol, "tau", "rho"],
                           ascending=[False, True, True]).iloc[0]


def rentang_puncak(dfk, kol, n=10):
    v = dfk.nlargest(n, kol)[kol]
    return float(v.max() - v.min())


# ===========================================================================
# BAGIAN 4 -- UJI SANITAS SINTETIS
# ===========================================================================

def _kanvas(h=200, w=260):
    return np.zeros((h, w), dtype=np.uint8)


def _cakram(m, cy, cx, r, nilai=NUKLEUS):
    Y, X = np.ogrid[:m.shape[0], :m.shape[1]]
    m[(Y - cy) ** 2 + (X - cx) ** 2 <= r * r] = nilai


def _batang(m, cy, x0, x1, setengah, nilai=NUKLEUS):
    m[cy - setengah:cy + setengah + 1, x0:x1] = nilai


def _garis_tebal(m, p0, p1, setengah, nilai=NUKLEUS):
    (y0, x0), (y1, x1) = p0, p1
    for t in np.linspace(0, 1, 600):
        yy = int(round(y0 + (y1 - y0) * t))
        xx = int(round(x0 + (x1 - x0) * t))
        m[yy - setengah:yy + setengah + 1, xx - setengah:xx + setengah + 1] = nilai


def _bungkus(m, tebal=6):
    nuk = (m == NUKLEUS)
    lebar = ndi.binary_dilation(nuk, structure=STRUKTUR8, iterations=tebal)
    m[lebar & ~nuk] = SITOPLASMA
    return m


def _rosette(rad=18, R=34, setengah=2, h=240, w=240):
    """Tiga lobus tersusun melingkar mengurung ruang di tengah -- geometri
    nukleus segmented yang justru dirusak oleh pengisian lubang."""
    m = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    pus = [(cy + int(R * np.sin(a)), cx + int(R * np.cos(a)))
           for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    for (y, x) in pus:
        _cakram(m, y, x, rad)
    for i in range(3):
        _garis_tebal(m, pus[i], pus[(i + 1) % 3], setengah)
    return _bungkus(m)


def _jalankan(m, aturan="selektif", lmaks=15, lfrak=0.5, batasi=True,
              min_frak=0.02, pad=True, tau=1.0, rho=0.0):
    nuk, isi, d, info = siapkan_nukleus(m, batasi, aturan, lmaks, lfrak, min_frak)
    if nuk is None or not nuk.any():
        return None
    rl, kej, bert, off, bbox = _persistensi_dari_nukleus(nuk, isi, pad)
    K = (np.array([[k[0], k[1], k[2], k[3], k[6]] for k in kej], np.float32)
         if kej else np.zeros((0, 5), np.float32))
    e = dict(r_lobus=rl, luas_nukleus=int(nuk.sum()), kejadian=K,
             bertahan=np.array(bert, np.float32))
    T = Turunan({"x": e}, ["x"])
    h = T.hitung(tau, rho, "min")
    return dict(r_lobus=rl, status=h["status_topologi"][0],
                n_lobus=int(h["n_lobus"][0]), rp=float(h["r_pisah"][0]),
                br=float(h["bridge_ratio"][0]),
                jarak=float(h["jarak_saddle_ke_lubang"][0]),
                lub_n=info["n_region"], lub_diisi=info["n_diisi"],
                lub_dibiarkan=info["n_dibiarkan"],
                lub_piks=info["piks_total"],
                komponen_mentah=d["komponen_mentah"],
                komponen_akhir=d["komponen_akhir"])


def jalankan_sanitas(out, verbose=True):
    print("\n=== UJI SANITAS SINTETIS ===")
    lolos = True
    baris = []

    def cek(nama, got, harap, tol=0.02, catatan=""):
        nonlocal lolos
        ok, pesan = True, []
        for k, v in harap.items():
            g = got.get(k)
            if k in ("rp", "br", "r_lobus"):
                if g is None or (isinstance(g, float) and math.isnan(g)):
                    ok = False; pesan.append("%s=NaN" % k)
                elif abs(g - v) > tol:
                    ok = False; pesan.append("%s %.4f != %.4f" % (k, g, v))
            elif g != v:
                ok = False; pesan.append("%s %r != %r" % (k, g, v))
        lolos &= ok
        if verbose:
            print("  [%s] %-31s r_lobus=%7.3f %-17s rp=%s br=%.3f lobus=%d "
                  "lubang %d (isi %d/biar %d) %s"
                  % ("OK " if ok else "GAGAL", nama, got["r_lobus"], got["status"],
                     ("%6.3f" % got["rp"]) if not math.isnan(got["rp"]) else "   nan",
                     got["br"], got["n_lobus"], got["lub_n"], got["lub_diisi"],
                     got["lub_dibiarkan"],
                     ("<- " + "; ".join(pesan)) if pesan else catatan))
        baris.append(dict(kasus=nama, lolos=ok, catatan="; ".join(pesan), **got))

    # ---- regresi Fase 2C: perilaku dasar tidak boleh berubah -------------
    m = _kanvas(); _cakram(m, 100, 130, 30); _bungkus(m)
    cek("cakram_konveks", _jalankan(m),
        dict(status="tak_pernah_pecah", n_lobus=1, br=1.0))

    m = _kanvas(w=300)
    _cakram(m, 100, 80, 20); _cakram(m, 100, 200, 20); _batang(m, 100, 80, 201, 4)
    _bungkus(m)
    cek("halter_leher9", _jalankan(m),
        dict(status="normal", n_lobus=2, rp=5.0))

    m = _kanvas(h=200, w=340)
    for cx in (60, 160, 260):
        _cakram(m, 100, cx, 20)
    _batang(m, 100, 60, 161, 3); _batang(m, 100, 160, 261, 8)
    _bungkus(m)
    cek("rantai_leher7_17", _jalankan(m),
        dict(status="normal", n_lobus=3, rp=4.0))

    m = _kanvas(h=220, w=300)
    _cakram(m, 110, 100, 28); _bungkus(m)
    _cakram(m, 35, 40, 16)
    cek("tetangga_di_luar_sel", _jalankan(m),
        dict(status="tak_pernah_pecah", n_lobus=1, komponen_mentah=2,
             komponen_akhir=1))

    m = _kanvas(h=220, w=300)
    _cakram(m, 110, 110, 28); _cakram(m, 110, 160, 20)
    m[80:141, 110:161] = np.where(m[80:141, 110:161] == 0, SITOPLASMA,
                                  m[80:141, 110:161])
    _bungkus(m)
    cek("dua_lobus_terpisah", _jalankan(m),
        dict(status="terpisah_di_nol", n_lobus=2, rp=0.0, br=0.0,
             komponen_akhir=2))

    # ---- BARU: aturan lubang --------------------------------------------
    # L1 lubang KECIL (13 px) berisi sitoplasma -> diisi karena di bawah batas
    m = _kanvas(); _cakram(m, 100, 130, 30); _bungkus(m)
    _cakram(m, 100, 130, 2, nilai=SITOPLASMA)
    cek("lubang_kecil_13px_diisi", _jalankan(m),
        dict(status="tak_pernah_pecah", n_lobus=1, lub_n=1, lub_diisi=1,
             lub_dibiarkan=0, lub_piks=13, r_lobus=30.017))

    # L2 lubang BESAR (441 px) berisi sitoplasma -> dibiarkan. Inti Fase 2D.
    m = _kanvas(); _cakram(m, 100, 130, 30); _bungkus(m)
    _cakram(m, 100, 130, 12, nilai=SITOPLASMA)
    g_sel = _jalankan(m, aturan="selektif")
    g_semua = _jalankan(m, aturan="semua")
    g_tidak = _jalankan(m, aturan="tidak")
    cek("lubang_besar_sitoplasma_biar", g_sel,
        dict(lub_n=1, lub_diisi=0, lub_dibiarkan=1, lub_piks=441,
             r_lobus=9.220))
    cek("aturan_semua_mengisi", g_semua,
        dict(lub_n=1, lub_diisi=1, lub_dibiarkan=0, r_lobus=30.017))
    cek("aturan_tidak_membiarkan", g_tidak,
        dict(lub_n=1, lub_diisi=0, lub_dibiarkan=1, r_lobus=9.220))

    # L3 lubang BESAR berisi LATAR -> diisi (artefak segmentasi, bukan jaringan)
    m = _kanvas(); _cakram(m, 100, 130, 30); _bungkus(m)
    _cakram(m, 100, 130, 12, nilai=LATAR)
    cek("lubang_besar_latar_diisi", _jalankan(m, aturan="selektif"),
        dict(status="tak_pernah_pecah", n_lobus=1, lub_n=1, lub_diisi=1,
             lub_dibiarkan=0, r_lobus=30.017))

    # L4 ROSETTE: tiga lobus melingkar. Pengisian menghapus lehernya.
    m = _rosette()
    r_sel = _jalankan(m, aturan="selektif")
    r_semua = _jalankan(m, aturan="semua")
    cek("rosette_selektif", r_sel, dict(lub_diisi=0, status="normal"))
    cek("rosette_diisi_semua", r_semua, dict(lub_diisi=1, status="normal"))
    # Yang diuji BUKAN arah jumlah lobus. Pengisian ternyata menambah lobus
    # semu di tengah SEKALIGUS melebarkan leher, jadi patokannya adalah
    # r_pisah dan bridge ratio: sel harus MELINTASI ambang 1/3.
    naik_br = r_semua["br"] - r_sel["br"]
    lipat_rp = r_semua["rp"] / max(r_sel["rp"], 1e-9)
    ok_ros = (r_sel["br"] < SEPERTIGA < r_semua["br"]) and (lipat_rp > 2.0)
    lolos &= bool(ok_ros)
    print("  [%s] %-31s r_pisah %.2f -> %.2f (%.1fx), bridge ratio %.3f -> %.3f "
          "(naik %.3f), lobus %d -> %d"
          % ("OK " if ok_ros else "GAGAL", "rosette_MEKANISME",
             r_sel["rp"], r_semua["rp"], lipat_rp, r_sel["br"], r_semua["br"],
             naik_br, r_sel["n_lobus"], r_semua["n_lobus"]))
    print("       -> moda kegagalan terbukti: nukleus segmented melintasi ambang")
    print("          1/3 dan jadi 'band' hanya karena lubangnya diisi. Pengisian")
    print("          juga menambah lobus semu di tengah, bukan menguranginya.")
    baris.append(dict(kasus="rosette_MEKANISME", lolos=ok_ros,
                      catatan="br %.4f -> %.4f" % (r_sel["br"], r_semua["br"]),
                      **r_semua))

    # L5 diagnostik jarak: saddle rosette-terisi harus DEKAT piksel terisi.
    # Ambang 6 px, bukan 3: saddle duduk di batas antara massa terisi dan
    # batang asli, jadi jaraknya beberapa piksel, bukan nol.
    if r_semua["status"] == "normal":
        dekat = r_semua["jarak"] <= 6.0
        lolos &= bool(dekat)
        print("  [%s] %-31s jarak saddle ke piksel terisi = %.2f px (harapan <= 6; "
              "pembanding rosette selektif: %s)"
              % ("OK " if dekat else "GAGAL", "rosette_jarak_saddle",
                 r_semua["jarak"],
                 "tanpa lubang terisi" if r_sel["jarak"] >= JARAK_JAUH - 1
                 else "%.2f px" % r_sel["jarak"]))
    else:
        print("  [INFO] rosette_jarak_saddle dilewati: status '%s'" % r_semua["status"])

    df = pd.DataFrame(baris)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "D0_uji_sanitas.csv", index=False)
    print("  -> %s" % ("SELURUH UJI LOLOS" if lolos else
                       "ADA UJI GAGAL -- JANGAN LANJUT KE DATA ASLI"))
    return lolos


# ===========================================================================
# BAGIAN 5 -- DATA DAN CACHE
# ===========================================================================

def indeks_mask(dirmask):
    peta, n = {}, 0
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
    for k in (s, s[:-6] if s.endswith("_ccrop") else None, s + "_ccrop"):
        if k and k in peta:
            return peta[k]
    return None


def nama_cache(cfg):
    batasi, aturan, lmaks, lfrak, mf, konek, prune, pad = cfg
    return ("cache_p2d_lub-%s_lm%d_lf%g_sel%d_pad%d_mf%g_conn%d_pr%g.pkl.gz"
            % (aturan, lmaks, lfrak, int(batasi), int(pad), mf, konek, prune))


def bangun_cache(df, peta, cfg, out, pekerja, paksa=False):
    out.mkdir(parents=True, exist_ok=True)
    fc = out / nama_cache(cfg)
    if fc.exists() and not paksa:
        with gzip.open(fc, "rb") as f:
            cache = pickle.load(f)
        if not [n for n in df["img_name"] if n not in cache]:
            print("  cache dimuat: %s" % fc.name)
            return cache, fc
        print("  cache tidak lengkap, membangun ulang")

    tugas, hilang = [], []
    for nm in df["img_name"]:
        p = cari_mask(peta, nm)
        if p is None:
            hilang.append(nm)
        else:
            tugas.append((nm, str(p), cfg))
    if hilang:
        print("  PERINGATAN: %d mask tidak ditemukan (contoh %s)"
              % (len(hilang), hilang[:3]))
    print("  menghitung %d sel, aturan lubang '%s', %d pekerja..."
          % (len(tugas), cfg[1], pekerja))
    t0 = time.time()
    cache, galat = {}, 0
    ex = None
    if pekerja <= 1:
        it = (proses_satu_sel(t) for t in tugas)
    else:
        ex = ProcessPoolExecutor(max_workers=pekerja)
        it = ex.map(proses_satu_sel, tugas, chunksize=24)
    for i, (nm, e) in enumerate(it):
        if e is None or "galat" in e:
            galat += 1
        else:
            cache[nm] = e
        if (i + 1) % 1000 == 0:
            print("    %d/%d (%.0f dtk)" % (i + 1, len(tugas), time.time() - t0))
    if ex is not None:
        ex.shutdown()
    dt = time.time() - t0
    print("  selesai: %d sel, %d galat, %.0f dtk (%.1f ms/sel)"
          % (len(cache), galat, dt, 1000 * dt / max(len(tugas), 1)))
    with gzip.open(fc, "wb", compresslevel=5) as f:
        pickle.dump(cache, f, protocol=4)
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


# ===========================================================================
# BAGIAN 6 -- T1 SENSUS LUBANG (KRITERIA PRIMER)
# ===========================================================================

def tugas_sensus(cache, meta, out):
    print("\n=== T1  SENSUS LUBANG (KRITERIA PRIMER, MEKANISTIK) ===")
    nama = list(meta["img_name"])
    lab = dict(zip(meta["img_name"], meta["label"]))
    ns = dict(zip(meta["img_name"], meta["nucleus_shape"]))
    kel = dict(zip(meta["img_name"], meta["kelompok"]))

    komp = np.zeros(6, dtype=np.int64)
    baris_sel, baris_reg = [], []
    for nm in nama:
        e = cache[nm]
        komp += np.asarray(e["lub_komposisi"], dtype=np.int64)
        baris_sel.append(dict(img_name=nm, label=lab[nm], nucleus_shape=ns[nm],
                              kelompok=kel[nm], lub_n=e["lub_n"],
                              lub_piks=e["lub_piks"],
                              lub_luas_maks=e["lub_luas_maks"],
                              lub_diisi_n=e["lub_diisi_n"],
                              lub_dibiarkan_n=e["lub_dibiarkan_n"],
                              luas_nukleus=e["luas_nukleus"]))
        for (luas, dom, frakj, diisi) in e["lub_region"]:
            baris_reg.append(dict(img_name=nm, label=lab[nm],
                                  nucleus_shape=ns[nm], kelompok=kel[nm],
                                  luas=luas, nilai_dominan=NAMA_NILAI.get(dom, dom),
                                  frak_jaringan=frakj, diisi=diisi))
    dsel = pd.DataFrame(baris_sel)
    dreg = pd.DataFrame(baris_reg)
    dsel.to_csv(out / "D1_lubang_per_sel.csv", index=False)
    dreg.to_csv(out / "D1_lubang_per_region.csv", index=False)

    total = int(komp.sum())
    print("  sel dengan >=1 region tertutup: %d dari %d (%.2f%%)"
          "   [Fase 2C mencatat 1556 dari 10298]"
          % (int((dsel["lub_n"] > 0).sum()), len(dsel),
             100 * (dsel["lub_n"] > 0).mean()))
    print("  total region tertutup %d, total piksel %d" % (len(dreg), total))
    print("\n  KOMPOSISI PIKSEL DI DALAM REGION TERTUTUP:")
    tab = []
    for v in range(6):
        pers = 100.0 * komp[v] / max(total, 1)
        if komp[v]:
            print("    %-12s %9d  %6.2f%%" % (NAMA_NILAI[v], komp[v], pers))
        tab.append(dict(nilai=NAMA_NILAI[v], piksel=int(komp[v]), persen=pers))
    frak_jar = 100.0 * komp[list(NILAI_JARINGAN)].sum() / max(total, 1)
    print("\n  -> %.2f%% piksel di dalam 'lubang' berlabel sitoplasma atau vakuola."
          % frak_jar)
    if frak_jar > 60:
        print("     Mengisinya SALAH SECARA FAKTUAL: itu jaringan yang memang")
        print("     bukan nukleus, bukan artefak segmentasi. Aturan 'semua'")
        print("     yang dipakai Fase 2A-2C harus ditinggalkan, dan alasan ini")
        print("     berdiri sendiri tanpa menyentuh AUC A-vs-B.")
    elif frak_jar < 25:
        print("     Lubang didominasi latar/derau; pengisian penuh bisa")
        print("     dipertahankan dan dugaan Fase 2C TIDAK terbukti.")
    else:
        print("     Campuran. Aturan selektif adalah pilihan yang tepat.")

    if len(dreg):
        print("\n  sebaran luas region (piksel): min %d | q25 %.0f | median %.0f | "
              "q75 %.0f | p95 %.0f | maks %d"
              % (dreg["luas"].min(), dreg["luas"].quantile(.25),
                 dreg["luas"].median(), dreg["luas"].quantile(.75),
                 dreg["luas"].quantile(.95), dreg["luas"].max()))
        print("  region diisi %d, dibiarkan %d"
              % (int(dreg["diisi"].sum()), int((dreg["diisi"] == 0).sum())))
        print("\n  sel terdampak per nucleus_shape:")
        t = (dsel[dsel["lub_n"] > 0].groupby("nucleus_shape")
             .agg(n_sel=("img_name", "count"), rerata_region=("lub_n", "mean"),
                  rerata_piks=("lub_piks", "mean"),
                  luas_maks=("lub_luas_maks", "max")))
        t["persen_dari_bentuk"] = [
            100.0 * v / max(int((dsel["nucleus_shape"] == s).sum()), 1)
            for s, v in zip(t.index, t["n_sel"])]
        print(t.to_string(float_format=lambda v: "%.2f" % v))
    return dsel, dreg, frak_jar, pd.DataFrame(tab)


# ===========================================================================
# BAGIAN 7 -- AMBANG, PLATEAU, BOOTSTRAP
# ===========================================================================

def analisis_ambang(br_a, br_b, eps=1e-3):
    h = dict(n_A=len(br_a), n_B=len(br_b))
    if roc_auc_score is None or len(br_a) < 2 or len(br_b) < 2:
        h["auc"] = np.nan
        return h
    y = np.r_[np.ones(len(br_a)), np.zeros(len(br_b))]
    s = np.r_[br_a, br_b]
    h["auc"] = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)
    J = tpr - fpr
    fin = np.isfinite(thr) & (thr <= 1.0 + 1e-12)
    if not fin.any():
        fin = np.isfinite(thr)
    Jf, tf = J[fin], thr[fin]
    k = int(np.argmax(Jf))
    h["J_max"], h["ambang"] = float(Jf[k]), float(tf[k])
    nilai = np.unique(s)
    atas = nilai[nilai > h["ambang"]]
    h["ambang_atas"] = float(atas.min()) if atas.size else 1.0
    h["lebar_persis"] = h["ambang_atas"] - h["ambang"]
    h["sepertiga_di_interval_persis"] = bool(h["ambang"] <= SEPERTIGA < h["ambang_atas"])
    ok = Jf >= (h["J_max"] - eps)
    h["plateau_lo"], h["plateau_hi"] = float(tf[ok].min()), float(tf[ok].max())
    h["lebar_plateau_eps"] = h["plateau_hi"] - h["plateau_lo"]
    h["sepertiga_di_plateau_eps"] = bool(h["plateau_lo"] <= SEPERTIGA <= h["plateau_hi"])
    h["J_sepertiga"] = float((br_a >= SEPERTIGA).mean() - (br_b >= SEPERTIGA).mean())
    h["defisit_J_sepertiga"] = h["J_max"] - h["J_sepertiga"]
    h["akurasi_sepertiga"] = float(((s >= SEPERTIGA) == (y == 1)).mean())
    h["akurasi_ambang"] = float(((s >= h["ambang"]) == (y == 1)).mean())
    h["sens_band"] = float((br_a >= SEPERTIGA).mean())
    h["spes_band"] = float((br_b < SEPERTIGA).mean())
    return h


def bootstrap_ambang(br_a, br_b, n=2000, benih=0):
    if roc_curve is None or len(br_a) < 2 or len(br_b) < 2:
        return np.nan, np.nan, np.nan
    r = np.random.default_rng(benih)
    out = np.empty(n)
    for i in range(n):
        a = br_a[r.integers(0, len(br_a), len(br_a))]
        b = br_b[r.integers(0, len(br_b), len(br_b))]
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        s = np.r_[a, b]
        fpr, tpr, thr = roc_curve(y, s)
        J = tpr - fpr
        fin = np.isfinite(thr) & (thr <= 1.0 + 1e-12)
        if not fin.any():
            fin = np.isfinite(thr)
        out[i] = thr[fin][int(np.argmax(J[fin]))]
    return (float(np.median(out)), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)))


def tugas_ambang(T, meta, out, pilih, aturan="min", n_boot=2000):
    print("\n=== T4  AMBANG, PLATEAU, BOOTSTRAP ===")
    kel = meta["kelompok"].values
    split = meta["split"].astype(str).values if "split" in meta else None
    isA, isB = kel == "A_sepakat_band", kel == "B_sepakat_segmented"
    h = T.hitung(pilih[0], pilih[1], aturan)
    br, st = h["bridge_ratio"], h["status_topologi"]
    a = analisis_ambang(br[isA], br[isB])
    med, lo, hi = bootstrap_ambang(br[isA], br[isB], n_boot)

    print("  AUC %.4f  [Fase 2C: %.4f, selisih %+.4f]"
          % (a["auc"], REF_2C["auc"], a["auc"] - REF_2C["auc"]))
    print("  Youden %.5f (J=%.4f) | interval optimal PERSIS [%.5f, %.5f) lebar %.5f"
          % (a["ambang"], a["J_max"], a["ambang"], a["ambang_atas"],
             a["lebar_persis"]))
    print("  plateau eps [%.4f, %.4f] lebar %.4f | 1/3 di plateau: %s"
          % (a["plateau_lo"], a["plateau_hi"], a["lebar_plateau_eps"],
             "YA" if a["sepertiga_di_plateau_eps"] else "TIDAK"))
    print("  bootstrap median %.4f IK95 [%.4f, %.4f] | 1/3 di dalam IK: %s"
          % (med, lo, hi, "YA" if lo <= SEPERTIGA <= hi else "TIDAK"))
    print("  akurasi @1/3 %.4f vs @optimum %.4f (ongkos memakai 1/3 = %.4f)"
          % (a["akurasi_sepertiga"], a["akurasi_ambang"],
             a["akurasi_ambang"] - a["akurasi_sepertiga"]))
    print("  sensitivitas band %.4f | spesifisitas band %.4f"
          % (a["sens_band"], a["spes_band"]))

    r = dict(a)
    r.update(tau=pilih[0], rho=pilih[1], boot_median=med, boot_lo=lo,
             boot_hi=hi, sepertiga_di_IK=bool(lo <= SEPERTIGA <= hi))

    if split is not None:
        trA, trB = isA & (split == "train"), isB & (split == "train")
        teA, teB = isA & (split == "test"), isB & (split == "test")
        if trA.sum() > 1 and trB.sum() > 1 and teA.sum() > 1 and teB.sum() > 1:
            at = analisis_ambang(br[trA], br[trB])
            y = np.r_[np.ones(int(teA.sum())), np.zeros(int(teB.sum()))]
            s = np.r_[br[teA], br[teB]]
            r.update(ambang_train=at["ambang"],
                     auc_test=float(roc_auc_score(y, s)),
                     akurasi_test_ambang=float(((s >= at["ambang"]) == (y == 1)).mean()),
                     akurasi_test_sepertiga=float(((s >= SEPERTIGA) == (y == 1)).mean()),
                     n_train=int(trA.sum() + trB.sum()), n_test=int(len(y)))
            print("  train->test: ambang train %.4f | AUC test %.4f | "
                  "akurasi test @ambang %.4f, @1/3 %.4f"
                  % (r["ambang_train"], r["auc_test"],
                     r["akurasi_test_ambang"], r["akurasi_test_sepertiga"]))

    m = st != "tak_pernah_pecah"
    aa = analisis_ambang(br[isA & m], br[isB & m])
    r.update(auc_tanpa_takpecah=aa.get("auc", np.nan),
             akurasi13_tanpa_takpecah=aa.get("akurasi_sepertiga", np.nan))
    print("  [sensitivitas] tanpa sel 'tak_pernah_pecah': AUC %.4f, akurasi@1/3 %.4f"
          % (aa.get("auc", np.nan), aa.get("akurasi_sepertiga", np.nan)))
    pd.DataFrame([r]).to_csv(out / "D4_ambang_terpilih.csv", index=False)

    # kurva Youden
    t = np.linspace(0, 1, 1001)
    J = np.array([(br[isA] >= x).mean() - (br[isB] >= x).mean() for x in t])
    pd.DataFrame({"ambang": t, "J": J}).to_csv(out / "D4_kurva_youden.csv",
                                               index=False)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(t, J, lw=1.6, color="#2b5d8a")
    if np.isfinite(lo) and np.isfinite(hi):
        ax.axvspan(lo, hi, color="#8ab6d6", alpha=.25, label="IK95 bootstrap")
    ax.axvspan(a["plateau_lo"], a["plateau_hi"], color="#ffd27f", alpha=.6,
               label="plateau J dalam 0.001")
    ax.axvline(a["ambang"], color="#c0392b", lw=1.2,
               label="optimum %.4f" % a["ambang"])
    ax.axvline(SEPERTIGA, color="#27795b", lw=1.2, ls="--", label="1/3")
    ax.set_xlabel("ambang bridge ratio"); ax.set_ylabel("Youden J = TPR - FPR")
    ax.set_title("Kurva Youden, tau=%g rho=%g (lubang: %s)"
                 % (pilih[0], pilih[1], out.name), fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "F2D_kurva_youden.png", dpi=140)
    plt.close(fig)
    return r


# ===========================================================================
# BAGIAN 8 -- T2 ABLASI ATURAN LUBANG
# ===========================================================================

def tugas_ablasi_lubang(df, peta, out, pekerja, cfg_dasar, aturan_saddle="min",
                        paksa=False, ambang_jaringan=60.0):
    print("\n=== T2  ABLASI ATURAN LUBANG ===")
    print("  Aturan dipilih dari KRITERIA INDEPENDEN (mekanistik + skor")
    print("  non-neutrofil). Kolom *_KONSEKUENSI dicetak setelahnya dan tidak")
    print("  pernah ikut memilih.")
    batasi, _, lmaks, lfrak, mf, konek, prune, pad = cfg_dasar

    simpan = {}
    baris = []
    cache_semua, dasar_nama = None, []
    for at in ATURAN_LUBANG:
        cfg = (batasi, at, lmaks, lfrak, mf, konek, prune, pad)
        cache, _ = bangun_cache(df, peta, cfg, out, pekerja, paksa=paksa)
        nm = [n for n in df["img_name"] if n in cache]
        if at == "semua":
            cache_semua, dasar_nama = cache, nm
        meta = df.set_index("img_name").loc[nm].reset_index()
        T = Turunan(cache, nm)
        pen = Penilai(meta)
        dfk = grid_kalibrasi(T, pen, aturan_saddle)
        b = pilih_terbaik(dfk, "skor_br")
        h = T.hitung(float(b["tau"]), float(b["rho"]), aturan_saddle)
        kel = meta["kelompok"].values
        isA, isB = kel == "A_sepakat_band", kel == "B_sepakat_segmented"
        br = h["bridge_ratio"]
        a = analisis_ambang(br[isA], br[isB])
        simpan[at] = dict(meta=meta, br=br, st=h["status_topologi"],
                          jar=h["jarak_saddle_ke_lubang"], nm=nm, dfk=dfk,
                          tau=float(b["tau"]), rho=float(b["rho"]))
        r = dict(aturan_lubang=at, tau=float(b["tau"]), rho=float(b["rho"]),
                 skor_br=float(b["skor_br"]),
                 rec_harus_pecah=float(b["rec_harus_pecah"]),
                 rec_harus_utuh=float(b["rec_harus_utuh"]),
                 skor_makro3_pada_tau_br=float(b["skor_makro3"]),
                 skor_makro3_terbaik=float(pilih_terbaik(dfk, "skor_makro3")["skor_makro3"]),
                 limf_tak_pecah=float(b["limf_tak_pecah"]),
                 limf_lobus=float(b["limf_lobus"]), eo_lobus=float(b["eo_lobus"]),
                 frak_takpecah=float((h["status_topologi"] == "tak_pernah_pecah").mean()),
                 r_lobus_mean=float(h["r_lobus"].mean()),
                 r_lobus_maks=float(h["r_lobus"].max()),
                 auc_KONSEKUENSI=a.get("auc", np.nan),
                 akurasi13_KONSEKUENSI=a.get("akurasi_sepertiga", np.nan),
                 spes_band_KONSEKUENSI=a.get("spes_band", np.nan),
                 sens_band_KONSEKUENSI=a.get("sens_band", np.nan))
        baris.append(r)
        print("\n  %-9s KRITERIA: skor_br %.4f (harus pecah %.4f, harus utuh %.4f), "
              "makro3 %.4f, limfosit %.4f"
              % (at, r["skor_br"], r["rec_harus_pecah"], r["rec_harus_utuh"],
                 r["skor_makro3_terbaik"], r["limf_tak_pecah"]))
        print("  %-9s konsekuensi: AUC %.4f | akurasi@1/3 %.4f | spesifisitas B %.4f"
              % ("", r["auc_KONSEKUENSI"], r["akurasi13_KONSEKUENSI"],
                 r["spes_band_KONSEKUENSI"]))

    dfa = pd.DataFrame(baris)
    dfa.to_csv(out / "D2_ablasi_lubang.csv", index=False)

    # ---- PEMILIHAN ATURAN, BERLAPIS DAN EKSPLISIT ------------------------
    # Lapis 1 (mekanistik, menggugurkan): bila mayoritas piksel di dalam
    #   region tertutup berlabel jaringan, aturan 'semua' salah secara
    #   faktual dan digugurkan, berapa pun skornya.
    # Lapis 2 (kriteria independen): skor_br, lalu skor_makro3.
    # Lapis 3 (prinsip): bila masih seri, 'selektif' menang atas 'tidak',
    #   karena 'selektif' tetap memperbaiki artefak segmentasi yang nyata.
    # AUC A-vs-B tidak pernah masuk ke mana pun di sini.
    komp = np.zeros(6, dtype=np.int64)
    for n in dasar_nama:
        komp += np.asarray(cache_semua[n]["lub_komposisi"], dtype=np.int64)
    tot = int(komp.sum())
    frak_jar = 100.0 * komp[list(NILAI_JARINGAN)].sum() / max(tot, 1)
    print("\n  PEMILIHAN ATURAN")
    print("    lapis 1 (mekanistik): %.2f%% piksel lubang berlabel jaringan"
          % frak_jar)
    gugur = []
    if tot > 0 and frak_jar >= ambang_jaringan:
        gugur.append("semua")
        print("      -> aturan 'semua' DIGUGURKAN: mengisi jaringan nyata")
    elif tot == 0:
        print("      -> tidak ada region tertutup sama sekali; lapis 1 netral")
    else:
        print("      -> lubang tidak didominasi jaringan; lapis 1 netral")
    layak = dfa[~dfa["aturan_lubang"].isin(gugur)].copy()
    if not len(layak):
        layak = dfa.copy()
    layak["_prinsip"] = (layak["aturan_lubang"] == "selektif").astype(int)
    layak = layak.sort_values(["skor_br", "skor_makro3_terbaik", "_prinsip"],
                              ascending=False)
    menang = str(layak.iloc[0]["aturan_lubang"])
    seri = int((layak["skor_br"] == layak.iloc[0]["skor_br"]).sum())
    print("    lapis 2 (skor non-neutrofil): urutan %s"
          % " > ".join(layak["aturan_lubang"].tolist()))
    if seri > 1:
        print("      -> %d aturan SERI pada skor_br; tie-break memakai makro3 "
              "lalu prinsip" % seri)
    print("  -> aturan terpilih menurut kriteria independen: '%s'" % menang)

    # ---- perbandingan BERPASANGAN terhadap aturan 'semua' (perilaku 2C)
    print("\n  PERBANDINGAN BERPASANGAN pada ekor B (basis: aturan 'semua' = Fase 2C)")
    dasar = simpan["semua"]
    kelb = dasar["meta"]["kelompok"].values
    isB0 = kelb == "B_sepakat_segmented"
    isA0 = kelb == "A_sepakat_band"
    salah0_B = isB0 & (dasar["br"] >= SEPERTIGA)
    salah0_A = isA0 & (dasar["br"] < SEPERTIGA)
    pas = []
    for at in ATURAN_LUBANG:
        if at == "semua":
            continue
        s = simpan[at]
        if s["nm"] != dasar["nm"]:
            peta_i = {n: i for i, n in enumerate(s["nm"])}
            ok = np.array([n in peta_i for n in dasar["nm"]])
            ix = np.array([peta_i.get(n, 0) for n in dasar["nm"]])
            br2 = np.where(ok, s["br"][ix], np.nan)
        else:
            br2 = s["br"]
        salah1_B = isB0 & (br2 >= SEPERTIGA)
        salah1_A = isA0 & (br2 < SEPERTIGA)
        r = dict(aturan_lubang=at,
                 B_salah_sebelum=int(salah0_B.sum()), B_salah_sesudah=int(salah1_B.sum()),
                 B_diperbaiki=int((salah0_B & ~salah1_B).sum()),
                 B_dirusak=int((~salah0_B & salah1_B & isB0).sum()),
                 A_salah_sebelum=int(salah0_A.sum()), A_salah_sesudah=int(salah1_A.sum()),
                 A_diperbaiki=int((salah0_A & ~salah1_A).sum()),
                 A_dirusak=int((~salah0_A & salah1_A & isA0).sum()))
        pas.append(r)
        print("    %-9s ekor B %d -> %d (diperbaiki %d, dirusak %d) | "
              "galat A %d -> %d (diperbaiki %d, dirusak %d)"
              % (at, r["B_salah_sebelum"], r["B_salah_sesudah"],
                 r["B_diperbaiki"], r["B_dirusak"], r["A_salah_sebelum"],
                 r["A_salah_sesudah"], r["A_diperbaiki"], r["A_dirusak"]))
    dfp = pd.DataFrame(pas)
    dfp.to_csv(out / "D2_berpasangan.csv", index=False)

    # ---- diagnostik asal-usul leher, DIJALANKAN PADA ATURAN 'semua'
    # (perilaku Fase 2C). Di sinilah pertanyaannya berarti: dari 195 galat
    # ekor B Fase 2C, berapa yang saddle-nya menempel pada lubang terisi?
    dd = pd.DataFrame(dict(kelompok=kelb, br=dasar["br"], st=dasar["st"],
                           jarak_saddle_ke_lubang=dasar["jar"]))
    dd["pred_band"] = dd["br"] >= SEPERTIGA
    Bd = dd[dd["kelompok"] == "B_sepakat_segmented"]
    Ad = dd[dd["kelompok"] == "A_sepakat_band"]
    dfd = diagnostik_jarak(
        [("B salah, leher terukur", Bd[Bd["pred_band"] & (Bd["st"] == "normal")]),
         ("B benar segmented", Bd[(~Bd["pred_band"]) & (Bd["st"] == "normal")]),
         ("A benar band", Ad[Ad["pred_band"] & (Ad["st"] == "normal")])],
        "T2.2  ASAL-USUL LEHER pada aturan 'semua' (= perilaku Fase 2C)")
    dfd.to_csv(out / "D2_asal_usul_leher.csv", index=False)
    return dfa, dfp, dfd, menang


# ===========================================================================
# BAGIAN 9 -- T3 KALIBRASI
# ===========================================================================

def tugas_kalibrasi(T, meta, out, aturan="min"):
    print("\n=== T3  KALIBRASI DUA SISI (sel non-neutrofil, independen) ===")
    pen = Penilai(meta)
    print("  sel non-neutrofil: %d" % pen.n_non)
    dfk = grid_kalibrasi(T, pen, aturan)
    dfk.to_csv(out / "D3_kalibrasi_grid.csv", index=False)
    bl, bb = pilih_terbaik(dfk, "skor_makro3"), pilih_terbaik(dfk, "skor_br")
    for nm_, b, kol in (("JUMLAH LOBUS (skor makro-3)", bl, "skor_makro3"),
                        ("BRIDGE RATIO (skor status)", bb, "skor_br")):
        rp = rentang_puncak(dfk, kol)
        pojok = (b["tau"] == TAU_GRID[-1]) or (b["rho"] == RHO_GRID[-1])
        print("  -- %s: tau=%g rho=%g, skor %.4f" % (nm_, b["tau"], b["rho"], b[kol]))
        print("     rentang 10 teratas %.4f%s | di batas atas grid: %s"
              % (rp, "  <- DATAR, laporkan WILAYAH bukan titik" if rp < 0.01 else "",
                 "YA, perluas grid" if pojok else "tidak"))
    print("     limfosit tak pecah %.4f (lobus %.4f) | eosinofil lobus %.4f"
          % (bb["limf_tak_pecah"], bb["limf_lobus"], bb["eo_lobus"]))
    print("\n  sepuluh teratas menurut skor_makro3:")
    print(dfk.nlargest(10, "skor_makro3")[
        ["tau", "rho", "skor_makro3", "rec_t1", "rec_t2", "rec_t3",
         "skor_br", "limf_tak_pecah"]].to_string(
        index=False, float_format=lambda v: "%.4f" % v))
    print("\n  pembanding Fase 2C (lubang diisi semua): "
          "lobus tau=3 rho=0.4 makro3=0.7826 | bridge tau=0.75 rho=0.2 skor_br=0.9721")
    return dfk, (float(bl["tau"]), float(bl["rho"])), (float(bb["tau"]), float(bb["rho"]))


# ===========================================================================
# BAGIAN 10 -- T5 AUDIT EKOR B
# ===========================================================================

def tugas_audit(dfin, out):
    print("\n=== T5  AUDIT EKOR B + DIAGNOSTIK JARAK SADDLE KE LUBANG ===")
    d = dfin.copy()
    d["pred_band"] = d["bridge_ratio"] >= SEPERTIGA
    A = d[d["kelompok"] == "A_sepakat_band"]
    B = d[d["kelompok"] == "B_sepakat_segmented"]
    print("  A (band)      benar %.2f%%  (%d salah)"
          % (100 * A["pred_band"].mean(), int((~A["pred_band"]).sum())))
    print("  B (segmented) benar %.2f%%  (%d salah)   [Fase 2C: %d salah]"
          % (100 * (~B["pred_band"]).mean(), int(B["pred_band"].sum()),
             REF_2C["ekor_B"]))
    salah = B[B["pred_band"]].copy()
    for k, v in salah["status_topologi"].value_counts().items():
        print("    ekor B %-18s %4d (%.1f%%)" % (k, v, 100 * v / max(len(salah), 1)))
    nm = salah[salah["status_topologi"] == "normal"]
    if len(nm):
        print("    leher lebar: r_pisah median %.2f px (q25 %.2f, q75 %.2f), "
              "r_lobus mean %.2f"
              % (nm["r_pisah"].median(), nm["r_pisah"].quantile(.25),
                 nm["r_pisah"].quantile(.75), nm["r_lobus"].mean()))

    dd = diagnostik_jarak(
        [("B salah, leher terukur", nm),
         ("B benar segmented", B[(~B["pred_band"]) &
                                 (B["status_topologi"] == "normal")]),
         ("A benar band", A[A["pred_band"] &
                            (A["status_topologi"] == "normal")])],
        "UJI LANGSUNG: apakah leher lebar itu buatan pengisian lubang?")
    salah.to_csv(out / "D5_ekor_B.csv", index=False)
    dd.to_csv(out / "D5_diagnostik_jarak.csv", index=False)
    return salah, dd


def diagnostik_jarak(kelompok_data, judul_blok):
    """
    Sebaran jarak dari saddle terpilih ke piksel lubang TERISI terdekat.
    Saddle yang selnya tidak punya lubang terisi sama sekali diberi
    JARAK_JAUH dan dihitung terpisah, supaya tidak mencemari median.
    """
    print("\n  %s" % judul_blok)
    baris = []
    for judul, s in kelompok_data:
        if len(s):
            j = pd.to_numeric(s["jarak_saddle_ke_lubang"], errors="coerce").dropna()
        else:
            j = pd.Series(dtype=float)
        ada = j[j < JARAK_JAUH - 1]
        n_tanpa = int((j >= JARAK_JAUH - 1).sum())
        r = dict(kelompok=judul, n=len(s), n_dengan_saddle=len(j),
                 n_tanpa_lubang_terisi=n_tanpa, n_dengan_lubang_terisi=len(ada),
                 frak_sel_ada_lubang=(len(ada) / len(j)) if len(j) else np.nan,
                 median_jarak=float(ada.median()) if len(ada) else np.nan,
                 frak_dekat_3px=float((ada <= 3).mean()) if len(ada) else np.nan,
                 frak_dekat_6px=float((ada <= 6).mean()) if len(ada) else np.nan)
        baris.append(r)
        print("    %-24s n=%4d | punya lubang terisi %4d (%s) | median jarak %s "
              "| <=3px %s | <=6px %s"
              % (judul, len(s), len(ada),
                 ("%.1f%%" % (100 * r["frak_sel_ada_lubang"]))
                 if not np.isnan(r["frak_sel_ada_lubang"]) else "n/a",
                 ("%6.2f px" % r["median_jarak"]) if len(ada) else "     n/a",
                 ("%.1f%%" % (100 * r["frak_dekat_3px"])) if len(ada) else "n/a",
                 ("%.1f%%" % (100 * r["frak_dekat_6px"])) if len(ada) else "n/a"))
    print("    Bila baris pertama jauh lebih sering punya lubang terisi DEKAT")
    print("    saddle-nya daripada dua pembanding, leher lebar itu memang buatan.")
    return pd.DataFrame(baris)


# ===========================================================================
# BAGIAN 11 -- T6 DEFERRAL
# ===========================================================================

def tugas_deferral(dfin, out, ambang=SEPERTIGA):
    print("\n=== T6  ANALISIS DEFERRAL / SELECTIVE PREDICTION ===")
    neu = dfin[dfin["label"] == "Neutrophil"].copy()
    neu["keyakinan"] = (neu["bridge_ratio"] - ambang).abs()
    neu = neu.sort_values("keyakinan", ascending=False).reset_index(drop=True)
    ab = neu["kelompok"].isin(["A_sepakat_band", "B_sepakat_segmented"])
    benar = np.where(neu["kelompok"] == "A_sepakat_band",
                     neu["bridge_ratio"] >= ambang, neu["bridge_ratio"] < ambang)
    neu["benar"] = np.where(ab, benar.astype(float), np.nan)
    N = len(neu)
    basis = {k: float((neu["kelompok"] == k).mean()) for k in URUT_KELOMPOK}

    baris = []
    for cak in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]:
        k = int(round(cak * N))
        simpan, tunda = neu.iloc[:k], neu.iloc[k:]
        sab = simpan[simpan["kelompok"].isin(["A_sepakat_band", "B_sepakat_segmented"])]
        r = dict(cakupan=cak, n_simpan=k, n_tunda=N - k, n_AB_simpan=len(sab),
                 akurasi=float(sab["benar"].mean()) if len(sab) else np.nan)
        a_ = sab[sab["kelompok"] == "A_sepakat_band"]
        b_ = sab[sab["kelompok"] == "B_sepakat_segmented"]
        r["sens_band"] = float(a_["benar"].mean()) if len(a_) else np.nan
        r["spes_band"] = float(b_["benar"].mean()) if len(b_) else np.nan
        r["akurasi_seimbang"] = float(np.nanmean([r["sens_band"], r["spes_band"]]))
        for kk in URUT_KELOMPOK:
            pangsa = float((tunda["kelompok"] == kk).mean()) if len(tunda) else np.nan
            r["pangsa_tunda_" + kk[0]] = pangsa
            r["pengayaan_" + kk[0]] = (pangsa / basis[kk]) if basis[kk] > 0 else np.nan
        baris.append(r)
    dfd = pd.DataFrame(baris)
    dfd.to_csv(out / "D6_deferral.csv", index=False)
    kol = [c for c in ["cakupan", "n_tunda", "akurasi", "akurasi_seimbang",
                       "sens_band", "spes_band", "pengayaan_C", "pengayaan_D",
                       "pengayaan_F"] if c in dfd.columns]
    print(dfd[kol].to_string(index=False, float_format=lambda v: "%.4f" % v))
    b0 = dfd.iloc[0]
    for _, r in dfd.iterrows():
        if abs(r["cakupan"] - 0.8) < 1e-9:
            print("\n  Pada cakupan 80%%: akurasi %.4f -> %.4f (+%.2f poin), "
                  "spesifisitas segmented %.4f -> %.4f"
                  % (b0["akurasi"], r["akurasi"],
                     100 * (r["akurasi"] - b0["akurasi"]),
                     b0["spes_band"], r["spes_band"]))
            print("  Kelompok C terwakili %.2fx lipat di dalam 20%% yang ditunda "
                  "(D %.2fx, F %.2fx)."
                  % (r["pengayaan_C"], r["pengayaan_D"], r["pengayaan_F"]))
            print("  Artinya: sel yang mesin ragukan adalah sel yang memang")
            print("  diperselisihkan manusia -- bukan sel acak.")

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax[0].plot(100 * dfd["cakupan"], 100 * dfd["akurasi"], "o-",
               color="#2b5d8a", label="akurasi A vs B")
    ax[0].plot(100 * dfd["cakupan"], 100 * dfd["akurasi_seimbang"], "s--",
               color="#c0392b", label="akurasi seimbang")
    ax[0].set_xlabel("cakupan: % sel yang tetap diputuskan mesin")
    ax[0].set_ylabel("akurasi (%)")
    ax[0].set_title("Kurva risiko-cakupan", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].invert_xaxis(); ax[0].grid(alpha=.3)
    for kk, c in (("C", "#e08a1e"), ("D", "#7b4fa8"), ("F", "#2f8f5b")):
        col = "pengayaan_" + kk
        if col in dfd:
            ax[1].plot(100 * (1 - dfd["cakupan"]), dfd[col], "o-", color=c,
                       label="kelompok " + kk)
    ax[1].axhline(1.0, color="k", lw=.8, ls=":")
    ax[1].set_xlabel("% sel yang ditunda ke pemeriksa manusia")
    ax[1].set_ylabel("pengayaan (x lipat terhadap basis)")
    ax[1].set_title("Siapa yang ditunda mesin?", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "F2D_deferral.png", dpi=140)
    plt.close(fig)
    return dfd


# ===========================================================================
# BAGIAN 12 -- TABEL DAN LAPORAN
# ===========================================================================

def df_ke_md(df, nd=4):
    d = df.copy()
    for c in d.columns:
        d[c] = (d[c].map(lambda v: "" if pd.isna(v) else "%.*f" % (nd, v))
                if pd.api.types.is_float_dtype(d[c]) else d[c].astype(str))
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |",
                      "|" + "|".join(["---"] * len(d.columns)) + "|"] +
                     ["| " + " | ".join(str(r[c]) for c in d.columns) + " |"
                      for _, r in d.iterrows()])


def tabel_kelompok(dfin):
    b = []
    for k in URUT_KELOMPOK:
        s = dfin[dfin["kelompok"] == k]
        if not len(s):
            continue
        p = float((s["bridge_ratio"] >= SEPERTIGA).mean())
        ent = (0.0 if p in (0.0, 1.0)
               else -(p * math.log2(p) + (1 - p) * math.log2(1 - p)))
        b.append(dict(kelompok=k, n=len(s), q25=s["bridge_ratio"].quantile(.25),
                      median=s["bridge_ratio"].median(),
                      q75=s["bridge_ratio"].quantile(.75), p_band=p, entropi=ent,
                      zona_23_43=float(s["bridge_ratio"].between(.23, .43).mean()),
                      tak_pecah=float((s["status_topologi"] == "tak_pernah_pecah").mean()),
                      mean_lobus=float(s["n_lobus"].mean())))
    t = pd.DataFrame(b)
    mA = t.loc[t["kelompok"] == "A_sepakat_band", "median"]
    mB = t.loc[t["kelompok"] == "B_sepakat_segmented", "median"]
    if len(mA) and len(mB) and abs(float(mB.iloc[0]) - float(mA.iloc[0])) > 1e-9:
        t["posisi_AB"] = ((t["median"] - float(mA.iloc[0])) /
                          (float(mB.iloc[0]) - float(mA.iloc[0])))
    return t


def tabel_bentuk(dfl):
    return pd.DataFrame([dict(
        nucleus_shape=s, n=len(x), mean_lobus=float(x["n_lobus"].mean()),
        median_lobus=float(x["n_lobus"].median()),
        hanya_1_lobus=float((x["n_lobus"] == 1).mean()),
        median_br=float(x["bridge_ratio"].median()))
        for s in URUT_BENTUK for x in [dfl[dfl["nucleus_shape"] == s]] if len(x)])


def tabel_kelas(dfl):
    return pd.DataFrame([dict(
        kelas=k, n=len(x), median_br=float(x["bridge_ratio"].median()),
        tak_pernah_pecah=float((x["status_topologi"] == "tak_pernah_pecah").mean()),
        mean_lobus=float(x["n_lobus"].mean()),
        terpisah_di_nol=float((x["status_topologi"] == "terpisah_di_nol").mean()))
        for k in ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
        for x in [dfl[dfl["label"] == k]] if len(x)])


def tulis_laporan(out, ctx):
    L = ["# LAPORAN FASE 2D\n",
         "Dihasilkan `fase2d_lubang_dan_deferral.py` pada %s.\n"
         % time.strftime("%Y-%m-%d %H:%M"),
         "Aturan saddle: **%s**. Angka memakai titik desimal.\n"
         % ctx.get("aturan_saddle", "min"),
         "\n## 0. Konfigurasi\n"]
    for k in ["n_sel", "aturan_lubang", "lubang_maks_px", "lubang_frak_sel",
              "tau_lobus", "rho_lobus", "tau_br", "rho_br",
              "sanitas_lolos", "replikasi_kelompok", "tugas_dijalankan",
              "parameter_cadangan"]:
        if k in ctx:
            L.append("- **%s**: %s" % (k, ctx[k]))
    if ctx.get("parameter_cadangan"):
        L.append("\n> **PERINGATAN.** Tugas `kalibrasi` tidak dijalankan, jadi "
                 "tau/rho di atas adalah nilai cadangan Fase 2C yang "
                 "dikalibrasi pada aturan lubang `semua`. Angka pada laporan "
                 "ini belum sah dibawa ke pembimbing.\n")
    if "tugas_dijalankan" in ctx and ctx["tugas_dijalankan"] != "semua":
        L.append("\n> Laporan ini **parsial**: hanya bagian dari tugas yang "
                 "dijalankan, sehingga sebagian bab sengaja kosong.\n")

    if "frak_jar" in ctx:
        L.append("\n## 1. Sensus lubang -- kriteria primer\n")
        L.append("**%.2f%% piksel di dalam region tertutup berlabel sitoplasma "
                 "atau vakuola.** Kesimpulan aturan lubang diambil dari angka "
                 "ini dan dari skor kalibrasi non-neutrofil, bukan dari AUC "
                 "A-vs-B.\n" % ctx["frak_jar"])
        if "sensus_komposisi" in ctx:
            L.append(df_ke_md(ctx["sensus_komposisi"]))
    if "abl" in ctx:
        L.append("\n## 2. Ablasi aturan lubang\n")
        L.append("Kolom berakhiran `_KONSEKUENSI` **tidak** dipakai memilih aturan.\n")
        L.append(df_ke_md(ctx["abl"]))
    if "pas" in ctx:
        L.append("\n### 2.1 Perbandingan berpasangan terhadap aturan `semua` (Fase 2C)\n")
        L.append(df_ke_md(ctx["pas"]))
    if "asal" in ctx:
        L.append("\n### 2.2 Asal-usul leher, diukur pada aturan `semua` (Fase 2C)\n")
        L.append("Jarak dari saddle terpilih ke piksel lubang terisi terdekat. "
                 "Kalau galat ekor B jauh lebih sering punya lubang terisi tepat "
                 "di lehernya dibanding sel B yang benar, leher itu buatan.\n")
        L.append(df_ke_md(ctx["asal"]))
    if "kal" in ctx:
        L.append("\n## 3. Kalibrasi dua sisi\n")
        L.append("### 3.1 Sepuluh teratas menurut skor makro-3 (jumlah lobus)\n")
        L.append(df_ke_md(ctx["kal"].nlargest(10, "skor_makro3")[
            ["tau", "rho", "skor_makro3", "rec_t1", "rec_t2", "rec_t3",
             "skor_br", "limf_tak_pecah"]]))
        L.append("\n### 3.2 Sepuluh teratas menurut skor status (bridge ratio)\n")
        L.append(df_ke_md(ctx["kal"].nlargest(10, "skor_br")[
            ["tau", "rho", "skor_br", "rec_harus_pecah", "rec_harus_utuh",
             "skor_makro3", "limf_tak_pecah"]]))
    if "amb" in ctx:
        L.append("\n## 4. Ambang dan plateau\n")
        L.append(df_ke_md(pd.DataFrame([ctx["amb"]]).T.reset_index().rename(
            columns={"index": "metrik", 0: "nilai"})))
        L.append("\n> Ambang 1/3 harus ditulis sebagai pilihan yang ongkosnya "
                 "kecil dan terletak di dalam IK bootstrap, **bukan** sebagai "
                 "argmax Youden. Interval optimal persisnya terlalu sempit untuk "
                 "memuat 1/3.\n")
    if "kel" in ctx:
        L.append("\n## 5. Kelompok neutrofil (parameter bridge ratio)\n")
        L.append(df_ke_md(ctx["kel"]))
    if "bentuk" in ctx:
        L.append("\n## 6. Jumlah lobus per nucleus_shape (parameter LOBUS)\n")
        L.append(df_ke_md(ctx["bentuk"]))
    if "kelas" in ctx:
        L.append("\n## 7. Kontrol lintas kelas sel (parameter LOBUS)\n")
        L.append(df_ke_md(ctx["kelas"]))
    if "ekor" in ctx:
        L.append("\n## 8. Ekor B\n")
        e = ctx["ekor"]
        L.append("- sel B salah klasifikasi: **%d** (Fase 2C: %d)"
                 % (len(e), REF_2C["ekor_B"]))
        for k, v in e["status_topologi"].value_counts().items():
            L.append("  - %s: %d (%.1f%%)" % (k, v, 100 * v / max(len(e), 1)))
        if "diag" in ctx:
            L.append("\n### 8.1 Jarak saddle ke piksel lubang terisi\n")
            L.append(df_ke_md(ctx["diag"]))
    if "def" in ctx:
        L.append("\n## 9. Analisis deferral\n")
        L.append(df_ke_md(ctx["def"]))
    L.append("\n## 10. Yang harus dijawab berikutnya\n")
    L.append("- Apakah sensus membenarkan aturan selektif, dan apakah batas "
             "`--lubang-maks-px` dan `--lubang-frak-sel` perlu digeser?")
    L.append("- Berapa galat ekor B yang tersisa setelah lubang dibereskan, "
             "dan apa penyebabnya sekarang?")
    L.append("- Fase 2E: sumbu pematangan ig, dan penulisan ulang Bagian 14.\n")
    (out / "LAPORAN_FASE2D.md").write_text("\n".join(L), encoding="utf-8")
    print("\n  laporan ditulis: %s" % (out / "LAPORAN_FASE2D.md"))


# ===========================================================================
# BAGIAN 13 -- MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fase 2D: aturan pengisian lubang dan analisis deferral.")
    ap.add_argument("--dir-kerja", default=".")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir-mask", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pekerja", type=int, default=7)
    ap.add_argument("--sampel", type=int, default=0)
    ap.add_argument("--tugas", default="semua")
    ap.add_argument("--aturan-lubang", default="selektif", choices=list(ATURAN_LUBANG))
    ap.add_argument("--lubang-maks-px", type=int, default=15)
    ap.add_argument("--lubang-frak-sel", type=float, default=0.5)
    ap.add_argument("--min-frak-komp", type=float, default=0.02)
    ap.add_argument("--prune", type=float, default=0.05)
    ap.add_argument("--aturan-saddle", default="min", choices=["min", "max"])
    ap.add_argument("--tau-lobus", type=float, default=None)
    ap.add_argument("--rho-lobus", type=float, default=None)
    ap.add_argument("--tau-br", type=float, default=None)
    ap.add_argument("--rho-br", type=float, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--paksa-cache", action="store_true")
    args = ap.parse_args()

    dk = Path(args.dir_kerja)
    out = Path(args.out) if args.out else dk / "hasil_fase2d"
    out.mkdir(parents=True, exist_ok=True)
    tug = set(t.strip() for t in args.tugas.split(","))
    mau = lambda t: ("semua" in tug) or (t in tug)

    print("=" * 76)
    print("FASE 2D -- aturan pengisian lubang, dan analisis deferral")
    print("=" * 76)
    print("keluaran: %s" % out.resolve())

    # ---- 1. SANITAS (selalu; pelajaran 1, Bagian 16)
    lolos = jalankan_sanitas(out)
    if not lolos:
        print("\nHENTI: uji sanitas gagal. Perbaiki implementasi dulu.")
        sys.exit(2)
    if tug == {"sanitas"}:
        return

    # ---- 2. DATA
    csvp = Path(args.csv) if args.csv else dk / "pbc_attr_v1_ccrop_all.csv"
    if not csvp.exists():
        kand = list(dk.rglob("pbc_attr_v1_ccrop_all.csv"))
        if not kand:
            print("HENTI: CSV atribut tidak ditemukan. Pakai --csv.")
            sys.exit(2)
        csvp = kand[0]
    df = pd.read_csv(csvp)
    dm = Path(args.dir_mask) if args.dir_mask else dk / "pbcseg_final_v1"
    if not dm.exists():
        print("HENTI: folder mask tidak ditemukan. Pakai --dir-mask.")
        sys.exit(2)
    peta, nunik = indeks_mask(dm)
    print("\nCSV %s (%d baris) | mask %d berkas unik" % (csvp.name, len(df), nunik))

    pref, kel = beri_kelompok(df)
    df["prefix"], df["kelompok"] = pref, kel
    penuh = args.sampel <= 0
    rep = None
    if not penuh:
        kunci = df["label"].astype(str) + "|" + df["kelompok"].astype(str)
        npg = max(4, args.sampel // max(kunci.nunique(), 1))
        amb = []
        for _, s in df.groupby(kunci, sort=False):
            amb += s.sample(min(len(s), npg), random_state=0).index.tolist()
        df = df.loc[amb].reset_index(drop=True)
        print("MODE SAMPEL: %d sel dari %d grup (label x kelompok)"
              % (len(df), kunci.nunique()))
    else:
        rep = all(int((df["kelompok"] == k).sum()) == v
                  for k, v in JUMLAH_KELOMPOK_HARAPAN.items())
        print("replikasi kelompok Fase 1: %s" % ("LOLOS" if rep else "GAGAL"))

    cfg = (True, args.aturan_lubang, args.lubang_maks_px, args.lubang_frak_sel,
           args.min_frak_komp, 8, args.prune, True)
    ctx = dict(n_sel=len(df), aturan_lubang=args.aturan_lubang,
               lubang_maks_px=args.lubang_maks_px,
               lubang_frak_sel=args.lubang_frak_sel, sanitas_lolos=lolos,
               replikasi_kelompok=rep, aturan_saddle=args.aturan_saddle,
               tugas_dijalankan=args.tugas)

    # ---- 3. ABLASI (membangun cache ketiga aturan; yang terpilih dipakai ulang)
    if mau("ablasi"):
        dfa, dfp, dfd2, menang = tugas_ablasi_lubang(
            df, peta, out, args.pekerja, cfg, args.aturan_saddle, args.paksa_cache)
        ctx["abl"], ctx["pas"], ctx["asal"] = dfa, dfp, dfd2
        if menang != args.aturan_lubang:
            print("  CATATAN: aturan terbaik menurut kriteria adalah '%s', "
                  "sedangkan analisis utama memakai '%s' sesuai --aturan-lubang."
                  % (menang, args.aturan_lubang))

    # ---- 4. CACHE UTAMA
    cache, _ = bangun_cache(df, peta, cfg, out, args.pekerja, paksa=args.paksa_cache)
    nama = [n for n in df["img_name"] if n in cache]
    meta = df.set_index("img_name").loc[nama].reset_index()
    T = Turunan(cache, nama)
    print("cache siap: %d sel" % len(nama))

    diag = pd.DataFrame([{
        "sel_total": len(nama),
        "komponen_mentah_>1": int(sum(cache[n]["komponen_mentah"] > 1 for n in nama)),
        "komponen_akhir_>1": int(sum(cache[n]["komponen_akhir"] > 1 for n in nama)),
        "sel_buang_piksel_luar_sel": int(sum(cache[n]["piks_buang_luar_sel"] > 0 for n in nama)),
        "sel_dengan_lubang": int(sum(cache[n]["lub_n"] > 0 for n in nama)),
        "sel_lubang_diisi": int(sum(cache[n]["lub_diisi_n"] > 0 for n in nama)),
        "sel_lubang_dibiarkan": int(sum(cache[n]["lub_dibiarkan_n"] > 0 for n in nama)),
        "r_lobus_maks": float(max(cache[n]["r_lobus"] for n in nama)),
    }])
    print("\n=== DIAGNOSTIK PRAPROSES ===")
    for k, v in diag.iloc[0].items():
        print("  %-30s %s" % (k, ("%.4f" % v) if k == "r_lobus_maks" else "%d" % int(v)))
    print("  (Fase 2C: 1556 sel berlubang, r_lobus maks 59.03)")
    diag.to_csv(out / "D1_diagnostik_praproses.csv", index=False)
    ctx["diag_pra"] = diag

    # ---- 5. SENSUS
    if mau("sensus"):
        dsel, dreg, frak, tabkomp = tugas_sensus(cache, meta, out)
        ctx["frak_jar"], ctx["sensus_komposisi"] = frak, tabkomp

    # ---- 6. KALIBRASI
    pl, pb = (args.tau_lobus, args.rho_lobus), (args.tau_br, args.rho_br)
    if mau("kalibrasi"):
        dfk, bl, bb = tugas_kalibrasi(T, meta, out, args.aturan_saddle)
        ctx["kal"] = dfk
        if pl[0] is None:
            pl = bl
        if pb[0] is None:
            pb = bb
    cadangan = (pl[0] is None) or (pb[0] is None)
    pl = (float(pl[0]) if pl[0] is not None else 3.0,
          float(pl[1]) if pl[1] is not None else 0.4)
    pb = (float(pb[0]) if pb[0] is not None else 0.75,
          float(pb[1]) if pb[1] is not None else 0.2)
    if cadangan and not mau("kalibrasi"):
        print("\n  *** PERINGATAN: tugas 'kalibrasi' tidak dijalankan dan")
        print("      --tau-*/--rho-* tidak diberikan, jadi skrip memakai")
        print("      parameter CADANGAN dari Fase 2C (lobus 3/0.4, bridge")
        print("      0.75/0.2). Parameter itu dikalibrasi pada aturan lubang")
        print("      'semua' dan belum tentu cocok di sini. Jalankan tugas")
        print("      'kalibrasi', atau sebutkan parameternya secara eksplisit,")
        print("      sebelum angka mana pun dibawa ke pembimbing. ***")
    ctx.update(tau_lobus=pl[0], rho_lobus=pl[1], tau_br=pb[0], rho_br=pb[1],
               parameter_cadangan=bool(cadangan and not mau("kalibrasi")))
    print("\nPARAMETER: lobus (tau=%g rho=%g) | bridge ratio (tau=%g rho=%g)"
          % (pl[0], pl[1], pb[0], pb[1]))

    # ---- 7. DATA INTI
    hb = T.frame(pb[0], pb[1], args.aturan_saddle)
    hl = T.frame(pl[0], pl[1], args.aturan_saddle)
    dfin = meta.merge(hb, on="img_name", how="left")
    dfl = meta.merge(hl, on="img_name", how="left")
    dfin["n_lobus_kal"] = dfl["n_lobus"].values
    for c, v in (("tau_br", pb[0]), ("rho_br", pb[1]),
                 ("tau_lobus", pl[0]), ("rho_lobus", pl[1])):
        dfin[c] = v
    dfin["aturan_lubang"] = args.aturan_lubang
    dfin.to_csv(out / "bridge_ratio_2d.csv", index=False)
    print("data inti disimpan: bridge_ratio_2d.csv (%d baris)" % len(dfin))

    ctx["bentuk"], ctx["kelas"] = tabel_bentuk(dfl), tabel_kelas(dfl)
    ctx["kel"] = tabel_kelompok(dfin)
    print("\n=== JUMLAH LOBUS PER nucleus_shape (parameter LOBUS) ===")
    print(ctx["bentuk"].to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\n=== KONTROL LINTAS KELAS (parameter LOBUS) ===")
    print(ctx["kelas"].to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\n=== KELOMPOK NEUTROFIL (parameter BRIDGE RATIO) ===")
    print(ctx["kel"].to_string(index=False, float_format=lambda v: "%.4f" % v))
    pc = ctx["kel"].loc[ctx["kel"]["kelompok"] == "C_konflik_SNE_band", "posisi_AB"]
    if len(pc):
        print("  posisi kelompok C pada sumbu A->B: %.4f   "
              "[Fase 2C %.4f | Fase 2B 0.328 | bridge ratio v1 0.339]"
              % (float(pc.iloc[0]), REF_2C["posisi_C"]))

    # ---- 8. AMBANG, AUDIT, DEFERRAL, LAPORAN
    if mau("ambang"):
        ctx["amb"] = tugas_ambang(T, meta, out, pb, args.aturan_saddle, args.boot)
    if mau("audit"):
        ctx["ekor"], ctx["diag"] = tugas_audit(dfin, out)
    if mau("deferral"):
        ctx["def"] = tugas_deferral(dfin, out)
    if mau("laporan"):
        tulis_laporan(out, ctx)

    print("\nSELESAI. Kirim LAPORAN_FASE2D.md beserta D0-D6*.csv.")


if __name__ == "__main__":
    main()
