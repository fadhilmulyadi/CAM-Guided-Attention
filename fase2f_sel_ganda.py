#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fase2f_sel_ganda.py — Fase 2F bagian 1: M11 (sensitivitas terpisah_di_nol) dan M9 (sel ganda)

URUTAN KERJA
  0. Uji sanitas sintetis (berhenti bila ada yang gagal).
  1. M11 — tanpa mask, hanya dari bridge_ratio_2d.csv:
       S0 baseline (sekaligus uji reproduksi terhadap angka Fase 2D),
       S1 buang seluruh sel terpisah_di_nol,
       S2 batas pesimis: sel B terpisah_di_nol dianggap salah (br := 1).
  2. M9 — detektor sel ganda:
       a. Persistensi topologis dijalankan pada MASK SEL (bukan nukleus). Dua sel yang
          bersentuhan membentuk 'halter' pada distance transform sel; badan kedua muncul
          sebagai penggabungan dengan persistensi relatif f = persistence / r_sel tinggi.
       b. Kalibrasi ambang f SEPENUHNYA INDEPENDEN dari A/B neutrofil:
            - negatif nyata : limfosit + monosit dengan tepat 1 komponen inti
            - kriteria      : Neyman–Pearson, FPR <= --fpr (bawaan 1%) pada negatif nyata.
              Ambang = nilai TERKECIL yang memenuhi batas itu, dihitung dari FPR empiris,
              BUKAN np.quantile: kuantil tidak menjamin FPR <= alfa untuk aturan 'f > t'
              bila sebaran f menumpuk di satu nilai (mis. nol).
            - positif sintetis : komposit dua sel non-neutrofil asli yang ditempelkan
            - positif nyata : limfosit/monosit dengan >=2 komponen inti besar
              (inti limfosit/monosit tidak berlobus; dua inti = dua sel) -> validasi
          AUC A-vs-B TIDAK dipakai di lapis mana pun.
       c. Sel yang terdeteksi dipartisi dengan watershed pada DT sel; hanya badan yang
          memuat pusat bingkai (citra center-crop) yang dipertahankan; bridge ratio dihitung ulang.
       d. Validasi partisi pada komposit: bridge ratio setelah koreksi harus sama dengan
          bridge ratio sel tunggal aslinya.
       e. Skenario S3a/S3b/S4, sapuan sensitivitas terhadap ambang (F8), montase, dan
          berkas sensus visual (F7) untuk diisi manusia.

DIUJI DI MANA
  Uji sanitas (10 kasus sintetis berjawaban analitis) lolos, dan SELURUH pipeline sudah
  dijalankan ujung-ke-ujung pada dataset mask sintetis 706 sel dengan 37 sel ganda yang
  ditanam sengaja: ke-37 tertangkap, nol positif palsu, partisi memulihkan bridge ratio
  sel tunggal aslinya dengan IoU inti 1,000. Yang BELUM teruji adalah perilaku pada mask
  asli — itulah yang dijalankan sekarang.

CONTOH
  python fase2f_sel_ganda.py --tugas sanitas
  python fase2f_sel_ganda.py --tugas m11 > log_2f_m11.txt 2>&1
  python fase2f_sel_ganda.py --tugas semua --pekerja 7 --dir-mask pbcseg_final_v1 ^
         --dir-citra <folder_citra_RGB_PBC> > log_2f.txt 2>&1
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import rankdata
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

S8 = np.ones((3, 3), bool)
KELAS_KALIBRASI = ("Lymphocyte", "Monocyte")
SEPERTIGA = 1.0 / 3.0

# Angka Fase 2D final yang harus direproduksi oleh S0 (uji reproduksi analisis)
HARAPAN_S0 = {
    "auc": (0.96674, 5e-5),
    "youden": (0.376355, 1e-5),
    "acc_13": (0.932833, 1e-5),
    "acc_test_13": (0.932380, 1e-5),
    "posisi_C": (0.3247, 5e-4),
    "def90_acc": (0.9579, 5e-4),
    "def90_peng_C": (2.3709, 5e-3),
}

# Sel dari adjudikasi montase Fase 2E (Bagian 12.1) — VALIDASI, bukan kalibrasi
ADJ_A_GANDA = ["BNE_333793", "BNE_839717", "BNE_356501", "BNE_51939", "BNE_359949", "BNE_290973"]
ADJ_B_GANDA = ["SNE_691743", "SNE_699973", "SNE_119786", "SNE_316202", "SNE_905668"]


# =============================================================================
# 0. LAPORAN
# =============================================================================
class Laporan:
    def __init__(self, path):
        self.path = path
        self.baris = []
        self.peringatan_list = []

    def tulis(self, s=""):
        print(s, flush=True)
        self.baris.append(str(s))

    def peringatan(self, s):
        self.peringatan_list.append(s)
        self.tulis(f"\n> **PERINGATAN:** {s}\n")

    def tabel(self, df, desimal=4):
        self.tulis(tabel_md(df, desimal))
        self.tulis()

    def simpan(self):
        teks = list(self.baris)
        if self.peringatan_list:
            teks += ["", "## RANGKUMAN PERINGATAN (dibaca sebelum menarik kesimpulan)", ""]
            teks += [f"{i + 1}. {p}" for i, p in enumerate(self.peringatan_list)]
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(teks) + "\n")


def tabel_md(df, desimal=4):
    kol = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "YA" if v else "TIDAK"
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "–"
            return f"{v:.{desimal}f}"
        return str(v)

    baris = ["| " + " | ".join(kol) + " |", "|" + "---|" * len(kol)]
    for _, r in df.iterrows():
        baris.append("| " + " | ".join(fmt(r[c]) for c in df.columns) + " |")
    return "\n".join(baris)


# =============================================================================
# 1. GEOMETRI: PERSISTENSI TOPOLOGIS PADA DISTANCE TRANSFORM
# =============================================================================
def crop_pad(b):
    ys, xs = np.nonzero(b)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return np.pad(b[y0:y1, x0:x1], 1), (int(y0) - 1, int(x0) - 1)


def tempel(potong, off, shape):
    out = np.zeros(shape, bool)
    h, w = potong.shape
    y0, x0 = off
    ya, xa = max(y0, 0), max(x0, 0)
    yb, xb = min(y0 + h, shape[0]), min(x0 + w, shape[1])
    if yb > ya and xb > xa:
        out[ya:yb, xa:xb] = potong[ya - y0:yb - y0, xa - x0:xb - x0]
    return out


def persistensi(b, prune=0.05):
    """b: bool, sudah di-pad (tepi False). Superlevel-set filtration DT, union-find 8-konektivitas.
    kejadian = (saddle, persistence, puncak_mati, puncak_utama, idx_saddle, idx_puncak_mati)."""
    dt = ndi.distance_transform_edt(b)
    H, W = dt.shape
    flat = dt.ravel()
    fg = np.flatnonzero(flat > 0)
    if fg.size == 0:
        return dict(dt=dt, W=W, r_maks=0.0, kej=[], bertahan=[])
    urut = fg[np.argsort(-flat[fg], kind="stable")].tolist()
    nilai = flat.tolist()
    N = H * W
    induk = [-1] * N
    pk = [0.0] * N
    pki = [-1] * N
    lahir = [0] * N
    offs = (-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1)
    kej = []
    for urutan, p in enumerate(urut):
        v = nilai[p]
        akar = []
        for o in offs:
            q = p + o
            if induk[q] != -1:
                r = q
                while induk[r] != r:
                    r = induk[r]
                while induk[q] != r:
                    nx = induk[q]
                    induk[q] = r
                    q = nx
                if r not in akar:
                    akar.append(r)
        if not akar:
            induk[p] = p
            pk[p] = v
            pki[p] = p
            lahir[p] = urutan
            continue
        if len(akar) > 1:
            akar.sort(key=lambda r: (-pk[r], lahir[r]))
            utama = akar[0]
            for r in akar[1:]:
                pers = pk[r] - v
                if pers >= prune:
                    kej.append((v, pers, pk[r], pk[utama], p, pki[r]))
                induk[r] = utama
        else:
            utama = akar[0]
        induk[p] = utama
    akhir = set()
    for p in fg.tolist():
        r = p
        while induk[r] != r:
            r = induk[r]
        akhir.add(r)
    bertahan = sorted([(pk[r], pki[r]) for r in akhir], key=lambda x: -x[0])
    return dict(dt=dt, W=W, r_maks=float(flat.max()), kej=kej, bertahan=bertahan)


def turunkan(res, tau, rho, aturan="min"):
    r = res["r_maks"]
    sg = [k for k in res["kej"] if k[1] >= tau and k[2] >= rho * r]
    sb = [p for p, _ in res["bertahan"] if p >= tau and p >= rho * r]
    n = len(sg) + len(sb)
    if len(sb) >= 2:
        return 0.0, 0.0, n, "terpisah_di_nol"
    if sg:
        rp = min(k[0] for k in sg) if aturan == "min" else max(k[0] for k in sg)
        return rp, rp / r, n, "normal"
    return float("nan"), 1.0, n, "tak_pernah_pecah"


def komponen_sel(mask):
    """Komponen sel target (M4): komponen terhubung mask in {1,2,5} yang memuat pusat bingkai."""
    H, W = mask.shape
    cy, cx = H // 2, W // 2
    sel = np.isin(mask, (1, 2, 5))
    lab, n = ndi.label(sel, structure=S8)
    if n == 0:
        return sel, True
    l = lab[cy, cx]
    fb = False
    if l == 0:
        fb = True
        win = lab[max(cy - 20, 0):cy + 21, max(cx - 20, 0):cx + 21]
        v = win[win > 0]
        l = int(np.bincount(v).argmax()) if v.size else int(np.bincount(lab.ravel())[1:].argmax()) + 1
    return lab == l, fb


def inti_M4(mask, batas, min_frak=0.02):
    """Nukleus diiris dengan batas, buang komponen < 2% luas total. Lubang TIDAK diisi."""
    nuk = (mask == 2) & batas
    if not nuk.any():
        return nuk
    lab, n = ndi.label(nuk, structure=S8)
    if n > 1:
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        simpan = np.flatnonzero(sz >= min_frak * sz.sum())
        nuk = np.isin(lab, simpan)
    return nuk


def ukur_inti(nuk, prm):
    if not nuk.any():
        return dict(r_lobus=np.nan, r_pisah=np.nan, br=np.nan, status="kosong",
                    n_lobus_kal=0, luas_nukleus=0)
    c, _ = crop_pad(nuk)
    res = persistensi(c, prm["prune"])
    rp, br, _, st = turunkan(res, prm["tau_br"], prm["rho_br"])
    _, _, nl, _ = turunkan(res, prm["tau_lobus"], prm["rho_lobus"])
    return dict(r_lobus=res["r_maks"], r_pisah=rp, br=br, status=st,
                n_lobus_kal=nl, luas_nukleus=int(nuk.sum()))


def fitur_badan(sel):
    """Persistensi pada mask SEL. f_badan = persistence terbesar / r_sel."""
    kosong = dict(f_badan=0.0, puncak_rel_badan=0.0, saddle_rel_badan=0.0, r_sel=0.0, luas_sel=0)
    if not sel.any():
        return kosong, None, None
    c, off = crop_pad(sel)
    res = persistensi(c, prune=0.5)
    r = res["r_maks"]
    if res["kej"]:
        k = max(res["kej"], key=lambda k: k[1])
        f, pr, sr = k[1] / r, k[2] / r, k[0] / r
    else:
        f = pr = sr = 0.0
    return dict(f_badan=f, puncak_rel_badan=pr, saddle_rel_badan=sr, r_sel=r,
                luas_sel=int(sel.sum())), res, off


def pisah_badan(sel, res, off, t, pusat):
    """Watershed pada -DT sel dengan penanda = puncak utama + puncak badan yang f > t.
    Kembalikan badan yang memuat pusat bingkai."""
    if res is None:
        return sel.copy()
    from skimage.segmentation import watershed
    dt = res["dt"]
    r = res["r_maks"]
    penanda = [res["bertahan"][0][1]] + [k[5] for k in res["kej"] if k[1] / r > t]
    if len(penanda) == 1:
        return sel.copy()
    mk = np.zeros(dt.shape, np.int32)
    for i, p in enumerate(penanda):
        mk.flat[p] = i + 1
    lab = watershed(-dt, mk, mask=dt > 0)
    py, px = pusat[0] - off[0], pusat[1] - off[1]
    if 0 <= py < lab.shape[0] and 0 <= px < lab.shape[1] and lab[py, px] > 0:
        pilih = lab[py, px]
    else:
        ys, xs = np.unravel_index(penanda, dt.shape)
        pilih = int(np.argmin((ys - py) ** 2 + (xs - px) ** 2)) + 1
    return tempel(lab == pilih, off, sel.shape) & sel


def geser(a, dy, dx):
    out = np.zeros_like(a)
    H, W = a.shape
    if abs(dy) >= H or abs(dx) >= W:
        return out
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        a[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def komposit(m1, m2, fr, sudut):
    """Tempelkan sel target m2 di samping sel target m1, jarak pusat = fr * (R1 + R2).
    Sel 1 tetap di atas (tidak tertutup)."""
    s1, _ = komponen_sel(m1)
    s2, _ = komponen_sel(m2)
    L1 = np.where(s1, m1, 0).astype(np.uint8)
    L2 = np.where(s2, m2, 0).astype(np.uint8)
    c1 = np.array(ndi.center_of_mass(s1))
    c2 = np.array(ndi.center_of_mass(s2))
    R1 = math.sqrt(s1.sum() / math.pi)
    R2 = math.sqrt(s2.sum() / math.pi)
    d = fr * (R1 + R2)
    tujuan = c1 + d * np.array([math.sin(sudut), math.cos(sudut)])
    dy, dx = np.round(tujuan - c2).astype(int)
    L2g = geser(L2, int(dy), int(dx))
    kan = L1.copy()
    isi = (kan == 0) & (L2g > 0)
    kan[isi] = L2g[isi]
    return kan, L1, s1, L2g


# =============================================================================
# 2. BERKAS
# =============================================================================
def kunci_nama(s):
    b = os.path.splitext(os.path.basename(str(s)))[0]
    if b.endswith("_mask"):
        b = b[:-5]
    if b.endswith("_ccrop"):
        b = b[:-6]
    return b


def indeks_berkas(akar, pola, buang_mask=False):
    idx = {}
    for p in pola:
        for f in glob.glob(os.path.join(akar, "**", p), recursive=True):
            nama = os.path.basename(f)
            if buang_mask and "_mask" in nama:
                continue
            k = kunci_nama(f)
            if k not in idx or "_ccrop" in nama:
                idx[k] = f
    return idx


def baca_mask(p):
    from PIL import Image
    a = np.array(Image.open(p))
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.uint8)


def baca_rgb(p, shape):
    from PIL import Image
    a = np.array(Image.open(p).convert("RGB"))
    H, W = shape
    h, w = a.shape[:2]
    if (h, w) != (H, W):
        y0, x0 = max((h - H) // 2, 0), max((w - W) // 2, 0)
        a = a[y0:y0 + H, x0:x0 + W]
    return a


# =============================================================================
# 3. PEKERJA (top-level agar aman untuk multiprocessing di Windows)
# =============================================================================
def kerja_sel(arg):
    img, mpath, prm, t = arg
    out = {"img_name": img, "galat": ""}
    try:
        mask = baca_mask(mpath)
        H, W = mask.shape
        sel, fb = komponen_sel(mask)
        fitur, res, off = fitur_badan(sel)
        if t is None:
            out.update(fitur)
            out["pusat_fallback"] = fb
            nuk = inti_M4(mask, sel)
            lab, n = ndi.label(nuk, structure=S8)
            sz = np.sort(np.bincount(lab.ravel())[1:])[::-1] if n else np.array([0])
            out["n_komp_inti"] = int(n)
            out["frak_komp2"] = float(sz[1] / sz.sum()) if n > 1 else 0.0
            for k, v in ukur_inti(nuk, prm).items():
                out["ori_" + k] = v
        else:
            badan = pisah_badan(sel, res, off, t, (H // 2, W // 2))
            nuk = inti_M4(mask, badan)
            for k, v in ukur_inti(nuk, prm).items():
                out["kor_" + k] = v
            out["kor_luas_badan"] = int(badan.sum())
    except Exception as e:  # dicatat, tidak menghentikan jalanan
        out["galat"] = repr(e)
    return out


def kerja_komposit(arg):
    i, p1, p2, fr, sudut, prm, t = arg
    out = {"i": i, "fr": fr, "sudut": sudut, "p1": p1, "p2": p2, "galat": ""}
    try:
        m1, m2 = baca_mask(p1), baca_mask(p2)
        out.update(ukur_komposit(m1, m2, fr, sudut, prm, t))
    except Exception as e:
        out["galat"] = repr(e)
    return out


def ukur_komposit(m1, m2, fr, sudut, prm, t):
    kan, L1, s1, L2g = komposit(m1, m2, fr, sudut)
    H, W = kan.shape
    sel, _ = komponen_sel(kan)
    out = {"nyambung": bool((sel & (L2g > 0) & (L1 == 0)).any())}
    fitur, res, off = fitur_badan(sel)
    out.update(fitur)
    o = ukur_inti(inti_M4(kan, sel), prm)
    out["st_gabung"], out["br_gabung"] = o["status"], o["br"]
    if t is not None:
        ref_nuk = inti_M4(m1, s1)
        ref = ukur_inti(ref_nuk, prm)
        badan = pisah_badan(sel, res, off, t, (H // 2, W // 2))
        nk = inti_M4(kan, badan)
        kor = ukur_inti(nk, prm)
        out.update(terdeteksi=bool(fitur["f_badan"] > t),
                   iou_inti=float((nk & ref_nuk).sum() / max((nk | ref_nuk).sum(), 1)),
                   br_ref=ref["br"], st_ref=ref["status"], br_kor=kor["br"], st_kor=kor["status"])
    return out


def jalankan_pool(fungsi, tugas, pekerja, label, lap):
    t0 = time.time()
    hasil = []
    n = len(tugas)

    def maju(i):
        # baris teks untuk log file; bar tqdm otomatis mati bila stderr bukan terminal (disable=None)
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"  {label}: {i + 1}/{n} ({(i + 1) / n:.0%}) {dt:.0f} s, "
                  f"sisa ~{dt * (n - i - 1) / (i + 1):.0f} s", flush=True)

    if pekerja <= 1:
        for i, a in enumerate(tqdm(tugas, desc=label, smoothing=0.05, disable=None)):
            hasil.append(fungsi(a))
            maju(i)
    else:
        with Pool(pekerja) as pool:
            for i, h in enumerate(tqdm(pool.imap_unordered(fungsi, tugas, chunksize=8), total=n,
                                       desc=label, smoothing=0.05, disable=None)):
                hasil.append(h)
                maju(i)
    dt = time.time() - t0
    lap.tulis(f"- {label}: {len(tugas)} tugas, {dt:.1f} detik ({1000 * dt / max(len(tugas), 1):.1f} ms/tugas)")
    return hasil


# =============================================================================
# 4. ANALISIS AMBANG / KELOMPOK / DEFERRAL
# =============================================================================
def auc(pos, neg):
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden(pos, neg):
    """Positif bila skor >= t. Kembalikan (t, J, tpr, fpr); t terkecil yang mencapai J maks."""
    t = np.unique(np.concatenate([pos, neg]))
    ps, ns = np.sort(pos), np.sort(neg)
    tpr = 1 - np.searchsorted(ps, t, "left") / len(ps)
    fpr = 1 - np.searchsorted(ns, t, "left") / len(ns)
    J = tpr - fpr
    i = int(np.argmax(J))
    return float(t[i]), float(J[i]), float(tpr[i]), float(fpr[i])


def akurasi(A, B, t):
    return float(((A >= t).sum() + (B < t).sum()) / (len(A) + len(B)))


def ambang_np(fneg, alfa):
    """Ambang Neyman–Pearson untuk aturan 'deteksi bila f > t': t TERKECIL yang membuat
    FPR empiris pada negatif nyata <= alfa. Dipakai menggantikan np.quantile karena
    kuantil tidak menjamin FPR <= alfa bila sebaran f menumpuk di satu nilai (mis. nol)."""
    f = np.asarray(fneg, float)
    kand = np.unique(np.concatenate([[0.0], f]))
    for t in kand:
        if float((f > t).mean()) <= alfa:
            return float(t), float((f > t).mean())
    return float(kand[-1]), float((f > kand[-1]).mean())


def ringkas_cepat(d, kol):
    """AUC / akurasi / posisi C tanpa bootstrap — untuk sapuan ambang yang murah."""
    x = d[np.isfinite(d[kol].astype(float))]
    kel = x["kelompok"].astype(str).str[0]
    A = x.loc[kel == "A", kol].to_numpy(float)
    B = x.loc[kel == "B", kol].to_numpy(float)
    C = x.loc[kel == "C", kol].to_numpy(float)
    if not len(A) or not len(B):
        return dict(n_A=len(A), n_B=len(B), auc=np.nan, acc_13=np.nan, spes_band=np.nan, posisi_C=np.nan)
    mA, mB = float(np.median(A)), float(np.median(B))
    return dict(n_A=len(A), n_B=len(B), auc=auc(A, B), acc_13=akurasi(A, B, SEPERTIGA),
                spes_band=float((B < SEPERTIGA).mean()),
                posisi_C=float((mA - np.median(C)) / (mA - mB)) if len(C) and mA != mB else np.nan)


def entropi(p):
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def analisis(df, kol_br, kol_status, nama, rng, nboot=2000):
    """df: neutrofil saja, dengan kolom kelompok, split. Kembalikan ringkasan + tabel."""
    hilang = ~np.isfinite(df[kol_br].astype(float))
    d = df[~hilang].copy()
    d["kel"] = d["kelompok"].astype(str).str[0]
    n_buang = int(hilang.sum())
    buang_kel = df.loc[hilang, "kelompok"].astype(str).str[0].value_counts().to_dict() if n_buang else {}
    A = d.loc[d.kel == "A", kol_br].to_numpy(float)
    B = d.loc[d.kel == "B", kol_br].to_numpy(float)
    t, J, tpr, fpr = youden(A, B)
    boot = []
    for _ in range(nboot):
        boot.append(youden(rng.choice(A, len(A)), rng.choice(B, len(B)))[0])
    lo, med, hi = np.percentile(boot, [2.5, 50, 97.5])
    tr = d[d.split == "train"]
    te = d[d.split == "test"]
    At, Bt = tr.loc[tr.kel == "A", kol_br].to_numpy(float), tr.loc[tr.kel == "B", kol_br].to_numpy(float)
    Ae, Be = te.loc[te.kel == "A", kol_br].to_numpy(float), te.loc[te.kel == "B", kol_br].to_numpy(float)
    t_tr = youden(At, Bt)[0]

    medA, medB = np.median(A), np.median(B)
    baris = []
    for g in sorted(d["kelompok"].dropna().unique()):
        x = d.loc[d.kelompok == g, kol_br].to_numpy(float)
        st = d.loc[d.kelompok == g, kol_status]
        p = float((x >= SEPERTIGA).mean())
        baris.append(dict(kelompok=g, n=len(x), q25=np.percentile(x, 25), median=np.median(x),
                          q75=np.percentile(x, 75), p_band=p, entropi=entropi(p),
                          zona_023_043=float(((x >= 0.23) & (x <= 0.43)).mean()),
                          tak_pecah=float((st == "tak_pernah_pecah").mean()),
                          terpisah0=int((st == "terpisah_di_nol").sum()),
                          posisi_AB=float((medA - np.median(x)) / (medA - medB))))
    tk = pd.DataFrame(baris)

    # deferral pada seluruh neutrofil
    k = (d[kol_br].astype(float) - SEPERTIGA).abs().to_numpy()
    urut = np.argsort(-k, kind="stable")
    n = len(d)
    frak_C = float((d.kel == "C").mean())
    frak_A = float((d.kel == "A").mean())
    bd = []
    for c in (1.0, 0.95, 0.90, 0.80, 0.70, 0.50):
        ns = int(round(n * c))
        s = d.iloc[urut[:ns]]
        u = d.iloc[urut[ns:]]
        As, Bs = s.loc[s.kel == "A", kol_br].to_numpy(float), s.loc[s.kel == "B", kol_br].to_numpy(float)
        bd.append(dict(cakupan=c, n_simpan=ns, n_tunda=n - ns, n_AB=len(As) + len(Bs),
                       akurasi=akurasi(As, Bs, SEPERTIGA),
                       spes_band=float((Bs < SEPERTIGA).mean()) if len(Bs) else np.nan,
                       sens_band=float((As >= SEPERTIGA).mean()) if len(As) else np.nan,
                       peng_A=float((u.kel == "A").mean() / frak_A) if len(u) else np.nan,
                       peng_C=float((u.kel == "C").mean() / frak_C) if len(u) else np.nan))
    tdef = pd.DataFrame(bd)
    r90 = tdef[tdef.cakupan == 0.90].iloc[0]
    posC = tk.loc[tk.kelompok.str.startswith("C"), "posisi_AB"]
    zC = tk.loc[tk.kelompok.str.startswith("C"), "zona_023_043"]
    zA = tk.loc[tk.kelompok.str.startswith("A"), "zona_023_043"]
    ring = dict(skenario=nama, n_A=len(A), n_B=len(B), auc=auc(A, B), youden=t, J=J,
                boot_med=med, ik_lo=lo, ik_hi=hi, sepertiga_di_ik=bool(lo <= SEPERTIGA <= hi),
                acc_13=akurasi(A, B, SEPERTIGA), acc_youden=akurasi(A, B, t),
                sens_band=float((A >= SEPERTIGA).mean()), spes_band=float((B < SEPERTIGA).mean()),
                galat_A=int((A < SEPERTIGA).sum()), galat_B=int((B >= SEPERTIGA).sum()),
                auc_test=auc(Ae, Be) if len(Ae) and len(Be) else np.nan,
                ambang_train=t_tr,
                acc_test_train=akurasi(Ae, Be, t_tr) if len(Ae) else np.nan,
                acc_test_13=akurasi(Ae, Be, SEPERTIGA) if len(Ae) else np.nan,
                posisi_C=float(posC.iloc[0]) if len(posC) else np.nan,
                zona_C_per_A=float(zC.iloc[0] / zA.iloc[0]) if len(zC) and len(zA) and zA.iloc[0] > 0 else np.nan,
                def90_acc=float(r90.akurasi), def90_peng_C=float(r90.peng_C),
                n_dibuang=n_buang, dibuang_per_kel=str(buang_kel))
    return ring, tk, tdef


def laporkan_skenario(lap, ring, tk, tdef):
    lap.tulis(f"#### {ring['skenario']}")
    lap.tulis()
    lap.tulis(f"A = {ring['n_A']}, B = {ring['n_B']} | AUC **{ring['auc']:.5f}** | "
              f"Youden {ring['youden']:.6f} (J {ring['J']:.5f}) | bootstrap median {ring['boot_med']:.6f}, "
              f"IK95 [{ring['ik_lo']:.6f}; {ring['ik_hi']:.6f}] | 1/3 di IK95: "
              f"**{'YA' if ring['sepertiga_di_ik'] else 'TIDAK'}**")
    lap.tulis(f"akurasi @1/3 **{ring['acc_13']:.6f}** | @Youden {ring['acc_youden']:.6f} | "
              f"galat A {ring['galat_A']}, galat B {ring['galat_B']} | sens band {ring['sens_band']:.4f}, "
              f"spes band {ring['spes_band']:.4f}")
    lap.tulis(f"train→test: ambang train {ring['ambang_train']:.6f}, AUC test {ring['auc_test']:.5f}, "
              f"akurasi test @ambang {ring['acc_test_train']:.6f}, @1/3 {ring['acc_test_13']:.6f}")
    if ring.get("n_dibuang"):
        lap.tulis(f"sel dengan bridge ratio tak-hingga/NaN yang tidak masuk analisis: "
                  f"**{ring['n_dibuang']}** {ring['dibuang_per_kel']}")
    lap.tulis()
    lap.tabel(tk)
    lap.tabel(tdef)


# =============================================================================
# 5. UJI SANITAS
# =============================================================================
def cakram(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def uji_sanitas(lap, prm):
    lap.tulis("## 0. Uji sanitas")
    lap.tulis()
    hasil = []

    def cek(nama, kondisi, detail):
        hasil.append(dict(uji=nama, lolos=bool(kondisi), detail=detail))

    tol = 0.02
    # 1 halter, leher 9 baris
    b = np.zeros((80, 160), bool)
    b |= cakram(b.shape, 40, 40, 20) | cakram(b.shape, 40, 120, 20)
    b[36:45, 40:121] = True
    m = ukur_inti(b, prm)
    cek("halter leher 9 baris", abs(m["r_pisah"] - 5.0) < tol and m["n_lobus_kal"] == 2
        and abs(m["r_lobus"] - math.sqrt(401)) < tol,
        f"r_pisah {m['r_pisah']:.4f} (harap 5), r_lobus {m['r_lobus']:.4f} (harap 20,025), lobus {m['n_lobus_kal']}")
    # 2 rantai leher 7 dan 17: min 4, max 9
    b = np.zeros((80, 262), bool)
    for x in (40, 130, 220):
        b |= cakram(b.shape, 40, x, 22)
    b[37:44, 40:131] = True
    b[32:49, 130:221] = True
    c, _ = crop_pad(b)
    res = persistensi(c)
    rmin = turunkan(res, prm["tau_br"], prm["rho_br"], "min")
    rmax = turunkan(res, prm["tau_br"], prm["rho_br"], "max")
    cek("rantai leher 7 & 17 (min vs max)", abs(rmin[0] - 4) < tol and abs(rmax[0] - 9) < tol and rmin[2] == 3,
        f"min {rmin[0]:.4f} (harap 4), max {rmax[0]:.4f} (harap 9), lobus {rmin[2]}")
    # 3 cakram konveks
    m = ukur_inti(cakram((80, 80), 40, 40, 30), prm)
    cek("cakram konveks", m["status"] == "tak_pernah_pecah" and m["n_lobus_kal"] == 1, f"status {m['status']}")
    # 4 dua cakram terpisah
    b = cakram((60, 140), 30, 30, 20) | cakram((60, 140), 30, 105, 20)
    m = ukur_inti(b, prm)
    cek("dua cakram terpisah", m["status"] == "terpisah_di_nol" and m["br"] == 0, f"status {m['status']}")

    # 5 sel ganda bersentuhan: masing-masing inti cakram
    S = (360, 360)
    mk = np.zeros(S, np.uint8)
    mk[cakram(S, 180, 180, 45)] = 1
    mk[cakram(S, 180, 261, 45) & (mk == 0)] = 1
    mk[cakram(S, 180, 180, 20)] = 2
    mk[cakram(S, 180, 261, 20)] = 2
    sel, _ = komponen_sel(mk)
    fit, res, off = fitur_badan(sel)
    ori = ukur_inti(inti_M4(mk, sel), prm)
    badan = pisah_badan(sel, res, off, 0.3, (180, 180))
    kor = ukur_inti(inti_M4(mk, badan), prm)
    cek("sel ganda terdeteksi dan dikoreksi",
        fit["f_badan"] > 0.4 and ori["status"] == "terpisah_di_nol" and kor["status"] == "tak_pernah_pecah"
        and abs(kor["luas_nukleus"] - cakram(S, 180, 180, 20).sum()) == 0,
        f"f {fit['f_badan']:.3f}, status {ori['status']} → {kor['status']}, luas inti {ori['luas_nukleus']} → {kor['luas_nukleus']}")
    # 6 sel tunggal dengan tonjolan kecil
    mk = np.zeros(S, np.uint8)
    mk[cakram(S, 180, 180, 45) | cakram(S, 180, 227, 8)] = 1
    mk[cakram(S, 180, 180, 20)] = 2
    sel, _ = komponen_sel(mk)
    fit, _, _ = fitur_badan(sel)
    cek("tonjolan kecil bukan sel kedua", fit["f_badan"] < 0.2, f"f {fit['f_badan']:.3f} (harap < 0,2)")
    # 7 sel tunggal, inti bilobed: badan tidak terdeteksi, br terjaga
    mk = np.zeros(S, np.uint8)
    mk[cakram(S, 180, 180, 50)] = 1
    nk = cakram(S, 180, 160, 14) | cakram(S, 180, 200, 14)
    nk[178:183, 160:201] = True
    mk[nk] = 2
    sel, _ = komponen_sel(mk)
    fit, res, off = fitur_badan(sel)
    ori = ukur_inti(inti_M4(mk, sel), prm)
    kor = ukur_inti(inti_M4(mk, pisah_badan(sel, res, off, 0.3, (180, 180))), prm)
    cek("bilobed dalam sel tunggal tak tersentuh",
        fit["f_badan"] < 0.1 and abs(ori["r_pisah"] - 3.0) < tol and ori["n_lobus_kal"] == 2
        and abs(kor["br"] - ori["br"]) < 1e-12,
        f"f {fit['f_badan']:.3f}, r_pisah {ori['r_pisah']:.4f} (harap 3), br {ori['br']:.4f} → {kor['br']:.4f}")
    # 8 komposit: menempel vs berjauhan
    m1 = np.zeros(S, np.uint8)
    m1[cakram(S, 180, 180, 40)] = 1
    m1[cakram(S, 180, 180, 18)] = 2
    ok = ukur_komposit(m1, m1.copy(), 0.9, 0.0, prm, 0.3)
    jauh = ukur_komposit(m1, m1.copy(), 1.3, 0.0, prm, None)
    cek("komposit menempel: terdeteksi, partisi memulihkan sel 1",
        ok["nyambung"] and ok["terdeteksi"] and ok["st_gabung"] == "terpisah_di_nol"
        and ok["st_kor"] == ok["st_ref"] and ok["iou_inti"] > 0.999,
        f"f {ok['f_badan']:.3f}, {ok['st_gabung']} → {ok['st_kor']} (ref {ok['st_ref']}), IoU {ok['iou_inti']:.4f}")
    cek("komposit berjauhan tidak menyambung", not jauh["nyambung"] and jauh["st_gabung"] == "tak_pernah_pecah",
        f"nyambung {jauh['nyambung']}, status {jauh['st_gabung']}")
    # 9 analisis: kasus trivial
    rng = np.random.default_rng(0)
    dd = pd.DataFrame(dict(kelompok=["A_x"] * 20 + ["B_x"] * 20 + ["C_x"] * 10,
                           split=(["train", "test"] * 25), br=[0.6] * 20 + [0.1] * 20 + [0.35] * 10,
                           st=["normal"] * 50))
    r, _, _ = analisis(dd, "br", "st", "uji", rng, nboot=50)
    cek("analisis kasus terpisah sempurna", r["auc"] == 1.0 and r["acc_13"] == 1.0 and abs(r["posisi_C"] - 0.5) < 1e-9,
        f"AUC {r['auc']}, akurasi {r['acc_13']}, posisi C {r['posisi_C']:.3f}")

    t = pd.DataFrame(hasil)
    lap.tabel(t)
    if not t.lolos.all():
        lap.peringatan("UJI SANITAS GAGAL — jalanan dihentikan sebelum menyentuh data asli.")
        return False
    lap.tulis(f"**Seluruh {len(t)} uji LOLOS.**")
    lap.tulis()
    return True


# =============================================================================
# 6. MONTASE
# =============================================================================
def montase(item, path, judul, t, idx_mask, idx_citra, kolom=6):
    if not item:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(item)
    baris = math.ceil(n / kolom)
    fig, axs = plt.subplots(baris, kolom, figsize=(kolom * 3.0, baris * 3.4))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, it in zip(axs, item):
        k = kunci_nama(it["img_name"])
        mask = baca_mask(idx_mask[k])
        H, W = mask.shape
        sel, _ = komponen_sel(mask)
        _, res, off = fitur_badan(sel)
        badan = pisah_badan(sel, res, off, t, (H // 2, W // 2))
        nuk = inti_M4(mask, sel)
        if k in idx_citra:
            ax.imshow(baca_rgb(idx_citra[k], mask.shape))
        else:
            ax.imshow(mask, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
        for arr, warna, lw in ((nuk, "red", 0.6), (sel, "yellow", 0.5), (badan, "cyan", 0.9)):
            if arr.any() and not arr.all():
                ax.contour(arr.astype(float), levels=[0.5], colors=warna, linewidths=lw)
        ax.plot(W // 2, H // 2, "+", color="lime", ms=6)
        ax.set_title(it["teks"], fontsize=6)
    fig.suptitle(judul + "\nmerah = inti (M4) | kuning = komponen sel | cyan = badan yang dipertahankan | + = pusat",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# 7. M9
# =============================================================================
def jalankan_m9(args, lap, df, prm, rng, skenario):
    try:
        import skimage  # noqa: F401
    except ImportError:
        lap.peringatan("scikit-image tidak terpasang. Jalankan: pip install scikit-image")
        return
    lap.tulis("## 2. M9 — Detektor sel ganda")
    lap.tulis()
    idx_mask = indeks_berkas(args.dir_mask, ("*_mask.png",))
    idx_citra = indeks_berkas(args.dir_citra, ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"),
                              buang_mask=True) if args.dir_citra else {}
    df = df.copy()
    df["kunci"] = df.img_name.map(kunci_nama)
    ada = df.kunci.isin(idx_mask)
    lap.tulis(f"- mask terindeks {len(idx_mask)}, cocok dengan CSV {int(ada.sum())}/{len(df)}; "
              f"citra RGB terindeks {len(idx_citra)}, cocok {int(df.kunci.isin(idx_citra).sum())}")
    if ada.sum() == 0:
        lap.peringatan("Tidak satu pun mask ditemukan. Periksa --dir-mask.")
        return
    if (~ada).any():
        lap.peringatan(f"{int((~ada).sum())} baris CSV tanpa mask; dikeluarkan dari M9.")
    kerja = df[ada]
    if args.sampel:
        kerja = pd.concat([kerja[kerja.status_topologi.eq("terpisah_di_nol") & kerja.label.eq("Neutrophil")],
                           kerja.sample(min(args.sampel, len(kerja)), random_state=args.seed)]).drop_duplicates("img_name")
        lap.peringatan(f"Mode --sampel: hanya {len(kerja)} sel. Angka kalibrasi TIDAK sah untuk dilaporkan.")

    # ---------------- pass 1 ----------------
    f_cache = os.path.join(args.keluar, "F1_fitur_sel.csv")
    fit = None
    if os.path.exists(f_cache) and not args.paksa and not args.sampel:
        tmp = pd.read_csv(f_cache)
        perlu = set(kerja.img_name)
        kurang = perlu - set(tmp.get("img_name", pd.Series(dtype=str)))
        if kurang or "f_badan" not in tmp.columns:
            lap.peringatan(f"Cache {f_cache} tidak cocok dengan jalanan ini ({len(kurang)} sel hilang). "
                           "Cache diabaikan dan fitur dihitung ulang.")
        else:
            fit = tmp
            lap.tulis(f"- fitur sel dimuat dari cache {f_cache} ({len(fit)} baris). "
                      "Hapus berkas itu atau pakai `--paksa` bila mask/parameter berubah.")
    if fit is None:
        tugas = [(r.img_name, idx_mask[r.kunci], prm, None) for r in kerja.itertuples()]
        fit = pd.DataFrame(jalankan_pool(kerja_sel, tugas, args.pekerja, "pass 1 (fitur badan + inti asli)", lap))
        if not args.sampel:
            fit.to_csv(f_cache, index=False)
    gal = fit[fit.galat.fillna("").astype(str).str.len() > 0]
    if len(gal):
        lap.peringatan(f"{len(gal)} sel gagal diproses di pass 1. Contoh: {gal.galat.iloc[0]}")
    fit = fit[fit.galat.fillna("").astype(str).str.len() == 0]
    m = df.merge(fit.drop(columns=["galat"]), on="img_name", how="inner")

    # ---------------- uji reproduksi ----------------
    lap.tulis()
    lap.tulis("### 2.1 Uji reproduksi implementasi terhadap bridge_ratio_2d.csv")
    lap.tulis()
    sama_br = np.isclose(m.ori_br.astype(float), m.bridge_ratio.astype(float), atol=1e-6, equal_nan=True)
    sama_st = m.ori_status.eq(m.status_topologi)
    sama_nl = m.ori_n_lobus_kal.eq(m.n_lobus_kal) if "n_lobus_kal" in m else pd.Series(True, index=m.index)
    sama_luas = m.ori_luas_nukleus.eq(m.luas_nukleus) if "luas_nukleus" in m else pd.Series(True, index=m.index)
    rep = pd.DataFrame([dict(ukuran=u, cocok=int(s.sum()), n=len(s), frak=float(s.mean()))
                        for u, s in (("bridge_ratio (|Δ|<1e-6)", pd.Series(sama_br)), ("status_topologi", sama_st),
                                     ("n_lobus_kal", sama_nl), ("luas_nukleus", sama_luas))])
    lap.tabel(rep, 5)
    frak_rep = float(np.mean(sama_br))
    if frak_rep < 0.99:
        beda = m.loc[~sama_br, ["img_name", "label", "bridge_ratio", "ori_br", "status_topologi", "ori_status",
                                "luas_nukleus", "ori_luas_nukleus"]].head(12)
        lap.tabel(beda)
        lap.peringatan(f"Implementasi ulang hanya mereproduksi {frak_rep:.2%} bridge ratio Fase 2D. "
                       "Skenario S3b (seluruhnya implementasi ulang) adalah pembanding yang konsisten secara internal; "
                       "S3a (campuran CSV + koreksi) hanya sah bila reproduksi >= 99%.")
    else:
        lap.tulis(f"Reproduksi {frak_rep:.2%} → implementasi ulang setara dengan pipeline Fase 2D.")
    lap.tulis()

    # ---------------- kalibrasi ----------------
    lap.tulis("### 2.2 Kalibrasi ambang f_badan (independen dari A/B neutrofil)")
    lap.tulis()
    kal = m[m.label.isin(KELAS_KALIBRASI)]
    neg = kal[kal.n_komp_inti == 1]
    pos_nyata = kal[(kal.n_komp_inti >= 2) & (kal.frak_komp2 >= 0.25)]
    lap.tulis(f"- negatif nyata (limfosit+monosit, 1 komponen inti): **{len(neg)}**")
    lap.tulis(f"- positif nyata (limfosit+monosit, ≥2 komponen inti, komponen kedua ≥25%): **{len(pos_nyata)}**")
    if len(neg) < 50:
        lap.peringatan(f"Hanya {len(neg)} sel negatif untuk kalibrasi. Ambang Neyman–Pearson pada FPR "
                       f"{args.fpr:.1%} ditentukan oleh kurang dari satu sel; M9 dihentikan.")
        return

    nk = args.n_sintetis
    pas = rng.choice(len(neg), size=(nk, 2), replace=True)
    pas = pas[pas[:, 0] != pas[:, 1]]
    fr = rng.uniform(0.70, 0.98, len(pas))
    sd = rng.uniform(0, 2 * math.pi, len(pas))
    nama_neg = neg.kunci.to_numpy()
    tugas_k = [(i, idx_mask[nama_neg[a]], idx_mask[nama_neg[b]], float(fr[i]), float(sd[i]), prm, None)
               for i, (a, b) in enumerate(pas)]
    kom = pd.DataFrame(jalankan_pool(kerja_komposit, tugas_k, args.pekerja, "komposit sintetis", lap))
    if len(kom) == 0 or "galat" not in kom.columns:
        lap.peringatan("Tidak satu pun komposit sintetis berhasil dibuat. M9 dihentikan.")
        return
    n_galat_kom = int((kom.galat.fillna("").astype(str).str.len() > 0).sum())
    if n_galat_kom:
        lap.peringatan(f"{n_galat_kom} komposit gagal dibuat; dikeluarkan dari kalibrasi.")
    kom = kom[kom.galat.fillna("").astype(str).str.len() == 0]
    kom_ok = kom[kom.nyambung]
    lap.tulis(f"- komposit menyambung dengan sel pusat: {len(kom_ok)}/{len(kom)} "
              f"(sisanya sudah dipisahkan M4, tidak relevan)")
    lap.tulis(f"- status inti komposit sebelum koreksi: "
              + ", ".join(f"{k} {v}" for k, v in kom_ok.st_gabung.value_counts().items()))

    fneg = neg.f_badan.to_numpy(float)
    t_np, fpr_capai = ambang_np(fneg, args.fpr)
    frak_neg_positif = float((fneg > 0).mean())
    t_j = youden(kom_ok.f_badan.to_numpy(float), fneg)[0] if len(kom_ok) else np.nan
    sapu = []
    neu_all = m[m.label == "Neutrophil"]
    for t in sorted(set([round(x, 2) for x in np.arange(0.10, 0.65, 0.05)] + [t_np])):
        sapu.append(dict(t=t, fpr_neg=float((fneg > t).mean()),
                         tpr_sintetis=float((kom_ok.f_badan > t).mean()) if len(kom_ok) else np.nan,
                         tpr_pos_nyata=float((pos_nyata.f_badan > t).mean()) if len(pos_nyata) else np.nan,
                         neutrofil_terdeteksi=int((neu_all.f_badan > t).sum()),
                         dari_terpisah0=int(((neu_all.f_badan > t) & neu_all.status_topologi.eq("terpisah_di_nol")).sum()),
                         terpilih=bool(t == t_np)))
    lap.tabel(pd.DataFrame(sapu))
    t = t_np
    lap.tulis(f"**Ambang terpilih (Neyman–Pearson, FPR ≤ {args.fpr:.1%} pada negatif nyata): t = {t:.4f}, "
              f"FPR yang benar-benar tercapai {fpr_capai:.3%}** ({int(round(fpr_capai * len(neg)))} dari {len(neg)} "
              f"negatif). Ambang diambil sebagai nilai terkecil yang memenuhi batas itu, bukan kuantil: kuantil "
              f"tidak menjamin FPR ≤ α ketika f menumpuk di satu nilai. Negatif dengan f > 0: "
              f"{frak_neg_positif:.1%}. Pembanding Youden pada sintetis-vs-negatif: {t_j:.4f} (TIDAK dipakai; "
              f"bergantung pada sebaran jarak tempel yang saya pilih sendiri).")
    lap.tulis()
    if not len(kom_ok):
        lap.peringatan("Tidak ada komposit yang menyambung; sensitivitas detektor tidak terukur pada lapis sintetis.")
    if t <= 0.0:
        lap.peringatan("Ambang jatuh tepat di t = 0, artinya aturan menjadi 'ada penggabungan badan sama sekali'. "
                       "Itu sah menurut kriteria FPR, tetapi lapis negatif tidak lagi membatasi apa pun. "
                       "Periksa `neutrofil_terdeteksi` pada tabel sapuan dan montase F3 sebelum memakai S3/S4.")
    elif frak_neg_positif < 0.05:
        lap.peringatan(f"Hanya {frak_neg_positif:.1%} negatif nyata yang punya f > 0, sehingga batas FPR 1% "
                       "ditentukan oleh sedikit sekali sel. Ambang ini bergantung pada ekor yang tipis; "
                       "laporkan juga hasil pada ambang tetangga dari tabel sapuan.")
    if len(kom_ok):
        kom_ok = kom_ok.assign(bin_fr=pd.cut(kom_ok.fr, [0.70, 0.80, 0.90, 0.98]))
        tb = kom_ok.groupby("bin_fr", observed=True).agg(n=("f_badan", "size"),
                                                        tpr=("f_badan", lambda x: float((x > t).mean())),
                                                        f_median=("f_badan", "median")).reset_index()
        tb["bin_fr"] = tb.bin_fr.astype(str)
        lap.tulis("Sensitivitas pada komposit menurut jarak tempel (fr kecil = tumpang tindih berat):")
        lap.tabel(tb)

    # validasi partisi pada komposit
    sub = kom_ok.head(min(400, len(kom_ok)))
    tugas_v = [(int(r.i), r.p1, r.p2, float(r.fr), float(r.sudut), prm, t) for r in sub.itertuples()]
    val = pd.DataFrame(jalankan_pool(kerja_komposit, tugas_v, args.pekerja, "validasi partisi komposit", lap))
    if len(val) and "galat" in val.columns:
        val = val[val.galat.fillna("").astype(str).str.len() == 0]
    vd = val[val.terdeteksi] if len(val) and "terdeteksi" in val.columns else val
    if len(vd):
        pulih_st = float(vd.st_kor.eq(vd.st_ref).mean())
        pulih_br = float(np.isclose(vd.br_kor.astype(float), vd.br_ref.astype(float), atol=0.01).mean())
        lap.tulis(f"- Komposit terdeteksi {len(vd)}/{len(val)}. Setelah partisi: status inti sama dengan sel tunggal "
                  f"aslinya **{pulih_st:.2%}**, bridge ratio dalam ±0,01 **{pulih_br:.2%}**, "
                  f"IoU inti median **{vd.iou_inti.median():.4f}** (q10 {vd.iou_inti.quantile(0.1):.4f}).")
        if pulih_st < 0.9:
            lap.peringatan(f"Partisi hanya memulihkan status {pulih_st:.1%} komposit. Koreksi br pada sel "
                           "terdeteksi belum bisa dipercaya penuh.")
    lap.tulis()

    # ---------------- penerapan ----------------
    lap.tulis("### 2.3 Penerapan pada seluruh sel")
    lap.tulis()
    m["terdeteksi"] = m.f_badan > t
    per_kelas = m.groupby("label").agg(n=("img_name", "size"), terdeteksi=("terdeteksi", "sum"),
                                       frak=("terdeteksi", "mean")).reset_index()
    lap.tabel(per_kelas)

    tugas2 = [(r.img_name, idx_mask[r.kunci], prm, t) for r in m[m.terdeteksi].itertuples()]
    p2 = pd.DataFrame(jalankan_pool(kerja_sel, tugas2, args.pekerja, "pass 2 (koreksi sel terdeteksi)", lap))
    if len(p2):
        p2 = p2[p2.galat.fillna("").astype(str).str.len() == 0].drop(columns=["galat"])
        m = m.merge(p2, on="img_name", how="left")
    else:
        for c in ("kor_br", "kor_status", "kor_luas_nukleus", "kor_r_pisah", "kor_n_lobus_kal", "kor_luas_badan"):
            m[c] = np.nan

    neu = m[m.label == "Neutrophil"].copy()
    neu["kel"] = neu.kelompok.astype(str).str[0]
    med_luas = neu.luas_nukleus.median()
    tk = neu.groupby("kelompok").agg(
        n=("img_name", "size"),
        terpisah0=("status_topologi", lambda s: int(s.eq("terpisah_di_nol").sum())),
        terdeteksi=("terdeteksi", "sum")).reset_index()
    tk["terdeteksi_dan_terpisah0"] = [int((neu.kelompok.eq(g) & neu.terdeteksi & neu.status_topologi.eq("terpisah_di_nol")).sum())
                                      for g in tk.kelompok]
    tk["terdeteksi_bukan_terpisah0"] = tk.terdeteksi - tk.terdeteksi_dan_terpisah0
    lap.tabel(tk)
    n0 = int(neu.status_topologi.eq("terpisah_di_nol").sum())
    n0d = int((neu.status_topologi.eq("terpisah_di_nol") & neu.terdeteksi).sum())
    lap.tulis(f"**Dari {n0} neutrofil `terpisah_di_nol`, {n0d} ({n0d / max(n0, 1):.1%}) terdeteksi sebagai sel ganda.**")
    lap.tulis("Angka ini **batas bawah**, bukan hitungan sebenarnya. Sel negatif yang dipakai untuk mengkalibrasi "
              "ambang bisa saja sendiri memuat sel ganda yang intinya tertutup atau terbuang filter 2%; kontaminasi "
              "semacam itu menaikkan ambang, sehingga detektor melewatkan sel, bukan mengarang sel. Montase F2 "
              "(`terpisah_di_nol` yang TIDAK terdeteksi) adalah tempat memeriksa sel yang terlewat.")
    lap.tulis(f"Median luas inti neutrofil {med_luas:.0f} px; median luas inti sel terdeteksi "
              f"{neu.loc[neu.terdeteksi, 'luas_nukleus'].median():.0f} px; setelah koreksi "
              f"{neu.loc[neu.terdeteksi, 'kor_luas_nukleus'].median():.0f} px.")
    lap.tulis()
    trans = neu[neu.terdeteksi].groupby(["status_topologi", "kor_status"]).size().reset_index(name="n")
    lap.tulis("Transisi status sel neutrofil terdeteksi (lama → setelah koreksi):")
    lap.tabel(trans)

    # validasi adjudikasi 2E
    adj = []
    for daftar, sumber in ((ADJ_A_GANDA, "A yakin-salah (2E)"), (ADJ_B_GANDA, "B yakin-benar (2E)")):
        for nm in daftar:
            r = m[m.kunci == nm]
            if len(r):
                r = r.iloc[0]
                adj.append(dict(sel=nm, sumber=sumber, f_badan=r.f_badan, terdeteksi=bool(r.terdeteksi),
                                luas_inti=r.luas_nukleus, br_lama=r.bridge_ratio,
                                br_kor=r.get("kor_br", np.nan), status_kor=r.get("kor_status", "")))
            else:
                adj.append(dict(sel=nm, sumber=sumber, f_badan=np.nan, terdeteksi=False, luas_inti=np.nan,
                                br_lama=np.nan, br_kor=np.nan, status_kor="TIDAK DITEMUKAN"))
    lap.tulis("Validasi terhadap 11 sel ganda dari adjudikasi montase Fase 2E (tidak dipakai untuk kalibrasi):")
    lap.tabel(pd.DataFrame(adj))

    # ---------------- skenario setelah koreksi ----------------
    lap.tulis("### 2.4 Skenario setelah koreksi sel ganda")
    lap.tulis()
    neu["br_S3a"] = np.where(neu.terdeteksi, neu.kor_br, neu.bridge_ratio)
    neu["st_S3a"] = np.where(neu.terdeteksi, neu.kor_status, neu.status_topologi)
    neu["br_S3b"] = np.where(neu.terdeteksi, neu.kor_br, neu.ori_br)
    neu["st_S3b"] = np.where(neu.terdeteksi, neu.kor_status, neu.ori_status)
    for nama, kb, ks, dd in (
            ("S3a koreksi sel ganda (CSV + koreksi)", "br_S3a", "st_S3a", neu),
            ("S3b koreksi sel ganda (seluruhnya implementasi ulang)", "br_S3b", "st_S3b", neu),
            ("S4 buang sel terdeteksi ganda", "bridge_ratio", "status_topologi", neu[~neu.terdeteksi])):
        ring, tkk, tdef = analisis(dd, kb, ks, nama, rng)
        skenario.append(ring)
        laporkan_skenario(lap, ring, tkk, tdef)

    lap.tulis("**Seberapa besar kesimpulan bergantung pada ambang t?** Tabel berikut mengulang S4 (buang sel "
              "terdeteksi, tanpa koreksi) pada seluruh ambang sapuan. Murah karena hanya membuang baris. "
              "Bila AUC dan posisi C nyaris rata di seluruh baris, pilihan ambang bukan titik rapuh.")
    lap.tulis()
    sens = []
    for tt in sorted(set([round(x, 2) for x in np.arange(0.10, 0.65, 0.05)] + [t])):
        dd = neu[~(neu.f_badan > tt)]
        r = ringkas_cepat(dd, "bridge_ratio")
        r.update(t=tt, n_dibuang=int(len(neu) - len(dd)), terpilih=bool(tt == t))
        sens.append(r)
    sens = pd.DataFrame(sens)[["t", "n_dibuang", "n_A", "n_B", "auc", "acc_13", "spes_band", "posisi_C", "terpilih"]]
    lap.tabel(sens)
    sens.to_csv(os.path.join(args.keluar, "F8_sensitivitas_ambang.csv"), index=False)

    neu.to_csv(os.path.join(args.keluar, "F2_neutrofil_sel_ganda.csv"), index=False)
    m.drop(columns=["kunci"]).to_csv(os.path.join(args.keluar, "F3_semua_sel_fitur_badan.csv"), index=False)
    pd.DataFrame(sapu).to_csv(os.path.join(args.keluar, "F4_sapuan_ambang.csv"), index=False)
    kom.to_csv(os.path.join(args.keluar, "F5_komposit.csv"), index=False)
    val.to_csv(os.path.join(args.keluar, "F6_validasi_partisi.csv"), index=False)

    # ---------------- sensus visual ----------------
    lap.tulis("### 2.5 Montase dan berkas sensus visual")
    lap.tulis()

    def teks(r):
        kb = getattr(r, "kor_br", np.nan)
        kb = float(kb) if pd.notna(kb) else float(r.bridge_ratio)
        return (f"{r.kunci} {str(r.kelompok)[:1]} f={r.f_badan:.2f}\n"
                f"br {r.bridge_ratio:.2f}→{kb:.2f} inti {int(r.luas_nukleus)}")

    def it(dd):
        return [dict(img_name=r.img_name, teks=teks(r)) for r in dd.itertuples()]

    g1 = neu[neu.terdeteksi].sort_values("f_badan", ascending=False)
    g2 = neu[neu.status_topologi.eq("terpisah_di_nol") & ~neu.terdeteksi].sort_values("luas_nukleus", ascending=False)
    g3 = neu[(neu.f_badan > t - 0.05) & (neu.f_badan <= t + 0.05)].sort_values("f_badan")
    g4 = pos_nyata.assign(kelompok=pos_nyata.label).sort_values("f_badan")
    for dd, nama, judul in ((g1, "montase_F1_neutrofil_terdeteksi.png", f"Neutrofil terdeteksi ganda (f > {t:.3f})"),
                            (g2, "montase_F2_terpisah0_tidak_terdeteksi.png", "Neutrofil terpisah_di_nol yang TIDAK terdeteksi"),
                            (g3, "montase_F3_dekat_ambang.png", f"Neutrofil dengan f di sekitar ambang {t:.3f} ± 0,05"),
                            (g4, "montase_F4_limfomono_dua_inti.png", "Positif nyata: limfosit/monosit dengan ≥2 inti besar")):
        pilih = dd.head(36)
        if len(pilih):
            dd2 = pilih if "kor_br" in pilih else pilih.assign(kor_br=np.nan)
            montase(it(dd2), os.path.join(args.keluar, nama), f"{judul} — {len(pilih)} dari {len(dd)}",
                    t, idx_mask, idx_citra)
            lap.tulis(f"- `{nama}`: {len(pilih)} dari {len(dd)} sel")
    sensus = pd.concat([g1.assign(sumber="F1_terdeteksi"), g2.assign(sumber="F2_terpisah0_lolos"),
                        g3.assign(sumber="F3_dekat_ambang")]).drop_duplicates("img_name")
    sensus = sensus[["img_name", "sumber", "kelompok", "split", "status_topologi", "bridge_ratio", "f_badan",
                     "terdeteksi", "kor_br", "kor_status", "luas_nukleus", "kor_luas_nukleus"]].copy()
    sensus["keputusan_manusia"] = ""
    sensus.to_csv(os.path.join(args.keluar, "F7_sensus_visual.csv"), index=False)
    lap.tulis(f"- `F7_sensus_visual.csv`: {len(sensus)} sel. Isi kolom `keputusan_manusia` dengan "
              "GANDA / TUNGGAL / RAGU berdasarkan citra RGB.")
    lap.tulis()


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tugas", choices=["sanitas", "m11", "m9", "semua"], default="semua")
    ap.add_argument("--dir-hasil", default="hasil_fase2d")
    ap.add_argument("--csv", default=None, help="bawaan: <dir-hasil>/bridge_ratio_2d.csv")
    ap.add_argument("--dir-mask", default="pbcseg_final_v1")
    ap.add_argument("--dir-citra", default=None)
    ap.add_argument("--keluar", default="hasil_fase2f")
    ap.add_argument("--pekerja", type=int, default=7)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--n-sintetis", type=int, default=1000)
    ap.add_argument("--sampel", type=int, default=0)
    ap.add_argument("--paksa", action="store_true", help="hitung ulang cache fitur sel")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tau-br", type=float, default=1.25)
    ap.add_argument("--rho-br", type=float, default=0.2)
    ap.add_argument("--tau-lobus", type=float, default=4.0)
    ap.add_argument("--rho-lobus", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.keluar, exist_ok=True)
    lap = Laporan(os.path.join(args.keluar, "LAPORAN_FASE2F.md"))
    prm = dict(tau_br=args.tau_br, rho_br=args.rho_br, tau_lobus=args.tau_lobus,
               rho_lobus=args.rho_lobus, prune=0.05)
    rng = np.random.default_rng(args.seed)
    lap.tulis("# LAPORAN FASE 2F — M11 (sensitivitas terpisah_di_nol) dan M9 (sel ganda)")
    lap.tulis()
    lap.tulis(f"Dijalankan {time.strftime('%Y-%m-%d %H:%M')} | parameter: τ_br {args.tau_br}, ρ_br {args.rho_br}, "
              f"τ_lobus {args.tau_lobus}, ρ_lobus {args.rho_lobus}, aturan lubang `tidak`, saddle `min` | "
              f"FPR target {args.fpr} | seed {args.seed}")
    lap.tulis()
    try:
        if not uji_sanitas(lap, prm):
            sys.exit(1)
        if args.tugas == "sanitas":
            return
        path_csv = args.csv or os.path.join(args.dir_hasil, "bridge_ratio_2d.csv")
        df = pd.read_csv(path_csv)
        neu = df[df.label == "Neutrophil"].copy()
        lap.tulis(f"Data: `{path_csv}` — {len(df)} baris, neutrofil {len(neu)}")
        lap.tulis()

        # ---------------- M11 ----------------
        lap.tulis("## 1. M11 — Sensitivitas terhadap `terpisah_di_nol`")
        lap.tulis()
        skenario = []
        ring, tk, tdef = analisis(neu, "bridge_ratio", "status_topologi", "S0 baseline Fase 2D", rng)
        skenario.append(ring)
        laporkan_skenario(lap, ring, tk, tdef)
        cek = []
        for k, (harap, tol) in HARAPAN_S0.items():
            cek.append(dict(ukuran=k, diperoleh=ring[k], harapan_2D=harap, cocok=bool(abs(ring[k] - harap) <= tol)))
        cek = pd.DataFrame(cek)
        lap.tulis("Uji reproduksi analisis terhadap angka Fase 2D:")
        lap.tabel(cek, 6)
        if not cek.cocok.all():
            lap.peringatan("S0 tidak mereproduksi angka Fase 2D pada: " + ", ".join(cek.loc[~cek.cocok, "ukuran"])
                           + ". Periksa konvensi (>= vs >) sebelum membaca selisih antar-skenario.")

        tanpa0 = neu[~neu.status_topologi.eq("terpisah_di_nol")]
        ring, tk, tdef = analisis(tanpa0, "bridge_ratio", "status_topologi", "S1 buang terpisah_di_nol", rng)
        skenario.append(ring)
        laporkan_skenario(lap, ring, tk, tdef)

        pes = neu.copy()
        pes["br_pes"] = np.where(pes.kelompok.astype(str).str.startswith("B") &
                                 pes.status_topologi.eq("terpisah_di_nol"), 1.0, pes.bridge_ratio)
        ring, tk, tdef = analisis(pes, "br_pes", "status_topologi", "S2 pesimis (B terpisah_di_nol dianggap salah)", rng)
        skenario.append(ring)
        laporkan_skenario(lap, ring, tk, tdef)

        # ---------------- M9 ----------------
        if args.tugas in ("m9", "semua"):
            jalankan_m9(args, lap, df, prm, rng, skenario)

        # ---------------- ringkasan ----------------
        lap.tulis("## 3. Ringkasan lintas skenario")
        lap.tulis()
        rs = pd.DataFrame(skenario)
        kol = ["skenario", "n_A", "n_B", "auc", "youden", "ik_lo", "ik_hi", "sepertiga_di_ik", "acc_13",
               "galat_A", "galat_B", "spes_band", "auc_test", "acc_test_13", "posisi_C", "zona_C_per_A",
               "def90_acc", "def90_peng_C"]
        lap.tabel(rs[kol])
        rs.to_csv(os.path.join(args.keluar, "F0_ringkasan_skenario.csv"), index=False)
        b0 = rs.iloc[0]
        for _, r in rs.iloc[1:].iterrows():
            lap.tulis(f"- **{r.skenario}**: ΔAUC {r.auc - b0.auc:+.4f}, Δakurasi@1/3 {r.acc_13 - b0.acc_13:+.4f}, "
                      f"Δspesifisitas {r.spes_band - b0.spes_band:+.4f}, Δposisi C {r.posisi_C - b0.posisi_C:+.4f}")
        lap.tulis()
        lap.tulis("Cara membaca: S1 dan S2 mengapit dampak maksimum `terpisah_di_nol` tanpa perlu tahu mana yang "
                  "benar-benar sel ganda. Bila selisih S0–S2 kecil, M9 tidak mengancam klaim apa pun. S3/S4 adalah "
                  "koreksi yang sebenarnya, dan baru boleh dilaporkan setelah sensus visual F7 diisi.")
    finally:
        lap.simpan()
        print(f"\nLaporan: {lap.path}")


if __name__ == "__main__":
    main()
