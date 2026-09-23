#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fase2e_audit_derau_label.py
============================================================================
FASE 2E -- Audit lantai derau: apakah sisa galat itu salah LABEL atau salah
           MASK? Plus adjudikasi visual untuk pembimbing.

LATAR BELAKANG
--------------
Fase 2D menyelesaikan soal lubang: 100% piksel region tertutup berlabel
jaringan (frak_jaringan minimum 0.9677 di seluruh 2.366 region), jadi tidak
ada artefak segmentasi untuk diperbaiki dan aturan lubang runtuh jadi
'tidak'. Hipotesis lubang terkonfirmasi di tingkat sel: 67.6% galat ekor B
punya lubang terisi, versus 14.5% sel B benar (OR 12.3).

Yang tersisa TIDAK bisa dijelaskan lubang. Kurva deferral mendatar:
membuang separuh sel paling ragu hanya menaikkan akurasi 0.9293 -> 0.9754,
menyisakan ~35 sel YAKIN TAPI SALAH. Itu tanda lantai, bukan kesalahan
metode di dekat ambang.

Untuk sel semacam itu ada dua kemungkinan yang harus dipisahkan:
  (1) DERAU LABEL -- dua anotasi (prefix PBC + nucleus_shape WBCAtt) sama-
      sama bilang segmented, tapi intinya memang tidak tersegmentasi.
      Konsekuensi: ada plafon akurasi yang tak bisa dilewati metode apa pun.
  (2) MASK SALAH -- wbcsegmentor melebur lobus jadi satu.
      Konsekuensi: ini batas pipeline, bukan batas data.

Keduanya hanya bisa dipisahkan dengan MELIHAT citra RGB asli berdampingan
dengan mask-nya. Skrip ini menyiapkan bahan itu.

ATURAN MAIN
-----------
Skrip ini TIDAK menyetel apa pun dan TIDAK memilih parameter. Ia hanya
mengukur, mengurutkan, dan menggambar. Semua parameter dibaca dari keluaran
Fase 2D. Tidak ada keputusan yang diambil dari AUC.

YANG DIKERJAKAN
---------------
  T1  lantai derau: akurasi per desil keyakinan, sel yakin-tapi-salah,
      taksiran plafon yang tak bisa dilewati metode apa pun
  T2  profil sel yakin-tapi-salah: geometri, atribut lain, split
  T3  montase adjudikasi RGB + mask berdampingan, dengan saddle ditandai
      -> lembar kerja untuk pembimbing memutuskan label vs mask
  T4  perbandingan antar-aturan lubang di tingkat sel: sel mana yang
      berubah vonis, khususnya 14 sel A yang rusak karena lubang
      tidak diisi
  T5  ringkasan angka siap-pakai untuk penulisan ulang Bagian 14

PEMAKAIAN
---------
  python fase2e_audit_derau_label.py --tugas sanitas          # wajib, cepat

  # setelah menjalankan ulang Fase 2D dengan --aturan-lubang tidak
  python fase2e_audit_derau_label.py \
      --dir-hasil hasil_fase2d --dir-mask pbcseg_final_v1 \
      --dir-citra PBC_dataset_normal_DIB

  # tanpa citra RGB (montase jadi mask saja, tetap berguna tapi lebih lemah)
  python fase2e_audit_derau_label.py --dir-hasil hasil_fase2d

TUGAS: sanitas, lantai, profil, montase, aturan, ringkas, semua
============================================================================
"""

import argparse
import glob
import gzip
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
SEPERTIGA = 1.0 / 3.0
JARAK_JAUH = 999.0
LATAR, SITOPLASMA, NUKLEUS, TROMBOSIT, SEL_LAIN, VAKUOLA = 0, 1, 2, 3, 4, 5

URUT_KELOMPOK = ["A_sepakat_band", "B_sepakat_segmented", "C_konflik_SNE_band",
                 "D_konflik_BNE_segmented", "E_lainnya", "F_tak_bersubtipe"]

WARNA_MASK = {LATAR: (16, 16, 22), SITOPLASMA: (118, 140, 176),
              NUKLEUS: (238, 238, 248), TROMBOSIT: (92, 70, 70),
              SEL_LAIN: (56, 46, 56), VAKUOLA: (176, 72, 72)}

# Pembanding Fase 2D (dicetak berdampingan, tidak pernah dipakai menyetel)
REF_2D = dict(akurasi13=0.9292769656262347, ekor_B=124,
              galat_A=55, auc=0.9628528411786411, posisi_C=0.3262)


# ===========================================================================
# BAGIAN 1 -- PEMUATAN
# ===========================================================================

def baca_mask(path):
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def indeks_berkas(dirnya, pola):
    """Indeks {stem_bersih: path} untuk mask (*.png) atau citra (*.jpg/png)."""
    peta, n = {}, 0
    if dirnya is None:
        return peta, 0
    p = Path(dirnya)
    if not p.exists():
        return peta, 0
    for ext in pola:
        for f in p.rglob("*" + ext):
            s = f.stem
            if s.endswith("_mask"):
                s = s[:-5]
            if s not in peta:
                peta[s] = f
                n += 1
            if s.endswith("_ccrop"):
                peta.setdefault(s[:-6], f)
    return peta, n


def cari(peta, img_name):
    s = Path(str(img_name)).stem
    for k in (s, s[:-6] if s.endswith("_ccrop") else None, s + "_ccrop"):
        if k and k in peta:
            return peta[k]
    return None


def muat_cache(dir_hasil):
    """Muat semua cache Fase 2D yang ada, kunci = nama aturan lubang."""
    out = {}
    for f in sorted(glob.glob(str(Path(dir_hasil) / "cache_p2d_lub-*.pkl.gz"))):
        m = re.search(r"cache_p2d_lub-([a-z]+)_", Path(f).name)
        if not m:
            continue
        try:
            with gzip.open(f, "rb") as fh:
                out[m.group(1)] = pickle.load(fh)
        except Exception as e:                              # pragma: no cover
            print("  PERINGATAN: gagal memuat %s (%s)" % (Path(f).name, e))
    return out


class Turunan:
    """Salinan minimal dari Fase 2D: hanya yang dibutuhkan untuk menurunkan
    ulang status pada cache aturan lain. Tidak menghitung pohon merge."""

    def __init__(self, cache, nama_urut):
        self.nama = list(nama_urut)
        N = len(self.nama)
        self.N = N
        self.rl = np.zeros(N)
        sad, per, pma = [], [], []
        off = [0]
        bts, boff = [], [0]
        for i, nm in enumerate(self.nama):
            e = cache[nm]
            self.rl[i] = e["r_lobus"]
            K = e["kejadian"]
            if K.shape[0]:
                sad.append(K[:, 0]); per.append(K[:, 1]); pma.append(K[:, 2])
            off.append(off[-1] + int(K.shape[0]))
            Bt = e["bertahan"]
            bts.append(Bt)
            boff.append(boff[-1] + int(Bt.size))
        cat = lambda L: (np.concatenate(L).astype(np.float64) if L else np.zeros(0))
        self.sad, self.per, self.pma = cat(sad), cat(per), cat(pma)
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
        if self.sad.size:
            mev = (self.per >= tau) & (self.pma >= rho * self.rl_ev)
            nkej[self.idx_ada] = np.add.reduceat(mev.astype(np.int64), self.start)
            smin[self.idx_ada] = np.minimum.reduceat(
                np.where(mev, self.sad, np.inf), self.start)
            smax[self.idx_ada] = np.maximum.reduceat(
                np.where(mev, self.sad, -np.inf), self.start)
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
        return dict(img_name=np.array(self.nama, dtype=object),
                    r_lobus=self.rl.copy(), r_pisah=rp, bridge_ratio=br,
                    n_lobus=n_lobus, status_topologi=st)


# ===========================================================================
# BAGIAN 2 -- T1 LANTAI DERAU
# ===========================================================================

def sisi_benar(kelompok, br, ambang=SEPERTIGA):
    """True bila vonis geometri sepakat dengan kelompok konsensus."""
    return np.where(kelompok == "A_sepakat_band", br >= ambang, br < ambang)


def tugas_lantai(df, out, ambang=SEPERTIGA, n_desil=10, diam=False):
    """diam=True dipakai oleh uji sanitas: menghitung tanpa mencetak tabel."""
    P = (lambda *a: None) if diam else print
    print_ = P
    P("\n=== T1  LANTAI DERAU LABEL ===")
    ab = df[df["kelompok"].isin(["A_sepakat_band", "B_sepakat_segmented"])].copy()
    ab["keyakinan"] = (ab["bridge_ratio"] - ambang).abs()
    ab["benar"] = sisi_benar(ab["kelompok"].values, ab["bridge_ratio"].values, ambang)
    n = len(ab)
    P("  sel A+B: %d | akurasi keseluruhan %.4f (%d galat)"
          % (n, ab["benar"].mean(), int((~ab["benar"]).sum())))
    P("  [Fase 2D: akurasi %.4f]" % REF_2D["akurasi13"])

    ab = ab.sort_values("keyakinan", ascending=False).reset_index(drop=True)
    ab["desil"] = (np.arange(n) * n_desil // n) + 1     # 1 = paling yakin
    t = ab.groupby("desil").agg(
        n=("benar", "size"), akurasi=("benar", "mean"),
        galat=("benar", lambda s: int((~s.astype(bool)).sum())),
        keyakinan_min=("keyakinan", "min"), keyakinan_maks=("keyakinan", "max"),
        br_med=("bridge_ratio", "median")).reset_index()
    P("\n  akurasi per desil keyakinan (desil 1 = paling yakin):")
    P(t.to_string(index=False, float_format=lambda v: "%.4f" % v))
    t.to_csv(out / "E1_desil_keyakinan.csv", index=False)

    d1 = t.iloc[0]
    P("\n  DESIL PALING YAKIN: %d sel, akurasi %.4f, %d galat."
          % (d1["n"], d1["akurasi"], d1["galat"]))
    if d1["galat"] == 0:
        P("     Tidak ada sel yakin-tapi-salah. Tidak ada bukti lantai derau;")
        P("     sisa galat konsisten dengan kesalahan metode di dekat ambang.")
    else:
        P("     Ada %d sel yang SANGAT yakin dan tetap salah. Itu tidak bisa\n     dijelaskan sebagai kesalahan metode di dekat ambang." % int(d1["galat"]))

    # --- taksiran plafon: galat yang bertahan pada cakupan menurun
    P("\n  galat yang bertahan saat sel paling ragu dibuang:")
    baris = []
    for cak in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        k = max(int(round(cak * n)), 1)
        s = ab.iloc[:k]
        r = dict(cakupan=cak, n_simpan=k, akurasi=float(s["benar"].mean()),
                 galat=int((~s["benar"]).sum()),
                 galat_per_1000=1000.0 * (~s["benar"]).mean())
        baris.append(r)
        P("    cakupan %4.0f%%: %5d sel, akurasi %.4f, %4d galat (%.1f per 1000)"
              % (100 * cak, k, r["akurasi"], r["galat"], r["galat_per_1000"]))
    dfc = pd.DataFrame(baris)
    dfc.to_csv(out / "E1_kurva_lantai.csv", index=False)
    P("\n  Bila angka 'galat per 1000' berhenti turun, itu lantai. Galat yang")
    P("  bertahan di cakupan rendah adalah sel yakin-tapi-salah, dan hanya")
    P("  pemeriksaan visual yang bisa memutuskan label salah atau mask salah.")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(100 * dfc["cakupan"], dfc["galat_per_1000"], "o-", color="#2b5d8a")
    ax.set_xlabel("cakupan (% sel paling yakin yang dipertahankan)")
    ax.set_ylabel("galat per 1000 sel")
    ax.set_title("Lantai derau: laju galat di antara sel paling yakin", fontsize=10)
    ax.invert_xaxis(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "F2E_lantai.png", dpi=140)
    plt.close(fig)
    return ab, t, dfc


# ===========================================================================
# BAGIAN 3 -- T2 PROFIL SEL YAKIN-TAPI-SALAH
# ===========================================================================

ATRIBUT_KANDIDAT = ["cell_size", "cell_shape", "nuclear_cytoplasmic_ratio",
                    "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture",
                    "cytoplasm_colour", "granule_type", "granule_colour",
                    "granularity"]


def tugas_profil(ab, out, batas_yakin=0.5, ambang=SEPERTIGA):
    """
    Sel yakin-tapi-salah = vonis geometri salah DAN bridge ratio jauh dari
    ambang (|br - 1/3| >= batas_yakin * jarak maksimum yang mungkin ke sisi itu).
    Dinyatakan lebih sederhana: br >= 0.5 untuk sel B, br <= 0.2 untuk sel A.
    """
    print("\n=== T2  PROFIL SEL YAKIN-TAPI-SALAH ===")
    salah = ab[~ab["benar"]].copy()
    B = salah[salah["kelompok"] == "B_sepakat_segmented"]
    A = salah[salah["kelompok"] == "A_sepakat_band"]
    yakinB = B[B["bridge_ratio"] >= 0.50]
    yakinA = A[A["bridge_ratio"] <= 0.20]
    print("  galat sisi B (disebut band): %d, di antaranya %d dengan br >= 0.50"
          % (len(B), len(yakinB)))
    print("  galat sisi A (disebut segmented): %d, di antaranya %d dengan br <= 0.20"
          % (len(A), len(yakinA)))

    for judul, s in (("B yakin-salah (br >= 0.50)", yakinB),
                     ("A yakin-salah (br <= 0.20)", yakinA)):
        if not len(s):
            print("\n  %s: tidak ada" % judul)
            continue
        print("\n  %s -- n=%d" % (judul, len(s)))
        print("    bridge ratio: med %.3f (q25 %.3f, q75 %.3f, maks %.3f)"
              % (s["bridge_ratio"].median(), s["bridge_ratio"].quantile(.25),
                 s["bridge_ratio"].quantile(.75), s["bridge_ratio"].max()))
        for k in ("r_pisah", "r_lobus", "luas_nukleus", "n_lobus"):
            if k in s:
                print("    %-13s med %8.2f (q25 %8.2f, q75 %8.2f)"
                      % (k, s[k].median(), s[k].quantile(.25), s[k].quantile(.75)))
        if "status_topologi" in s:
            vc = s["status_topologi"].value_counts()
            print("    status: " + ", ".join("%s %d" % (k, v) for k, v in vc.items()))
        if "nucleus_shape" in s:
            vc = s["nucleus_shape"].value_counts()
            print("    nucleus_shape: " + ", ".join("%s %d" % (k, v) for k, v in vc.items()))
        if "split" in s:
            vc = s["split"].value_counts()
            print("    split: " + ", ".join("%s %d" % (k, v) for k, v in vc.items()))

    # --- apakah sel yakin-salah beda secara atribut lain dari sel yang benar?
    print("\n  Atribut lain: apakah sel yakin-salah menyimpang dari sel sekelompok")
    print("  yang benar? Penyimpangan mendukung DERAU LABEL; kemiripan mendukung")
    print("  MASK SALAH (sel normal, mask-nya yang gagal).")
    baris = []
    for nama_grup, kel, s in (("B yakin-salah", "B_sepakat_segmented", yakinB),
                              ("A yakin-salah", "A_sepakat_band", yakinA)):
        if not len(s):
            continue
        benar_ref = ab[(ab["kelompok"] == kel) & ab["benar"]]
        for kol in ATRIBUT_KANDIDAT:
            if kol not in ab.columns:
                continue
            if ab[kol].nunique(dropna=True) < 2:
                continue                      # atribut degenerat, tidak informatif
            modus = s[kol].mode()
            modus = modus.iloc[0] if len(modus) else None
            p_s = float((s[kol] == modus).mean()) if modus is not None else np.nan
            p_b = float((benar_ref[kol] == modus).mean()) if modus is not None else np.nan
            baris.append(dict(grup=nama_grup, atribut=kol, nilai_modus=modus,
                              frak_yakin_salah=p_s, frak_benar=p_b,
                              selisih=p_s - p_b))
    if baris:
        dfa = pd.DataFrame(baris).sort_values("selisih", key=abs, ascending=False)
        print(dfa.head(12).to_string(index=False, float_format=lambda v: "%.4f" % v))
        dfa.to_csv(out / "E2_atribut_yakin_salah.csv", index=False)
    else:
        print("    (tidak ada atribut non-degenerat untuk dibandingkan)")

    kolom = [c for c in ["img_name", "kelompok", "nucleus_shape", "split",
                         "bridge_ratio", "r_pisah", "r_lobus", "n_lobus",
                         "status_topologi", "luas_nukleus", "keyakinan"]
             if c in salah.columns]
    pd.concat([yakinB, yakinA])[kolom].to_csv(out / "E2_yakin_salah.csv", index=False)
    salah[kolom].to_csv(out / "E2_semua_galat.csv", index=False)
    return yakinB, yakinA


# ===========================================================================
# BAGIAN 4 -- T3 MONTASE ADJUDIKASI
# ===========================================================================

def _mask_rgb(sub):
    rgb = np.zeros(sub.shape + (3,), dtype=np.uint8)
    for v, c in WARNA_MASK.items():
        rgb[sub == v] = c
    return rgb


def _potong(m, bbox, margin=14):
    y0, x0, y1, x1 = bbox
    h, w = m.shape[:2]
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(h, y1 + margin); x1 = min(w, x1 + margin)
    return (y0, x0, y1, x1)


def montase_adjudikasi(sub, cache, peta_mask, peta_citra, out, judul, berkas,
                       tau, rho, n=18):
    """Dua panel per sel: citra RGB asli (bila ada) dan mask berwarna, dengan
    saddle terpilih ditandai. Inilah lembar kerja adjudikasi."""
    if sub is None or not len(sub) or "img_name" not in sub.columns:
        print("  (montase %s dilewati: tidak ada sel)" % berkas)
        return 0
    sub = sub.head(n)
    baris = int(math.ceil(len(sub) / 3.0))
    fig, axes = plt.subplots(baris, 6, figsize=(17, 3.25 * baris))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    dipakai = 0
    for i, (_, r) in enumerate(sub.iterrows()):
        nm = r["img_name"]
        e = cache.get(nm)
        pm = cari(peta_mask, nm)
        if e is None or pm is None:
            continue
        m = baca_mask(pm)
        y0, x0, y1, x1 = _potong(m, e["bbox"])
        ax_c = axes[i // 3, (i % 3) * 2]
        ax_m = axes[i // 3, (i % 3) * 2 + 1]

        pc = cari(peta_citra, nm) if peta_citra else None
        if pc is not None:
            try:
                cit = np.asarray(Image.open(pc).convert("RGB"))
                if cit.shape[:2] == m.shape[:2]:
                    ax_c.imshow(cit[y0:y1, x0:x1])
                else:
                    ax_c.imshow(cit)
                ax_c.set_title("RGB | %s" % nm, fontsize=6.5)
            except Exception:
                ax_c.text(.5, .5, "citra gagal dibaca", ha="center", fontsize=7)
        else:
            ax_c.text(.5, .5, "citra RGB\ntidak ditemukan", ha="center",
                      va="center", fontsize=7, color="#888888")
            ax_c.set_title(str(nm), fontsize=6.5)
        ax_c.axis("off")

        ax_m.imshow(_mask_rgb(m[y0:y1, x0:x1]))
        K, P = e["kejadian"], e["pos"]
        if K.shape[0] and r.get("status_topologi") == "normal":
            sel = (K[:, 1] >= tau) & (K[:, 2] >= rho * e["r_lobus"])
            if sel.any():
                j = int(np.argmin(K[sel, 0]))
                yy, xx = P[sel][j]
                ax_m.plot(xx - x0, yy - y0, "+", color="#ff2d2d", ms=14, mew=2.4)
        ax_m.set_title("mask | br=%.2f rp=%s lob=%d"
                       % (r["bridge_ratio"],
                          ("%.1f" % r["r_pisah"]) if not pd.isna(r.get("r_pisah"))
                          else "nan", int(r.get("n_lobus", 0))), fontsize=7)
        ax_m.axis("off")
        dipakai += 1
    fig.suptitle(judul + "   (kiri: citra asli, kanan: mask; + = saddle terpilih)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / berkas, dpi=125)
    plt.close(fig)
    print("  montase disimpan: %s (%d sel)" % (berkas, dipakai))
    return dipakai


def tugas_montase(ab, yakinB, yakinA, cache, peta_mask, peta_citra, out,
                  tau, rho, n_panel=18):
    print("\n=== T3  MONTASE ADJUDIKASI ===")
    if not peta_citra:
        print("  PERINGATAN: folder citra RGB tidak diberikan atau kosong.")
        print("  Montase tetap dibuat dari mask saja, tapi TIDAK bisa memisahkan")
        print("  'label salah' dari 'mask salah'. Pakai --dir-citra bila ada.")
    urut = lambda d, naik: (d.sort_values("bridge_ratio", ascending=naik)
                            if len(d) and "bridge_ratio" in d.columns else d)
    montase_adjudikasi(urut(yakinB, False),
                       cache, peta_mask, peta_citra, out,
                       "SEL B YAKIN-SALAH: dua anotasi bilang segmented, geometri bilang band",
                       "montase_E_B_yakin_salah.png", tau, rho, n_panel)
    montase_adjudikasi(urut(yakinA, True),
                       cache, peta_mask, peta_citra, out,
                       "SEL A YAKIN-SALAH: dua anotasi bilang band, geometri bilang segmented",
                       "montase_E_A_yakin_salah.png", tau, rho, n_panel)
    # pembanding: sel yang paling yakin DAN benar, supaya pembimbing punya
    # patokan seperti apa rupa sel yang metodenya bekerja
    for kel, berkas, judul in (
            ("B_sepakat_segmented", "montase_E_B_yakin_benar.png",
             "PEMBANDING: sel B paling yakin dan BENAR"),
            ("A_sepakat_band", "montase_E_A_yakin_benar.png",
             "PEMBANDING: sel A paling yakin dan BENAR")):
        s = ab[(ab["kelompok"] == kel) & ab["benar"]].nlargest(n_panel, "keyakinan")
        montase_adjudikasi(s, cache, peta_mask, peta_citra, out, judul,
                           berkas, tau, rho, n_panel)


# ===========================================================================
# BAGIAN 5 -- T4 PERBANDINGAN ANTAR-ATURAN LUBANG
# ===========================================================================

def tugas_aturan(df, caches, out, tau, rho, saddle="min", acuan="tidak",
                 peta_mask=None, peta_citra=None, n_panel=18):
    print("\n=== T4  PERBANDINGAN ANTAR-ATURAN LUBANG DI TINGKAT SEL ===")
    tersedia = [k for k in ("semua", "tidak", "selektif") if k in caches]
    if len(tersedia) < 2:
        print("  hanya %d cache tersedia (%s); perbandingan dilewati."
              % (len(tersedia), ", ".join(tersedia) or "tidak ada"))
        return None
    print("  cache tersedia: %s | acuan: '%s'" % (", ".join(tersedia), acuan))
    if acuan not in tersedia:
        acuan = tersedia[0]

    kel = dict(zip(df["img_name"], df["kelompok"]))
    hasil = {}
    for at in tersedia:
        c = caches[at]
        nm = [n for n in df["img_name"] if n in c]
        T = Turunan(c, nm)
        h = T.hitung(tau, rho, saddle)
        hasil[at] = pd.DataFrame({
            "img_name": h["img_name"], "br_" + at: h["bridge_ratio"],
            "st_" + at: h["status_topologi"], "rp_" + at: h["r_pisah"],
            "lob_" + at: h["n_lobus"]})

    g = hasil[tersedia[0]]
    for at in tersedia[1:]:
        g = g.merge(hasil[at], on="img_name", how="inner")
    g["kelompok"] = g["img_name"].map(kel)
    g = g[g["kelompok"].isin(["A_sepakat_band", "B_sepakat_segmented"])].copy()
    for at in tersedia:
        g["benar_" + at] = sisi_benar(g["kelompok"].values, g["br_" + at].values)

    print("\n  galat per aturan (pada tau=%g rho=%g yang sama untuk semua):"
          % (tau, rho))
    ring = []
    for at in tersedia:
        A = g[g["kelompok"] == "A_sepakat_band"]
        B = g[g["kelompok"] == "B_sepakat_segmented"]
        r = dict(aturan=at, galat_A=int((~A["benar_" + at]).sum()),
                 galat_B=int((~B["benar_" + at]).sum()))
        r["total"] = r["galat_A"] + r["galat_B"]
        ring.append(r)
        print("    %-9s galat A %3d | galat B %3d | total %3d"
              % (at, r["galat_A"], r["galat_B"], r["total"]))
    pd.DataFrame(ring).to_csv(out / "E4_galat_per_aturan.csv", index=False)
    print("  CATATAN: angka ini memakai satu (tau,rho) untuk semua aturan, jadi")
    print("  tidak identik dengan tabel Fase 2D yang mengkalibrasi ulang tiap")
    print("  aturan. Yang dibandingkan di sini adalah pengaruh lubang saja.")

    lain = [a for a in tersedia if a != acuan]
    pindah_semua = []
    for at in lain:
        rusak = g[g["benar_" + at] & ~g["benar_" + acuan]]
        pulih = g[~g["benar_" + at] & g["benar_" + acuan]]
        print("\n  '%s' -> '%s': %d sel PULIH, %d sel RUSAK"
              % (at, acuan, len(pulih), len(rusak)))
        for judul, s in (("pulih", pulih), ("rusak", rusak)):
            if len(s):
                vc = s["kelompok"].value_counts()
                print("     %s: %s" % (judul, ", ".join("%s %d" % (k, v)
                                                        for k, v in vc.items())))
        for s, tag in ((pulih, "pulih"), (rusak, "rusak")):
            if len(s):
                t = s.copy(); t["arah"] = tag; t["dibanding"] = at
                pindah_semua.append(t)

    if pindah_semua:
        dp = pd.concat(pindah_semua, ignore_index=True)
        dp.to_csv(out / "E4_sel_berpindah.csv", index=False)
        # montase sel yang RUSAK oleh aturan acuan -- ini yang harus dibela
        rusak = dp[dp["arah"] == "rusak"].drop_duplicates("img_name")
        if len(rusak) and peta_mask:
            m = rusak.rename(columns={"br_" + acuan: "bridge_ratio",
                                      "rp_" + acuan: "r_pisah",
                                      "st_" + acuan: "status_topologi",
                                      "lob_" + acuan: "n_lobus"})
            montase_adjudikasi(m, caches[acuan], peta_mask, peta_citra, out,
                               "SEL YANG RUSAK oleh aturan '%s' (benar pada aturan lain)"
                               % acuan, "montase_E_rusak_oleh_%s.png" % acuan,
                               tau, rho, n_panel)
        return dp
    return None


# ===========================================================================
# BAGIAN 6 -- T5 RINGKASAN UNTUK BAGIAN 14
# ===========================================================================

def banding_ambang(ab, ambang=SEPERTIGA):
    """
    Bandingkan ambang 1/3 dengan argmax Youden, DIHITUNG ULANG dari data ini,
    dan nyatakan selisihnya dalam satuan SEL. Tanpa sklearn supaya tidak
    menambah kebergantungan.
    """
    A = ab.loc[ab["kelompok"] == "A_sepakat_band", "bridge_ratio"].values
    B = ab.loc[ab["kelompok"] == "B_sepakat_segmented", "bridge_ratio"].values
    if len(A) < 2 or len(B) < 2:
        return None
    kand = np.unique(np.r_[A, B])
    tpr = np.array([(A >= t).mean() for t in kand])
    fpr = np.array([(B >= t).mean() for t in kand])
    J = tpr - fpr
    k = int(np.argmax(J))
    t_opt = float(kand[k])
    n = len(A) + len(B)
    akur = lambda t: (int((A >= t).sum()) + int((B < t).sum())) / float(n)
    a13, aopt = akur(ambang), akur(t_opt)
    return dict(n=n, ambang_youden=t_opt, J_maks=float(J[k]),
                J_sepertiga=float((A >= ambang).mean() - (B >= ambang).mean()),
                akurasi_sepertiga=a13, akurasi_youden=aopt,
                selisih_sel=int(round((a13 - aopt) * n)))


def tugas_ringkas(df, ab, desil, kurva, yakinB, yakinA, out, ctx):
    print("\n=== T5  RINGKASAN ANGKA UNTUK PENULISAN ULANG BAGIAN 14 ===")
    L = ["# ANGKA SIAP-PAKAI, FASE 2E\n",
         "Dihasilkan %s. Setiap klaim ditulis bersama angka pendukungnya dan",
         "bersama keterbatasannya. Tidak ada klaim di sini yang bersandar pada",
         "AUC A-vs-B sebagai kriteria seleksi.\n"]
    L[1] = L[1] % time.strftime("%Y-%m-%d %H:%M")

    ab_ok = ab["benar"].mean()
    n_ab = len(ab)
    d1 = desil.iloc[0]
    kmin = kurva.iloc[-1]

    L.append("\n## Klaim 1 -- posisi kelompok C stabil lintas metode\n")
    pc = ctx.get("posisi_C")
    if pc is not None:
        L.append("Median bridge ratio kelompok C berada pada **%.4f** sepanjang "
                 "sumbu A->B. Empat versi metode dengan praproses yang berbeda "
                 "jauh memberi 0.339 (v1), 0.328 (2B), 0.3214 (2C), %.4f (2D/2E). "
                 "Rentang %.3f poin.\n" % (pc, pc, abs(0.339 - min(pc, 0.3214))))
    L.append("Ini klaim terkuat: yang diukur bukan artefak satu pipeline.\n")

    L.append("\n## Klaim 2 -- ambang 1/3 bukan argmax, tapi ongkosnya sebesar derau\n")
    ba = banding_ambang(ab)
    if ba is None:
        L.append("(kelompok A atau B terlalu kecil untuk membandingkan ambang)\n")
    else:
        d_ = ba["selisih_sel"]
        arah = ("lebih benar pada %d sel dari %d" % (d_, ba["n"]) if d_ > 0 else
                ("lebih salah pada %d sel dari %d" % (-d_, ba["n"]) if d_ < 0
                 else "tidak berbeda sama sekali (nol sel dari %d)" % ba["n"]))
        L.append("Argmax Youden yang dihitung ulang dari data ini jatuh di "
                 "**%.4f**, bukan di 1/3. Namun selisih akurasinya bisa "
                 "dinyatakan dalam satuan sel: memakai 1/3 **%s** "
                 "dibanding memakai argmax Youden (%.4f versus %.4f). Youden "
                 "memaksimalkan TPR-FPR, bukan akurasi, dan kelompoknya tidak "
                 "seimbang, jadi kedua fakta itu tidak bertentangan.\n"
                 % (ba["ambang_youden"], arah,
                    ba["akurasi_sepertiga"], ba["akurasi_youden"]))
        print("  banding ambang: Youden %.4f | akurasi @1/3 %.4f vs @Youden %.4f "
              "| selisih %+d sel dari %d"
              % (ba["ambang_youden"], ba["akurasi_sepertiga"],
                 ba["akurasi_youden"], ba["selisih_sel"], ba["n"]))
        pd.DataFrame([ba]).to_csv(out / "E5_banding_ambang.csv", index=False)
    L.append("> Jangan tulis \"ambang optimalnya sepertiga\". Tulis bahwa memilih "
             "1/3 adalah keputusan yang ongkosnya tidak terukur pada data ini.\n")

    L.append("\n## Klaim 3 -- pengisian lubang adalah kesalahan praproses, bukan pilihan\n")
    L.append("Sensus Fase 2D: seluruh piksel region tertutup di dalam nukleus "
             "berlabel sitoplasma atau vakuola, dengan fraksi jaringan minimum "
             "0.9677 pada 2366 region. Tidak ada artefak segmentasi untuk "
             "diperbaiki. Di tingkat sel, 67.6% galat ekor B Fase 2C punya lubang "
             "terisi di intinya, versus 14.5% pada sel B yang benar (odds ratio "
             "12.3) dan 6.7% pada sel A yang benar (odds ratio 29.0).\n")

    L.append("\n## Klaim 4 -- ada lantai yang tidak bisa dilewati metode apa pun\n")
    L.append("Akurasi A-vs-B **%.4f** pada %d sel. Membuang sel paling ragu tidak "
             "menghapus galat: pada cakupan %.0f%% masih tersisa %d galat "
             "(%.1f per 1000). Desil paling yakin sendiri menyisakan **%d galat "
             "dari %d sel**.\n"
             % (ab_ok, n_ab, 100 * kmin["cakupan"], kmin["galat"],
                kmin["galat_per_1000"], int(d1["galat"]), int(d1["n"])))
    L.append("Sel-sel itu tidak berada di perbatasan, jadi tidak bisa disebut "
             "kesalahan metode di dekat ambang. Kandidatnya: **%d sel B dengan "
             "bridge ratio >= 0.50** dan **%d sel A dengan bridge ratio <= 0.20**. "
             "Montase adjudikasi menyertakan citra RGB berdampingan dengan mask "
             "supaya bisa dipisahkan mana label yang salah dan mana mask yang "
             "salah.\n" % (len(yakinB), len(yakinA)))
    L.append("> Sampai adjudikasi visual selesai, tulis ini sebagai **dua "
             "kemungkinan yang belum dipisahkan**, bukan sebagai derau label.\n")

    L.append("\n## Yang BELUM boleh diklaim\n")
    L.append("- Belum boleh menyebut sisa galat sebagai derau label sebelum "
             "montase diperiksa mata manusia.")
    L.append("- Belum ada validasi eksternal lintas tahap pematangan; folder `ig` "
             "belum punya mask.")
    L.append("- Ambang 1/3 belum diuji pada dataset di luar PBC.\n")

    (out / "ANGKA_UNTUK_BAGIAN14.md").write_text("\n".join(L), encoding="utf-8")
    print("  ditulis: %s" % (out / "ANGKA_UNTUK_BAGIAN14.md"))
    for baris in L[4:]:
        if baris.startswith("## ") or baris.startswith("\n## "):
            print("   " + baris.strip())


# ===========================================================================
# BAGIAN 7 -- UJI SANITAS
# ===========================================================================

def jalankan_sanitas(out):
    print("\n=== UJI SANITAS ===")
    lolos = True

    def cek(nama, ok, pesan=""):
        nonlocal lolos
        lolos &= bool(ok)
        print("  [%s] %-34s %s" % ("OK " if ok else "GAGAL", nama, pesan))

    # 1. sisi_benar: A benar bila br >= 1/3, B benar bila br < 1/3
    kel = np.array(["A_sepakat_band", "A_sepakat_band",
                    "B_sepakat_segmented", "B_sepakat_segmented"])
    br = np.array([0.90, 0.10, 0.10, 0.90])
    got = sisi_benar(kel, br)
    cek("sisi_benar", list(got) == [True, False, True, False],
        "harapan T,F,T,F -> dapat %s" % list(got))

    # 2. sel tepat DI ambang dihitung band (>=), jadi A benar dan B salah
    got = sisi_benar(np.array(["A_sepakat_band", "B_sepakat_segmented"]),
                     np.array([SEPERTIGA, SEPERTIGA]))
    cek("perlakuan tepat di ambang", list(got) == [True, False],
        "br == 1/3 dihitung band")

    # 3. desil: 100 sel, 10 galat semuanya di sel paling TIDAK yakin ->
    #    desil 1 harus berakurasi 1.0 dan desil 10 berakurasi 0.0
    n = 100
    d = pd.DataFrame({
        "img_name": ["x%03d" % i for i in range(n)],
        "kelompok": ["B_sepakat_segmented"] * n,
        # 90 sel jelas segmented (br kecil), 10 sel tepat di atas ambang -> salah
        "bridge_ratio": np.r_[np.linspace(0.01, 0.30, 90),
                              np.full(10, SEPERTIGA + 0.001)],
        "r_pisah": 1.0, "r_lobus": 10.0, "n_lobus": 2,
        "status_topologi": "normal", "luas_nukleus": 100,
        "nucleus_shape": "segmented-bilobed", "split": "train"})
    tmp = Path(out)
    tmp.mkdir(parents=True, exist_ok=True)
    ab, desil, kurva = tugas_lantai(d, tmp, n_desil=10, diam=True)
    ok = (abs(float(desil.iloc[0]["akurasi"]) - 1.0) < 1e-9 and
          abs(float(desil.iloc[-1]["akurasi"]) - 0.0) < 1e-9 and
          int((~ab["benar"]).sum()) == 10)
    cek("desil keyakinan", ok,
        "desil1 %.2f, desil10 %.2f, total galat %d"
        % (desil.iloc[0]["akurasi"], desil.iloc[-1]["akurasi"],
           int((~ab["benar"]).sum())))

    # 4. kurva lantai: pada cakupan 0.9 seluruh galat harus sudah terbuang
    c90 = kurva[abs(kurva["cakupan"] - 0.9) < 1e-9].iloc[0]
    cek("kurva lantai membuang yang ragu", int(c90["galat"]) == 0,
        "cakupan 90%% -> %d galat (harapan 0)" % int(c90["galat"]))

    # 5. kasus berlawanan: galat ditaruh pada sel PALING yakin -> desil 1 buruk
    d2 = d.copy()
    d2["bridge_ratio"] = np.r_[np.full(10, 0.99), np.linspace(0.01, 0.30, 90)]
    ab2, desil2, kurva2 = tugas_lantai(d2, tmp, n_desil=10, diam=True)
    c50 = kurva2[abs(kurva2["cakupan"] - 0.5) < 1e-9].iloc[0]
    cek("galat yakin bertahan saat cakupan turun", int(c50["galat"]) == 10,
        "cakupan 50%% -> %d galat (harapan 10, semuanya bertahan)"
        % int(c50["galat"]))

    print("  -> %s" % ("SELURUH UJI LOLOS" if lolos
                       else "ADA UJI GAGAL -- JANGAN LANJUT KE DATA ASLI"))
    return lolos


# ===========================================================================
# BAGIAN 8 -- MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fase 2E: audit lantai derau dan adjudikasi visual.")
    ap.add_argument("--dir-hasil", default="hasil_fase2d",
                    help="folder keluaran Fase 2D (berisi bridge_ratio_2d.csv dan cache)")
    ap.add_argument("--dir-mask", default="pbcseg_final_v1")
    ap.add_argument("--dir-citra", default=None,
                    help="folder citra RGB asli PBC; tanpa ini montase hanya mask")
    ap.add_argument("--out", default="hasil_fase2e")
    ap.add_argument("--csv", default=None,
                    help="default: <dir-hasil>/bridge_ratio_2d.csv")
    ap.add_argument("--tugas", default="semua")
    ap.add_argument("--tau-br", type=float, default=None)
    ap.add_argument("--rho-br", type=float, default=None)
    ap.add_argument("--aturan-saddle", default="min", choices=["min", "max"])
    ap.add_argument("--acuan-lubang", default="tidak",
                    choices=["semua", "tidak", "selektif"],
                    help="aturan lubang yang dijadikan acuan pada T4")
    ap.add_argument("--n-panel", type=int, default=18)
    ap.add_argument("--desil", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tug = set(t.strip() for t in args.tugas.split(","))
    mau = lambda t: ("semua" in tug) or (t in tug)

    print("=" * 76)
    print("FASE 2E -- audit lantai derau label dan adjudikasi visual")
    print("=" * 76)
    print("keluaran: %s" % out.resolve())

    if not jalankan_sanitas(out / "_sanitas"):
        print("\nHENTI: uji sanitas gagal.")
        sys.exit(2)
    if tug == {"sanitas"}:
        return

    csvp = Path(args.csv) if args.csv else Path(args.dir_hasil) / "bridge_ratio_2d.csv"
    if not csvp.exists():
        print("HENTI: %s tidak ditemukan. Jalankan Fase 2D dulu, atau pakai --csv."
              % csvp)
        sys.exit(2)
    df = pd.read_csv(csvp)
    print("\ndata inti: %s (%d baris)" % (csvp.name, len(df)))
    if "aturan_lubang" in df.columns:
        at = sorted(set(df["aturan_lubang"].astype(str)))
        print("aturan lubang pada data ini: %s" % ", ".join(at))
        if at == ["selektif"]:
            print("  *** PERINGATAN: data ini dihasilkan dengan aturan 'selektif',")
            print("      padahal kriteria independen Fase 2D memilih 'tidak'.")
            print("      Jalankan ulang Fase 2D dengan --aturan-lubang tidak")
            print("      sebelum angka mana pun dibawa ke pembimbing. ***")

    tau = args.tau_br
    rho = args.rho_br
    if tau is None and "tau_br" in df.columns:
        tau = float(df["tau_br"].iloc[0])
    if rho is None and "rho_br" in df.columns:
        rho = float(df["rho_br"].iloc[0])
    if tau is None or rho is None:
        print("HENTI: tau/rho tidak ada di CSV dan tidak diberikan lewat argumen.")
        sys.exit(2)
    print("parameter bridge ratio: tau=%g rho=%g (dibaca dari Fase 2D)" % (tau, rho))

    peta_mask, nm_ = indeks_berkas(args.dir_mask, [".png"])
    peta_citra, nc_ = indeks_berkas(args.dir_citra, [".jpg", ".jpeg", ".png"])
    print("mask terindeks %d berkas | citra RGB terindeks %d berkas" % (nm_, nc_))
    caches = muat_cache(args.dir_hasil)
    print("cache Fase 2D dimuat: %s"
          % (", ".join("%s (%d sel)" % (k, len(v)) for k, v in caches.items())
             or "tidak ada"))

    aktif = None
    if "aturan_lubang" in df.columns:
        a0 = str(df["aturan_lubang"].iloc[0])
        aktif = caches.get(a0)
    if aktif is None and caches:
        aktif = caches.get(args.acuan_lubang) or list(caches.values())[0]

    ctx = dict()
    kelC = df[df["kelompok"] == "C_konflik_SNE_band"]
    kelA = df[df["kelompok"] == "A_sepakat_band"]
    kelB = df[df["kelompok"] == "B_sepakat_segmented"]
    if len(kelA) and len(kelB) and len(kelC):
        mA, mB = kelA["bridge_ratio"].median(), kelB["bridge_ratio"].median()
        if abs(mB - mA) > 1e-9:
            ctx["posisi_C"] = float((kelC["bridge_ratio"].median() - mA) / (mB - mA))
            print("posisi kelompok C pada sumbu A->B: %.4f   "
                  "[Fase 2D %.4f | 2B 0.328 | v1 0.339]"
                  % (ctx["posisi_C"], REF_2D["posisi_C"]))

    ab = desil = kurva = None
    yakinB = yakinA = pd.DataFrame()
    if mau("lantai"):
        ab, desil, kurva = tugas_lantai(df, out, n_desil=args.desil)
    if mau("profil"):
        if ab is None:
            ab, desil, kurva = tugas_lantai(df, out, n_desil=args.desil)
        yakinB, yakinA = tugas_profil(ab, out)
    if mau("montase"):
        if ab is None:
            ab, desil, kurva = tugas_lantai(df, out, n_desil=args.desil)
        if not len(yakinB) and not len(yakinA):
            # montase butuh daftar sel yakin-salah; hitung walau 'profil'
            # tidak diminta, supaya --tugas montase bisa berdiri sendiri
            yakinB, yakinA = tugas_profil(ab, out)
        if aktif is None:
            print("\n=== T3  MONTASE dilewati: cache Fase 2D tidak ditemukan ===")
        else:
            tugas_montase(ab, yakinB, yakinA, aktif, peta_mask, peta_citra,
                          out, tau, rho, args.n_panel)
    if mau("aturan"):
        tugas_aturan(df, caches, out, tau, rho, args.aturan_saddle,
                     args.acuan_lubang, peta_mask, peta_citra, args.n_panel)
    if mau("ringkas"):
        if ab is None:
            ab, desil, kurva = tugas_lantai(df, out, n_desil=args.desil)
        tugas_ringkas(df, ab, desil, kurva, yakinB, yakinA, out, ctx)

    print("\nSELESAI. Kirim ANGKA_UNTUK_BAGIAN14.md, E1-E4*.csv, dan keempat montase.")


if __name__ == "__main__":
    main()
