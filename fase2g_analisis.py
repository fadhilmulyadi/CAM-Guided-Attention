#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fase2g_analisis.py — Fase 2G bagian 2: QC mask, kontrol in-domain, dan uji monotonisitas.

DIJALANKAN DI LINGKUNGAN BIASA (CPU, tanpa GPU), bersama `fase2f_sel_ganda.py` di folder
yang sama — geometri persistensi diimpor dari sana, bukan disalin.

TIGA HAL, BERURUT, DAN URUTANNYA MENGIKAT
  1. LAPIS QC MASK. `ig` di luar domain latih wbcsegmentor. Bila masknya buruk,
     non-monotonisitas tidak membuktikan apa pun tentang bridge ratio. QC dirancang
     untuk menangkap KEGAGALAN MASK, bukan perbedaan biologi — promielosit memang
     berinti besar dan ber-rasio-NC tinggi, dan itu bukan cacat mask. Karena itu
     kriterianya sengaja tidak menyentuh ukuran inti atau rasio NC.
     Tingkat kelulusan QC PER TAHAP adalah hasil tersendiri: bila PMY jauh lebih
     sering gagal daripada SNE, itu batasan yang wajib dilaporkan, bukan disembunyikan.

  2. KONTROL IN-DOMAIN. Sebelum menyentuh `ig`, ukur dulu berapa banyak bridge ratio
     bergeser ketika mask DIPREDIKSI alih-alih DIANOTASI, pada 10.298 sel yang sama.
     Ini memberi plafon: `ig` tidak mungkin lebih baik daripada ini. Sekaligus menjawab
     "Pembanding wajib: pipeline dua tahap versus satu tahap" untuk Fase 3.

  3. MONOTONISITAS LIMA TAHAP. promielosit -> mielosit -> metamielosit -> band -> segmented.
     SELURUH tahap memakai mask PREDIKSI, termasuk BNE/SNE, supaya sumber mask tidak
     berubah di tengah sumbu. Statistik: Spearman dan Kendall tau-b dengan uji permutasi,
     plus Mann-Whitney antar-tahap bersebelahan dengan ukuran efek probability of superiority.

CONTOH
  python fase2g_analisis.py --tugas sanitas
  python fase2g_analisis.py --tugas semua --pekerja 7 > log_2g2.txt 2>&1
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import kendalltau, mannwhitneyu, rankdata, spearmanr

import fase2f_sel_ganda as F
from fase2f_sel_ganda import (S8, SEPERTIGA, Laporan, akurasi, auc, indeks_berkas, jalankan_pool,
                              komponen_sel, kunci_nama, baca_mask, baca_rgb, tabel_md, youden)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

TAHAP = {"PMY": (1, "1_promielosit"), "MY": (2, "2_mielosit"), "MMY": (3, "3_metamielosit"),
         "BNE": (4, "4_band"), "SNE": (5, "5_segmented")}
LUAR_SUMBU = {"IG": "ig_tak_bersubtipe", "NEUTROPHIL": "neutrofil_tak_bersubtipe"}


# =============================================================================
# 1. QC MASK — menangkap kegagalan mask, BUKAN perbedaan biologi
# =============================================================================
def ukur_qc(mask):
    """Ukuran yang tidak mengandaikan ukuran inti atau rasio NC tertentu, supaya sel
    imatur tidak dihukum karena memang imatur."""
    H, W = mask.shape
    sel, fb = komponen_sel(mask)
    nuk = (mask == 2) & sel
    if nuk.any():
        lab, n = ndi.label(nuk, structure=S8)
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        nuk = np.isin(lab, np.flatnonzero(sz >= 0.02 * sz.sum()))
        n_komp = int(len(np.flatnonzero(sz >= 0.02 * sz.sum())))
        frak_remah = float(1 - sz[sz >= 0.02 * sz.sum()].sum() / sz.sum())
    else:
        n_komp, frak_remah = 0, 0.0
    tepi = np.zeros_like(sel)
    tepi[0, :] = tepi[-1, :] = True
    tepi[:, 0] = tepi[:, -1] = True
    return dict(
        luas_sel=int(sel.sum()), luas_nukleus=int(nuk.sum()),
        rasio_nc=float(nuk.sum() / max(sel.sum(), 1)),
        pusat_fallback=bool(fb),
        frak_tepi=float((sel & tepi).sum() / max(tepi.sum(), 1)),
        n_komp_inti=n_komp, frak_remah_inti=frak_remah,
        frak_bingkai=float(sel.sum() / (H * W)))


def aturan_qc(d, batas):
    """d: DataFrame ukuran QC. Kembalikan Series bool + tabel sebab kegagalan."""
    sebab = pd.DataFrame(index=d.index)
    sebab["inti_hilang"] = d.luas_nukleus < batas["luas_nukleus_min"]
    sebab["pusat_meleset"] = d.pusat_fallback.astype(bool)
    sebab["sel_terpotong"] = d.frak_tepi > batas["frak_tepi_maks"]
    sebab["sel_terlalu_kecil"] = d.luas_sel < batas["luas_sel_min"]
    sebab["sel_terlalu_besar"] = d.luas_sel > batas["luas_sel_maks"]
    sebab["inti_remah"] = d.frak_remah_inti > batas["frak_remah_maks"]
    return ~sebab.any(axis=1), sebab


def kerja_qc(arg):
    img, path, prm, hitung_br = arg
    out = {"img_name": img, "galat": ""}
    try:
        mask = baca_mask(path)
        out.update(ukur_qc(mask))
        if hitung_br:
            sel, _ = komponen_sel(mask)
            for k, v in F.ukur_inti(F.inti_M4(mask, sel), prm).items():
                out["p_" + k] = v
    except Exception as e:
        out["galat"] = repr(e)
    return out


# =============================================================================
# 2. STATISTIK TREN
# =============================================================================
def tren_permutasi(tahap, nilai, n_perm=5000, seed=7):
    """Spearman dan Kendall tau-b, dengan nilai-p permutasi yang tidak bergantung
    asumsi distribusi. Permutasi dihitung pada peringkat, jadi cepat."""
    tahap = np.asarray(tahap, float)
    nilai = np.asarray(nilai, float)
    m = np.isfinite(tahap) & np.isfinite(nilai)
    tahap, nilai = tahap[m], nilai[m]
    n = len(nilai)
    if n < 20 or len(np.unique(tahap)) < 2:
        return dict(n=n, rho=np.nan, tau=np.nan, p_rho=np.nan, p_tau=np.nan, p_perm=np.nan)
    rho, p_rho = spearmanr(tahap, nilai)
    tau, p_tau = kendalltau(tahap, nilai)
    rx, ry = rankdata(tahap), rankdata(nilai)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    rng = np.random.default_rng(seed)
    nol = np.array([float(np.dot(rng.permutation(rx), ry) / n) for _ in range(n_perm)])
    p_perm = float((np.sum(np.abs(nol) >= abs(rho)) + 1) / (n_perm + 1))
    return dict(n=n, rho=float(rho), tau=float(tau), p_rho=float(p_rho),
                p_tau=float(p_tau), p_perm=p_perm)


def banding_pasangan(a, b):
    """Mann-Whitney + probability of superiority P(b > a) + 0.5 P(sama)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return dict(n_a=len(a), n_b=len(b), p=np.nan, ps=np.nan)
    u, p = mannwhitneyu(b, a, alternative="two-sided")
    return dict(n_a=len(a), n_b=len(b), p=float(p), ps=float(u / (len(a) * len(b))))


# =============================================================================
# 3. UJI SANITAS
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

    S = (360, 360)
    # 1 mask sehat
    mk = np.zeros(S, np.uint8)
    mk[cakram(S, 180, 180, 70)] = 1
    mk[cakram(S, 180, 180, 30)] = 2
    q = ukur_qc(mk)
    cek("mask sehat lolos seluruh kriteria QC",
        q["luas_nukleus"] > 2500 and not q["pusat_fallback"] and q["frak_tepi"] == 0
        and q["n_komp_inti"] == 1,
        f"inti {q['luas_nukleus']}, tepi {q['frak_tepi']:.3f}, komp {q['n_komp_inti']}")
    # 2 inti hilang
    mk2 = np.zeros(S, np.uint8)
    mk2[cakram(S, 180, 180, 70)] = 1
    q2 = ukur_qc(mk2)
    cek("inti hilang terdeteksi", q2["luas_nukleus"] == 0, f"inti {q2['luas_nukleus']}")
    # 3 sel terpotong tepi bingkai
    mk3 = np.zeros(S, np.uint8)
    mk3[cakram(S, 20, 180, 90)] = 1
    mk3[cakram(S, 20, 180, 35)] = 2
    q3 = ukur_qc(mk3)
    cek("sel terpotong tepi terdeteksi", q3["frak_tepi"] > 0.05, f"frak_tepi {q3['frak_tepi']:.4f}")
    # 4 sel imatur (inti besar, NC tinggi) TIDAK boleh dihukum
    mk4 = np.zeros(S, np.uint8)
    mk4[cakram(S, 180, 180, 75)] = 1
    mk4[cakram(S, 180, 180, 66)] = 2      # rasio NC ~0,77 seperti promielosit
    q4 = ukur_qc(mk4)
    batas = dict(luas_nukleus_min=500, frak_tepi_maks=0.05, luas_sel_min=2000,
                 luas_sel_maks=60000, frak_remah_maks=0.25)
    lolos4, _ = aturan_qc(pd.DataFrame([q4]), batas)
    cek("sel imatur berinti besar TIDAK dihukum QC",
        bool(lolos4.iloc[0]) and q4["rasio_nc"] > 0.7,
        f"rasio_NC {q4['rasio_nc']:.3f}, lolos {bool(lolos4.iloc[0])}")
    # 5 tren monoton sempurna terdeteksi
    rng = np.random.default_rng(0)
    th = np.repeat([1, 2, 3, 4, 5], 200)
    y = -th + rng.normal(0, 0.15, len(th))
    t = tren_permutasi(th, y, n_perm=500)
    cek("tren menurun kuat terdeteksi", t["rho"] < -0.9 and t["p_perm"] < 0.01,
        f"rho {t['rho']:.3f}, p_perm {t['p_perm']:.4f}")
    # 6 tidak ada tren -> tidak dilaporkan sebagai tren
    y2 = rng.normal(0, 1, len(th))
    t2 = tren_permutasi(th, y2, n_perm=500)
    cek("data tanpa tren tidak memberi tren palsu", abs(t2["rho"]) < 0.15 and t2["p_perm"] > 0.05,
        f"rho {t2['rho']:.3f}, p_perm {t2['p_perm']:.4f}")
    # 7 tren NON-MONOTON (U) tidak lolos sebagai monoton
    y3 = (th - 3.0) ** 2 + rng.normal(0, 0.15, len(th))
    t3 = tren_permutasi(th, y3, n_perm=500)
    cek("tren berbentuk U tidak terbaca monoton", abs(t3["rho"]) < 0.3,
        f"rho {t3['rho']:.3f} (harus dekat nol meski polanya kuat)")
    # 8 probability of superiority
    ps = banding_pasangan(np.zeros(100), np.ones(100))
    cek("probability of superiority benar", abs(ps["ps"] - 1.0) < 1e-9, f"PS {ps['ps']:.4f}")

    t = pd.DataFrame(hasil)
    lap.tabel(t)
    if not t.lolos.all():
        lap.peringatan("UJI SANITAS GAGAL — jalanan dihentikan.")
        return False
    lap.tulis(f"**Seluruh {len(t)} uji LOLOS.** Uji 4 dan 7 yang terpenting: QC tidak boleh "
              "menghukum sel imatur karena imatur, dan tren berbentuk U tidak boleh terbaca "
              "sebagai monoton.")
    lap.tulis()
    return True


# =============================================================================
# 4. MONTASE
# =============================================================================
def montase_tahap(baris, path, judul, idx_mask, idx_citra, kolom=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(baris)
    if not n:
        return
    br_ = math.ceil(n / kolom)
    fig, axs = plt.subplots(br_, kolom, figsize=(kolom * 3.0, br_ * 3.3))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, it in zip(axs, baris):
        k = it["kunci"]
        if k not in idx_mask:
            continue
        mask = baca_mask(idx_mask[k])
        sel, _ = komponen_sel(mask)
        nuk = F.inti_M4(mask, sel)
        if k in idx_citra:
            ax.imshow(baca_rgb(idx_citra[k], mask.shape))
        else:
            ax.imshow(mask, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
        for arr, warna, lw in ((nuk, "red", 0.8), (sel, "yellow", 0.5)):
            if arr.any() and not arr.all():
                ax.contour(arr.astype(float), levels=[0.5], colors=warna, linewidths=lw)
        ax.set_title(it["teks"], fontsize=6)
    fig.suptitle(judul + "\nmerah = inti (M4) | kuning = komponen sel", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tugas", choices=["sanitas", "kontrol", "ig", "semua"], default="semua")
    ap.add_argument("--dir-hasil", default="hasil_fase2d")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir-2g", default="hasil_fase2g")
    ap.add_argument("--dir-mask", default="pbcseg_final_v1")
    ap.add_argument("--keluar", default="hasil_fase2g_analisis")
    ap.add_argument("--pekerja", type=int, default=7)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tau-br", type=float, default=1.25)
    ap.add_argument("--rho-br", type=float, default=0.2)
    ap.add_argument("--tau-lobus", type=float, default=4.0)
    ap.add_argument("--rho-lobus", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.keluar, exist_ok=True)
    lap = Laporan(os.path.join(args.keluar, "LAPORAN_FASE2G.md"))
    prm = dict(tau_br=args.tau_br, rho_br=args.rho_br, tau_lobus=args.tau_lobus,
               rho_lobus=args.rho_lobus, prune=0.05)
    lap.tulis("# LAPORAN FASE 2G — validasi eksternal lintas tahap pematangan")
    lap.tulis()
    lap.tulis(f"Dijalankan {time.strftime('%Y-%m-%d %H:%M')} | τ_br {args.tau_br}, ρ_br {args.rho_br}, "
              f"τ_lobus {args.tau_lobus}, ρ_lobus {args.rho_lobus}, aturan lubang `tidak` | "
              f"permutasi {args.n_perm}")
    lap.tulis()
    try:
        if not uji_sanitas(lap, prm):
            sys.exit(1)
        if args.tugas == "sanitas":
            return

        dir_pbc = os.path.join(args.dir_2g, "mask_pred_pbc")
        dir_ig = os.path.join(args.dir_2g, "mask_pred_ig")
        path_csv = args.csv or os.path.join(args.dir_hasil, "bridge_ratio_2d.csv")
        df = pd.read_csv(path_csv)
        df["kunci"] = df.img_name.map(kunci_nama)
        idx_pred_pbc = indeks_berkas(dir_pbc, ("*_mask.png",)) if os.path.isdir(dir_pbc) else {}
        idx_pred_ig = indeks_berkas(dir_ig, ("*_mask.png",)) if os.path.isdir(dir_ig) else {}
        idx_citra_ig = indeks_berkas(dir_ig, ("*.jpg",), buang_mask=True) if os.path.isdir(dir_ig) else {}
        lap.tulis(f"Mask prediksi: PBC **{len(idx_pred_pbc)}**, ig **{len(idx_pred_ig)}** | "
                  f"citra ig **{len(idx_citra_ig)}** | CSV anotasi {len(df)} baris")
        lap.tulis()

        # ---------------- batas QC dari mask ANOTASI ----------------
        lap.tulis("## 1. Lapis QC — ambang dikalibrasi dari mask ANOTASI, diterapkan ke mask prediksi")
        lap.tulis()
        idx_anot = indeks_berkas(args.dir_mask, ("*_mask.png",))
        contoh = df.sample(min(1500, len(df)), random_state=args.seed)
        tug = [(r.img_name, idx_anot[r.kunci], prm, False) for r in contoh.itertuples()
               if r.kunci in idx_anot]
        qa = pd.DataFrame(jalankan_pool(kerja_qc, tug, args.pekerja, "QC pada mask anotasi", lap))
        qa = qa[qa.galat.fillna("").astype(str).str.len() == 0]
        batas = dict(luas_nukleus_min=500,
                     frak_tepi_maks=float(max(qa.frak_tepi.quantile(0.995), 0.02)),
                     luas_sel_min=float(qa.luas_sel.quantile(0.001) * 0.5),
                     luas_sel_maks=float(qa.luas_sel.quantile(0.999) * 2.0),
                     frak_remah_maks=0.25)
        lap.tulis("Ambang sengaja longgar dan hanya menyasar **kegagalan mask**. Ukuran inti dan "
                  "rasio NC TIDAK dipakai sebagai kriteria, karena sel imatur memang berinti besar "
                  "dan ber-rasio-NC tinggi — menghukumnya berarti membuang biologi, bukan derau.")
        lap.tulis()
        lap.tabel(pd.DataFrame([dict(kriteria=k, nilai=v) for k, v in batas.items()]))
        lolos_a, sebab_a = aturan_qc(qa, batas)
        lap.tulis(f"Kontrol: mask ANOTASI sendiri lolos QC **{lolos_a.mean():.2%}** "
                  f"({int(lolos_a.sum())}/{len(qa)}). Angka ini adalah plafon — mask prediksi "
                  "tidak mungkin melampauinya secara bermakna.")
        lap.tulis()

        ringkas = []

        # ---------------- 2. kontrol in-domain ----------------
        if args.tugas in ("kontrol", "semua") and idx_pred_pbc:
            lap.tulis("## 2. Kontrol in-domain — bridge ratio dari mask PREDIKSI versus mask ANOTASI")
            lap.tulis()
            tug = [(r.img_name, idx_pred_pbc[r.kunci], prm, True) for r in df.itertuples()
                   if r.kunci in idx_pred_pbc]
            qp = pd.DataFrame(jalankan_pool(kerja_qc, tug, args.pekerja, "QC + br pada mask prediksi PBC", lap))
            qp = qp[qp.galat.fillna("").astype(str).str.len() == 0]
            lolos_p, sebab_p = aturan_qc(qp, batas)
            qp["lolos_qc"] = lolos_p.values
            lap.tulis(f"Mask prediksi PBC lolos QC **{lolos_p.mean():.2%}** versus anotasi {lolos_a.mean():.2%}.")
            lap.tulis()
            lap.tabel(pd.DataFrame([dict(sebab=c, n=int(sebab_p[c].sum()),
                                         frak=float(sebab_p[c].mean())) for c in sebab_p.columns]))
            m = df.merge(qp, on="img_name", how="inner")
            ok = m[m.lolos_qc & np.isfinite(m.p_br.astype(float)) & np.isfinite(m.bridge_ratio.astype(float))]
            rho = spearmanr(ok.bridge_ratio, ok.p_br)[0]
            beda = (ok.p_br.astype(float) - ok.bridge_ratio.astype(float))
            pindah = float((ok.bridge_ratio.ge(SEPERTIGA) != ok.p_br.ge(SEPERTIGA)).mean())
            lap.tulis(f"Pada {len(ok)} sel lolos QC: Spearman(br_anotasi, br_prediksi) = **{rho:.4f}**, "
                      f"selisih median **{beda.median():+.4f}**, |selisih| median {beda.abs().median():.4f}, "
                      f"berpindah sisi ambang 1/3 **{pindah:.2%}**.")
            lap.tulis()
            per = ok.groupby("label").apply(
                lambda g: pd.Series(dict(n=len(g), rho=spearmanr(g.bridge_ratio, g.p_br)[0],
                                         beda_median=(g.p_br - g.bridge_ratio).median(),
                                         pindah=float((g.bridge_ratio.ge(SEPERTIGA) != g.p_br.ge(SEPERTIGA)).mean()))),
                include_groups=False).reset_index()
            lap.tabel(per)
            neu = ok[ok.label == "Neutrophil"].copy()
            neu["kel"] = neu.kelompok.astype(str).str[0]
            bar = []
            for nm, kol in (("mask anotasi (Fase 2D)", "bridge_ratio"), ("mask prediksi", "p_br")):
                A = neu.loc[neu.kel == "A", kol].to_numpy(float)
                B = neu.loc[neu.kel == "B", kol].to_numpy(float)
                if len(A) and len(B):
                    bar.append(dict(sumber_mask=nm, n_A=len(A), n_B=len(B), auc=auc(A, B),
                                    acc_13=akurasi(A, B, SEPERTIGA), youden=youden(A, B)[0]))
            tb = pd.DataFrame(bar)
            lap.tulis("**Ongkos pipeline dua tahap** (segmentasi → geometri) dibanding mask anotasi:")
            lap.tabel(tb, 5)
            if len(tb) == 2:
                d_auc = tb.auc.iloc[1] - tb.auc.iloc[0]
                lap.tulis(f"ΔAUC **{d_auc:+.4f}**, Δakurasi@1/3 **{tb.acc_13.iloc[1] - tb.acc_13.iloc[0]:+.4f}**. "
                          "Ini plafon untuk `ig`: sel di luar domain latih tidak mungkin lebih baik "
                          "daripada sel di dalam domain latih.")
                ringkas.append(dict(ukuran="ΔAUC pipeline dua tahap (in-domain)", nilai=d_auc))
            lap.tulis()
            m.drop(columns=["kunci"]).to_csv(os.path.join(args.keluar, "I1_kontrol_in_domain.csv"), index=False)

        # ---------------- 3. monotonisitas ----------------
        if args.tugas in ("ig", "semua") and idx_pred_ig:
            lap.tulis("## 3. Monotonisitas lintas lima tahap pematangan")
            lap.tulis()
            baris = []
            for k, p in idx_pred_ig.items():
                pre = k.split("_")[0].upper()
                baris.append(dict(kunci=k, path=p, prefix=pre))
            for k, p in idx_pred_pbc.items():
                pre = k.split("_")[0].upper()
                if pre in ("BNE", "SNE", "NEUTROPHIL"):
                    baris.append(dict(kunci=k, path=p, prefix=pre))
            d5 = pd.DataFrame(baris)
            d5["tahap_idx"] = d5.prefix.map(lambda p: TAHAP.get(p, (np.nan,))[0])
            d5["tahap"] = d5.prefix.map(lambda p: TAHAP[p][1] if p in TAHAP else LUAR_SUMBU.get(p, "?"))
            lap.tulis("Seluruh tahap memakai **mask prediksi**, termasuk BNE/SNE — supaya sumber "
                      "mask tidak berubah di tengah sumbu. Ini keputusan yang menentukan: "
                      "mencampur mask anotasi (BNE/SNE) dengan mask prediksi (PMY/MY/MMY) akan "
                      "membuat tren tak bisa dibedakan dari pergantian sumber mask.")
            lap.tulis()
            tug = [(r.kunci, r.path, prm, True) for r in d5.itertuples()]
            g5 = pd.DataFrame(jalankan_pool(kerja_qc, tug, args.pekerja, "QC + br lima tahap", lap))
            g5 = g5[g5.galat.fillna("").astype(str).str.len() == 0].rename(columns={"img_name": "kunci"})
            d5 = d5.merge(g5, on="kunci", how="inner")
            lolos5, sebab5 = aturan_qc(d5, batas)
            d5["lolos_qc"] = lolos5.values

            tq = d5.groupby("tahap").apply(
                lambda g: pd.Series(dict(n=len(g), lolos_qc=int(g.lolos_qc.sum()),
                                         frak_lolos=float(g.lolos_qc.mean()),
                                         inti_hilang=int((g.luas_nukleus < batas["luas_nukleus_min"]).sum()),
                                         pusat_meleset=int(g.pusat_fallback.sum()),
                                         terpotong=int((g.frak_tepi > batas["frak_tepi_maks"]).sum()))),
                include_groups=False).reset_index()
            lap.tulis("**Tingkat kelulusan QC per tahap** — ini hasil tersendiri, bukan sekadar penyaring:")
            lap.tabel(tq)
            rentang_qc = float(tq[tq.tahap.str.match(r"^[1-5]_")].frak_lolos.max()
                               - tq[tq.tahap.str.match(r"^[1-5]_")].frak_lolos.min())
            if rentang_qc > 0.15:
                lap.peringatan(f"Kelulusan QC berbeda {rentang_qc:.0%} antar tahap. Tren apa pun di bawah ini "
                               "sebagian bisa berasal dari kualitas mask yang tidak merata, bukan dari "
                               "morfologi. Laporkan bersama, jangan terpisah.")
            lap.tulis()

            for nama, sub in (("seluruh sel", d5), ("hanya sel lolos QC", d5[d5.lolos_qc])):
                s = sub[np.isfinite(sub.tahap_idx.astype(float))]
                lap.tulis(f"### 3.{1 if nama == 'seluruh sel' else 2} Tren pada {nama}")
                lap.tulis()
                t = s.groupby("tahap").apply(
                    lambda g: pd.Series(dict(n=len(g), q25=g.p_br.quantile(.25),
                                             median=g.p_br.median(), q75=g.p_br.quantile(.75),
                                             frak_ge_13=float(g.p_br.ge(SEPERTIGA).mean()),
                                             tak_pecah=float(g.p_status.eq("tak_pernah_pecah").mean()),
                                             terpisah0=float(g.p_status.eq("terpisah_di_nol").mean()),
                                             luas_inti_med=g.luas_nukleus.median(),
                                             rasio_nc_med=g.rasio_nc.median())),
                    include_groups=False).reset_index()
                lap.tabel(t)
                tr = tren_permutasi(s.tahap_idx, s.p_br, args.n_perm, args.seed)
                lap.tulis(f"Spearman ρ = **{tr['rho']:+.4f}** (p {tr['p_rho']:.2e}) | "
                          f"Kendall τ-b = **{tr['tau']:+.4f}** (p {tr['p_tau']:.2e}) | "
                          f"p permutasi {args.n_perm}× = **{tr['p_perm']:.2e}** | n = {tr['n']}")
                lap.tulis()
                pas = []
                urut = sorted(TAHAP.values())
                for (i1, n1), (i2, n2) in zip(urut[:-1], urut[1:]):
                    a = s.loc[s.tahap == n1, "p_br"]
                    b = s.loc[s.tahap == n2, "p_br"]
                    r = banding_pasangan(a, b)
                    pas.append(dict(pasangan=f"{n1} → {n2}", **r,
                                    arah="turun" if r["ps"] < 0.5 else "naik"))
                lap.tabel(pd.DataFrame(pas), 5)
                lap.tulis("`ps` = probability of superiority: peluang sel tahap berikutnya punya br "
                          "lebih tinggi. Monotonisitas menurun berarti seluruh `ps` di bawah 0,5.")
                lap.tulis()
                if nama == "hanya sel lolos QC":
                    ringkas.append(dict(ukuran="Spearman ρ lima tahap (lolos QC)", nilai=tr["rho"]))
                    ringkas.append(dict(ukuran="p permutasi", nilai=tr["p_perm"]))
                    semua_turun = all(p["ps"] < 0.5 for p in pas if np.isfinite(p["ps"]))
                    ringkas.append(dict(ukuran="seluruh pasangan bersebelahan menurun", nilai=float(semua_turun)))

            luar = d5[~np.isfinite(d5.tahap_idx.astype(float))]
            if len(luar):
                lap.tulis("Kelompok di luar sumbu (dilaporkan terpisah, tidak masuk uji tren):")
                lap.tabel(luar.groupby("tahap").apply(
                    lambda g: pd.Series(dict(n=len(g), lolos_qc=int(g.lolos_qc.sum()),
                                             median_br=g.p_br.median(),
                                             frak_ge_13=float(g.p_br.ge(SEPERTIGA).mean()))),
                    include_groups=False).reset_index())

            d5.to_csv(os.path.join(args.keluar, "I2_lima_tahap.csv"), index=False)

            lap.tulis("### 3.3 Montase per tahap")
            lap.tulis()
            idx_gab = dict(idx_pred_ig)
            idx_gab.update(idx_pred_pbc)
            for tnama in sorted(d5.tahap.unique()):
                sub = d5[d5.tahap == tnama].sort_values("p_br")
                if not len(sub):
                    continue
                amb = sub.iloc[np.linspace(0, len(sub) - 1, min(12, len(sub))).astype(int)]
                it = [dict(kunci=r.kunci,
                           teks=f"{r.kunci}\nbr {r.p_br:.2f} inti {int(r.luas_nukleus)} "
                                f"NC {r.rasio_nc:.2f}{'' if r.lolos_qc else ' [QC GAGAL]'}")
                      for r in amb.itertuples()]
                montase_tahap(it, os.path.join(args.keluar, f"montase_G_{tnama}.png"),
                              f"{tnama} — 12 sel dari {len(sub)}, diurut menurut bridge ratio",
                              idx_gab, idx_citra_ig)
                lap.tulis(f"- `montase_G_{tnama}.png`: 12 dari {len(sub)} sel")
            lap.tulis()

        if ringkas:
            lap.tulis("## 4. Ringkasan")
            lap.tulis()
            lap.tabel(pd.DataFrame(ringkas), 5)
            lap.tulis("Cara membaca: ΔAUC in-domain adalah plafon kualitas. Bila tren lima tahap "
                      "kuat DAN kelulusan QC merata antar tahap, validasi eksternal berhasil. "
                      "Bila kelulusan QC timpang, tren harus dilaporkan bersama ketimpangan itu.")
    finally:
        lap.simpan()
        print(f"\nLaporan: {lap.path}")


if __name__ == "__main__":
    main()
