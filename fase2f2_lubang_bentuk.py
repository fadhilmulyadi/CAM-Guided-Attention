#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fase2f2_lubang_bentuk.py — Fase 2F bagian 2: M10 (aturan lubang berbasis bentuk)

MEMBUTUHKAN `fase2f_sel_ganda.py` DI FOLDER YANG SAMA. Geometri persistensi diimpor dari
sana, bukan disalin, supaya identik dengan jalanan yang sudah mereproduksi Fase 2D 99,99%.

GAGASAN POKOK
  Aturan `tidak` dan `semua` bukan dua pilihan terpisah, melainkan DUA UJUNG dari satu
  keluarga berparameter tunggal:

      isi lubang bila  lebar_maks(lubang) <= L          (lebar_maks = 2 x EDT maksimum)

      L = 0    -> tidak ada lubang diisi          == aturan `tidak`
      L = ~    -> seluruh lubang diisi            == aturan `semua`

  Rasionalnya dari Bagian 12.4: lubang TIPIS-MEMANJANG adalah intrusi anotasi (harus
  diisi, karena nukleusnya sebenarnya menyambung), lubang KOMPAK adalah ruang sitoplasma
  sejati (jangan diisi). `lebar_maks` memisahkan keduanya secara langsung: celah 2x20 px
  memberi lebar_maks 2,00 dan aspek 10,0; cakram r=5 memberi lebar_maks 10,2 dan aspek 0,78.

TIGA LAPIS, DENGAN PINTU KELUAR DITETAPKAN SEBELUM JALANAN
  1. MEKANISTIK (bisa menggugurkan, TANPA label apa pun): apakah sebaran `lebar_maks`
     pada seluruh region lubang bimodal? Bila menyatu tanpa celah, tidak ada L yang
     bisa dibenarkan secara mekanistik dan M10 DITUTUP SEBAGAI NEGATIF.
  2. KALIBRASI INDEPENDEN: skor kecocokan jumlah lobus terhadap `nucleus_shape`, HANYA
     pada sel non-neutrofil. Pintu yang sama dengan Fase 2B/2C. Dilaporkan sebagai
     WILAYAH, bukan argmax (permukaannya datar; Pelajaran 7).
  3. UJI DUA SISI WAJIB: aturan apa pun harus memperbaiki sel A yang dirusak `tidak`
     TANPA merusak sel B yang dipulihkan `tidak`. Kedua mekanisme berbeda (Bagian 11.6:
     sel A lewat pembilang r_pisah, sel B lewat penyebut r_lobus).

  AUC A-vs-B TIDAK DIPAKAI DI LAPIS MANA PUN. Ia hanya dilaporkan di akhir.

BAGIAN KEDUA — ALTERNATIF TANPA PARAMETER BARU
  Bila aturan biner memang tidak bisa memuaskan kedua sisi (dugaan Bagian 12.4), maka
  sel yang VONISNYA BERUBAH antar-aturan adalah sel yang bentuk lubangnya memang ambigu.
  Itu sinyal ketidakpastian yang bisa langsung masuk ke mesin deferral yang sudah
  tervalidasi di Fase 2D, TANPA menambah satu parameter pun ke pipeline.
  Diuji: apakah `peka_aturan` mengungguli |br - 1/3| sebagai kriteria penundaan.

CONTOH
  python fase2f2_lubang_bentuk.py --tugas sanitas
  python fase2f2_lubang_bentuk.py --tugas semua --pekerja 7 ^
      --dir-mask pbcseg_final_v1 > log_2f2.txt 2>&1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import fase2f_sel_ganda as F
from fase2f_sel_ganda import (S8, SEPERTIGA, Laporan, akurasi, auc, crop_pad, indeks_berkas,
                              jalankan_pool, komponen_sel, kunci_nama, baca_mask, baca_rgb,
                              persistensi, tabel_md, turunkan, youden, analisis, laporkan_skenario)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# L yang disapu. 0 == aturan `tidak`, 999 == aturan `semua`.
DAFTAR_L = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 999.0)

# Harapan jumlah lobus dari nucleus_shape (Fase 2B). `irregular` dan `band` dikecualikan.
HARAP_LOBUS = {"unsegmented-round": (1, 1), "unsegmented-indented": (1, 1),
               "segmented-bilobed": (2, 2), "segmented-multilobed": (3, 99)}

# Sel dari Bagian 11.6 — uji dua sisi. WAJIB: perbaiki A tanpa merusak B.
A_DIRUSAK = ["BNE_313131", "BNE_411518", "BNE_741104", "BNE_482975", "BNE_792019", "BNE_866984",
             "BNE_396853", "BNE_544250", "BNE_379425", "BNE_815174", "BNE_848315", "BNE_723408",
             "BNE_598279", "BNE_196260"]
B_DIRUSAK = ["SNE_206630", "SNE_234830", "SNE_868570", "SNE_916750"]


# =============================================================================
# 1. GEOMETRI LUBANG
# =============================================================================
def ukur_lubang(nuk):
    """Kembalikan (label_lubang, daftar dict per region). 4-konektivitas untuk lubang,
    pasangan dari 8-konektivitas pada latar depan."""
    terisi = ndi.binary_fill_holes(nuk)
    lub = terisi & ~nuk
    if not lub.any():
        return np.zeros_like(nuk, np.int32), []
    lab, n = ndi.label(lub)  # struktur bawaan = 4-konektivitas
    out = []
    for i in range(1, n + 1):
        reg = lab == i
        dt = ndi.distance_transform_edt(np.pad(reg, 1))
        lebar = float(2 * dt.max())
        luas = int(reg.sum())
        out.append(dict(idx=i, luas=luas, lebar_maks=lebar,
                        aspek=float(luas / lebar ** 2) if lebar > 0 else np.nan))
    return lab, out


def nukleus_beraturan(mask, batas, L, min_frak=0.02):
    """Nukleus M4 dengan lubang berlebar_maks <= L diisi. L=0 -> `tidak`, L besar -> `semua`."""
    nuk = (mask == 2) & batas
    if not nuk.any():
        return nuk
    lab, n = ndi.label(nuk, structure=S8)
    if n > 1:
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        nuk = np.isin(lab, np.flatnonzero(sz >= min_frak * sz.sum()))
    if L <= 0:
        return nuk
    lab_l, reg = ukur_lubang(nuk)
    isi = [r["idx"] for r in reg if r["lebar_maks"] <= L]
    if isi:
        nuk = nuk | np.isin(lab_l, isi)
    return nuk


def saddle_terpilih(nuk, prm):
    """Posisi (y,x) saddle yang dipilih aturan `min`, pada koordinat citra penuh."""
    if not nuk.any():
        return None
    c, off = crop_pad(nuk)
    res = persistensi(c, prm["prune"])
    r = res["r_maks"]
    sg = [k for k in res["kej"] if k[1] >= prm["tau_br"] and k[2] >= prm["rho_br"] * r]
    sb = [p for p, _ in res["bertahan"] if p >= prm["tau_br"] and p >= prm["rho_br"] * r]
    if len(sb) >= 2 or not sg:
        return None
    k = min(sg, key=lambda k: k[0])
    y, x = np.unravel_index(k[4], c.shape)
    return (int(y) + off[0], int(x) + off[1])


# =============================================================================
# 2. PEKERJA
# =============================================================================
def kerja(arg):
    img, mpath, prm, daftar_L = arg
    out = {"img_name": img, "galat": ""}
    try:
        mask = baca_mask(mpath)
        sel, _ = komponen_sel(mask)
        # lubang diukur pada nukleus TANPA pengisian (basis aturan `tidak`)
        nuk0 = nukleus_beraturan(mask, sel, 0.0)
        lab_l, reg = ukur_lubang(nuk0)
        sp = saddle_terpilih(nuk0, prm)
        for r in reg:
            if sp is None:
                r["jarak_saddle"] = np.nan
            else:
                ys, xs = np.nonzero(lab_l == r["idx"])
                r["jarak_saddle"] = float(np.sqrt((ys - sp[0]) ** 2 + (xs - sp[1]) ** 2).min())
        out["lub_n"] = len(reg)
        out["lub_region"] = reg
        out["lub_lebar_maks"] = max([r["lebar_maks"] for r in reg], default=0.0)
        out["lub_piks"] = int(sum(r["luas"] for r in reg))
        for L in daftar_L:
            u = F.ukur_inti(nukleus_beraturan(mask, sel, L), prm)
            tag = f"{L:g}"
            out[f"br_{tag}"] = u["br"]
            out[f"st_{tag}"] = u["status"]
            out[f"rp_{tag}"] = u["r_pisah"]
            out[f"rl_{tag}"] = u["r_lobus"]
            out[f"nl_{tag}"] = u["n_lobus_kal"]
            out[f"lu_{tag}"] = u["luas_nukleus"]
    except Exception as e:
        out["galat"] = repr(e)
    return out


# =============================================================================
# 3. SKOR KALIBRASI (non-neutrofil saja)
# =============================================================================
def skor_lobus(df, kol_nl):
    """Makro-rata empat kelas nucleus_shape: fraksi sel yang jumlah lobusnya sesuai harapan.
    Dua sisi: menyaring terlalu agresif menjatuhkan bilobed/multilobed, terlalu longgar
    menjatuhkan round/indented."""
    per = {}
    for bentuk, (lo, hi) in HARAP_LOBUS.items():
        s = df[df.nucleus_shape == bentuk]
        if len(s):
            x = s[kol_nl].to_numpy(float)
            per[bentuk] = float(((x >= lo) & (x <= hi)).mean())
    return (float(np.mean(list(per.values()))) if per else np.nan), per


# =============================================================================
# 4. UJI SANITAS
# =============================================================================
def uji_sanitas(lap, prm):
    lap.tulis("## 0. Uji sanitas")
    lap.tulis()
    hasil = []

    def cek(nama, kondisi, detail):
        hasil.append(dict(uji=nama, lolos=bool(kondisi), detail=detail))

    S = (220, 220)
    yy, xx = np.ogrid[:S[0], :S[1]]

    # 1 ukuran bentuk pada geometri berjawaban analitis
    nuk = np.zeros(S, bool)
    nuk[(yy - 110) ** 2 + (xx - 110) ** 2 <= 60 ** 2] = True
    slot = np.zeros(S, bool)
    slot[109:111, 70:150] = True          # celah 2 x 80
    kompak = np.zeros(S, bool)
    kompak[(yy - 70) ** 2 + (xx - 110) ** 2 <= 9 ** 2] = True   # cakram r=9
    n1 = nuk & ~slot & ~kompak
    _, reg = ukur_lubang(n1)
    reg = sorted(reg, key=lambda r: r["lebar_maks"])
    cek("lebar_maks memisahkan celah tipis dari lubang kompak",
        len(reg) == 2 and abs(reg[0]["lebar_maks"] - 2.0) < 0.05 and abs(reg[1]["lebar_maks"] - 18.0) < 0.6
        and reg[0]["aspek"] > 8 and reg[1]["aspek"] < 1.2,
        f"n={len(reg)}, lebar {[round(r['lebar_maks'], 2) for r in reg]}, aspek {[round(r['aspek'], 2) for r in reg]}")

    # 2 L=0 identik `tidak`, L besar identik `semua`
    mk = np.zeros(S, np.uint8)
    mk[nuk] = 1
    mk[n1] = 2
    sel, _ = komponen_sel(mk)
    a0 = nukleus_beraturan(mk, sel, 0.0)
    a9 = nukleus_beraturan(mk, sel, 999.0)
    a2 = nukleus_beraturan(mk, sel, 3.0)
    cek("L=0 == `tidak`, L besar == `semua`",
        int(a0.sum()) == int(n1.sum()) and int(a9.sum()) == int((n1 | slot | kompak).sum())
        and int(a2.sum()) == int((n1 | slot).sum()),
        f"L0 {int(a0.sum())} (harap {int(n1.sum())}), L999 {int(a9.sum())} "
        f"(harap {int((n1 | slot | kompak).sum())}), L3 {int(a2.sum())} (harap {int((n1 | slot).sum())})")

    # 3 MEKANISME PEMBILANG (Bagian 11.6, sisi A): slot tipis memotong leher.
    #   Geometri dihitung lebih dulu, bukan ditebak (Pelajaran 5). Nilai eksak:
    #   lebar lubang 2,00 | r_pisah 2,000 -> 11,000 | r_lobus 30,017 tidak berubah.
    #   CATATAN: lubang harus memotong leher MELINTANG. Lubang yang sejajar leher hanya
    #   menyisakan dua lintasan paralel, dan yang lebih lebar tetap menentukan r_pisah.
    b = np.zeros(S, bool)
    b[(yy - 110) ** 2 + (xx - 70) ** 2 <= 30 ** 2] = True
    b[(yy - 110) ** 2 + (xx - 150) ** 2 <= 30 ** 2] = True
    b[100:121, 70:151] = True                      # leher tebal 21 baris
    slot = np.zeros(S, bool)
    slot[103:119, 108:110] = True                  # intrusi 2 px melintang, menyisakan sliver
    mk = np.zeros(S, np.uint8)
    mk[b] = 1
    mk[b & ~slot] = 2
    sel, _ = komponen_sel(mk)
    n0 = nukleus_beraturan(mk, sel, 0.0)
    _, reg_u = ukur_lubang(n0)
    u0 = F.ukur_inti(n0, prm)
    u9 = F.ukur_inti(nukleus_beraturan(mk, sel, 999.0), prm)
    cek("pembilang: slot tipis di leher menjatuhkan r_pisah 2,000 → 11,000, r_lobus tetap",
        abs(u0["r_pisah"] - 2.0) < 0.02 and abs(u9["r_pisah"] - 11.0) < 0.02
        and abs(u0["r_lobus"] - u9["r_lobus"]) < 0.02 and abs(reg_u[0]["lebar_maks"] - 2.0) < 0.02,
        f"lebar lubang {reg_u[0]['lebar_maks']:.2f}, r_pisah {u0['r_pisah']:.3f} → {u9['r_pisah']:.3f}, "
        f"r_lobus {u0['r_lobus']:.3f} → {u9['r_lobus']:.3f}, br {u0['br']:.4f} → {u9['br']:.4f}")

    # 4 MEKANISME PENYEBUT (Bagian 11.6, sisi B): lubang kompak di dalam lobus DOMINAN.
    #   Nilai eksak: lebar lubang 30,07 | r_pisah 5,000 -> 5,000 | r_lobus 25,020 -> 40,012.
    b2 = np.zeros(S, bool)
    b2[(yy - 110) ** 2 + (xx - 70) ** 2 <= 40 ** 2] = True
    b2[(yy - 110) ** 2 + (xx - 165) ** 2 <= 25 ** 2] = True
    b2[106:115, 70:166] = True                     # leher tipis 9 baris
    lub = np.zeros(S, bool)
    lub[(yy - 110) ** 2 + (xx - 70) ** 2 <= 15 ** 2] = True
    mk = np.zeros(S, np.uint8)
    mk[b2] = 1
    mk[b2 & ~lub] = 2
    sel, _ = komponen_sel(mk)
    n0b = nukleus_beraturan(mk, sel, 0.0)
    _, reg_v = ukur_lubang(n0b)
    v0 = F.ukur_inti(n0b, prm)
    v9 = F.ukur_inti(nukleus_beraturan(mk, sel, 999.0), prm)
    cek("penyebut: lubang kompak menaikkan r_lobus 25,020 → 40,012, r_pisah 5,000 tidak bergerak",
        abs(v0["r_pisah"] - v9["r_pisah"]) < 1e-9 and abs(v0["r_lobus"] - 25.020) < 0.02
        and abs(v9["r_lobus"] - 40.012) < 0.02 and abs(reg_v[0]["lebar_maks"] - 30.07) < 0.05,
        f"lebar lubang {reg_v[0]['lebar_maks']:.2f}, r_pisah {v0['r_pisah']:.3f} → {v9['r_pisah']:.3f}, "
        f"r_lobus {v0['r_lobus']:.3f} → {v9['r_lobus']:.3f}, br {v0['br']:.4f} → {v9['br']:.4f}")

    # 5 dua mekanisme berlawanan arah pada br, DAN lebar_maks memisahkan keduanya
    cek("kedua mekanisme melawan arah, dan lebar_maks memisahkannya (hipotesis M10 in miniatur)",
        (u9["br"] > u0["br"]) and (v9["br"] < v0["br"])
        and reg_u[0]["lebar_maks"] < 3 < 20 < reg_v[0]["lebar_maks"],
        f"pembilang br {u0['br']:.4f} → {u9['br']:.4f} (NAIK, lubang {reg_u[0]['lebar_maks']:.1f} px) | "
        f"penyebut br {v0['br']:.4f} → {v9['br']:.4f} (TURUN, lubang {reg_v[0]['lebar_maks']:.1f} px)")

    # 6 skor lobus dua sisi
    dd = pd.DataFrame(dict(nucleus_shape=["unsegmented-round"] * 10 + ["segmented-bilobed"] * 10,
                           nl_ketat=[1] * 10 + [1] * 10, nl_pas=[1] * 10 + [2] * 10))
    s_ketat, _ = skor_lobus(dd, "nl_ketat")
    s_pas, _ = skor_lobus(dd, "nl_pas")
    cek("skor lobus menghukum penyaringan agresif", abs(s_ketat - 0.5) < 1e-9 and abs(s_pas - 1.0) < 1e-9,
        f"skor ketat {s_ketat:.3f} (harap 0,5), skor pas {s_pas:.3f} (harap 1,0)")

    t = pd.DataFrame(hasil)
    lap.tabel(t)
    if not t.lolos.all():
        lap.peringatan("UJI SANITAS GAGAL — jalanan dihentikan sebelum menyentuh data asli.")
        return False
    lap.tulis(f"**Seluruh {len(t)} uji LOLOS.** Uji 3–5 memastikan skrip benar-benar mereproduksi "
              "kedua mekanisme Bagian 11.6, termasuk fakta bahwa keduanya menggerakkan `br` ke arah "
              "berlawanan — itulah yang membuat aturan biner tidak bisa memuaskan keduanya.")
    lap.tulis()
    return True


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tugas", choices=["sanitas", "semua"], default="semua")
    ap.add_argument("--dir-hasil", default="hasil_fase2d")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir-mask", default="pbcseg_final_v1")
    ap.add_argument("--dir-citra", default=None)
    ap.add_argument("--keluar", default="hasil_fase2f2")
    ap.add_argument("--pekerja", type=int, default=7)
    ap.add_argument("--sampel", type=int, default=0)
    ap.add_argument("--paksa", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tau-br", type=float, default=1.25)
    ap.add_argument("--rho-br", type=float, default=0.2)
    ap.add_argument("--tau-lobus", type=float, default=4.0)
    ap.add_argument("--rho-lobus", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.keluar, exist_ok=True)
    lap = Laporan(os.path.join(args.keluar, "LAPORAN_FASE2F2.md"))
    prm = dict(tau_br=args.tau_br, rho_br=args.rho_br, tau_lobus=args.tau_lobus,
               rho_lobus=args.rho_lobus, prune=0.05)
    rng = np.random.default_rng(args.seed)
    lap.tulis("# LAPORAN FASE 2F-2 — M10: aturan lubang berbasis bentuk")
    lap.tulis()
    lap.tulis(f"Dijalankan {time.strftime('%Y-%m-%d %H:%M')} | τ_br {args.tau_br}, ρ_br {args.rho_br}, "
              f"τ_lobus {args.tau_lobus}, ρ_lobus {args.rho_lobus} | keluarga aturan: isi lubang bila "
              f"lebar_maks ≤ L | L = 0 ≡ `tidak`, L = 999 ≡ `semua`")
    lap.tulis()
    try:
        if not uji_sanitas(lap, prm):
            sys.exit(1)
        if args.tugas == "sanitas":
            return

        path_csv = args.csv or os.path.join(args.dir_hasil, "bridge_ratio_2d.csv")
        df = pd.read_csv(path_csv)
        df["kunci"] = df.img_name.map(kunci_nama)
        idx_mask = indeks_berkas(args.dir_mask, ("*_mask.png",))
        ada = df.kunci.isin(idx_mask)
        lap.tulis(f"Data: `{path_csv}` — {len(df)} baris | mask terindeks {len(idx_mask)}, cocok {int(ada.sum())}")
        if not ada.any():
            lap.peringatan("Tidak satu pun mask ditemukan. Periksa --dir-mask.")
            return
        if (~ada).any():
            lap.peringatan(f"{int((~ada).sum())} baris CSV tanpa mask; dikeluarkan.")
        kerja_df = df[ada]
        if args.sampel:
            kerja_df = kerja_df.sample(min(args.sampel, len(kerja_df)), random_state=args.seed)
            lap.peringatan(f"Mode --sampel: {len(kerja_df)} sel. Angka TIDAK sah untuk dilaporkan.")
        lap.tulis()

        cache = os.path.join(args.keluar, "G1_lubang_per_sel.csv")
        cache_reg = os.path.join(args.keluar, "G2_lubang_per_region.csv")
        if os.path.exists(cache) and not args.paksa and not args.sampel:
            h = pd.read_csv(cache)
            reg = pd.read_csv(cache_reg)
            lap.tulis(f"- dimuat dari cache: {len(h)} sel, {len(reg)} region lubang")
        else:
            tugas = [(r.img_name, idx_mask[r.kunci], prm, DAFTAR_L) for r in kerja_df.itertuples()]
            hasil = jalankan_pool(kerja, tugas, args.pekerja, f"sapuan {len(DAFTAR_L)} nilai L", lap)
            baris_reg = []
            for o in hasil:
                for r in o.pop("lub_region", []) or []:
                    baris_reg.append(dict(img_name=o["img_name"], **r))
            h = pd.DataFrame(hasil)
            reg = pd.DataFrame(baris_reg)
            if not args.sampel:
                h.to_csv(cache, index=False)
                reg.to_csv(cache_reg, index=False)
        gal = h[h.galat.fillna("").astype(str).str.len() > 0]
        if len(gal):
            lap.peringatan(f"{len(gal)} sel gagal. Contoh: {gal.galat.iloc[0]}")
        h = h[h.galat.fillna("").astype(str).str.len() == 0]
        m = df.merge(h.drop(columns=["galat"]), on="img_name", how="inner")
        neu = m[m.label == "Neutrophil"].copy()
        non = m[m.label != "Neutrophil"].copy()

        # ---- uji reproduksi: L=0 harus == kolom CSV Fase 2D ----
        rep = float(np.isclose(m["br_0"].astype(float), m.bridge_ratio.astype(float),
                               atol=1e-6, equal_nan=True).mean())
        lap.tulis(f"- uji reproduksi: L = 0 mereproduksi `bridge_ratio` Fase 2D pada **{rep:.2%}** sel")
        if rep < 0.99:
            lap.peringatan(f"L=0 hanya mereproduksi {rep:.2%} nilai Fase 2D. Seluruh selisih antar-L "
                           "harus dibaca sebagai selisih internal skrip ini saja.")
        lap.tulis()

        # =====================================================================
        lap.tulis("## 1. LAPIS MEKANISTIK — apakah bentuk lubang bimodal? (tanpa label apa pun)")
        lap.tulis()
        lap.tulis(f"Region lubang terukur: **{len(reg)}** pada {int((h.lub_n > 0).sum())} sel.")
        lap.tulis()
        tb = []
        tepi = [0, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 12, 99]
        for lo, hi in zip(tepi[:-1], tepi[1:]):
            s = reg[(reg.lebar_maks > lo) & (reg.lebar_maks <= hi)]
            tb.append(dict(lebar_maks=f"({lo:g}; {hi:g}]", n=len(s), frak=len(s) / max(len(reg), 1),
                           luas_median=s.luas.median() if len(s) else np.nan,
                           aspek_median=s.aspek.median() if len(s) else np.nan,
                           jarak_saddle_median=s.jarak_saddle.median() if len(s) else np.nan))
        lap.tabel(pd.DataFrame(tb))
        q = reg.lebar_maks.quantile([.1, .25, .5, .75, .9, .95, .99]).to_dict()
        lap.tulis("Kuantil `lebar_maks`: " + ", ".join(f"q{int(k * 100)} {v:.2f}" for k, v in q.items()))
        # uji celah: adakah bin kosong / lembah di antara dua puncak?
        hist, _ = np.histogram(reg.lebar_maks.clip(0, 20), bins=40, range=(0, 20))
        puncak = [i for i in range(1, 39) if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > len(reg) * 0.01]
        lembah_kosong = int((hist[1:-1] == 0).sum())
        lap.tulis(f"Histogram 40 bin pada [0; 20] px: **{len(puncak)} puncak lokal**, "
                  f"**{lembah_kosong} bin kosong di tengah sebaran**.")
        bimodal = len(puncak) >= 2 and lembah_kosong > 0
        lap.tulis(f"**Putusan lapis mekanistik: sebaran {'BIMODAL' if bimodal else 'MENYATU (unimodal)'}.**")
        if not bimodal:
            lap.tulis()
            lap.tulis("> Tidak ada celah alami pada `lebar_maks`. Setiap nilai L yang dipilih akan menjadi "
                      "potongan sembarang di tengah sebaran yang kontinu — persis jenis parameter yang "
                      "Bagian 19 melarang ditambahkan. Lapis 2 dan 3 tetap dijalankan sebagai bukti, "
                      "tetapi **M10 sudah gagal di pintu pertama**.")
        lap.tulis()

        # =====================================================================
        lap.tulis("## 2. LAPIS KALIBRASI INDEPENDEN — skor lobus pada sel NON-NEUTROFIL")
        lap.tulis()
        lap.tulis(f"n = {len(non)} sel non-neutrofil. Tidak satu pun masuk kelompok A atau B. "
                  "Harapan: round 1, indented 1, bilobed 2, multilobed ≥3. `irregular` dan `band` dikecualikan.")
        lap.tulis()
        tb = []
        for L in DAFTAR_L:
            tag = f"{L:g}"
            s, per = skor_lobus(non, f"nl_{tag}")
            tb.append(dict(L=L, skor_makro=s, **{k.split("-")[-1]: v for k, v in per.items()},
                           frak_lubang_diisi=float((reg.lebar_maks <= L).mean())))
        tkal = pd.DataFrame(tb)
        lap.tabel(tkal, 4)
        best = tkal.loc[tkal.skor_makro.idxmax()]
        rentang = tkal.skor_makro.max() - tkal.skor_makro.min()
        lap.tulis(f"Skor terbaik L = {best.L:g} ({best.skor_makro:.4f}); rentang seluruh permukaan "
                  f"**{rentang:.4f}**. Laporkan sebagai wilayah, bukan argmax.")
        if rentang < 0.01:
            lap.peringatan(f"Permukaan kalibrasi datar (rentang {rentang:.4f}). Argmax di sini mengejar "
                           "derau — persis kesalahan Fase 2A-rev. Tidak boleh dipakai memilih L.")
        lap.tulis()

        # =====================================================================
        lap.tulis("## 3. UJI DUA SISI — perbaiki sel A tanpa merusak sel B")
        lap.tulis()
        tb = []
        for L in DAFTAR_L:
            tag = f"{L:g}"
            r = {"L": L}
            for nm, daftar, arah in (("A_dirusak_tidak", A_DIRUSAK, "band"), ("B_dirusak_tidak", B_DIRUSAK, "seg")):
                s = neu[neu.kunci.isin(daftar)]
                x = s[f"br_{tag}"].astype(float)
                r[nm + "_n"] = len(s)
                r[nm + "_benar"] = int((x >= SEPERTIGA).sum()) if arah == "band" else int((x < SEPERTIGA).sum())
            A = neu[neu.kelompok.astype(str).str.startswith("A")][f"br_{tag}"].astype(float)
            B = neu[neu.kelompok.astype(str).str.startswith("B")][f"br_{tag}"].astype(float)
            r["galat_A"] = int((A < SEPERTIGA).sum())
            r["galat_B"] = int((B >= SEPERTIGA).sum())
            r["galat_total"] = r["galat_A"] + r["galat_B"]
            tb.append(r)
        tdua = pd.DataFrame(tb)
        lap.tabel(tdua)
        n_A_ket = int(neu.kunci.isin(A_DIRUSAK).sum())
        n_B_ket = int(neu.kunci.isin(B_DIRUSAK).sum())
        terukur = n_A_ket >= len(A_DIRUSAK) // 2 and n_B_ket >= 2
        lap.tulis(f"Sel acuan yang ditemukan: **{n_A_ket} dari {len(A_DIRUSAK)}** sisi A, "
                  f"**{n_B_ket} dari {len(B_DIRUSAK)}** sisi B.")
        if not terukur:
            lap.peringatan(f"Sel acuan Bagian 11.6 tidak ditemukan di data ini ({n_A_ket} A, {n_B_ket} B). "
                           "Uji dua sisi TIDAK TERUKUR — dan itu berbeda dari 'gagal'. Periksa pencocokan "
                           "nama (`kunci`) sebelum membaca putusan M10.")
        lap.tulis()
        lap.tulis("Kolom `*_benar` adalah jumlah sel yang divonis BENAR pada L itu. Aturan yang berhasil "
                  "harus menaikkan `A_dirusak_tidak_benar` ke arah 14 **tanpa** menurunkan "
                  "`B_dirusak_tidak_benar` dari 4. `galat_A` dan `galat_B` disertakan sebagai konteks, "
                  "**bukan** sebagai kriteria pemilihan.")
        lap.tulis()
        ada_menang = bool(((tdua.A_dirusak_tidak_benar > tdua.A_dirusak_tidak_benar.iloc[0]) &
                           (tdua.B_dirusak_tidak_benar >= tdua.B_dirusak_tidak_benar.iloc[0])).any())
        lap.tulis(f"**Ada L yang memperbaiki sisi A tanpa merusak sisi B: "
                  f"{'YA' if ada_menang else 'TIDAK TERUKUR' if not terukur else 'TIDAK'}.**")
        lap.tulis()

        # =====================================================================
        lap.tulis("## 4. PUTUSAN M10")
        lap.tulis()
        if not terukur:
            lap.tulis("**PUTUSAN DITANGGUHKAN.** Uji dua sisi tidak terukur karena sel acuan Bagian 11.6 "
                      "tidak ditemukan, jadi M10 tidak boleh ditutup ke arah mana pun dari jalanan ini. "
                      "Lapis 1 dan 2 tetap sah dan tercatat di atas.")
            lap.tulis()
            lolos = None
        else:
            lolos = bimodal and ada_menang and rentang >= 0.01
        if lolos is None:
            pass
        elif lolos:
            lap.tulis("**M10 LOLOS ketiga lapis.** Aturan bentuk boleh dipertimbangkan sebagai pengganti "
                      "`tidak`. Jalankan ulang Fase 2D/2E pada L terpilih sebelum melaporkan angka apa pun.")
        else:
            sebab = []
            if not bimodal:
                sebab.append("sebaran `lebar_maks` menyatu tanpa celah alami (lapis mekanistik)")
            if rentang < 0.01:
                sebab.append(f"permukaan kalibrasi datar, rentang {rentang:.4f} (lapis kalibrasi)")
            if not ada_menang:
                sebab.append("tidak ada L yang memperbaiki sisi A tanpa merusak sisi B (uji dua sisi)")
            lap.tulis("**M10 DITUTUP SEBAGAI NEGATIF.** Sebab: " + "; ".join(sebab) + ".")
            lap.tulis()
            lap.tulis("`tidak` tetap aturan final, dan sekarang ia final karena **bukti**, bukan karena "
                      "tie-break diam-diam yang memilih perilaku lama (Pelajaran 7).")
        lap.tulis()

        # =====================================================================
        lap.tulis("## 5. ALTERNATIF TANPA PARAMETER BARU — ketidaksepakatan aturan sebagai ketidakpastian")
        lap.tulis()
        neu["vonis_0"] = neu.br_0.astype(float) >= SEPERTIGA
        neu["vonis_999"] = neu["br_999"].astype(float) >= SEPERTIGA
        neu["peka"] = neu.vonis_0 != neu.vonis_999
        neu["kel"] = neu.kelompok.astype(str).str[0]
        AB = neu[neu.kel.isin(("A", "B"))].copy()
        AB["benar"] = np.where(AB.kel == "A", AB.vonis_0, ~AB.vonis_0)
        n_peka = int(neu.peka.sum())
        lap.tulis(f"Sel yang vonisnya BERUBAH antara `tidak` dan `semua`: **{n_peka}** dari {len(neu)} "
                  f"neutrofil ({n_peka / max(len(neu), 1):.2%}).")
        lap.tulis()
        tb = []
        for g in sorted(neu.kelompok.dropna().unique()):
            s = neu[neu.kelompok == g]
            tb.append(dict(kelompok=g, n=len(s), peka=int(s.peka.sum()), frak_peka=float(s.peka.mean())))
        tp = pd.DataFrame(tb)
        tp["pengayaan"] = tp.frak_peka / (n_peka / max(len(neu), 1))
        lap.tabel(tp)
        pekaAB = AB[AB.peka]
        lap.tulis(f"Akurasi pada sel peka-aturan: **{pekaAB.benar.mean():.4f}** (n = {len(pekaAB)}); "
                  f"pada sel tidak-peka: **{AB[~AB.peka].benar.mean():.4f}** (n = {int((~AB.peka).sum())}).")
        lap.tulis()
        lap.tulis("Perbandingan kriteria penundaan pada cakupan yang sama:")
        lap.tulis()
        k_jarak = (AB.br_0.astype(float) - SEPERTIGA).abs().to_numpy()
        tb = []
        for c in (1.0, 0.99, 0.975, 0.95, 0.90, 0.85):
            ns = int(round(len(AB) * c))
            # (a) jarak ke ambang
            u1 = np.argsort(-k_jarak, kind="stable")[:ns]
            # (b) peka-aturan lebih dulu ditunda, lalu jarak ke ambang
            ku = np.lexsort((-k_jarak, AB.peka.to_numpy()))[:ns]
            tb.append(dict(cakupan=c, n_simpan=ns,
                           akurasi_jarak=float(AB.benar.to_numpy()[u1].mean()),
                           akurasi_peka_lalu_jarak=float(AB.benar.to_numpy()[ku].mean())))
        tdef = pd.DataFrame(tb)
        tdef["selisih"] = tdef.akurasi_peka_lalu_jarak - tdef.akurasi_jarak
        lap.tabel(tdef, 5)
        menang = bool((tdef.selisih > 0.001).any())
        lap.tulis(f"**Ketidaksepakatan aturan mengungguli |br − 1/3| sebagai kriteria penundaan: "
                  f"{'YA' if menang else 'TIDAK'}.**")
        if menang:
            lap.tulis("Artinya ada sinyal ketidakpastian tambahan yang bisa dipakai **tanpa satu pun "
                      "parameter baru di pipeline** — kedua aturan sudah ada sejak Fase 2D, dan yang "
                      "dipakai hanya fakta bahwa keduanya tidak sepakat. Ini jalur keluar dari ketegangan "
                      "Bagian 12.4 yang tidak menuntut aturan biner mana pun menang.")
        else:
            lap.tulis("Ketidaksepakatan aturan tidak membawa informasi melebihi jarak ke ambang. "
                      "Jalur ini ditutup juga, dan mesin deferral Fase 2D tetap seperti apa adanya.")
        lap.tulis()

        # ---- simpan ----
        m.drop(columns=["kunci"]).to_csv(os.path.join(args.keluar, "G3_sapuan_L_per_sel.csv"), index=False)
        neu.drop(columns=["kunci"]).to_csv(os.path.join(args.keluar, "G4_neutrofil_peka_aturan.csv"), index=False)
        tkal.to_csv(os.path.join(args.keluar, "G5_kalibrasi_lobus.csv"), index=False)
        tdua.to_csv(os.path.join(args.keluar, "G6_uji_dua_sisi.csv"), index=False)
        tdef.to_csv(os.path.join(args.keluar, "G7_deferral_banding.csv"), index=False)

        # ---- konteks AUC, dilaporkan PALING AKHIR dan tidak dipakai memilih apa pun ----
        lap.tulis("## 6. Konteks AUC (dilaporkan terakhir, TIDAK dipakai memilih L)")
        lap.tulis()
        tb = []
        for L in DAFTAR_L:
            tag = f"{L:g}"
            d = neu[np.isfinite(neu[f"br_{tag}"].astype(float))]
            A = d[d.kel == "A"][f"br_{tag}"].to_numpy(float)
            B = d[d.kel == "B"][f"br_{tag}"].to_numpy(float)
            C = d[d.kel == "C"][f"br_{tag}"].to_numpy(float)
            mA, mB = np.median(A), np.median(B)
            tb.append(dict(L=L, auc=auc(A, B), acc_13=akurasi(A, B, SEPERTIGA),
                           youden=youden(A, B)[0],
                           posisi_C=float((mA - np.median(C)) / (mA - mB)) if mA != mB else np.nan))
        lap.tabel(pd.DataFrame(tb), 5)
        lap.tulis("Tabel ini hanya konteks. Bila L dengan AUC tertinggi berbeda dari putusan Bagian 4, "
                  "**putusan Bagian 4 yang berlaku** — itulah inti Pelajaran 3.")
    finally:
        lap.simpan()
        print(f"\nLaporan: {lap.path}")


if __name__ == "__main__":
    main()
