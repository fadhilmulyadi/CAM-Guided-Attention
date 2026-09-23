#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kemas_hasil.py
============================================================================
Gabungkan seluruh keluaran teks Fase 2D dan 2E jadi SATU berkas Markdown,
supaya muat di batas 20 lampiran per pesan.

Setelah menjalankan ini, yang perlu dikirim hanya:
    1. KIRIM_2D_2E.md              <- berkas hasil skrip ini
    2..6. kelima montase PNG dari hasil_fase2e

Gambar PNG tidak bisa digabung (harus dilihat aslinya), jadi tetap dikirim
terpisah. Cache .pkl.gz dan bridge_ratio_2d.csv sengaja TIDAK disertakan.

PEMAKAIAN
---------
    python kemas_hasil.py
    python kemas_hasil.py --dir-2d hasil_fase2d --dir-2e hasil_fase2e \
                          --log log_2d.txt log_2e.txt --out KIRIM_2D_2E.md
============================================================================
"""

import argparse
import io
from pathlib import Path

import pandas as pd

# Berkas yang disertakan, berurut. (nama, wajib?)
BERKAS_2D = [
    ("LAPORAN_FASE2D.md", True),
    ("D1_diagnostik_praproses.csv", True),
    ("D2_ablasi_lubang.csv", True),
    ("D2_berpasangan.csv", True),
    ("D2_asal_usul_leher.csv", True),
    ("D3_kalibrasi_grid.csv", True),
    ("D4_ambang_terpilih.csv", True),
    ("D4_kurva_youden.csv", False),
    ("D5_diagnostik_jarak.csv", True),
    ("D5_ekor_B.csv", True),
    ("D6_deferral.csv", True),
    ("D0_uji_sanitas.csv", False),
]
BERKAS_2E = [
    ("ANGKA_UNTUK_BAGIAN14.md", True),
    ("E1_desil_keyakinan.csv", True),
    ("E1_kurva_lantai.csv", True),
    ("E2_yakin_salah.csv", True),
    ("E2_atribut_yakin_salah.csv", False),
    ("E4_galat_per_aturan.csv", False),
    ("E4_sel_berpindah.csv", False),
    ("E5_banding_ambang.csv", False),
]
# Kolom yang dibuang dari tabel besar supaya berkas tidak membengkak.
KOLOM_BUANG = ["path", "cell_size", "cell_shape", "nuclear_cytoplasmic_ratio",
               "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture",
               "cytoplasm_colour", "granule_type", "granule_colour",
               "granularity"]
MAKS_BARIS = 200


def csv_ke_md(path, maks=MAKS_BARIS):
    df = pd.read_csv(path)
    n0, k0 = len(df), len(df.columns)
    buang = [c for c in KOLOM_BUANG if c in df.columns]
    if buang and k0 > 12:
        df = df.drop(columns=buang)
    catatan = ""
    if len(df) > maks:
        langkah = max(1, len(df) // maks)
        df = df.iloc[::langkah].head(maks)
        catatan = ("\n> Disubsampel tiap %d baris: %d dari %d baris ditampilkan.\n"
                   % (langkah, len(df), n0))
    if buang and k0 > 12:
        catatan += ("\n> Kolom atribut konstan dibuang (%s).\n" % ", ".join(buang))
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else ("%.6g" % v))
        else:
            df[c] = df[c].astype(str)
    kol = [str(c) for c in df.columns]
    baris = ["| " + " | ".join(kol) + " |",
             "|" + "|".join(["---"] * len(kol)) + "|"]
    for _, r in df.iterrows():
        baris.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return ("%d baris x %d kolom%s\n\n" % (n0, k0, catatan)) + "\n".join(baris)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-2d", default="hasil_fase2d")
    ap.add_argument("--dir-2e", default="hasil_fase2e")
    ap.add_argument("--log", nargs="*", default=["log_2d.txt", "log_2e.txt"])
    ap.add_argument("--out", default="KIRIM_2D_2E.md")
    ap.add_argument("--maks-baris", type=int, default=MAKS_BARIS)
    args = ap.parse_args()

    L, hilang, masuk = [], [], []
    L.append("# PAKET HASIL FASE 2D + 2E\n")
    L.append("Berkas ini digabung otomatis oleh `kemas_hasil.py`. Setiap bagian "
             "di bawah adalah isi satu berkas keluaran, apa adanya.\n")

    for judul, dirnya, daftar in (("FASE 2D", args.dir_2d, BERKAS_2D),
                                  ("FASE 2E", args.dir_2e, BERKAS_2E)):
        L.append("\n\n" + "=" * 70 + "\n# %s\n" % judul)
        d = Path(dirnya)
        if not d.exists():
            L.append("\n**Folder `%s` tidak ditemukan.**\n" % dirnya)
            hilang.append(dirnya + "/ (folder)")
            continue
        for nama, wajib in daftar:
            p = d / nama
            if not p.exists():
                if wajib:
                    hilang.append("%s/%s" % (dirnya, nama))
                    L.append("\n## %s\n\n**TIDAK DITEMUKAN.**\n" % nama)
                continue
            L.append("\n## %s\n" % nama)
            try:
                if p.suffix.lower() == ".csv":
                    L.append(csv_ke_md(p, args.maks_baris))
                else:
                    L.append(p.read_text(encoding="utf-8", errors="replace"))
                masuk.append("%s/%s" % (dirnya, nama))
            except Exception as e:
                L.append("**Gagal dibaca: %s**" % e)
                hilang.append("%s/%s (gagal dibaca)" % (dirnya, nama))

    logs = [Path(x) for x in (args.log or []) if Path(x).exists()]
    if logs:
        L.append("\n\n" + "=" * 70 + "\n# KELUARAN KONSOL\n")
        for p in logs:
            teks = p.read_text(encoding="utf-8", errors="replace")
            L.append("\n## %s\n\n```\n%s\n```\n" % (p.name, teks.strip()))
            masuk.append(str(p))
    else:
        L.append("\n\n> Berkas log konsol tidak ditemukan. Kalau ada, jalankan "
                 "ulang dengan `--log <berkas>`; catatan penting seperti aturan "
                 "lubang yang terpilih hanya muncul di konsol.\n")

    out = Path(args.out)
    out.write_text("\n".join(L), encoding="utf-8")
    ukur = out.stat().st_size
    print("=" * 66)
    print("PAKET DITULIS: %s  (%.2f MB, %d berkas digabung)"
          % (out, ukur / 1e6, len(masuk)))
    print("=" * 66)
    for m in masuk:
        print("  + %s" % m)
    if hilang:
        print("\n  BERKAS WAJIB YANG HILANG:")
        for h in hilang:
            print("  - %s" % h)
        print("  Periksa apakah Fase 2D/2E sudah dijalankan sampai selesai.")
    if ukur > 3e6:
        print("\n  PERINGATAN: paket > 3 MB. Turunkan --maks-baris (mis. 80).")

    d2e = Path(args.dir_2e)
    png = sorted(d2e.glob("montase_E_*.png")) if d2e.exists() else []
    print("\nYANG PERLU DIKIRIM (%d berkas, batas 20):" % (1 + len(png)))
    print("  1. %s" % out.name)
    if png:
        for i, p in enumerate(png, start=2):
            print("  %d. %s/%s  (%.2f MB)" % (i, args.dir_2e, p.name,
                                              p.stat().st_size / 1e6))
    else:
        print("  2-6. montase_E_*.png dari %s  (belum ada)" % args.dir_2e)
    print("\nJANGAN kirim: cache_p2d_*.pkl.gz, bridge_ratio_2d.csv, "
          "D1_lubang_per_*.csv, folder _sanitas/")


if __name__ == "__main__":
    main()
