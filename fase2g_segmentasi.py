#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fase2g_segmentasi.py — Fase 2G bagian 1: menyegmentasi folder `ig` dengan wbcsegmentor.

DIJALANKAN DI LINGKUNGAN `wbcsegmentor` (butuh GPU), BUKAN di lingkungan skrip 2D/2E/2F.

EMPAT KEPUTUSAN RANCANGAN YANG DIAMBIL DARI MEMBACA KODE REPO
  1. Citra `ig` berukuran 360 x 363 (PBC mentah), sedangkan model memaksa keluaran
     360 x 360 (`out_resolution=360`). Memasukkan 360x363 apa adanya menghasilkan mask
     yang TIDAK SEJAJAR dengan citranya. Karena itu citra di-crop dulu ke 360x360
     memakai konvensi `_ccrop` WBCAtt+ — dan offset crop itu DIUKUR, bukan ditebak,
     dengan membandingkan PBC mentah terhadap ccrop WBCAtt+ pada sel yang sama.
  2. Dipakai wrapper dari kode LATIH (`out_resolution=360`, forward berbatch), yaitu
     yang dipakai `results_pbc_eval.ipynb` dan menghasilkan mIoU 93,67 — bukan wrapper
     "in the wild" yang menangani ukuran sembarang. Masukan kita berukuran tetap dan
     persis sama dengan domain latih, jadi jalur eval yang benar.
  3. Model TIDAK di-`.bfloat16()`; hanya autocast bfloat16 dipakai saat inferensi.
     Itu persis jalur `results_pbc_eval.ipynb`. Wrapper "in the wild" memakai
     `.bfloat16()` penuh dan bukan jalur yang angkanya dipublikasikan.
  4. BNE dan SNE JUGA disegmentasi ulang. Bila PMY/MY/MMY memakai mask prediksi
     sementara BNE/SNE memakai mask anotasi, tren lima tahap jadi campuran dua sumbu
     — perubahan sumber mask dan perubahan tahap pematangan tidak bisa dipisahkan.
     Seluruh lima tahap HARUS memakai mask dari sumber yang sama.

GERBANG WAJIB
  `--tugas verifikasi` mereproduksi mIoU 93,67 pada test split memakai mask anotasi
  sebagai acuan. Bila meleset jauh, setup salah dan tidak ada gunanya menyentuh `ig`.

URUTAN
  python fase2g_segmentasi.py --tugas cek        --dir-pbc <PBC_asli> --dir-ccrop pbcseg_final_v1
  python fase2g_segmentasi.py --tugas verifikasi --dir-ccrop pbcseg_final_v1 --csv <attr csv>
  python fase2g_segmentasi.py --tugas pbc        --dir-ccrop pbcseg_final_v1 --csv <attr csv>
  python fase2g_segmentasi.py --tugas ig         --dir-pbc <PBC_asli>

KELUARAN
  <--keluar>/mask_pred_pbc/<kunci>_ccrop_mask.png   mask prediksi untuk sel WBCAtt+
  <--keluar>/mask_pred_ig/<kunci>_ccrop_mask.png    mask prediksi untuk sel folder ig
  <--keluar>/mask_pred_ig/<kunci>_ccrop.jpg         citra ig yang sudah di-crop 360x360
  <--keluar>/H*_*.csv                               log, daftar, dan hasil verifikasi
Keluarannya sengaja memakai konvensi nama `_ccrop_mask.png` supaya skrip geometri
Fase 2F bisa dijalankan di atasnya TANPA satu baris pun diubah.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_NAME = "facebook/mask2former-swin-tiny-ade-semantic"
KELAS = ["bg", "cyto", "nucleus", "platelets", "rwbc", "vacuoles"]
MIOU_TERBIT = 93.67          # angka README repo, dipakai sebagai gerbang
IOU_TERBIT = {"cyto": 96.67, "nucleus": 98.14, "platelets": 92.91, "rwbc": 99.43, "vacuoles": 81.20}
PREFIX_IG = ("PMY", "MY", "MMY", "IG")
PREFIX_NEU = ("BNE", "SNE", "NEUTROPHIL")


def log(s=""):
    print(s, flush=True)


# =============================================================================
# 1. WRAPPER — salinan verbatim dari explore_m2f_finetune_v2.py
# =============================================================================
def bangun_wrapper(dir_model):
    """Coba impor dari repo; kalau gagal (mis. tensorboard tidak terpasang), pakai
    salinan verbatim. Sumber yang dipakai selalu dicetak."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import Mask2FormerForUniversalSegmentation

    if dir_model and dir_model not in sys.path:
        sys.path.insert(0, dir_model)
    for modul in ("explore_m2f_finetune", "explore_m2f_finetune_v2"):
        try:
            m = __import__(modul)
            log(f"  wrapper: diimpor dari repo ({modul}.py)")
            return m.Mask2FormerWrapper
        except Exception as e:
            log(f"  wrapper: impor {modul} gagal ({type(e).__name__}); mencoba berikutnya")

    class Mask2FormerWrapper(nn.Module):
        """Salinan verbatim dari explore_m2f_finetune_v2.py."""

        def __init__(self, model_name, num_classes, out_resolution):
            super().__init__()
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
            self.out_resolution = out_resolution
            self.num_classes = num_classes

        def forward(self, images):
            outputs = self.model(pixel_values=images)
            cls_logits = outputs.class_queries_logits
            mask_logits = outputs.masks_queries_logits
            cls_probs = F.softmax(cls_logits, dim=-1)
            mask_probs = torch.sigmoid(mask_logits)
            b, q, h_small, w_small = mask_probs.shape
            mask_probs_flat = mask_probs.view(b, q, h_small * w_small)
            semantic_map = torch.bmm(cls_probs[:, :, :self.num_classes].transpose(1, 2), mask_probs_flat)
            semantic_map = semantic_map.view(b, self.num_classes, h_small, w_small)
            return F.interpolate(semantic_map, size=(self.out_resolution, self.out_resolution),
                                 mode="bilinear", align_corners=False)

    log("  wrapper: memakai salinan verbatim di dalam skrip ini")
    return Mask2FormerWrapper


def muat_model(args):
    import torch
    from torchvision.transforms import v2
    Wrapper = bangun_wrapper(args.dir_model)
    ckpt = args.ckpt or os.path.join(args.dir_model, "model_epoch=050.ckpt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint tidak ditemukan: {ckpt}\n"
                         "Unduh dari https://huggingface.co/apple2373/wbcsegmentor_m2f_tiny")
    log(f"  checkpoint: {ckpt}")
    seg = Wrapper(MODEL_NAME, num_classes=6, out_resolution=360).to("cuda")
    cp = torch.load(ckpt, weights_only=False)
    log(f"  load_state_dict: {seg.load_state_dict(cp['model_state_dict'])}")
    seg = seg.cuda().eval()          # TANPA .bfloat16() — jalur results_pbc_eval.ipynb
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(1024, 1024), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return seg, transform


# =============================================================================
# 2. CITRA
# =============================================================================
def kunci_nama(p):
    b = os.path.splitext(os.path.basename(str(p)))[0]
    for akhiran in ("_mask", "_ccrop"):
        if b.endswith(akhiran):
            b = b[: -len(akhiran)]
    return b


def potong_ccrop(im, dy, dx=0, sisi=360):
    """Crop ke sisi x sisi memakai offset yang SUDAH DIUKUR."""
    w, h = im.size
    if (w, h) == (sisi, sisi):
        return im
    return im.crop((dx, dy, dx + sisi, dy + sisi))


def ukur_offset_ccrop(dir_pbc, dir_ccrop, n=25):
    """Tentukan offset crop WBCAtt+ secara empiris: cari (dy,dx) yang membuat PBC mentah
    identik dengan ccrop. Tidak menebak."""
    ccrops = {}
    for f in glob.glob(os.path.join(dir_ccrop, "**", "*.jpg"), recursive=True):
        if "_mask" in os.path.basename(f):
            continue
        ccrops.setdefault(kunci_nama(f), f)
    mentah = {}
    for f in glob.glob(os.path.join(dir_pbc, "**", "*.jpg"), recursive=True):
        mentah.setdefault(kunci_nama(f), f)
    sama = [k for k in ccrops if k in mentah]
    log(f"  ccrop terindeks {len(ccrops)}, PBC mentah terindeks {len(mentah)}, beririsan {len(sama)}")
    if not sama:
        return None, None, None
    suara, rinci, per_offset = {}, [], {}
    for k in sorted(sama)[:n]:
        a = np.array(Image.open(ccrops[k]).convert("RGB"))
        b = np.array(Image.open(mentah[k]).convert("RGB"))
        H, W = a.shape[:2]
        skor_semua = {}
        for dy in range(max(b.shape[0] - H, 0) + 1):
            for dx in range(max(b.shape[1] - W, 0) + 1):
                skor_semua[(dy, dx)] = float(
                    np.abs(b[dy:dy + H, dx:dx + W].astype(np.int16) - a.astype(np.int16)).mean())
        terbaik = min(skor_semua, key=skor_semua.get)
        suara[terbaik] = suara.get(terbaik, 0) + 1
        for o, s in skor_semua.items():
            per_offset.setdefault(o, []).append(s)
        rinci.append((k, terbaik, skor_semua[terbaik], b.shape[:2], a.shape[:2]))
    return suara, rinci, per_offset


def daftar_ig(dir_pbc):
    out = []
    for f in glob.glob(os.path.join(dir_pbc, "**", "*.jpg"), recursive=True):
        nm = os.path.basename(f)
        pre = nm.split("_")[0].upper()
        if pre in PREFIX_IG:
            out.append((kunci_nama(f), pre, f))
    return sorted(set(out))


# =============================================================================
# 3. INFERENSI BERBATCH DENGAN CHECKPOINT
# =============================================================================
def inferensi(seg, transform, tugas, dir_out, batch, dy, dx, simpan_citra=False):
    """tugas: daftar (kunci, path). Melewati berkas yang sudah ada -> bisa dilanjutkan."""
    import torch
    os.makedirs(dir_out, exist_ok=True)
    sisa = [(k, p) for k, p in tugas if not os.path.exists(os.path.join(dir_out, f"{k}_ccrop_mask.png"))]
    log(f"  {len(tugas)} citra, {len(tugas) - len(sisa)} sudah ada, {len(sisa)} dikerjakan")
    t0 = time.time()
    gagal = []
    for i in range(0, len(sisa), batch):
        potongan = sisa[i:i + batch]
        tensors, kunci_ok = [], []
        for k, p in potongan:
            try:
                im = potong_ccrop(Image.open(p).convert("RGB"), dy, dx)
                tensors.append(transform(im))
                kunci_ok.append((k, im))
            except Exception as e:
                gagal.append((k, repr(e)))
        if not tensors:
            continue
        x = torch.stack(tensors).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = seg(x).argmax(dim=1).cpu().numpy().astype(np.uint8)
        for (k, im), pm in zip(kunci_ok, pred):
            Image.fromarray(pm).save(os.path.join(dir_out, f"{k}_ccrop_mask.png"))
            if simpan_citra:
                im.save(os.path.join(dir_out, f"{k}_ccrop.jpg"), quality=95)
        n = i + len(potongan)
        if n % (batch * 20) < batch or n >= len(sisa):
            dt = time.time() - t0
            log(f"    {n}/{len(sisa)} ({dt:.0f} s, sisa ~{dt / max(n, 1) * (len(sisa) - n):.0f} s)")
    if gagal:
        log(f"  GAGAL {len(gagal)} citra. Contoh: {gagal[0]}")
    return gagal


# =============================================================================
# 4. TUGAS
# =============================================================================
def tugas_cek(args):
    log("## Pemeriksaan lingkungan dan konvensi crop\n")
    ok = True
    try:
        import torch
        log(f"- torch {torch.__version__} | CUDA tersedia: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"  GPU: {torch.cuda.get_device_name(0)}, "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            log("  PERINGATAN: CUDA tidak tersedia. Inferensi akan sangat lambat atau gagal.")
            ok = False
    except Exception as e:
        log(f"- torch TIDAK terpasang: {e}")
        ok = False
    try:
        import transformers
        log(f"- transformers {transformers.__version__}")
    except Exception as e:
        log(f"- transformers TIDAK terpasang: {e}")
        ok = False
    ckpt = args.ckpt or os.path.join(args.dir_model, "model_epoch=050.ckpt")
    log(f"- checkpoint {'ADA' if os.path.exists(ckpt) else 'TIDAK ADA'}: {ckpt}")
    ok = ok and os.path.exists(ckpt)

    log("\n## Offset crop `_ccrop` — diukur, bukan ditebak\n")
    if not (args.dir_pbc and args.dir_ccrop):
        log("  --dir-pbc dan --dir-ccrop diperlukan untuk pengukuran ini.")
        return
    suara, rinci, per_offset = ukur_offset_ccrop(args.dir_pbc, args.dir_ccrop, args.n_cek)
    if not suara:
        log("  Tidak ada sel yang bisa dibandingkan. Periksa kedua folder.")
        return
    log(f"  {'offset':<16}{'galat piksel rata-rata':>24}{'menang pada':>14}")
    urut = sorted(per_offset.items(), key=lambda x: np.mean(x[1]))
    for o, s in urut:
        log(f"  dy={o[0]}, dx={o[1]:<10}{np.mean(s):>24.3f}{suara.get(o, 0):>10} sel")
    menang = max(suara, key=suara.get)
    sepakat = suara[menang] / sum(suara.values())
    g_menang = float(np.mean(per_offset[menang]))
    g_kedua = float(np.mean(urut[1][1])) if len(urut) > 1 else float("inf")
    margin = g_kedua / g_menang if g_menang > 0 else float("inf")
    log(f"\n  Offset terpilih: **dy={menang[0]}, dx={menang[1]}** | kesepakatan {sepakat:.0%} | "
        f"galat {g_menang:.3f} versus pesaing terdekat {g_kedua:.3f} (**{margin:.1f}x lebih baik**)")
    log("  Galat tidak nol itu wajar: ccrop WBCAtt+ adalah JPEG yang dikompres ulang. "
        "Yang menentukan adalah jarak ke offset pesaing, bukan nilai mutlaknya.")
    if margin < 1.5:
        log("  PERINGATAN: offset pemenang nyaris seimbang dengan pesaingnya. Kemungkinan ccrop "
            "bukan crop murni (ada resize). JANGAN lanjut sebelum ini dipahami.")
    if sepakat < 0.95:
        log("  PERINGATAN: offset tidak konsisten antar sel. JANGAN lanjut sebelum ini dipahami.")
    with open(os.path.join(args.keluar, "H0_offset_ccrop.json"), "w") as f:
        json.dump({"dy": menang[0], "dx": menang[1], "kesepakatan": sepakat,
                   "galat_piksel": g_menang, "galat_pesaing": g_kedua, "margin": margin}, f, indent=2)
    log(f"\n  Tersimpan di {os.path.join(args.keluar, 'H0_offset_ccrop.json')}. "
        f"Pakai --dy {menang[0]} --dx {menang[1]} pada tugas berikutnya.")
    log(f"\nSiap lanjut: {'YA' if ok else 'TIDAK — perbaiki yang di atas dulu'}")


def tugas_verifikasi(args):
    """GERBANG: reproduksi mIoU 93,67 pada test split. Tanpa ini, sisanya tidak bermakna."""
    import pandas as pd
    import torch
    log("## GERBANG — reproduksi mIoU terbitan pada test split\n")
    df = pd.read_csv(args.csv)
    df = df[df.split == "test"]
    if args.n_verif:
        df = df.sample(min(args.n_verif, len(df)), random_state=42)
    log(f"  {len(df)} sel test")
    idx = {}
    for f in glob.glob(os.path.join(args.dir_ccrop, "**", "*.png"), recursive=True):
        if "_mask" in os.path.basename(f):
            idx[kunci_nama(f)] = f
    img_idx = {}
    for f in glob.glob(os.path.join(args.dir_ccrop, "**", "*.jpg"), recursive=True):
        if "_mask" not in os.path.basename(f):
            img_idx[kunci_nama(f)] = f
    seg, transform = muat_model(args)
    conf = np.zeros((6, 6), np.int64)
    t0 = time.time()
    n = 0
    for i in range(0, len(df), args.batch):
        potongan = df.iloc[i:i + args.batch]
        tensors, gts = [], []
        for r in potongan.itertuples():
            k = kunci_nama(r.img_name)
            if k not in idx or k not in img_idx:
                continue
            tensors.append(transform(Image.open(img_idx[k]).convert("RGB")))
            g = np.array(Image.open(idx[k]))
            gts.append(g[..., 0] if g.ndim == 3 else g)
        if not tensors:
            continue
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = seg(torch.stack(tensors).cuda()).argmax(dim=1).cpu().numpy()
        for p, g in zip(pred, gts):
            m = (g >= 0) & (g < 6)
            conf += np.bincount(6 * g[m].ravel().astype(np.int64) + p[m].ravel().astype(np.int64),
                                minlength=36).reshape(6, 6)
        n += len(tensors)
        if n % (args.batch * 20) < args.batch:
            log(f"    {n}/{len(df)} ({time.time() - t0:.0f} s)")
    tp = np.diag(conf).astype(float)
    iou = tp / (conf.sum(1) + conf.sum(0) - tp)
    miou = float(np.nanmean(iou[1:]) * 100)
    log(f"\n  {'kelas':<12}{'IoU diperoleh':>15}{'IoU terbitan':>15}{'selisih':>10}")
    lolos = True
    for i, nm in enumerate(KELAS):
        t = IOU_TERBIT.get(nm)
        d = f"{100 * iou[i] - t:+.2f}" if t else "–"
        log(f"  {nm:<12}{100 * iou[i]:>15.2f}{(t if t else float('nan')):>15.2f}{d:>10}")
        if t and abs(100 * iou[i] - t) > 3.0:
            lolos = False
    log(f"\n  mIoU (tanpa background): **{miou:.2f}** versus terbitan {MIOU_TERBIT:.2f} "
        f"(selisih {miou - MIOU_TERBIT:+.2f})")
    if abs(miou - MIOU_TERBIT) > 2.0:
        lolos = False
    log(f"\n  **GERBANG {'LOLOS' if lolos else 'GAGAL'}.**")
    if not lolos:
        log("  Setup tidak mereproduksi angka terbitan. JANGAN lanjut ke `ig` — "
            "non-monotonisitas apa pun nanti tidak bisa dibedakan dari salah setup.")
    pd.DataFrame([dict(kelas=KELAS[i], iou=100 * iou[i], iou_terbit=IOU_TERBIT.get(KELAS[i], np.nan))
                  for i in range(6)] + [dict(kelas="mIoU", iou=miou, iou_terbit=MIOU_TERBIT)]
                 ).to_csv(os.path.join(args.keluar, "H1_verifikasi_miou.csv"), index=False)
    return lolos


def tugas_pbc(args):
    """Kontrol in-domain: segmentasi ulang seluruh sel WBCAtt+ yang SUDAH punya mask anotasi.
    Dari sini kita ukur berapa banyak bridge ratio bergeser ketika mask diprediksi."""
    import pandas as pd
    log("## Kontrol in-domain — segmentasi ulang sel WBCAtt+\n")
    df = pd.read_csv(args.csv)
    img_idx = {}
    for f in glob.glob(os.path.join(args.dir_ccrop, "**", "*.jpg"), recursive=True):
        if "_mask" not in os.path.basename(f):
            img_idx[kunci_nama(f)] = f
    tugas = [(kunci_nama(r.img_name), img_idx[kunci_nama(r.img_name)])
             for r in df.itertuples() if kunci_nama(r.img_name) in img_idx]
    if args.hanya_neutrofil:
        tugas = [t for t in tugas if t[0].split("_")[0].upper() in PREFIX_NEU]
        log("  dibatasi ke neutrofil saja")
    log(f"  {len(tugas)} sel")
    seg, transform = muat_model(args)
    inferensi(seg, transform, tugas, os.path.join(args.keluar, "mask_pred_pbc"),
              args.batch, 0, 0, simpan_citra=False)


def tugas_ig(args):
    log("## Segmentasi folder `ig`\n")
    d = daftar_ig(args.dir_pbc)
    if not d:
        raise SystemExit(f"Tidak ada citra PMY/MY/MMY/IG di {args.dir_pbc}")
    hit = {}
    for k, pre, f in d:
        hit[pre] = hit.get(pre, 0) + 1
    log("  " + " | ".join(f"{k} {v}" for k, v in sorted(hit.items())))
    log(f"  total {len(d)} citra | offset crop dy={args.dy}, dx={args.dx}")
    if args.dy is None:
        raise SystemExit("--dy belum ditentukan. Jalankan `--tugas cek` lebih dulu.")
    seg, transform = muat_model(args)
    inferensi(seg, transform, [(k, f) for k, _, f in d],
              os.path.join(args.keluar, "mask_pred_ig"), args.batch, args.dy, args.dx,
              simpan_citra=True)
    import pandas as pd
    pd.DataFrame([dict(img_name=f"{k}_ccrop.jpg", kunci=k, prefix=pre, path_asal=f,
                       tahap={"PMY": "1_promielosit", "MY": "2_mielosit", "MMY": "3_metamielosit",
                              "IG": "X_ig_tak_bersubtipe"}[pre])
                  for k, pre, f in d]).to_csv(os.path.join(args.keluar, "H2_daftar_ig.csv"), index=False)
    log(f"\n  Daftar tersimpan: {os.path.join(args.keluar, 'H2_daftar_ig.csv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tugas", choices=["cek", "verifikasi", "pbc", "ig"], required=True)
    ap.add_argument("--dir-model", default="./m2f_tiny_1024_color_20260414_091312")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dir-pbc", default=None, help="folder PBC asli (berisi subfolder ig/, neutrophil/, ...)")
    ap.add_argument("--dir-ccrop", default="pbcseg_final_v1", help="folder WBCAtt+ (citra ccrop + mask anotasi)")
    ap.add_argument("--csv", default="pbc_attr_v1_ccrop_all.csv")
    ap.add_argument("--keluar", default="hasil_fase2g")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dy", type=int, default=None)
    ap.add_argument("--dx", type=int, default=0)
    ap.add_argument("--n-cek", type=int, default=25)
    ap.add_argument("--n-verif", type=int, default=400, help="0 = seluruh test split")
    ap.add_argument("--hanya-neutrofil", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.keluar, exist_ok=True)
    log(f"# FASE 2G — segmentasi | tugas `{args.tugas}` | {time.strftime('%Y-%m-%d %H:%M')}\n")
    if args.tugas == "cek":
        tugas_cek(args)
    elif args.tugas == "verifikasi":
        tugas_verifikasi(args)
    elif args.tugas == "pbc":
        tugas_pbc(args)
    else:
        tugas_ig(args)


if __name__ == "__main__":
    main()