# LAPORAN FASE 2F-2 — M10: aturan lubang berbasis bentuk

Dijalankan 2026-09-21 22:10 | τ_br 1.25, ρ_br 0.2, τ_lobus 4.0, ρ_lobus 0.4 | keluarga aturan: isi lubang bila lebar_maks ≤ L | L = 0 ≡ `tidak`, L = 999 ≡ `semua`

## 0. Uji sanitas

| uji | lolos | detail |
|---|---|---|
| lebar_maks memisahkan celah tipis dari lubang kompak | YA | n=2, lebar [2.0, 18.11], aspek [40.0, 0.77] |
| L=0 == `tidak`, L besar == `semua` | YA | L0 10876 (harap 10876), L999 11289 (harap 11289), L3 11036 (harap 11036) |
| pembilang: slot tipis di leher menjatuhkan r_pisah 2,000 → 11,000, r_lobus tetap | YA | lebar lubang 2.00, r_pisah 2.000 → 11.000, r_lobus 30.017 → 30.017, br 0.0666 → 0.3665 |
| penyebut: lubang kompak menaikkan r_lobus 25,020 → 40,012, r_pisah 5,000 tidak bergerak | YA | lebar lubang 30.07, r_pisah 5.000 → 5.000, r_lobus 25.020 → 40.012, br 0.1998 → 0.1250 |
| kedua mekanisme melawan arah, dan lebar_maks memisahkannya (hipotesis M10 in miniatur) | YA | pembilang br 0.0666 → 0.3665 (NAIK, lubang 2.0 px) | penyebut br 0.1998 → 0.1250 (TURUN, lubang 30.1 px) |
| skor lobus menghukum penyaringan agresif | YA | skor ketat 0.500 (harap 0,5), skor pas 1.000 (harap 1,0) |

**Seluruh 6 uji LOLOS.** Uji 3–5 memastikan skrip benar-benar mereproduksi kedua mekanisme Bagian 11.6, termasuk fakta bahwa keduanya menggerakkan `br` ke arah berlawanan — itulah yang membuat aturan biner tidak bisa memuaskan keduanya.

Data: `hasil_fase2d\bridge_ratio_2d.csv` — 10298 baris | mask terindeks 10298, cocok 10298

- sapuan 12 nilai L: 10298 tugas, 850.3 detik (82.6 ms/tugas)
- uji reproduksi: L = 0 mereproduksi `bridge_ratio` Fase 2D pada **99.99%** sel

## 1. LAPIS MEKANISTIK — apakah bentuk lubang bimodal? (tanpa label apa pun)

Region lubang terukur: **2366** pada 1556 sel.

| lebar_maks | n | frak | luas_median | aspek_median | jarak_saddle_median |
|---|---|---|---|---|---|
| (0; 1] | 0 | 0.0000 | – | – | – |
| (1; 1.5] | 0 | 0.0000 | – | – | – |
| (1.5; 2] | 818 | 0.3457 | 1.0000 | 0.2500 | 20.2485 |
| (2; 2.5] | 0 | 0.0000 | – | – | – |
| (2.5; 3] | 132 | 0.0558 | 15.0000 | 1.8750 | 15.0373 |
| (3; 4] | 198 | 0.0837 | 28.0000 | 1.7500 | 12.6491 |
| (4; 5] | 112 | 0.0473 | 48.5000 | 2.4250 | 12.0416 |
| (5; 6] | 209 | 0.0883 | 52.0000 | 1.5625 | 13.9284 |
| (6; 8] | 205 | 0.0866 | 77.0000 | 1.4808 | 12.3683 |
| (8; 12] | 351 | 0.1484 | 135.0000 | 1.4044 | 11.4018 |
| (12; 99] | 341 | 0.1441 | 443.0000 | 1.4875 | 8.2462 |

Kuantil `lebar_maks`: q10 2.00, q25 2.00, q50 4.47, q75 8.94, q90 14.42, q95 18.44, q99 28.50
Histogram 40 bin pada [0; 20] px: **11 puncak lokal**, **13 bin kosong di tengah sebaran**.
**Putusan lapis mekanistik: sebaran BIMODAL.**

## 2. LAPIS KALIBRASI INDEPENDEN — skor lobus pada sel NON-NEUTROFIL

n = 6969 sel non-neutrofil. Tidak satu pun masuk kelompok A atau B. Harapan: round 1, indented 1, bilobed 2, multilobed ≥3. `irregular` dan `band` dikecualikan.

| L | skor_makro | round | indented | bilobed | multilobed | frak_lubang_diisi |
|---|---|---|---|---|---|---|
| 0.0000 | 0.8319 | 0.9925 | 0.7582 | 0.8412 | 0.7358 | 0.0000 |
| 1.0000 | 0.8319 | 0.9925 | 0.7582 | 0.8412 | 0.7358 | 0.0000 |
| 1.5000 | 0.8319 | 0.9925 | 0.7582 | 0.8412 | 0.7358 | 0.0000 |
| 2.0000 | 0.8298 | 0.9925 | 0.7597 | 0.8396 | 0.7275 | 0.3457 |
| 2.5000 | 0.8298 | 0.9925 | 0.7597 | 0.8396 | 0.7275 | 0.3457 |
| 3.0000 | 0.8295 | 0.9925 | 0.7612 | 0.8404 | 0.7239 | 0.4015 |
| 4.0000 | 0.8272 | 0.9925 | 0.7635 | 0.8372 | 0.7156 | 0.4852 |
| 5.0000 | 0.8259 | 0.9925 | 0.7672 | 0.8332 | 0.7109 | 0.5325 |
| 6.0000 | 0.8229 | 0.9934 | 0.7710 | 0.8292 | 0.6979 | 0.6209 |
| 8.0000 | 0.8208 | 0.9934 | 0.7762 | 0.8228 | 0.6908 | 0.7075 |
| 12.0000 | 0.8196 | 0.9934 | 0.7882 | 0.8156 | 0.6813 | 0.8559 |
| 999.0000 | 0.8120 | 0.9944 | 0.8001 | 0.8019 | 0.6517 | 1.0000 |

Skor terbaik L = 0 (0.8319); rentang seluruh permukaan **0.0199**. Laporkan sebagai wilayah, bukan argmax.

## 3. UJI DUA SISI — perbaiki sel A tanpa merusak sel B

| L | A_dirusak_tidak_n | A_dirusak_tidak_benar | B_dirusak_tidak_n | B_dirusak_tidak_benar | galat_A | galat_B | galat_total |
|---|---|---|---|---|---|---|---|
| 0.0000 | 14.0000 | 0.0000 | 4.0000 | 0.0000 | 59.0000 | 110.0000 | 169.0000 |
| 1.0000 | 14.0000 | 0.0000 | 4.0000 | 0.0000 | 59.0000 | 110.0000 | 169.0000 |
| 1.5000 | 14.0000 | 0.0000 | 4.0000 | 0.0000 | 59.0000 | 110.0000 | 169.0000 |
| 2.0000 | 14.0000 | 5.0000 | 4.0000 | 0.0000 | 54.0000 | 120.0000 | 174.0000 |
| 2.5000 | 14.0000 | 5.0000 | 4.0000 | 0.0000 | 54.0000 | 120.0000 | 174.0000 |
| 3.0000 | 14.0000 | 6.0000 | 4.0000 | 0.0000 | 52.0000 | 125.0000 | 177.0000 |
| 4.0000 | 14.0000 | 9.0000 | 4.0000 | 0.0000 | 50.0000 | 129.0000 | 179.0000 |
| 5.0000 | 14.0000 | 10.0000 | 4.0000 | 0.0000 | 49.0000 | 131.0000 | 180.0000 |
| 6.0000 | 14.0000 | 12.0000 | 4.0000 | 1.0000 | 47.0000 | 136.0000 | 183.0000 |
| 8.0000 | 14.0000 | 13.0000 | 4.0000 | 2.0000 | 46.0000 | 140.0000 | 186.0000 |
| 12.0000 | 14.0000 | 14.0000 | 4.0000 | 4.0000 | 45.0000 | 155.0000 | 200.0000 |
| 999.0000 | 14.0000 | 14.0000 | 4.0000 | 4.0000 | 45.0000 | 196.0000 | 241.0000 |

Sel acuan yang ditemukan: **14 dari 14** sisi A, **4 dari 4** sisi B.

Kolom `*_benar` adalah jumlah sel yang divonis BENAR pada L itu. Aturan yang berhasil harus menaikkan `A_dirusak_tidak_benar` ke arah 14 **tanpa** menurunkan `B_dirusak_tidak_benar` dari 4. `galat_A` dan `galat_B` disertakan sebagai konteks, **bukan** sebagai kriteria pemilihan.

**Ada L yang memperbaiki sisi A tanpa merusak sisi B: YA.**

## 4. PUTUSAN M10

**M10 LOLOS ketiga lapis.** Aturan bentuk boleh dipertimbangkan sebagai pengganti `tidak`. Jalankan ulang Fase 2D/2E pada L terpilih sebelum melaporkan angka apa pun.

## 5. ALTERNATIF TANPA PARAMETER BARU — ketidaksepakatan aturan sebagai ketidakpastian

Sel yang vonisnya BERUBAH antara `tidak` dan `semua`: **150** dari 3329 neutrofil (4.51%).

| kelompok | n | peka | frak_peka | pengayaan |
|---|---|---|---|---|
| A_sepakat_band | 1555 | 14 | 0.0090 | 0.1998 |
| B_sepakat_segmented | 976 | 94 | 0.0963 | 2.1375 |
| C_konflik_SNE_band | 662 | 27 | 0.0408 | 0.9052 |
| D_konflik_BNE_segmented | 48 | 9 | 0.1875 | 4.1612 |
| E_lainnya | 38 | 0 | 0.0000 | 0.0000 |
| F_tak_bersubtipe | 50 | 6 | 0.1200 | 2.6632 |

Akurasi pada sel peka-aturan: **0.8333** (n = 108); pada sel tidak-peka: **0.9377** (n = 2423).

Perbandingan kriteria penundaan pada cakupan yang sama:

| cakupan | n_simpan | akurasi_jarak | akurasi_peka_lalu_jarak | selisih |
|---|---|---|---|---|
| 1.00000 | 2531.00000 | 0.93323 | 0.93323 | 0.00000 |
| 0.99000 | 2506.00000 | 0.93815 | 0.93615 | -0.00200 |
| 0.97500 | 2468.00000 | 0.94571 | 0.93801 | -0.00770 |
| 0.95000 | 2404.00000 | 0.95466 | 0.94218 | -0.01248 |
| 0.90000 | 2278.00000 | 0.96181 | 0.96049 | -0.00132 |
| 0.85000 | 2151.00000 | 0.96513 | 0.96513 | 0.00000 |

**Ketidaksepakatan aturan mengungguli |br − 1/3| sebagai kriteria penundaan: TIDAK.**
Ketidaksepakatan aturan tidak membawa informasi melebihi jarak ke ambang. Jalur ini ditutup juga, dan mesin deferral Fase 2D tetap seperti apa adanya.

## 6. Konteks AUC (dilaporkan terakhir, TIDAK dipakai memilih L)

| L | auc | acc_13 | youden | posisi_C |
|---|---|---|---|---|
| 0.00000 | 0.96629 | 0.93323 | 0.37635 | 0.32471 |
| 1.00000 | 0.96629 | 0.93323 | 0.37635 | 0.32471 |
| 1.50000 | 0.96629 | 0.93323 | 0.37635 | 0.32471 |
| 2.00000 | 0.96308 | 0.93125 | 0.37635 | 0.32689 |
| 2.50000 | 0.96308 | 0.93125 | 0.37635 | 0.32689 |
| 3.00000 | 0.96277 | 0.93007 | 0.37635 | 0.32590 |
| 4.00000 | 0.96146 | 0.92928 | 0.37635 | 0.32866 |
| 5.00000 | 0.96038 | 0.92888 | 0.37635 | 0.32897 |
| 6.00000 | 0.95767 | 0.92770 | 0.37635 | 0.32982 |
| 8.00000 | 0.95533 | 0.92651 | 0.37635 | 0.33026 |
| 12.00000 | 0.93990 | 0.92098 | 0.37148 | 0.33242 |
| 999.00000 | 0.90504 | 0.90478 | 0.37148 | 0.32212 |

Tabel ini hanya konteks. Bila L dengan AUC tertinggi berbeda dari putusan Bagian 4, **putusan Bagian 4 yang berlaku** — itulah inti Pelajaran 3.
