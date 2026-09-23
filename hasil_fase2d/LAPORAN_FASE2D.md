# LAPORAN FASE 2D

Dihasilkan `fase2d_lubang_dan_deferral.py` pada 2026-09-21 00:27.

Aturan saddle: **min**. Angka memakai titik desimal.


## 0. Konfigurasi

- **n_sel**: 10298
- **aturan_lubang**: tidak
- **lubang_maks_px**: 15
- **lubang_frak_sel**: 0.5
- **tau_lobus**: 4.0
- **rho_lobus**: 0.4
- **tau_br**: 1.25
- **rho_br**: 0.2
- **sanitas_lolos**: True
- **replikasi_kelompok**: True
- **tugas_dijalankan**: semua
- **parameter_cadangan**: False

## 1. Sensus lubang -- kriteria primer

**100.00% piksel di dalam region tertutup berlabel sitoplasma atau vakuola.** Kesimpulan aturan lubang diambil dari angka ini dan dari skor kalibrasi non-neutrofil, bukan dari AUC A-vs-B.

| nilai | piksel | persen |
|---|---|---|
| latar | 0 | 0.0000 |
| sitoplasma | 262162 | 88.4445 |
| nukleus | 3 | 0.0010 |
| trombosit | 0 | 0.0000 |
| sel_lain | 0 | 0.0000 |
| vakuola | 34249 | 11.5544 |

## 2. Ablasi aturan lubang

Kolom berakhiran `_KONSEKUENSI` **tidak** dipakai memilih aturan.

| aturan_lubang | tau | rho | skor_br | rec_harus_pecah | rec_harus_utuh | skor_makro3_pada_tau_br | skor_makro3_terbaik | limf_tak_pecah | limf_lobus | eo_lobus | frak_takpecah | r_lobus_mean | r_lobus_maks | auc_KONSEKUENSI | akurasi13_KONSEKUENSI | spes_band_KONSEKUENSI | sens_band_KONSEKUENSI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semua | 0.7500 | 0.2000 | 0.9721 | 0.9611 | 0.9831 | 0.7287 | 0.7826 | 0.9852 | 1.0222 | 2.5630 | 0.2412 | 29.4028 | 59.0339 | 0.9072 | 0.9044 | 0.8002 | 0.9698 |
| tidak | 1.2500 | 0.2000 | 0.9861 | 0.9853 | 0.9868 | 0.7641 | 0.8130 | 0.9868 | 1.0222 | 2.5194 | 0.2264 | 28.6000 | 59.0339 | 0.9667 | 0.9328 | 0.8873 | 0.9614 |
| selektif | 1.2500 | 0.2000 | 0.9850 | 0.9832 | 0.9868 | 0.7638 | 0.8100 | 0.9868 | 1.0222 | 2.5162 | 0.2276 | 28.6258 | 59.0339 | 0.9629 | 0.9293 | 0.8730 | 0.9646 |

### 2.1 Perbandingan berpasangan terhadap aturan `semua` (Fase 2C)

| aturan_lubang | B_salah_sebelum | B_salah_sesudah | B_diperbaiki | B_dirusak | A_salah_sebelum | A_salah_sesudah | A_diperbaiki | A_dirusak |
|---|---|---|---|---|---|---|---|---|
| tidak | 195 | 110 | 90 | 5 | 47 | 60 | 1 | 14 |
| selektif | 195 | 124 | 77 | 6 | 47 | 55 | 1 | 9 |

### 2.2 Asal-usul leher, diukur pada aturan `semua` (Fase 2C)

Jarak dari saddle terpilih ke piksel lubang terisi terdekat. Kalau galat ekor B jauh lebih sering punya lubang terisi tepat di lehernya dibanding sel B yang benar, leher itu buatan.

| kelompok | n | n_dengan_saddle | n_tanpa_lubang_terisi | n_dengan_lubang_terisi | frak_sel_ada_lubang | median_jarak | frak_dekat_3px | frak_dekat_6px |
|---|---|---|---|---|---|---|---|---|
| B salah, leher terukur | 173 | 173 | 56 | 117 | 0.6763 | 8.5440 | 0.3077 | 0.4274 |
| B benar segmented | 716 | 716 | 612 | 104 | 0.1453 | 27.2304 | 0.1154 | 0.1827 |
| A benar band | 1413 | 1413 | 1318 | 95 | 0.0672 | 15.2315 | 0.0842 | 0.1474 |

## 3. Kalibrasi dua sisi

### 3.1 Sepuluh teratas menurut skor makro-3 (jumlah lobus)

| tau | rho | skor_makro3 | rec_t1 | rec_t2 | rec_t3 | skor_br | limf_tak_pecah |
|---|---|---|---|---|---|---|---|
| 4.0000 | 0.4000 | 0.8130 | 0.8620 | 0.8412 | 0.7358 | 0.9745 | 0.9918 |
| 4.0000 | 0.3000 | 0.8123 | 0.8607 | 0.8320 | 0.7441 | 0.9751 | 0.9918 |
| 4.0000 | 0.2000 | 0.8109 | 0.8599 | 0.8288 | 0.7441 | 0.9753 | 0.9918 |
| 4.0000 | 0.0000 | 0.8106 | 0.8595 | 0.8284 | 0.7441 | 0.9753 | 0.9918 |
| 4.0000 | 0.1000 | 0.8106 | 0.8595 | 0.8284 | 0.7441 | 0.9753 | 0.9918 |
| 3.0000 | 0.4000 | 0.8100 | 0.8340 | 0.8128 | 0.7832 | 0.9793 | 0.9909 |
| 5.0000 | 0.3000 | 0.8096 | 0.8841 | 0.8540 | 0.6908 | 0.9706 | 0.9926 |
| 5.0000 | 0.0000 | 0.8088 | 0.8832 | 0.8524 | 0.6908 | 0.9706 | 0.9926 |
| 5.0000 | 0.1000 | 0.8088 | 0.8832 | 0.8524 | 0.6908 | 0.9706 | 0.9926 |
| 5.0000 | 0.2000 | 0.8088 | 0.8832 | 0.8524 | 0.6908 | 0.9706 | 0.9926 |

### 3.2 Sepuluh teratas menurut skor status (bridge ratio)

| tau | rho | skor_br | rec_harus_pecah | rec_harus_utuh | skor_makro3 | limf_tak_pecah |
|---|---|---|---|---|---|---|
| 1.2500 | 0.2000 | 0.9861 | 0.9853 | 0.9868 | 0.7641 | 0.9868 |
| 1.5000 | 0.2000 | 0.9860 | 0.9823 | 0.9896 | 0.7720 | 0.9893 |
| 1.2500 | 0.3000 | 0.9859 | 0.9850 | 0.9868 | 0.7677 | 0.9868 |
| 1.5000 | 0.3000 | 0.9858 | 0.9820 | 0.9896 | 0.7756 | 0.9893 |
| 1.2500 | 0.4000 | 0.9856 | 0.9844 | 0.9868 | 0.7756 | 0.9868 |
| 1.2500 | 0.0000 | 0.9856 | 0.9853 | 0.9859 | 0.7605 | 0.9868 |
| 1.2500 | 0.1000 | 0.9856 | 0.9853 | 0.9859 | 0.7609 | 0.9868 |
| 1.5000 | 0.4000 | 0.9855 | 0.9814 | 0.9896 | 0.7830 | 0.9893 |
| 1.5000 | 0.0000 | 0.9855 | 0.9823 | 0.9887 | 0.7694 | 0.9893 |
| 1.5000 | 0.1000 | 0.9855 | 0.9823 | 0.9887 | 0.7697 | 0.9893 |

## 4. Ambang dan plateau

| metrik | nilai |
|---|---|
| n_A | 1555 |
| n_B | 976 |
| auc | 0.9667364002951875 |
| J_max | 0.860253149544041 |
| ambang | 0.3763549647474907 |
| ambang_atas | 0.3796455716576034 |
| lebar_persis | 0.003290606910112681 |
| sepertiga_di_interval_persis | False |
| plateau_lo | 0.35777087639996635 |
| plateau_hi | 0.38390264701787685 |
| lebar_plateau_eps | 0.0261317706179105 |
| sepertiga_di_plateau_eps | False |
| J_sepertiga | 0.8487098729639977 |
| defisit_J_sepertiga | 0.011543276580043282 |
| akurasi_sepertiga | 0.9328328723824575 |
| akurasi_ambang | 0.9312524693796919 |
| sens_band | 0.9614147909967846 |
| spes_band | 0.8872950819672131 |
| tau | 1.25 |
| rho | 0.2 |
| boot_median | 0.37242264987106555 |
| boot_lo | 0.30759119560374354 |
| boot_hi | 0.39107694533126924 |
| sepertiga_di_IK | True |
| ambang_train | 0.3587657577822075 |
| auc_test | 0.9554185114906177 |
| akurasi_test_ambang | 0.9323797139141743 |
| akurasi_test_sepertiga | 0.9323797139141743 |
| n_train | 1509 |
| n_test | 769 |
| auc_tanpa_takpecah | 0.9666612183743658 |
| akurasi13_tanpa_takpecah | 0.9318823055219669 |

> Ambang 1/3 harus ditulis sebagai pilihan yang ongkosnya kecil dan terletak di dalam IK bootstrap, **bukan** sebagai argmax Youden. Interval optimal persisnya terlalu sempit untuk memuat 1/3.


## 5. Kelompok neutrofil (parameter bridge ratio)

| kelompok | n | q25 | median | q75 | p_band | entropi | zona_23_43 | tak_pecah | mean_lobus | posisi_AB |
|---|---|---|---|---|---|---|---|---|---|---|
| A_sepakat_band | 1555 | 0.4763 | 0.5663 | 0.6632 | 0.9614 | 0.2358 | 0.1241 | 0.0315 | 2.4997 | -0.0000 |
| B_sepakat_segmented | 976 | 0.0841 | 0.1313 | 0.2206 | 0.1127 | 0.5080 | 0.1855 | 0.0010 | 3.3648 | 1.0000 |
| C_konflik_SNE_band | 662 | 0.3530 | 0.4251 | 0.5392 | 0.7915 | 0.7385 | 0.4743 | 0.0242 | 2.9290 | 0.3247 |
| D_konflik_BNE_segmented | 48 | 0.2778 | 0.4418 | 0.5274 | 0.6875 | 0.8960 | 0.2292 | 0.0000 | 2.9375 | 0.2863 |
| E_lainnya | 38 | 0.6499 | 1.0000 | 1.0000 | 0.9737 | 0.1756 | 0.0263 | 0.6316 | 1.4737 | -0.9970 |
| F_tak_bersubtipe | 50 | 0.1059 | 0.2216 | 0.3749 | 0.3200 | 0.9044 | 0.3600 | 0.0000 | 3.5200 | 0.7924 |

## 6. Jumlah lobus per nucleus_shape (parameter LOBUS)

| nucleus_shape | n | mean_lobus | median_lobus | hanya_1_lobus | median_br |
|---|---|---|---|---|---|
| unsegmented-round | 1069 | 1.0112 | 1.0000 | 0.9925 | 1.0000 |
| unsegmented-indented | 1342 | 1.2869 | 1.0000 | 0.7586 | 1.0000 |
| irregular | 870 | 1.4437 | 1.0000 | 0.6563 | 1.0000 |
| unsegmented-band | 2614 | 2.0340 | 2.0000 | 0.1534 | 0.5303 |
| segmented-bilobed | 3119 | 2.1629 | 2.0000 | 0.0401 | 0.1262 |
| segmented-multilobed | 1284 | 2.9556 | 3.0000 | 0.0202 | 0.1038 |

## 7. Kontrol lintas kelas sel (parameter LOBUS)

| kelas | n | median_br | tak_pernah_pecah | mean_lobus | terpisah_di_nol |
|---|---|---|---|---|---|
| Basophil | 1218 | 0.4489 | 0.3506 | 1.9154 | 0.0952 |
| Eosinophil | 3117 | 0.1200 | 0.0491 | 2.2201 | 0.1203 |
| Lymphocyte | 1214 | 1.0000 | 0.9918 | 1.0132 | 0.0058 |
| Monocyte | 1420 | 1.0000 | 0.7366 | 1.3106 | 0.0092 |
| Neutrophil | 3329 | 0.4521 | 0.1117 | 2.2764 | 0.0228 |

## 8. Ekor B

- sel B salah klasifikasi: **110** (Fase 2C: 195)
  - normal: 109 (99.1%)
  - tak_pernah_pecah: 1 (0.9%)

### 8.1 Jarak saddle ke piksel lubang terisi

| kelompok | n | n_dengan_saddle | n_tanpa_lubang_terisi | n_dengan_lubang_terisi | frak_sel_ada_lubang | median_jarak | frak_dekat_3px | frak_dekat_6px |
|---|---|---|---|---|---|---|---|---|
| B salah, leher terukur | 109 | 109 | 109 | 0 | 0.0000 |  |  |  |
| B benar segmented | 801 | 801 | 801 | 0 | 0.0000 |  |  |  |
| A benar band | 1446 | 1446 | 1446 | 0 | 0.0000 |  |  |  |

## 9. Analisis deferral

| cakupan | n_simpan | n_tunda | n_AB_simpan | akurasi | sens_band | spes_band | akurasi_seimbang | pangsa_tunda_A | pengayaan_A | pangsa_tunda_B | pengayaan_B | pangsa_tunda_C | pengayaan_C | pangsa_tunda_D | pengayaan_D | pangsa_tunda_E | pengayaan_E | pangsa_tunda_F | pengayaan_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0000 | 3329 | 0 | 2531 | 0.9328 | 0.9614 | 0.8873 | 0.9244 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.9500 | 3163 | 166 | 2452 | 0.9498 | 0.9762 | 0.9075 | 0.9419 | 0.2651 | 0.5675 | 0.2108 | 0.7192 | 0.4639 | 2.3326 | 0.0301 | 2.0890 | 0.0000 | 0.0000 | 0.0301 | 2.0054 |
| 0.9000 | 2996 | 333 | 2374 | 0.9579 | 0.9797 | 0.9221 | 0.9509 | 0.2402 | 0.5143 | 0.2312 | 0.7887 | 0.4715 | 2.3709 | 0.0270 | 1.8744 | 0.0000 | 0.0000 | 0.0300 | 1.9994 |
| 0.8500 | 2830 | 499 | 2290 | 0.9616 | 0.9812 | 0.9288 | 0.9550 | 0.2445 | 0.5234 | 0.2385 | 0.8134 | 0.4669 | 2.3481 | 0.0220 | 1.5288 | 0.0020 | 0.1756 | 0.0261 | 1.7345 |
| 0.8000 | 2663 | 666 | 2189 | 0.9657 | 0.9818 | 0.9388 | 0.9603 | 0.2748 | 0.5882 | 0.2387 | 0.8143 | 0.4414 | 2.2199 | 0.0165 | 1.1455 | 0.0015 | 0.1315 | 0.0270 | 1.7995 |
| 0.7500 | 2497 | 832 | 2085 | 0.9659 | 0.9816 | 0.9396 | 0.9606 | 0.2981 | 0.6381 | 0.2380 | 0.8117 | 0.4207 | 2.1154 | 0.0168 | 1.1670 | 0.0024 | 0.2106 | 0.0240 | 1.6005 |
| 0.7000 | 2330 | 999 | 1965 | 0.9669 | 0.9829 | 0.9403 | 0.9616 | 0.3273 | 0.7008 | 0.2392 | 0.8160 | 0.3904 | 1.9632 | 0.0170 | 1.1802 | 0.0020 | 0.1754 | 0.0240 | 1.5995 |
| 0.6000 | 1997 | 1332 | 1709 | 0.9731 | 0.9814 | 0.9588 | 0.9701 | 0.3581 | 0.7667 | 0.2590 | 0.8834 | 0.3401 | 1.7102 | 0.0195 | 1.3538 | 0.0023 | 0.1973 | 0.0210 | 1.3996 |
| 0.5000 | 1664 | 1665 | 1422 | 0.9782 | 0.9859 | 0.9641 | 0.9750 | 0.3808 | 0.8152 | 0.2853 | 0.9731 | 0.2931 | 1.4739 | 0.0174 | 1.2080 | 0.0042 | 0.3683 | 0.0192 | 1.2796 |

## 10. Yang harus dijawab berikutnya

- Apakah sensus membenarkan aturan selektif, dan apakah batas `--lubang-maks-px` dan `--lubang-frak-sel` perlu digeser?
- Berapa galat ekor B yang tersisa setelah lubang dibereskan, dan apa penyebabnya sekarang?
- Fase 2E: sumbu pematangan ig, dan penulisan ulang Bagian 14.
