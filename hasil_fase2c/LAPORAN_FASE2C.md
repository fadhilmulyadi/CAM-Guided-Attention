# LAPORAN FASE 2C

Dihasilkan otomatis oleh `fase2c_kalibrasi_dua_sisi.py` pada 2026-09-18 07:18.

Aturan saddle: **min**. Angka memakai titik desimal.


## 0. Konfigurasi

- **n_sel**: 10298
- **praproses**: batasi_sel=True, min_frak=0.02, prune=0.05
- **tau_lobus**: 3.0
- **rho_lobus**: 0.4
- **tau_br**: 0.75
- **rho_br**: 0.2
- **sanitas_lolos**: True
- **replikasi_kelompok**: True

## 1. Diagnostik praproses M4 (pembatasan ke komponen sel target)

| sel_total | komponen_nukleus_mentah_>1 | komponen_nukleus_akhir_>1 | sel_kehilangan_piksel_luar_sel | piksel_dibuang_luar_sel_total | sel_kehilangan_serpihan_kecil | sel_dengan_lubang | pusat_fallback | pusat_beda_terbesar | r_lobus_maks |
|---|---|---|---|---|---|---|---|---|---|
| 10298 | 891 | 596 | 306 | 1301717 | 17 | 1556 | 16 | 38 | 59.0339 |

## 2. Kalibrasi dua sisi (M1, M2, M3)

Grid: tau [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] x rho [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] = 128 kombinasi. Dikalibrasi HANYA pada sel non-neutrofil, sehingga tidak ada sel yang dipakai untuk menyetel parameter sekaligus untuk menguji ambang.


### 2.1 Sepuluh kombinasi teratas menurut skor makro-3 (jumlah lobus)

| tau | rho | skor_makro3 | skor_mentah | rec_t1 | rec_t2 | rec_t3 | limf_tak_pecah | limf_lobus |
|---|---|---|---|---|---|---|---|---|
| 3.0000 | 0.4000 | 0.7826 | 0.8049 | 0.8611 | 0.7863 | 0.7002 | 0.9918 | 1.0124 |
| 2.5000 | 0.4000 | 0.7814 | 0.7952 | 0.8428 | 0.7715 | 0.7299 | 0.9918 | 1.0124 |
| 3.0000 | 0.3000 | 0.7805 | 0.7995 | 0.8603 | 0.7715 | 0.7097 | 0.9918 | 1.0132 |
| 4.0000 | 0.4000 | 0.7799 | 0.8150 | 0.8862 | 0.8019 | 0.6517 | 0.9926 | 1.0115 |
| 4.0000 | 0.3000 | 0.7796 | 0.8122 | 0.8857 | 0.7931 | 0.6600 | 0.9926 | 1.0124 |
| 4.0000 | 0.2000 | 0.7785 | 0.8108 | 0.8853 | 0.7903 | 0.6600 | 0.9926 | 1.0124 |
| 4.0000 | 0.0000 | 0.7782 | 0.8105 | 0.8845 | 0.7903 | 0.6600 | 0.9926 | 1.0124 |
| 4.0000 | 0.1000 | 0.7782 | 0.8105 | 0.8845 | 0.7903 | 0.6600 | 0.9926 | 1.0124 |
| 2.5000 | 0.3000 | 0.7780 | 0.7884 | 0.8411 | 0.7546 | 0.7382 | 0.9918 | 1.0132 |
| 3.0000 | 0.2000 | 0.7778 | 0.7960 | 0.8599 | 0.7638 | 0.7097 | 0.9918 | 1.0132 |

### 2.2 Sepuluh kombinasi teratas menurut skor status (bridge ratio)

| tau | rho | skor_br | rec_harus_pecah | rec_harus_utuh | skor_makro3 | limf_tak_pecah |
|---|---|---|---|---|---|---|
| 0.7500 | 0.2000 | 0.9721 | 0.9611 | 0.9831 | 0.7287 | 0.9852 |
| 0.7500 | 0.3000 | 0.9719 | 0.9608 | 0.9831 | 0.7336 | 0.9852 |
| 0.7500 | 0.0000 | 0.9719 | 0.9617 | 0.9821 | 0.7196 | 0.9852 |
| 0.7500 | 0.1000 | 0.9719 | 0.9617 | 0.9821 | 0.7219 | 0.9852 |
| 0.7500 | 0.4000 | 0.9715 | 0.9599 | 0.9831 | 0.7386 | 0.9852 |
| 0.5000 | 0.2000 | 0.9706 | 0.9685 | 0.9727 | 0.6995 | 0.9712 |
| 0.5000 | 0.3000 | 0.9703 | 0.9679 | 0.9727 | 0.7050 | 0.9712 |
| 0.5000 | 0.0000 | 0.9703 | 0.9688 | 0.9718 | 0.6840 | 0.9703 |
| 0.5000 | 0.1000 | 0.9703 | 0.9688 | 0.9718 | 0.6907 | 0.9712 |
| 1.0000 | 0.2000 | 0.9702 | 0.9536 | 0.9868 | 0.7422 | 0.9868 |

### 2.3 Pembanding parameter lama

| tau | rho | skor_mentah | skor_makro3 | skor_br | rec_t1 | rec_t2 | rec_t3 | limf_tak_pecah |
|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.3000 | 0.3415 | 0.4834 | 0.8150 | 0.3632 | 0.1034 | 0.9834 | 0.6219 |
| 1.0000 | 0.0000 | 0.7075 | 0.7370 | 0.9699 | 0.7502 | 0.6219 | 0.8389 | 0.9868 |
| 4.0000 | 0.5000 | 0.8140 | 0.7692 | 0.9457 | 0.8887 | 0.8123 | 0.6066 | 0.9934 |

## 3. Ambang, plateau, dan uji pergeseran-konstan (M5)

- nilai ambang unik di 128 kombinasi: **2**, rentang 0.0127, sd 0.0051
- lebar interval optimal PERSIS: median 0.0006, maks 0.0006
- lebar plateau (J dalam 0.001): median 0.0137, maks 0.0137
- 1/3 di interval optimal persis: 0/128 kombinasi
- **uji pergeseran-konstan**: sd[J_komb(t) - J_ref(t)] median 0.01261, maks 0.26091; rentang selisih maks 0.62451
- fraksi sel dengan bridge ratio identik terhadap referensi: median 0.8815

> **Kesimpulan M5.** Pergeseran J tidak konstan; invariansi ambang membawa informasi nyata.


### 3.1 Parameter terpilih

| metrik | nilai |
|---|---|
| n_A | 1555 |
| n_B | 976 |
| auc | 0.9072334088872488 |
| J_max | 0.7814684254915397 |
| ambang | 0.3587657577822075 |
| ambang_atas | 0.3593394089315999 |
| lebar_persis | 0.0005736511493923957 |
| sepertiga_di_interval_persis | False |
| plateau_lo | 0.3587657577822075 |
| plateau_hi | 0.37242264987106555 |
| lebar_plateau_eps | 0.013656892088858064 |
| sepertiga_di_plateau_eps | False |
| J_sepertiga | 0.7699798376469347 |
| defisit_J_sepertiga | 0.011488587844605047 |
| akurasi_sepertiga | 0.9043856183326748 |
| akurasi_ambang | 0.9067562228368234 |
| sens_band | 0.9697749196141479 |
| spes_band | 0.8002049180327869 |
| tau | 0.75 |
| rho | 0.2 |
| boot_median | 0.3700818721334576 |
| boot_lo | 0.31923475378704885 |
| boot_hi | 0.3878901011868793 |
| sepertiga_di_IK | True |
| auc_tanpa_takpecah | 0.9227298182131473 |
| ambang_tanpa_takpecah | 0.3587657577822075 |
| akurasi13_tanpa_takpecah | 0.9088649544324772 |
| ambang_train | 0.37147841246231744 |
| n_train | 1509 |
| auc_test | 0.8855154965211891 |
| n_test | 769 |
| akurasi_test_ambang | 0.9011703511053316 |
| akurasi_test_sepertiga | 0.9024707412223667 |

## 4. Kelompok neutrofil pada parameter terpilih

| kelompok | n | q25 | median | q75 | p_band | entropi | zona_28_38 | zona_23_43 | tak_pecah | mean_lobus | posisi_AB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_sepakat_band | 1555 | 0.4852 | 0.5810 | 0.6814 | 0.9698 | 0.1955 | 0.0399 | 0.1119 | 0.0611 | 2.6026 | -0.0000 |
| B_sepakat_segmented | 976 | 0.0864 | 0.1437 | 0.2751 | 0.1998 | 0.7215 | 0.0789 | 0.1773 | 0.0225 | 3.3586 | 1.0000 |
| C_konflik_SNE_band | 662 | 0.3670 | 0.4405 | 0.5630 | 0.8142 | 0.6926 | 0.2009 | 0.4335 | 0.0514 | 3.0317 | 0.3214 |
| D_konflik_BNE_segmented | 48 | 0.4111 | 0.4896 | 0.6295 | 0.8750 | 0.5436 | 0.1250 | 0.1875 | 0.0833 | 2.7500 | 0.2090 |
| E_lainnya | 38 | 0.6080 | 1.0000 | 1.0000 | 0.9737 | 0.1756 | 0.0263 | 0.0789 | 0.5526 | 1.6053 | -0.9580 |
| F_tak_bersubtipe | 50 | 0.1142 | 0.2546 | 0.4166 | 0.4000 | 0.9710 | 0.1000 | 0.3000 | 0.1000 | 3.3000 | 0.7465 |

## 5. Jumlah lobus per nucleus_shape

| nucleus_shape | n | mean_lobus | median_lobus | hanya_1_lobus | median_br |
|---|---|---|---|---|---|
| unsegmented-round | 1069 | 1.0084 | 1.0000 | 0.9944 | 1.0000 |
| unsegmented-indented | 1342 | 1.2832 | 1.0000 | 0.7548 | 1.0000 |
| irregular | 870 | 1.3931 | 1.0000 | 0.7000 | 1.0000 |
| unsegmented-band | 2614 | 2.0838 | 2.0000 | 0.1576 | 0.5492 |
| segmented-bilobed | 3119 | 2.1452 | 2.0000 | 0.0830 | 0.1368 |
| segmented-multilobed | 1284 | 2.8341 | 3.0000 | 0.0787 | 0.1085 |

## 6. Kontrol lintas kelas sel

| kelas | n | median_br | tak_pernah_pecah | mean_lobus | terpisah_di_nol |
|---|---|---|---|---|---|
| Basophil | 1218 | 0.5859 | 0.3933 | 1.8793 | 0.0952 |
| Eosinophil | 3117 | 0.1278 | 0.0834 | 2.1979 | 0.1203 |
| Lymphocyte | 1214 | 1.0000 | 0.9918 | 1.0124 | 0.0058 |
| Monocyte | 1420 | 1.0000 | 0.7535 | 1.2838 | 0.0092 |
| Neutrophil | 3329 | 0.4808 | 0.1334 | 2.2821 | 0.0228 |

## 7. Ablasi praproses (M6)

Target yang dijelaskan: AUC v1 = 0.9227 versus v2 sekitar 0.88.

| varian | tau | rho | diskret | batasi_sel | isi_lubang | pad | min_frak | n | auc | ambang | akurasi_sepertiga | r_lobus_maks | r_lobus_mean | frak_takpecah | frak_terpisah0 | pearson_v1 | spearman_v1 | pindah_sisi_13 | n_banding_v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | 0.7500 | 0.2000 | 0.0000 | 1 | 1 | 1 | 0.0200 | 3329 | 0.9072 | 0.3588 | 0.9044 | 50.5371 | 23.8534 | 0.0544 | 0.0231 | 0.6762 | 0.7145 | 445 | 3279 |
| tanpa_batas_sel | 0.7500 | 0.2000 | 0.0000 | 0 | 1 | 1 | 0.0200 | 3329 | 0.8822 | 0.3588 | 0.8886 | 51.8652 | 23.9534 | 0.0523 | 0.0469 | 0.7359 | 0.7806 | 382 | 3279 |
| tanpa_isi_lubang | 0.7500 | 0.2000 | 0.0000 | 1 | 0 | 1 | 0.0200 | 3329 | 0.9664 | 0.3588 | 0.9324 | 50.5371 | 22.8068 | 0.0186 | 0.0231 | 0.8374 | 0.8432 | 307 | 3279 |
| tanpa_pad | 0.7500 | 0.2000 | 0.0000 | 1 | 1 | 0 | 0.0200 | 3329 | 0.9072 | 0.3588 | 0.9044 | 50.5371 | 23.8719 | 0.0535 | 0.0231 | 0.6773 | 0.7151 | 443 | 3279 |
| tanpa_filter | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 1 | 0.0200 | 3329 | 0.9132 | 0.3588 | 0.8996 | 50.5371 | 23.8534 | 0.0099 | 0.0231 | 0.7594 | 0.7653 | 379 | 3279 |
| diskret_0p5 | 0.7500 | 0.2000 | 0.5000 | 1 | 1 | 1 | 0.0200 | 3329 | 0.9074 | 0.3706 | 0.9075 | 50.5371 | 23.8534 | 0.0544 | 0.0231 | 0.6794 | 0.7146 | 453 | 3279 |
| emulasi_v1 | 0.0000 | 0.0000 | 0.5000 | 0 | 0 | 0 | 0.0000 | 3329 | 0.9319 | 0.3696 | 0.9099 | 70.3847 | 23.0020 | 0.0024 | 0.0475 | 0.9438 | 0.9492 | 156 | 3279 |

## 8. Audit ekor B (M7)

- sel B salah klasifikasi: **195**
  - normal: 173 (88.7%)
  - tak_pernah_pecah: 22 (11.3%)

## 9. Yang harus dijawab berikutnya

- Apakah optimum kalibrasi masih di pojok grid? Bila ya, perluas lagi.
- Apakah plateau Youden cukup lebar sehingga 0.3588 harus ditulis sebagai interval, bukan titik?
- Apakah ablasi sudah menutup selisih AUC v1 vs v2 secara kuantitatif?
- Fase 2D (analisis deferral) dan Fase 2E (sumbu pematangan ig).
