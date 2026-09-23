# LAPORAN FASE 2G — validasi eksternal lintas tahap pematangan

Dijalankan 2026-09-23 11:49 | τ_br 1.25, ρ_br 0.2, τ_lobus 4.0, ρ_lobus 0.4, aturan lubang `tidak` | permutasi 5000

## 0. Uji sanitas

| uji | lolos | detail |
|---|---|---|
| mask sehat lolos seluruh kriteria QC | YA | inti 2821, tepi 0.000, komp 1 |
| inti hilang terdeteksi | YA | inti 0 |
| sel terpotong tepi terdeteksi | YA | frak_tepi 0.1219 |
| sel imatur berinti besar TIDAK dihukum QC | YA | rasio_NC 0.774, lolos True |
| tren menurun kuat terdeteksi | YA | rho -0.980, p_perm 0.0020 |
| data tanpa tren tidak memberi tren palsu | YA | rho -0.061, p_perm 0.0619 |
| tren berbentuk U tidak terbaca monoton | YA | rho -0.007 (harus dekat nol meski polanya kuat) |
| probability of superiority benar | YA | PS 1.0000 |

**Seluruh 8 uji LOLOS.** Uji 4 dan 7 yang terpenting: QC tidak boleh menghukum sel imatur karena imatur, dan tren berbentuk U tidak boleh terbaca sebagai monoton.

Mask prediksi: PBC **10298**, ig **2895** | citra ig **2895** | CSV anotasi 10298 baris

## 1. Lapis QC — ambang dikalibrasi dari mask ANOTASI, diterapkan ke mask prediksi

- QC pada mask anotasi: 1500 tugas, 1.6 detik (1.0 ms/tugas)
Ambang sengaja longgar dan hanya menyasar **kegagalan mask**. Ukuran inti dan rasio NC TIDAK dipakai sebagai kriteria, karena sel imatur memang berinti besar dan ber-rasio-NC tinggi — menghukumnya berarti membuang biologi, bukan derau.

| kriteria | nilai |
|---|---|
| luas_nukleus_min | 500.0000 |
| frak_tepi_maks | 0.0200 |
| luas_sel_min | 2374.3360 |
| luas_sel_maks | 67198.4380 |
| frak_remah_maks | 0.2500 |

Kontrol: mask ANOTASI sendiri lolos QC **99.87%** (1498/1500). Angka ini adalah plafon — mask prediksi tidak mungkin melampauinya secara bermakna.

## 2. Kontrol in-domain — bridge ratio dari mask PREDIKSI versus mask ANOTASI

- QC + br pada mask prediksi PBC: 10298 tugas, 15.2 detik (1.5 ms/tugas)
Mask prediksi PBC lolos QC **99.66%** versus anotasi 99.87%.

| sebab | n | frak |
|---|---|---|
| inti_hilang | 0 | 0.0000 |
| pusat_meleset | 16 | 0.0016 |
| sel_terpotong | 21 | 0.0020 |
| sel_terlalu_kecil | 0 | 0.0000 |
| sel_terlalu_besar | 0 | 0.0000 |
| inti_remah | 0 | 0.0000 |

Pada 10263 sel lolos QC: Spearman(br_anotasi, br_prediksi) = **0.9625**, selisih median **+0.0000**, |selisih| median 0.0022, berpindah sisi ambang 1/3 **3.86%**.

| label | n | rho | beda_median | pindah |
|---|---|---|---|---|
| Basophil | 1214.0000 | 0.8557 | 0.0000 | 0.1153 |
| Eosinophil | 3104.0000 | 0.9322 | 0.0000 | 0.0428 |
| Lymphocyte | 1208.0000 | 0.8764 | 0.0000 | 0.0000 |
| Monocyte | 1413.0000 | 0.9458 | 0.0000 | 0.0099 |
| Neutrophil | 3324.0000 | 0.9559 | 0.0000 | 0.0328 |

**Ongkos pipeline dua tahap** (segmentasi → geometri) dibanding mask anotasi:
| sumber_mask | n_A | n_B | auc | acc_13 | youden |
|---|---|---|---|---|---|
| mask anotasi (Fase 2D) | 1553 | 973 | 0.96736 | 0.93349 | 0.37635 |
| mask prediksi | 1553 | 973 | 0.96469 | 0.93587 | 0.38742 |

ΔAUC **-0.0027**, Δakurasi@1/3 **+0.0024**. Ini plafon untuk `ig`: sel di luar domain latih tidak mungkin lebih baik daripada sel di dalam domain latih.

## 3. Monotonisitas lintas lima tahap pematangan

Seluruh tahap memakai **mask prediksi**, termasuk BNE/SNE — supaya sumber mask tidak berubah di tengah sumbu. Ini keputusan yang menentukan: mencampur mask anotasi (BNE/SNE) dengan mask prediksi (PMY/MY/MMY) akan membuat tren tak bisa dibedakan dari pergantian sumber mask.

- QC + br lima tahap: 6224 tugas, 9.6 detik (1.5 ms/tugas)
**Tingkat kelulusan QC per tahap** — ini hasil tersendiri, bukan sekadar penyaring:
| tahap | n | lolos_qc | frak_lolos | inti_hilang | pusat_meleset | terpotong |
|---|---|---|---|---|---|---|
| 1_promielosit | 592.0000 | 580.0000 | 0.9797 | 0.0000 | 1.0000 | 10.0000 |
| 2_mielosit | 1137.0000 | 1130.0000 | 0.9938 | 0.0000 | 0.0000 | 7.0000 |
| 3_metamielosit | 1015.0000 | 1011.0000 | 0.9961 | 0.0000 | 0.0000 | 4.0000 |
| 4_band | 1633.0000 | 1631.0000 | 0.9988 | 0.0000 | 1.0000 | 1.0000 |
| 5_segmented | 1646.0000 | 1643.0000 | 0.9982 | 0.0000 | 0.0000 | 3.0000 |
| ig_tak_bersubtipe | 151.0000 | 151.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| neutrofil_tak_bersubtipe | 50.0000 | 50.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |


### 3.1 Tren pada seluruh sel

| tahap | n | q25 | median | q75 | frak_ge_13 | tak_pecah | terpisah0 | luas_inti_med | rasio_nc_med |
|---|---|---|---|---|---|---|---|---|---|
| 1_promielosit | 592.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9358 | 0.7990 | 0.0422 | 9875.0000 | 0.5265 |
| 2_mielosit | 1137.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9824 | 0.9164 | 0.0167 | 7770.0000 | 0.5381 |
| 3_metamielosit | 1015.0000 | 0.7931 | 1.0000 | 1.0000 | 0.9852 | 0.5419 | 0.0128 | 7146.0000 | 0.5118 |
| 4_band | 1633.0000 | 0.4813 | 0.5697 | 0.6662 | 0.9712 | 0.0410 | 0.0049 | 5632.0000 | 0.3723 |
| 5_segmented | 1646.0000 | 0.1185 | 0.2677 | 0.4271 | 0.4052 | 0.0109 | 0.0492 | 5766.5000 | 0.3608 |

Spearman ρ = **-0.8063** (p 0.00e+00) | Kendall τ-b = **-0.6928** (p 0.00e+00) | p permutasi 5000× = **2.00e-04** | n = 6023

| pasangan | n_a | n_b | p | ps | arah |
|---|---|---|---|---|---|
| 1_promielosit → 2_mielosit | 592 | 1137 | 0.00000 | 0.55980 | naik |
| 2_mielosit → 3_metamielosit | 1137 | 1015 | 0.00000 | 0.31787 | turun |
| 3_metamielosit → 4_band | 1015 | 1633 | 0.00000 | 0.10021 | turun |
| 4_band → 5_segmented | 1633 | 1646 | 0.00000 | 0.12828 | turun |

`ps` = probability of superiority: peluang sel tahap berikutnya punya br lebih tinggi. Monotonisitas menurun berarti seluruh `ps` di bawah 0,5.

### 3.2 Tren pada hanya sel lolos QC

| tahap | n | q25 | median | q75 | frak_ge_13 | tak_pecah | terpisah0 | luas_inti_med | rasio_nc_med |
|---|---|---|---|---|---|---|---|---|---|
| 1_promielosit | 580.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9534 | 0.8138 | 0.0241 | 9802.5000 | 0.5248 |
| 2_mielosit | 1130.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9885 | 0.9221 | 0.0106 | 7767.0000 | 0.5382 |
| 3_metamielosit | 1011.0000 | 0.7967 | 1.0000 | 1.0000 | 0.9891 | 0.5440 | 0.0089 | 7130.0000 | 0.5118 |
| 4_band | 1631.0000 | 0.4813 | 0.5697 | 0.6664 | 0.9718 | 0.0411 | 0.0043 | 5629.0000 | 0.3723 |
| 5_segmented | 1643.0000 | 0.1188 | 0.2684 | 0.4276 | 0.4060 | 0.0110 | 0.0475 | 5766.0000 | 0.3608 |

Spearman ρ = **-0.8215** (p 0.00e+00) | Kendall τ-b = **-0.7046** (p 0.00e+00) | p permutasi 5000× = **2.00e-04** | n = 5995

| pasangan | n_a | n_b | p | ps | arah |
|---|---|---|---|---|---|
| 1_promielosit → 2_mielosit | 580 | 1130 | 0.00000 | 0.55513 | naik |
| 2_mielosit → 3_metamielosit | 1130 | 1011 | 0.00000 | 0.31490 | turun |
| 3_metamielosit → 4_band | 1011 | 1631 | 0.00000 | 0.09675 | turun |
| 4_band → 5_segmented | 1631 | 1643 | 0.00000 | 0.12800 | turun |

`ps` = probability of superiority: peluang sel tahap berikutnya punya br lebih tinggi. Monotonisitas menurun berarti seluruh `ps` di bawah 0,5.

Kelompok di luar sumbu (dilaporkan terpisah, tidak masuk uji tren):
| tahap | n | lolos_qc | median_br | frak_ge_13 |
|---|---|---|---|---|
| ig_tak_bersubtipe | 151.0000 | 151.0000 | 1.0000 | 0.9934 |
| neutrofil_tak_bersubtipe | 50.0000 | 50.0000 | 0.2259 | 0.3600 |

### 3.3 Montase per tahap

- `montase_G_1_promielosit.png`: 12 dari 592 sel
- `montase_G_2_mielosit.png`: 12 dari 1137 sel
- `montase_G_3_metamielosit.png`: 12 dari 1015 sel
- `montase_G_4_band.png`: 12 dari 1633 sel
- `montase_G_5_segmented.png`: 12 dari 1646 sel
- `montase_G_ig_tak_bersubtipe.png`: 12 dari 151 sel
- `montase_G_neutrofil_tak_bersubtipe.png`: 12 dari 50 sel

## 4. Ringkasan

| ukuran | nilai |
|---|---|
| ΔAUC pipeline dua tahap (in-domain) | -0.00267 |
| Spearman ρ lima tahap (lolos QC) | -0.82150 |
| p permutasi | 0.00020 |
| seluruh pasangan bersebelahan menurun | 0.00000 |

Cara membaca: ΔAUC in-domain adalah plafon kualitas. Bila tren lima tahap kuat DAN kelulusan QC merata antar tahap, validasi eksternal berhasil. Bila kelulusan QC timpang, tren harus dilaporkan bersama ketimpangan itu.
