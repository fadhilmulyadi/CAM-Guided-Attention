# ANGKA SIAP-PAKAI, FASE 2E

Dihasilkan 2026-09-21 00:33. Setiap klaim ditulis bersama angka pendukungnya dan
bersama keterbatasannya. Tidak ada klaim di sini yang bersandar pada
AUC A-vs-B sebagai kriteria seleksi.


## Klaim 1 -- posisi kelompok C stabil lintas metode

Median bridge ratio kelompok C berada pada **0.3247** sepanjang sumbu A->B. Empat versi metode dengan praproses yang berbeda jauh memberi 0.339 (v1), 0.328 (2B), 0.3214 (2C), 0.3247 (2D/2E). Rentang 0.018 poin.

Ini klaim terkuat: yang diukur bukan artefak satu pipeline.


## Klaim 2 -- ambang 1/3 bukan argmax, tapi ongkosnya sebesar derau

Argmax Youden yang dihitung ulang dari data ini jatuh di **0.3764**, bukan di 1/3. Namun selisih akurasinya bisa dinyatakan dalam satuan sel: memakai 1/3 **lebih benar pada 4 sel dari 2531** dibanding memakai argmax Youden (0.9328 versus 0.9313). Youden memaksimalkan TPR-FPR, bukan akurasi, dan kelompoknya tidak seimbang, jadi kedua fakta itu tidak bertentangan.

> Jangan tulis "ambang optimalnya sepertiga". Tulis bahwa memilih 1/3 adalah keputusan yang ongkosnya tidak terukur pada data ini.


## Klaim 3 -- pengisian lubang adalah kesalahan praproses, bukan pilihan

Sensus Fase 2D: seluruh piksel region tertutup di dalam nukleus berlabel sitoplasma atau vakuola, dengan fraksi jaringan minimum 0.9677 pada 2366 region. Tidak ada artefak segmentasi untuk diperbaiki. Di tingkat sel, 67.6% galat ekor B Fase 2C punya lubang terisi di intinya, versus 14.5% pada sel B yang benar (odds ratio 12.3) dan 6.7% pada sel A yang benar (odds ratio 29.0).


## Klaim 4 -- ada lantai yang tidak bisa dilewati metode apa pun

Akurasi A-vs-B **0.9328** pada 2531 sel. Membuang sel paling ragu tidak menghapus galat: pada cakupan 10% masih tersisa 3 galat (11.9 per 1000). Desil paling yakin sendiri menyisakan **3 galat dari 254 sel**.

Sel-sel itu tidak berada di perbatasan, jadi tidak bisa disebut kesalahan metode di dekat ambang. Kandidatnya: **27 sel B dengan bridge ratio >= 0.50** dan **21 sel A dengan bridge ratio <= 0.20**. Montase adjudikasi menyertakan citra RGB berdampingan dengan mask supaya bisa dipisahkan mana label yang salah dan mana mask yang salah.

> Sampai adjudikasi visual selesai, tulis ini sebagai **dua kemungkinan yang belum dipisahkan**, bukan sebagai derau label.


## Yang BELUM boleh diklaim

- Belum boleh menyebut sisa galat sebagai derau label sebelum montase diperiksa mata manusia.
- Belum ada validasi eksternal lintas tahap pematangan; folder `ig` belum punya mask.
- Ambang 1/3 belum diuji pada dataset di luar PBC.
