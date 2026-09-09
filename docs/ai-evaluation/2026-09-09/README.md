# Evaluasi AI — 9 September 2026

Mulai dari [laporan evaluasi](laporan.md), [bahan konsultasi](bahan-konsultasi.md), atau [grafik perbandingan](figures/01_perbandingan_model.png).

**Accuracy, F1, dan kappa dalam paket ini berasal dari fixture sintetis; bukan akurasi sensor nyata.**
OLR dan Logistic Regression benar 36/38 window validation (94,74%), Decision Tree 37/38 (97,37%).
Data nyata mencakup 11 sesi tanpa ground truth; analisisnya deskriptif. Model aplikasi tidak diganti.

## Isi paket

- `tables/`: 13 tabel metrik/prediksi sintetis, statistik sensor, QC, dan inventaris hash.
- `figures/`: 21 grafik PNG 300 dpi, termasuk seluruh sesi sensor.
- `checks/`: bukti evaluasi lokal, reproduksi, paritas, dan pemeriksaan visual.
- `logs/repository-quality-full.log`: quality gate pada commit sumber laporan.
- `analysis.json` dan `model_specifications.json`: fakta analisis dan parameter tetap.
- `reproduce.py`: script analisis; `publication_manifest.json`: hash berkas yang dipublikasikan.

CSV sensor mentah, folder kerja hasil pipeline, cache, environment, dan lokasi folder pribadi tidak disertakan.
Bukti dalam `checks/reproduction.json` merekam pemeriksaan analisis lokal sebelum penyiapan publikasi.
Hash di bukti historis tersebut dapat berbeda untuk metadata yang lokasi foldernya disamarkan
dan CSV yang akhir barisnya dinormalisasi menjadi LF; isi numeriknya tetap sama.
Verifikasi script portabel dicatat terpisah dalam `checks/publication_verification.json`.

## Reproduksi lokal

Dari root repo dengan environment proyek dan CSV privat yang hash-nya sesuai inventaris:

```bash
ai/.venv/bin/python docs/ai-evaluation/2026-09-09/reproduce.py --repo . --raw /path/to/private-csv --output /tmp/ostosense-evaluation-new-run
```

Ganti `/path/to/private-csv` dengan folder sumber lokal. Gunakan folder output baru di luar repo.
Perintah memeriksa hash, menjalankan pipeline, menghitung metrik dan menggambar ulang hasil;
keluaran yang diharapkan adalah `ANALYSIS_COMPLETE`. Pemilik clone tanpa CSV privat dapat membaca
hasil yang dipublikasikan, tetapi tidak dapat mereproduksi bagian karakterisasi sensor nyata.
Dependensi mengikuti extra `pipeline` pada `ai/pyproject.toml`; tidak memerlukan toolchain baru.
Sumber CSV dan model tidak diubah. Output turunan dapat disimpan terpisah; tidak diperlukan rollback sumber.

Quality gate repo dijalankan terpisah:

```bash
./scripts/verify.sh
```

317 tes Python bukan 317 eksperimen sensor. Ketepatan risiko nyata dan integrasi pada alat fisik belum dibuktikan.
