# Runbook Shakedown Intake OSTOSENSE v0.1

**Status: DRAFT, belum dikunci. Untuk 3–5 sesi shakedown rekayasa dua-kanal
pertama.** Runbook ini menjelaskan cara menyiapkan, merekam, dan memeriksa
kualitas (QC) sesi integrasi logger kapasitif + LIG pertama sebelum ada model,
boundary, atau evaluasi performa apa pun.

Runbook ini **bukan** bukti akurasi AI, akurasi notifikasi, performa peringatan
dini, validitas sensor, validitas firmware ESP32, atau nilai klinis. Semua nilai
QC di sini bersifat provisional (`PROPOSED_PILOT_SETTING`) dan mengikuti DRAFT
Data Collection Protocol v0.1. Boundary produksi `B1/B2/B3` tetap
`PILOT_PENDING`.

> Semua identifier, nama field, kode issue, dan perintah shell ditulis dalam
> bahasa Inggris karena bersifat literal terhadap alat. Data shakedown **tidak
> boleh** dipakai untuk melatih atau mengevaluasi model.

---

## 1. Siapkan protocol manifest sebelum sesi

Sebelum merekam, isi satu baris `protocol_manifest.csv` **per sesi** memakai
header kanonis (lihat `docs/templates/protocol_manifest-shakedown-v0.1.example.csv`).
Header wajib persis:

```
session_id,protocol_version,planned_arm,planned_safe_horizon_s,target_fill_or_volume,injection_profile,injection_method,planned_flow_ml_min,physical_leak_observation_method,operator_id,bag_id,sensor_id,device_id
```

Aturan pengisian yang divalidasi QC (`raw_qc` v0.2.1):

- `protocol_version` harus cocok pola `^v0\.1-<label>$`, contoh
  `v0.1-shakedown-a`. `v0.1` polos ditolak.
- `planned_arm` harus enum Arm kontrak (`SAFE`, `LEAK_GRADUAL`, `LEAK_SUDDEN`,
  `FIELD`) dan **sama** dengan `arm` di `sessions.csv`.
- `operator_id`, `bag_id`, `sensor_id`, `device_id` non-kosong, CSV-safe, dan
  cocok dengan `sessions.csv`.
- `target_fill_or_volume` non-kosong untuk semua arm.
- `planned_safe_horizon_s`: **SAFE** wajib integer positif; **LEAK_GRADUAL**
  kosong (leak terencana) atau integer positif (non-leaking fill);
  **LEAK_SUDDEN** dan **FIELD** wajib kosong.
- **LEAK_GRADUAL**: `injection_profile` ∈ {`stepwise`, `continuous`};
  `injection_method` ∈ {`manual_syringe`, `pump`}; `planned_flow_ml_min` kosong
  atau angka positif; `physical_leak_observation_method` non-kosong.
- **SAFE**: `injection_profile`, `injection_method`, `planned_flow_ml_min`
  kosong; `physical_leak_observation_method` boleh kosong.
- **LEAK_SUDDEN**: `physical_leak_observation_method` non-kosong; field injeksi
  boleh kosong.

**Kepengarangan/otorisasi manifest sebelum sesi tidak dapat dibuktikan dari
file** — QC hanya memeriksa isi field. Simpan bukti prosedural (persetujuan
operator, catatan lab) di luar alat sebagai kewajiban prosedur, bukan klaim yang
divalidasi mesin.

Baris di file example ditandai `EXAMPLE_ONLY` dan **wajib diganti** dengan nilai
sesi nyata. Baris example bukan data nyata dan bukan ambang protokol yang
dibekukan.

## 2. Identifier yang stabil

Tetapkan `session_id`, `bag_id`, `sensor_id`, `device_id`, dan `operator_id`
yang stabil dan konsisten antara `sessions.csv` dan `protocol_manifest.csv`.
Jangan mengubah ID di tengah sesi. ID harus CSV-safe (tanpa koma atau baris
baru).

## 3. Logger dua-kanal tersinkron

Rekam kapasitif dan LIG dari satu logger tersinkron pada target **1 Hz**. Satu
baris `samples.csv` memuat kedua kanal pada timestamp yang sama beserta
`system_quality` yang diturunkan dari kualitas kanal (Data Contract v1.1).

## 4. Baseline 60 detik dan dry ≥ 120 detik

- Mulai sesi, jalankan kalibrasi, lalu diamkan rig kering (tanpa cairan) supaya
  jendela **baseline 60 detik** `[start, start+60000)` terisi penuh oleh sample
  1 Hz yang valid. QC menandai `BASELINE_WINDOW_INCOMPLETE` bila jendela tidak
  tercakup penuh (mis. hanya satu sample).
- Fase pre-injection dry minimal **120 detik** dihitung sejak sample pertama
  dengan kedua kanal `OK`. `INJECTION_START` pertama tidak boleh lebih awal dari
  itu; jika lebih pendek, QC menandai `PREINJECTION_DRY_TOO_SHORT`.

## 5. Injeksi dari dalam bag melalui simulasi stoma

Untuk arm leak, injeksikan cairan dari **dalam** bag melewati stoma tiruan
sesuai `injection_profile`/`injection_method` di manifest. Setiap langkah
injeksi dibuka `INJECTION_START` dan ditutup `INJECTION_END`. Pasangan harus
valid dan tidak tumpang tindih; jika tidak, QC menandai
`MALFORMED_REQUIRED_EVENTS`.

## 6. Event operator ber-timestamp RTC

Catat event lewat tombol/log operator dengan timestamp RTC yang sama dengan
`samples.csv`: `INJECTION_START`, `INJECTION_END`, `PHYSICAL_LEAK_OBSERVED`,
`LEAK_FLAG_FIRST`, `LEAK_FLAG_CONFIRMED`. Aturan per arm:

- **SAFE**: tanpa event injeksi dan tanpa `PHYSICAL_LEAK_OBSERVED`.
- **LEAK_GRADUAL** (horizon kosong): minimal satu pasang injeksi valid dan
  tepat satu `PHYSICAL_LEAK_OBSERVED`.
- **LEAK_GRADUAL** (horizon positif = non-leaking fill): minimal satu pasang
  injeksi valid; `PHYSICAL_LEAK_OBSERVED` opsional — jika terjadi, data
  dipertahankan dan QC hanya memberi WARNING `UNPLANNED_PHYSICAL_LEAK`.
- Pada arm **LEAK_GRADUAL**, `PHYSICAL_LEAK_OBSERVED` tidak boleh mendahului
  `INJECTION_START` pertama.
- **LEAK_SUDDEN**: tepat satu `PHYSICAL_LEAK_OBSERVED`; event injeksi opsional
  namun jika ada harus berpasangan valid.
- `end_reason = LEAK_CONFIRMED` wajib memiliki tepat satu
  `PHYSICAL_LEAK_OBSERVED`.
- Event `DEVICE_RESTART` di tengah sesi = error `DEVICE_RESTART_DURING_SESSION`;
  logger harus memulai sesi baru setelah restart.
- Jika kebocoran fisik tercatat, `LEAK_FLAG_FIRST`/`LEAK_FLAG_CONFIRMED` yang
  hilang hanya WARNING (`MISSING_LIG_FLAG_EVENT`) karena LIG masih menunggu
  kalibrasi hardware. Pada sesi tanpa kebocoran fisik, ketiadaan flag LIG adalah
  kondisi yang benar dan tidak menghasilkan warning ini.

## 7. Ground truth: visual + indicator paper + video

Tentukan `PHYSICAL_LEAK_OBSERVED` dari observasi independen (mata operator,
kertas indikator, dan/atau video), bukan dari LIG. Catat metode di
`physical_leak_observation_method`. LIG adalah kanal fail-safe, bukan ground
truth.

## 8. Salin file mentah secara immutable

Setelah sesi, salin `sessions.csv`, `samples.csv`, dan `events.csv` apa adanya
ke direktori arsip read-only. Jangan mengedit, memformat ulang, mengurutkan
ulang, atau memperbaiki file mentah. QC memverifikasi bahwa file input tidak
dimutasi.

## 9. Perintah raw_qc

Jalankan dari root repo (butuh Python 3.11+; tanpa dependency opsional):

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.raw_qc \
  --input <direktori-logger> \
  --config ../configs/raw-qc-v0.1.json \
  --output <direktori-qc> \
  --protocol-manifest <protocol_manifest.csv>
```

Tambahkan `--overwrite` hanya bila memang ingin menimpa artefak QC lama. Output:
`qc_sessions.csv`, `qc_issues.csv`, `qc_report.json` (deterministik; input sama
menghasilkan output byte-identical).

## 10. Interpretasi PASS, FAIL, PARTIAL

- **PASS**: `contract_status` dan `protocol_status` sama-sama `PASS`. Sesi
  memenuhi Data Contract v1.1 dan bagian Protocol v0.1 yang dapat dievaluasi.
- **FAIL**: ada error kontrak atau protokol. Sesi harus diulang.
- **PARTIAL**: kontrak `PASS` tetapi protokol `NOT_EVALUATED` karena
  `protocol_manifest.csv` tidak disertakan. Sesi terstruktur benar namun hanya
  sebagian dievaluasi — sertakan manifest untuk penilaian penuh. Untuk arm
  `LEAK_GRADUAL`, QC tidak menebak apakah sesi merupakan planned leak atau
  non-leaking fill ketika manifest tidak tersedia.

Exit code CLI: `0` bila semua sesi `PASS`, `2` bila ada `FAIL`/`PARTIAL`, `1`
untuk kegagalan invocation/input fatal. Baca `qc_issues.csv` untuk kode issue
dan `detail` per sesi.

## 11. Sesi gagal dipertahankan dan diulang

Sesi `FAIL` **tetap disimpan** dan diulang; jangan pernah menghapus atau
"memperbaiki" data mentah secara diam-diam. Arsipkan sesi gagal beserta output
QC-nya sebagai catatan rekayasa.

## 12. Data shakedown tidak masuk training/evaluasi

Data shakedown adalah karakterisasi rekayasa dan berada **di luar** partisi
`development`, `validation`, dan `final_test`. Jangan membuat
`partition_manifest.csv` untuk shakedown. Data ini **tidak boleh** memasuki
pelatihan atau evaluasi model, dan tidak boleh dipakai untuk klaim performa
OSTOSENSE.

---

Lolos QC shakedown hanya membuktikan mekanik intake dan QC deterministik. Itu
bukan bukti akurasi AI, akurasi notifikasi, performa peringatan dini, validitas
sensor, validitas firmware ESP32, atau nilai klinis.
