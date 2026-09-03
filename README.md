# OSTOSENSE AI

Repository ini berisi kontrak data, pipeline AI offline, kontrak keluaran AI
untuk software, dan kernel C++ sisi-host OSTOSENSE. Tujuan integrasinya adalah
agar backend, aplikasi, dan web dapat membaca satu kelas risiko berurutan:
`Safe`, `Monitor`, `Caution`, atau `Urgent`.

## Status saat ini

- Pipeline sintetis dari pembuatan data, ekstraksi fitur, pelabelan, matriks,
  pelatihan OLR, evaluasi, hingga ekspor parameter sudah tersedia.
- Data uji alat P001-P007 saat ini dipakai untuk pemeriksaan kualitas dan
  analisis deskriptif, bukan untuk klaim akurasi model.
- Inferensi Python dan kernel C++ telah diuji untuk paritas sisi-host.
- Belum ada model berbasis data nyata berlabel yang disetujui untuk mode LIVE.
  Karena itu, software wajib menampilkan `AI belum tersedia` pada mode LIVE
  saat ini.

Hasil sintetis hanya membuktikan mekanik pipeline. Hasil tersebut bukan bukti
akurasi OSTOSENSE, validitas sensor, kinerja notifikasi, atau manfaat klinis.

## Mulai cepat

Clone dan siapkan lingkungan Python dari Linux:

```bash
git clone https://github.com/rsyarsya/OSTOSENSE-AI.git
cd OSTOSENSE-AI/ai
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Jalankan seluruh tes dependency-free:

```bash
cd src
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -W error \
  -m unittest discover -s ../tests
```

Stack pelatihan dan visualisasi bersifat opsional dan terpin:

```bash
cd ..
.venv/bin/python -m pip install -e ".[pipeline]"
```

## Integrasi software

Mulai dari tiga artefak berikut:

1. [Kontrak integrasi software](docs/ai-software-integration-contract-v0.1.md)
   menjelaskan aturan tampilan, batas tanggung jawab, dan daftar `MUST FIX`.
2. [JSON Schema](ai/contracts/ai-runtime-output-v0.1.schema.json) adalah kontrak
   yang harus divalidasi backend/mobile/web.
3. [Contoh payload](ai/contracts/examples/) menunjukkan keadaan LIVE tanpa
   model dan hasil simulasi internal.

Menghasilkan payload LIVE saat belum ada model nyata yang disetujui:

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.runtime_output unavailable \
  --output /tmp/ostosense-ai-live.json
```

Aturan singkat untuk software:

- `prediction_available=false`: tampilkan `AI belum tersedia`.
- `model_status=TEST_ONLY`: tampilkan `Simulasi AI: <risk_class>` dan jangan
  memicu notifikasi pasien.
- Jangan mengubah kelas menjadi persentase risiko atau hitung mundur.
- Status kebocoran langsung dari LIG, keterisian kantong, dan kualitas sensor
  tetap merupakan data terpisah dari keluaran AI.

## Struktur repository

| Path | Isi |
|---|---|
| `ai/src/ostosense_contract/` | Skema dan logger data dasar |
| `ai/src/ostosense_ai/` | Pipeline, inferensi, QC, dan runtime output |
| `ai/contracts/` | JSON Schema dan contoh payload untuk software |
| `ai/configs/` | Konfigurasi pipeline yang dapat diaudit |
| `ai/tests/` | Unit test, integration test, dan fixture sintetis |
| `docs/` | Kontrak, rulebook, protokol, runbook, dan panduan integrasi |
| `firmware/include/ostosense/` | Kernel C++ portabel sisi-host |
| `firmware/tests/` | Tes C++ standalone |

Dokumentasi pipeline yang lebih lengkap tersedia di [ai/README.md](ai/README.md)
dan indeks [docs/README.md](docs/README.md).
