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
- Model sintetis dapat menerima fitur sensor nyata `Kap_7` untuk demonstrasi
  integrasi langsung. Keluarannya wajib ditandai `AI Eksperimental` dan belum
  boleh digunakan untuk notifikasi atau keputusan klinis.
- Belum ada model berbasis data nyata berlabel yang tervalidasi. Saat window
  sensor tidak layak, software wajib menampilkan `AI belum tersedia`.

Hasil sintetis hanya membuktikan mekanik pipeline. Hasil tersebut bukan bukti
akurasi OSTOSENSE, validitas sensor, kinerja notifikasi, atau manfaat klinis.

## Mulai cepat

Clone dan siapkan lingkungan Python dari Linux:

```bash
git clone https://github.com/rsyarsya/OSTOSENSE-AI.git
cd OSTOSENSE-AI/ai
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
```

Jalankan seluruh tes dependency-free:

```bash
cd src
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -W error \
  -m unittest discover -s ../tests
```

Stack pelatihan, visualisasi, dan quality tools bersifat opsional dan terpin:

```bash
cd ..
.venv/bin/python -m pip install -e ".[pipeline,quality]"
cd ..
./scripts/verify.sh
```

## Integrasi software

Mulai dari tiga artefak berikut:

1. [Kontrak integrasi software](docs/ai-software-integration-contract-v0.2.md)
   menjelaskan aturan tampilan, batas tanggung jawab, dan daftar `MUST FIX`.
2. [JSON Schema](ai/contracts/ai-runtime-output-v0.2.schema.json) adalah kontrak
   yang harus divalidasi backend/mobile/web.
3. [Contoh payload](ai/contracts/examples/v0.2/) menunjukkan keadaan LIVE tanpa
   hasil, simulasi internal, dan hasil sensor nyata yang belum tervalidasi.
4. [Kontrak input fitur](ai/contracts/ai-feature-input-v0.1.schema.json)
   mengunci urutan lima fitur dan rumus `raw - baseline` untuk komponen yang
   menjalankan reference emitter.

Menghasilkan payload LIVE saat belum ada model nyata yang disetujui:

```bash
cd ai/src
../.venv/bin/python -m ostosense_ai.runtime_output unavailable-v2 \
  --output /tmp/ostosense-ai-live.json
```

Membangun model dan satu payload simulasi deterministik untuk menguji integrasi
software dari ujung ke ujung:

```bash
cd "$(git rev-parse --show-toplevel)"
./scripts/build_engineering_demo.sh /tmp/ostosense-engineering-demo
```

Hasil utama berada di
`/tmp/ostosense-engineering-demo/runtime-engineering-test.json` dan wajib
ditampilkan sebagai `Simulasi AI`, bukan hasil pasien.

Aturan singkat untuk software:

- `prediction_available=false`: tampilkan `AI belum tersedia`.
- `model_status=TEST_ONLY`: tampilkan `Simulasi AI: <risk_class>` dan jangan
  memicu notifikasi pasien.
- `model_status=UNVALIDATED`: tampilkan
  `AI Eksperimental: <risk_class>` dan jangan memicu notifikasi pasien.
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
dan indeks [docs/README.md](docs/README.md). Status kesiapan yang memisahkan
quality repo dari validitas model tersedia di
[docs/repository-readiness-v0.1.md](docs/repository-readiness-v0.1.md).
