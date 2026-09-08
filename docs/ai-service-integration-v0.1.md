# Menjalankan dan Menghubungkan Model AI

## Apa yang sudah tersedia

Repo AI membawa model uji siap pakai, empat contoh keluaran, pengolah jendela
sensor, CLI tanpa dependensi pelatihan, dan layanan HTTP. Tim software tidak
perlu melatih model terlebih dahulu. Model tersedia juga dalam wheel Python.

Versi paket Python adalah `0.2.0`. Model artifact tetap `0.1.0`, runtime tetap
`0.2.0`, dan antarmuka layanan baru `0.1.0`; versi-versi ini mempunyai fungsi
berbeda. Pembaruan paket tidak menyatakan model klinis baru telah dilatih.

Model: `engineering-v0.3`, dilatih pada fixture sintetis v0.3. SHA-256:
`ceb2922841431a38c274e0ebd5c350873a28a95f40a6f6aab2194a1603b93cef`.
[File model dan catatan penggunaan](../ai/src/ostosense_ai/model_assets/engineering_v03/).

Ini belum berarti backend, aplikasi, web, atau ESP32 telah diubah/deploy.
Untuk pekerjaan pada ketiga repo software, gunakan
[daftar MUST FIX dan uji penerimaan](software-must-fix-2026-09-08.md).

## 1. Instalasi dan uji model

Jalankan dari root clone OSTOSENSE-AI. Gunakan Python 3.11 atau 3.12.

```bash
python3 -m venv ai/.venv
ai/.venv/bin/python -m pip install -e ./ai
ai/.venv/bin/python -m ostosense_ai.bundled_model
ai/.venv/bin/python -m ostosense_ai.integration_cli examples --output /tmp/ostosense-examples.json
```

Perintah ketiga memeriksa file model dan menghitung contoh Safe, Monitor,
Caution, Urgent. Perintah terakhir membuat satu JSON berisi empat input serta
keluaran acuan sebenarnya. Exit 0 berarti berhasil. File yang sudah ada ditolak;
pilih nama lain atau gunakan `--overwrite` setelah memeriksa file sebelumnya.
Contoh adalah simulasi internal, tidak boleh dimasukkan sebagai data pasien.

Untuk menjalankan pengolah raw secara lokal tanpa server:

```bash
ai/.venv/bin/python -m ostosense_ai.integration_cli evaluate-window --input ai/contracts/examples/service-v0.1/waiting-input.json --output /tmp/ostosense-waiting.json
```

Contoh tersebut sengaja belum mempunyai pembacaan: exit **2** dan alasan
`COLLECTING_BASELINE` adalah hasil yang benar, bukan kegagalan instalasi.
CLI memakai exit 0 untuk kelas tersedia, 2 untuk belum tersedia, dan 1 untuk
input salah/kegagalan file. Input tidak ditimpa. Byte keluaran identik untuk
input dan versi model yang sama; tidak ada jam eksekusi/random di keluaran.

## 2. Menjalankan layanan AI

Tambahkan dependensi kecil untuk HTTP, bukan paket training:

```bash
ai/.venv/bin/python -m pip install -c ai/requirements-serve.lock -e './ai[serve]'
```

Konfigurasi diberikan oleh pengelola server melalui environment, bukan GitHub:

| Variabel | Pengaturan |
|---|---|
| `OSTOSENSE_SERVICE_TOKEN` | Rahasia acak minimal 32 karakter, hanya untuk backend dan layanan AI. Jangan menaruhnya di browser/mobile. |
| `OSTOSENSE_ALLOW_TEST` | `true` hanya untuk lingkungan uji internal; default `false`. |
| `OSTOSENSE_ENABLE_EXPERIMENTAL` | `true` untuk pengukuran operator yang sudah disetujui; default `false`. |
| `HOST` | Default `127.0.0.1`; container menggunakan `0.0.0.0`. |
| `PORT` | Default `8000`, boleh mengikuti port hosting. |

Setelah pengelola memasang token pada environment terminal/server:

```bash
export OSTOSENSE_ALLOW_TEST=true
export OSTOSENSE_ENABLE_EXPERIMENTAL=false
ai/.venv/bin/python -m ostosense_ai.service
```

Layanan hanya bind loopback secara default. Untuk deployment, tersedia
[Dockerfile](../Dockerfile) yang berjalan sebagai pengguna non-root. Tidak perlu
memasang Docker hanya untuk menjalankan CLI/server Python. Akses lintas jaringan
harus melalui HTTPS atau jaringan internal terisolasi yang dikelola tim,
disertai autentikasi. Jangan membuka port ini langsung untuk aplikasi pasien.

`GET /health` tidak memerlukan token dan tidak mengembalikan data sensor.
HTTP 200 dengan `ready=true` berarti model berhasil dimuat; 503 berarti tidak
siap. Token yang hilang membuat server menolak startup. Pemeriksaan ini bukan
uji akurasi atau pernyataan perangkat medis siap digunakan.

Pada terminal pengujian yang mempunyai token layanan yang sama, periksa empat
kelas melalui HTTP yang sebenarnya:

```bash
ai/.venv/bin/python scripts/probe_ai_service.py --url http://127.0.0.1:8000
```

Harapannya empat baris `Simulasi AI: ... - sesuai model bawaan`, exit 0.
Probe tidak menyimpan/mengirim apa pun ke database atau aplikasi pasien. Jangan
menjalankannya terhadap lingkungan pasien; aktifkan mode uji hanya di staging.

## 3. Endpoint untuk backend

Semua endpoint `/v1/` memerlukan `Authorization: Bearer <token layanan>`.
Permintaan JSON dibatasi 512 KiB. Tidak ada penyimpanan database atau identitas
pasien di layanan AI, sehingga restart layanan tidak menghapus sesi backend.

| Endpoint | Masukan / keluaran |
|---|---|
| `GET /v1/examples` | Empat contoh model; hanya ketika mode uji aktif. |
| `POST /v1/predict-features` | [Feature-input v0.1](../ai/contracts/ai-feature-input-v0.1.schema.json), hanya sumber sintetis. Menghasilkan `TEST_ONLY`. Bukan jalan pintas untuk data nyata. |
| `POST /v1/evaluate-window` | [Raw-window v0.1](../ai/contracts/ai-raw-window-input-v0.1.schema.json). Menghasilkan kelas `UNVALIDATED` atau keadaan belum tersedia. |

Keluaran kedua POST mengikuti
[service-response v0.1](../ai/contracts/ai-service-response-v0.1.schema.json):

- `ai`: **objek runtime v0.2 asli, tanpa field tambahan**.
- `diagnostics`: alasan tersedia/belum tersedia, status pemeriksaan, dan
  pembanding kering jika perhitungan berhasil. Ini bukan metrik performa.
- `feature_input`: lima fitur yang benar-benar dipakai, atau null jika belum
  tersedia. Untuk audit backend, bukan persentase risiko di UI.

Backend membungkus `ai` dengan `device_id`, `session_id`, `acquisition_id`, dan
waktu server. Identitas tersebut berasal dari sesi backend yang telah
diotorisasi, bukan ditebak dari kelas AI. Validasi JSON Schema memakai registry
lokal untuk referensi antarfile, bukan mengunduh `ostosense.local`.
[Tipe TypeScript pendamping](../ai/contracts/typescript/ai-service-v0.1.ts)
tidak menggantikan validasi JSON saat menerima pesan.

HTTP 200 dapat berisi `prediction_available=false`: ini kondisi normal saat
menunggu. HTTP 401: autentikasi; 403: mode belum diaktifkan; 413: terlalu besar;
422: bentuk/nilai input salah; 503: model tidak tersedia. Pada error, timeout,
atau respons tidak valid, backend harus menghasilkan keadaan unavailable
kanonik dan alasan transport pendamping. Jangan mempertahankan kelas lama.

## 4. Data mentah dan awal pengukuran

Operator memulai sesi melalui backend dengan perangkat yang jelas, lalu
mengonfirmasi sambungan sensor dan kondisi kering. Catat identitas operator dan
waktu konfirmasi pada backend. Satu pemasangan/pengukuran mempunyai satu
`acquisition_id` baru; ID ini berbeda dari nama pasien atau ID perangkat tetap.

Raw transport dari ESP32 perlu membawa identitas perangkat, penanda boot/sesi
yang tidak dipakai ulang, `sample_no`, `elapsed_ms`, kelima kanal mentah, serta
status kanal. Firmware penulisan microSD dan transport tersebut tidak
diimplementasikan oleh repo AI. Simpan raw ke microSD/database tanpa menimpa
nilainya dengan hasil filtering.

Untuk permintaan AI, backend mengambil hanya `kap_7_raw` dan `kap_7_status`,
beserta nomor/waktu sampel. Awal waktu dinormalisasi sekali ke awal pengukuran
yang dikonfirmasi, bukan dimulai ulang pada setiap jendela. Nomor sampel boleh
mulai dari bilangan selain nol, tetapi harus berurutan di tiap rentang.

- `baseline_samples`: 600 sampel dari rentang `[0, 60000)` ms yang kering.
  Pembanding dihitung dari median 60 ringkasan satu detik dan tetap sama untuk
  seluruh pengukuran. Kirim data awal yang sama pada permintaan berikutnya.
- `window_samples`: 1.200 sampel dari `[akhir - 120000, akhir)` ms. Untuk akhir
  120000, ini `[0, 120000)`; berikutnya akhir 130000, 140000, dan seterusnya.
- Pengelompokan memakai detik berdasarkan waktu perangkat: tepat 10 sampel
  per detik, median nilai raw, timestamp ringkasan memakai timestamp sampel
  terakhir. Jadi ringkasan berada dalam konvensi fitur `(akhir - W, akhir]`.
- Interval raw 80-120 ms dan interval ringkasan 800-1200 ms diterima. Ini
  pengaturan rekayasa sementara, bukan hasil validasi sensor.
- Hitung delta `raw - pembanding`; gunakan mean, nilai terakhir, kemiringan OLS
  terhadap detik aktual, varians populasi, dan rentang. Urutan/rounding enam
  desimal memakai kernel Python yang sudah ada. Tidak ada `delta_norm` atau
  pencampuran LIG/Kap_4/Kap_5 ke fitur model ini.

Backend harus menunggu watermark waktu perangkat sebelum mengevaluasi akhir
jendela. Jangan mengganti waktu sampel dengan jam penerimaan Wi-Fi. Query 1.200
sampel harus dipaginasi jika database membatasi hasil per query menjadi 1.000.
Data baseline dan window yang bertumpang tindih wajib identik.

## 5. Kualitas, putus sambungan, dan tampilan

Konfirmasi operator wajib, termasuk bila firmware sudah mengirim `OK`.
`NOT_EVALUATED` hanya diterima untuk demo yang dikonfirmasi dan dilaporkan
sebagai `quality_verification=OPERATOR_ONLY`; data aslinya tidak diganti menjadi
`OK`. Bila semua status yang diterima adalah `OK`, keluaran bertanda `AUTOMATIC`:
artinya status berasal dari firmware, bukan diverifikasi mandiri oleh AI.
Aturan ini khusus adapter demo dan tidak mengubah aturan QC/Tier 1.

Status kerusakan/mentok/gap atau status asing, sampel hilang, duplikat, dan
waktu tidak berurutan menghentikan prediksi. Baseline yang bermasalah
memerlukan pengukuran kering baru. Masalah pada window dapat pulih setelah ada
jendela penuh yang bersih; sampel hilang tidak dibuat-buat.

| Keadaan | Tampilan yang diminta |
|---|---|
| Belum mulai / baseline / window | `AI belum tersedia`, disertai tahap yang sedang ditunggu. |
| Simulasi internal | `Simulasi AI: Safe/Monitor/Caution/Urgent`; hanya pada halaman uji. |
| Data nyata, siap dihitung | `AI Eksperimental: <kelas>`; tambahkan keterangan pemeriksaan operator jika belum otomatis. |
| Sensor gagal / AI gagal / data usang | Hapus kelas terkini; tampilkan alasan yang sesuai. |

UI memeriksa setiap 5 detik. Hasil tidak lebih dari 30 detik boleh ditampilkan
sebagai terkini; penilaian berdasarkan usia sampel/jendela dan sesi, bukan jam
fetch layar atau insert ulang. Waktu masa depan, replay data lama, respons
terlambat, atau perangkat lain tidak boleh menyegarkan kelas pasien. Hasil lama
boleh disimpan sebagai riwayat yang jelas waktunya.

Tidak ada notifikasi pasien, persentase risiko, hitung mundur, atau kesimpulan
kulit sehat dari AI ini. Peringatan sensor resistif merupakan jalur terpisah.
Model uji dapat menghasilkan kelas yang sama terus pada skala raw nyata;
jangan mengubah angka agar hasil tampak bervariasi.

## Verifikasi dan rollback

```bash
ai/.venv/bin/python -m pip install -c ai/requirements-serve.lock -e './ai[pipeline,quality,serve,serve-test]'
./scripts/verify.sh
```

Gate tersebut meliputi tes Python/HTTP, Ruff, Pyright, build wheel tanpa training
untuk pemuatan model, kernel C++, dan pemeriksaan repository. Pengujian Docker,
hosting, database, serta ESP32 dicatat terpisah: jangan menyebutnya lulus hanya
karena unit test lulus.

Rollback deployment: nonaktifkan mode uji/eksperimental, arahkan backend ke
unavailable, kemudian gunakan kembali revisi layanan sebelumnya. Jangan
menghapus raw sensor atau riwayat prediksi. Tidak ada migrasi database yang
dijalankan oleh paket ini. Perubahan software dan publikasi memerlukan
persetujuan terpisah sesuai ruang lingkup pekerjaan.
