# Kontrak Integrasi Keluaran AI OSTOSENSE v0.2

Dokumen ini adalah acuan integrasi baru untuk backend, aplikasi pasien, dan
dashboard web. Kontrak mesin berada di
`ai/contracts/ai-runtime-output-v0.2.schema.json`; tipe TypeScript berada di
`ai/contracts/typescript/ai-runtime-output-v0.2.ts`. Versi v0.1 tetap tersedia
untuk kompatibilitas dan tidak berubah.

## Batas penggunaan

OSTOSENSE belum mempunyai model empat kelas yang dilatih dan diuji dengan data
nyata berlabel. Model yang tersedia dilatih menggunakan fixture sintetis. Model
tersebut boleh menerima fitur sensor nyata hanya untuk membuktikan bahwa alur
hardware, AI, backend, dan tampilan dapat tersambung.

Keluaran tersebut wajib disebut **AI Eksperimental/Belum Tervalidasi**. Keluaran
tidak boleh memicu notifikasi pasien, menjadi dasar tindakan klinis, atau
ditampilkan sebagai persentase maupun perkiraan waktu kebocoran.

## Tiga keadaan yang sah

| Keadaan | Field utama | Tampilan software | Tindakan |
|---|---|---|---|
| Tidak ada prediksi | `LIVE`, `NONE`, `UNAVAILABLE`, `prediction_available=false` | `AI belum tersedia` | Kosongkan kelas lama dan jangan membuat notifikasi AI. |
| Simulasi internal | `ENGINEERING_TEST`, `SYNTHETIC_FIXTURE`, `TEST_ONLY` | `Simulasi AI: <kelas>` | Hanya halaman/test internal. |
| Sensor nyata eksperimental | `LIVE_EXPERIMENTAL`, `REAL_SENSOR`, `UNVALIDATED` | `AI Eksperimental: <kelas>` | Boleh untuk demo integrasi, tanpa notifikasi pasien. |

Kelas yang mungkin adalah `Safe=0`, `Monitor=1`, `Caution=2`, dan `Urgent=3`.
Kelas menunjukkan urutan keluaran model, bukan persentase kemungkinan bocor.

## Isi payload AI

| Field | Arti |
|---|---|
| `runtime_output_version` | Versi kontrak; integrasi ini memakai `0.2.0`. |
| `mode` | Keadaan operasi yang sah pada tabel di atas. |
| `data_source` | Asal masukan: tidak ada, fixture sintetis, atau sensor nyata. |
| `model_status` | `UNAVAILABLE`, `TEST_ONLY`, atau `UNVALIDATED`. |
| `prediction_available` | Penentu utama apakah kelas boleh ditampilkan. |
| `risk_class` / `risk_class_index` | Pasangan kelas dan indeks yang harus konsisten. |
| `source_window_end_ms` | Akhir window relatif terhadap awal sesi; bukan waktu UTC. |
| `model_input_channel` | `Kap_7` untuk mode sensor nyata. |
| `model_artifact_version` / `model_artifact_sha256` | Identitas model yang menghasilkan kelas. |
| `evidence_scope` | Batas bukti dari hasil tersebut. |
| `warning` | Peringatan audit yang wajib disimpan bersama hasil. |

Payload AI sengaja tidak membawa identitas perangkat/pasien atau waktu UTC.
Backend harus membungkusnya dengan informasi transport berikut:

```json
{
  "device_id": "device-001",
  "session_id": "session-001",
  "received_at": "2026-09-03T12:00:00Z",
  "ai": {
    "runtime_output_version": "0.2.0"
  }
}
```

Contoh di atas hanya menunjukkan envelope. Isi lengkap `ai` harus sama persis
dengan salah satu fixture di `ai/contracts/examples/v0.2/`. Backend menentukan
data usang dari `received_at` dan urutan sesi; `source_window_end_ms` tidak boleh
dianggap sebagai waktu kalender.

## Alur fitur nyata eksperimental

1. ESP32 merekam raw `Kap_7` pada 10 Hz dan menyimpan raw data tanpa smoothing
   permanen.
2. Sepuluh sampel lengkap diringkas menjadi satu nilai 1 Hz. Baseline sesi
   adalah median 60 detik awal pada kondisi kering.
3. Untuk setiap nilai 1 Hz, hitung `delta = Kap_7_raw - baseline_sesi`. Jangan
   membagi delta dengan skala baseline dan jangan memakai kolom
   `Kap_7_delta_norm` dari laporan pilot sebagai masukan model.
4. Setelah tersedia 120 detik data berkualitas baik, hitung lima fitur yang
   sudah dikunci: mean delta, delta terakhir, kemiringan delta per detik,
   varians populasi delta, dan rentang delta. Urutannya harus sama dengan
   `feature_order` pada kontrak input.
5. Perbarui window setiap 10 detik.
6. Jika window tidak lengkap, berisi status sensor buruk, atau nilai tidak
   finite, kirim keadaan `UNAVAILABLE`; jangan gunakan kelas sebelumnya.
7. Jalankan model dan kirim hanya kelas melalui payload v0.2.

Sensor resistif LIG tetap merupakan deteksi kontak cairan langsung yang
terpisah. Status LIG, keterisian kantong, kualitas sensor, dan aturan notifikasi
tidak boleh dimasukkan atau disimpulkan dari payload AI.

## Reference emitter

Input fitur eksperimental harus berbentuk:

```json
{
  "feature_input_version": "0.1.0",
  "data_source": "REAL_SENSOR",
  "model_input_channel": "Kap_7",
  "source_window_end_ms": 120000,
  "feature_basis": "RAW_MINUS_SESSION_BASELINE",
  "feature_order": [
    "cap_delta_mean",
    "cap_delta_last",
    "cap_delta_slope_per_s",
    "cap_delta_variance",
    "cap_delta_range"
  ],
  "features": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Struktur tersebut divalidasi oleh
`ai/contracts/ai-feature-input-v0.1.schema.json`. Nilai contoh nol hanya fixture
format, bukan pembacaan sensor atau contoh kondisi pasien.

Jalankan dari `ai/src`:

```bash
../.venv/bin/python -m ostosense_ai.runtime_output predict-live-experimental \
  --model /path/to/ordinal_model.json \
  --features /path/to/kap7_feature.json \
  --output /tmp/ostosense-ai-live-experimental.json
```

Perintah ini adalah reference emitter untuk integrasi, bukan aplikasi ESP32 dan
bukan bukti validitas prediksi.

Model v0.1 dilatih dengan fixture sintetis yang skalanya tidak membuktikan
kesetaraan fisik dengan `Kap_7` nyata. Karena itu, kelas
`LIVE_EXPERIMENTAL/UNVALIDATED` hanya membuktikan sambungan alur data dan belum
boleh ditafsirkan sebagai risiko kebocoran yang tervalidasi.

Untuk membuat model dan payload `TEST_ONLY` dari clone baru, jalankan dari root
repository setelah memasang extra `[pipeline]`:

```bash
cd "$(git rev-parse --show-toplevel)"
./scripts/build_engineering_demo.sh /tmp/ostosense-engineering-demo
```

File `runtime-engineering-test.json` pada direktori tersebut dapat dipakai untuk
uji backend, mobile, dan web. Script menolak direktori output yang sudah ada dan
tidak menulis model hasil generate ke repository.

## MUST FIX tim software

### Backend (`ostosense-be`)

1. Hapus fallback risk, volume, kelembapan, dan grafik prediksi 42 jam dari
   `sensor.service.ts`; jangan menggantikan data kosong dengan angka buatan.
2. Hentikan penentuan pasien kritis dengan ambang numerik 50/80 di
   `dashboard.service.ts`; gunakan kelas enum yang telah divalidasi.
3. Validasi setiap payload terhadap schema v0.2 sebelum disimpan.
4. Simpan versi kontrak, status model, kelas, hash model, source-window, waktu
   penerimaan, sesi, dan device sebagai data yang dapat diaudit.
5. Ganti koneksi MQTT publik tanpa enkripsi pada `mqtt.service.ts` dengan
   transport terenkripsi dan autentikasi perangkat sebelum data sensitif dipakai.

### Aplikasi pasien (`ostosense-mobile`)

1. Hapus fallback diam-diam dan grafik prediksi 42 jam dari `monitor.tsx`.
2. Render kelas hanya ketika `prediction_available=true`.
3. `TEST_ONLY` harus tampil sebagai `Simulasi AI`; `UNVALIDATED` sebagai
   `AI Eksperimental`; keduanya tidak boleh memicu notifikasi pasien.
4. Bedakan keadaan offline, data usang, sensor bermasalah, dan AI belum tersedia.

### Dashboard (`ostosense-website`)

1. Ganti tipe `risk` numerik dan ambang 50/80 dengan union empat kelas serta
   status ketersediaan.
2. Hapus nilai pasien fallback yang terlihat seperti data nyata.
3. Terapkan label `Simulasi AI`/`AI Eksperimental` dan keadaan unavailable yang
   sama dengan aplikasi.

Repo software tidak diubah oleh batch AI ini.

## Uji penerimaan konsumen

- Seluruh fixture v0.2 diterima; keempat kelas dirender pada mode `TEST_ONLY`
  dan `UNVALIDATED` sesuai label tampilannya.
- Payload dengan field tambahan, versi asing, kelas-indeks tidak cocok, channel
  selain `Kap_7`, hash tidak valid, atau kombinasi state silang ditolak.
- Payload unavailable langsung menghapus kelas lama dari tampilan.
- Tidak ada fixture yang menghasilkan risk percentage, countdown, notifikasi,
  LIG state, bag-fill estimate, atau clinical action.
