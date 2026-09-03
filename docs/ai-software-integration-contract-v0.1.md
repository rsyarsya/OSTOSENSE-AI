# Kontrak Integrasi Keluaran AI OSTOSENSE v0.1

> Versi ini dipertahankan untuk kompatibilitas. Integrasi baru menggunakan
> [v0.2](./ai-software-integration-contract-v0.2.md).

Dokumen ini menjelaskan data minimum yang boleh dikirim oleh komponen AI ke
backend, aplikasi, atau web. Kontrak mesin yang menjadi acuan tersedia di
`ai/contracts/ai-runtime-output-v0.1.schema.json`.

## Batas versi ini

Belum ada model berbasis data nyata yang disetujui untuk penggunaan langsung.
Karena itu, versi ini hanya memiliki dua keadaan:

1. `LIVE` tanpa model: AI tidak menghasilkan kelas risiko.
2. `ENGINEERING_TEST`: model dan masukan sintetis boleh menghasilkan kelas
   `Safe`, `Monitor`, `Caution`, atau `Urgent`, tetapi wajib ditandai
   `TEST_ONLY`.

Hasil uji sintetis tidak boleh ditampilkan sebagai kondisi pasien, dasar
notifikasi pasien, atau bukti kinerja klinis.

## Aturan tampilan

| Kondisi payload | Tampilan yang disarankan | Tindakan software |
|---|---|---|
| `prediction_available=false` | `AI belum tersedia` | Jangan menampilkan kelas atau membuat notifikasi AI. |
| `model_status=TEST_ONLY` | `Simulasi AI: <risk_class>` | Boleh dipakai pada halaman demo internal; jangan membuat notifikasi pasien. |
| Prediksi LIVE berbasis model nyata | Belum didukung kontrak v0.1 | Tolak sampai versi kontrak dan model nyata disetujui. |

`risk_class_index` hanya pasangan angka untuk kelas:

| `risk_class` | `risk_class_index` |
|---|---:|
| `Safe` | 0 |
| `Monitor` | 1 |
| `Caution` | 2 |
| `Urgent` | 3 |

Kelas AI bukan persentase kebocoran dan bukan hitung mundur waktu kebocoran.

## Arti field

| Field | Arti |
|---|---|
| `runtime_output_version` | Versi kontrak payload. |
| `mode` | `LIVE` atau `ENGINEERING_TEST`. |
| `data_source` | Sumber data yang diizinkan pada versi ini. |
| `model_status` | Ketersediaan dan batas penggunaan model. |
| `prediction_available` | Apakah `risk_class` benar-benar tersedia. |
| `risk_class` | Salah satu dari empat kelas berurutan, atau `null`. |
| `risk_class_index` | Indeks kelas 0-3, atau `null`. |
| `model_artifact_version` | Versi artefak model, atau `null`. |
| `evidence_scope` | Batas bukti dari hasil tersebut. |
| `warning` | Peringatan yang wajib diteruskan untuk audit. |

## Pemisahan tanggung jawab data

Jangan menggabungkan semua kondisi alat ke dalam payload AI.

| Informasi | Pemilik kontrak |
|---|---|
| Kelas risiko ordinal `Safe`-`Urgent` | AI runtime output ini |
| Cairan menyentuh sensor resistif LIG | Firmware/backend, sebagai status kebocoran langsung terpisah |
| Tingkat keterisian kantong | Firmware/backend dari kanal sensor kantong |
| Kualitas, kabel putus, nilai mentok, data usang | Firmware/backend sebagai status kualitas sistem |
| Identitas perangkat, sesi, pasien, dan waktu | Backend/device protocol |
| Aturan notifikasi dan eskalasi | Backend/aplikasi setelah kebijakan keselamatan disetujui |

## Contoh CLI

Dari direktori `ai/src`, hasil LIVE yang jujur saat model nyata belum tersedia:

```bash
../.venv/bin/python -m ostosense_ai.runtime_output unavailable \
  --output /tmp/ostosense-ai-live.json
```

Uji integrasi dengan artefak model dan fitur sintetis:

```bash
../.venv/bin/python -m ostosense_ai.runtime_output predict-test \
  --model /path/to/ordinal_model.json \
  --features /path/to/synthetic_feature.json \
  --output /tmp/ostosense-ai-test.json
```

Dokumen fitur harus memiliki tepat dua field:

```json
{
  "data_source": "SYNTHETIC_FIXTURE",
  "features": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Contoh payload yang sudah diperiksa berada di `ai/contracts/examples/`.

## MUST FIX sebelum integrasi software

1. Validasi setiap payload terhadap schema dan `runtime_output_version`.
2. Tampilkan kelas hanya ketika `prediction_available=true`.
3. Tampilkan label `Simulasi AI` untuk `TEST_ONLY` dan jangan memicu notifikasi pasien.
4. Hapus data contoh atau fallback yang diam-diam ditampilkan sebagai data nyata.
5. Jangan mengubah kelas atau nilai sensor menjadi persentase risiko, kelembapan,
   volume, atau hitung mundur seperti `42 jam` tanpa kontrak dan validasi terpisah.
6. Tampilkan kebocoran langsung dari LIG sebagai status terpisah dari prediksi AI.
7. Tampilkan keterisian kantong dan kualitas sensor sebagai status terpisah.
8. Ikat setiap data ke perangkat, sesi, pasien, dan waktu yang benar pada backend.
9. Gunakan autentikasi perangkat dan transport terenkripsi; jangan memakai broker
   publik untuk data pasien.
10. Tampilkan keadaan offline, data usang, dan AI tidak tersedia secara eksplisit;
    jangan menggantinya diam-diam dengan data demo.

### Pembagian pekerjaan per repo

**Backend (`ostosense-be`)**

- Sediakan satu endpoint atau jalur pesan untuk menerima payload AI dan validasi
  schema serta versinya sebelum menyimpan atau meneruskan data.
- Simpan hubungan `device`, sesi pengukuran, pasien, dan waktu pengambilan di
  luar payload AI; jangan menebak hubungan tersebut dari nilai sensor.
- Bedakan data terbaru, data usang, perangkat offline, dan AI tidak tersedia.
- Terapkan autentikasi perangkat dan transport terenkripsi. Jangan gunakan
  broker publik untuk data pasien.
- Jangan membuat kelas, persentase, atau hitung mundur pengganti ketika payload
  AI tidak tersedia.

**Aplikasi pasien (`ostosense-mobile`)**

- Baca kelas yang telah divalidasi backend hanya jika
  `prediction_available=true`.
- Untuk `TEST_ONLY`, tampilkan `Simulasi AI: <risk_class>` dan nonaktifkan
  notifikasi pasien berbasis AI.
- Tampilkan `AI belum tersedia`, offline, dan data usang sebagai keadaan yang
  berbeda dan mudah dipahami.
- Pisahkan kelas AI dari kebocoran langsung LIG, keterisian kantong, dan kualitas
  sensor.

**Dashboard web (`ostosense-website`)**

- Terapkan aturan kelas dan keadaan unavailable yang sama dengan aplikasi.
- Hapus nilai hard-coded atau fallback demo yang tampak seperti data pasien.
- Jangan menampilkan persentase risiko atau perkiraan waktu seperti `42 jam`
  dari kelas ordinal ini.
- Beri penanda yang jelas pada semua data `TEST_ONLY` dan jangan mencampurnya
  dengan riwayat pasien.

Repo website, mobile, dan backend tidak diubah dalam batch ini. Daftar di atas
adalah pekerjaan yang perlu diteruskan kepada tim software.
