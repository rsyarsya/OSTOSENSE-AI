# Pekerjaan Integrasi Tim Software

Status: daftar pekerjaan, **belum diterapkan pada repo software oleh batch AI
ini**. Temuan berasal dari pembacaan snapshot berikut, bukan inspeksi database
atau server produksi:

| Repo | Commit yang diperiksa |
|---|---|
| `dnday/ostosense-be` | `99fbb84eae8c7134e080e2dae8a74e42e861c194` |
| `dnday/ostosense-website` | `8ea8dfba97fb33f4dff9bd3188bcc4f67a35f8cf` |
| `dnday/ostosense-mobile` | `ed4deea5a2cd30c85f812849917f2b2958c2d688` |

## Mengapa AI belum tersedia

Backend sudah mempunyai penerimaan hasil AI di `src/ai/ai.controller.ts` dan
penyimpanan melalui `src/ai/ai.service.ts`. Namun jalur sensor dalam
`src/sensor/mqtt.service.ts` baru menyimpan pembacaan dan memeriksa ambang sensor;
belum memanggil model AI. Menambahkan field/komponen layar saja tidak menjalankan
model. Paket AI baru menyediakan model dan endpoint yang dapat dipanggil.

Panduan utama: [menjalankan layanan AI](ai-service-integration-v0.1.md).

## MUST FIX: Backend

1. Jalankan layanan Python sebagai layanan internal. Set URL dan token melalui
   environment backend, bukan variabel publik frontend. Gunakan timeout 5 detik.
2. Tambahkan operasi mulai/akhiri sesi pengukuran dengan autentikasi operator
   dan otorisasi perangkat/pasien. Rekam konfirmasi sambungan/kering, ID
   pengukuran baru, penanda boot, dan awal waktu perangkat. Tidak boleh memakai
   nama pasien atau satu konstanta perangkat untuk semua akun.
3. Tambahkan penyimpanan nomor sampel, waktu perangkat, identitas pengukuran,
   waktu penerimaan, dan status kanal pada raw transport. Field tidak tersedia
   tidak boleh diasumsikan 10 Hz/OK. Payload lama tetap dapat dipakai untuk
   tampilan sensor, tetapi AI harus unavailable bila syarat raw belum terpenuhi.
4. Setelah menyimpan raw, evaluasi batas jendela baru setiap 10 detik lewat
   `POST /v1/evaluate-window`. Ambil baseline awal yang sama dan jendela 120
   detik. Paginasikan query 1.200 sampel; jangan mengambil hanya 120 baris raw
   10 Hz seolah-olah dua menit. Jangan mengirim persentase volume atau LIG.
5. Validasi respons layanan terhadap schema lokal. Simpan objek `ai` melalui
   validator runtime yang ada. Simpan diagnostik, input fitur, ID pengukuran,
   waktu sumber dan waktu penerimaan sebagai kolom/metadata terpisah.
6. Terapkan kunci unik `(acquisition_id, source_window_end_ms, model_sha256)`
   untuk hasil yang tersedia. Pengiriman ulang tidak boleh membuat prediksi
   lama seolah baru. Tolak hasil respons sesi lama setelah sesi berganti;
   jangan biarkan respons lambat menimpa jendela yang lebih baru.
7. Model gagal, timeout, input buruk, atau sesi berakhir harus mengosongkan
   kelas terkini. Usia data ditentukan dari waktu/urutan sumber, bukan sekadar
   `received_at` baru. Restart perangkat harus memulai pengukuran kering baru.
8. Lindungi penulisan raw/AI dan pembacaan per pasien pada database. Migrasi
   `ai_predictions.sql` yang diperiksa belum mendefinisikan kebijakan RLS;
   konfigurasi produksi tidak dapat disimpulkan dari file ini. Batasi akses
   browser/mobile ke perangkat yang diotorisasi. Jangan menaruh service-role
   credential di klien.
9. Pertahankan penyimpanan sensor meskipun AI tidak tersedia. Jangan mengubah
   `TEST_ONLY`/`UNVALIDATED` menjadi pemicu notifikasi pasien atau status darurat.

## MUST FIX: Mobile

- `src/hooks/use-sensor-series.ts`: query sensor saat ini belum difilter ke
  sesi/perangkat. Batasi sesuai perangkat yang terikat ke akun; gunakan waktu
  sampel, bukan `Date.now()` saat fetch, untuk menilai keterbaruan data.
- `src/hooks/use-ai-prediction.ts`: gunakan binding perangkat dan pengukuran
  yang sama dengan sensor. Perbarui setiap 5 detik; kelas lebih dari 30 detik,
  waktu masa depan, error, atau sesi lama bukan kelas terkini.
- `src/app/index.tsx`: jangan menggabungkan Caution/Urgent eksperimental ke
  peringatan pasien. Badge besar tidak boleh default `Aman` akibat tidak ada
  hasil/data. Teks `Semua sensor berfungsi normal` memerlukan bukti kualitas,
  bukan sekadar aplikasi berhasil mengambil data.
- Beranda dan kartu monitor harus sama-sama menyertakan `Simulasi AI` atau
  `AI Eksperimental`. Jangan hanya menampilkan Safe/Monitor tanpa asal/status.
  `NOT_EVALUATED` bukan bukti sensor sehat; tampilkan keterangan operator.

## MUST FIX: Website

- `src/lib/ai-prediction.ts`: tangani query gagal, umur data, ID pengukuran, dan
  status model secara konsisten. Tidak boleh memakai kelas sesi lain.
- `src/lib/patient.ts`: ganti pemetaan nama pasien/ID uji tetap dengan binding
  perangkat yang diotorisasi dan tercatat. Akun lain tidak boleh membaca
  perangkat uji hanya karena nama tampilan cocok.
- `patient-detail-modal.tsx`: detail saat ini memperbarui data sensor, tetapi
  hasil AI perlu polling 5 detik juga, termasuk ketika modal tetap terbuka.
  Terapkan aturan yang sama pada dashboard dan workspace pasien.
- Tambahkan kendali operator `Mulai pengukuran` dan `Akhiri pengukuran` pada
  dashboard. Konfirmasi kering/sambungan disimpan di backend. Jangan mengirim
  token layanan AI dari JavaScript browser.
- Saat berganti pasien, segera kosongkan kelas lama sambil menunggu data baru.
  Kegagalan fetch harus tampil sebagai gagal/belum tersedia, bukan Safe.

## Uji penerimaan bersama

| Uji | Hasil yang harus dibuktikan |
|---|---|
| Empat contoh model | Kelas keluaran Python yang sebenarnya tersimpan dan tampil sama di web/mobile, seluruhnya berlabel simulasi internal. |
| Awal sesi nyata | Konfirmasi operator tercatat; baseline/window belum lengkap tidak mempunyai kelas. |
| Pengukuran cukup | Satu kelas eksperimental muncul dari panggilan layanan, dengan hash model dan waktu sumber yang sama. |
| Putus/mentok/sampel hilang | Tidak ada kelas baru palsu; kelas terkini dikosongkan. |
| NOT_EVALUATED | Tidak diterima tanpa operator; bila diterima, keterangan pemeriksaan manual tetap terlihat. |
| Data usang / masa depan | Refresh layar atau insert ulang tidak membuatnya terkini. |
| Dua perangkat/dua pasien | Baseline, fitur, kelas, dan akses data tidak tercampur. |
| Reset/restart/duplikasi | Sesi lama tidak mengisi sesi baru; tidak ada duplikasi hasil tersedia. |
| Layanan AI mati | Raw tetap tersimpan; UI menampilkan kegagalan AI tanpa status Safe pengganti. |
| Notifikasi | Tidak ada notifikasi pasien dari keluaran simulasi/eksperimental. |

Jalankan pada lingkungan uji dengan akun dan perangkat uji yang jelas. Rekam
commit empat repo, versi/model SHA, versi migrasi, serta bukti request-response
yang sudah disamarkan. Jangan mengunggah raw pasien, token, atau log berisi
rahasia. Publikasi ketiga repo, migrasi produksi, dan deployment dilakukan oleh
tim software setelah persetujuan, bukan otomatis oleh repo AI.

Terkait firmware: tim elektronis masih perlu membuktikan penulisan microSD,
sampling/timestamp, penanda restart, serta kiriman raw sesuai kontrak. Lulus uji
Python lokal tidak membuktikan bagian tersebut.
