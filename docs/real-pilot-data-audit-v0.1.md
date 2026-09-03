# Audit Data Pilot Nyata P001-P007 v0.1

## Ringkasan terverifikasi

Raw P001-P007 adalah hasil pembacaan ESP32 nyata lima kanal pada 10 Hz. Sebelas
file yang tersedia memiliki hash yang sama dengan hasil pemrosesan pilot
sebelumnya. Raw CSV tidak dimasukkan ke repository publik; inventaris mesin
tersedia di `ai/data-manifests/real-pilot-v0.1.inventory.json`.

| Item | Nilai |
|---|---:|
| Sesi/file | 11 |
| Baris raw | 21.887 |
| Kelompok 1 Hz lengkap | 2.184 |
| Sampel akhir parsial yang dilaporkan | 47 |
| Window 120 detik tanpa label | 94 |
| Sesi analisis deskriptif | 10 |

P001 menguji kondisi kering diam, P002 kering dengan gerakan, P003 pengisian
kantong bertahap, P004 kebocoran bertahap, P005 kebocoran mendadak, P006 gangguan
sensor, dan P007 kantong penuh. Pengulangan P004/P005 dipertahankan sebagai sesi
terpisah. P006 sengaja tidak digunakan dalam korelasi deskriptif.

## Yang dapat digunakan

- Pemeriksaan struktur dan timing logger.
- Pemeriksaan respons dan kestabilan kanal secara deskriptif.
- Pengujian alur raw 10 Hz menjadi ringkasan 1 Hz dan fitur tanpa label.
- Demonstrasi integrasi model dengan `Kap_7` yang selalu ditandai
  `LIVE_EXPERIMENTAL` dan `UNVALIDATED`.

## Yang tidak dapat disimpulkan

Tidak tersedia timestamp kejadian dan label risiko per window yang dapat
diaudit. Dua puluh detik awal juga belum diverifikasi sebagai kondisi kering.
Karena pengambilan tambahan tidak memungkinkan, data dipertahankan sebagai
`REAL_PILOT_UNLABELED`.

Nama skenario tidak boleh diubah menjadi label setiap window. Data juga tidak
boleh diaugmentasi untuk membuat kelas buatan atau dipakai menghitung akurasi,
Macro F1, kappa, lead time, performa notifikasi, maupun manfaat klinis.

## Keputusan integrasi

`Kap_7` dikunci sebagai kanal fitur untuk demo langsung eksperimental karena
merupakan kanal kapasitif utama pada pemetaan perangkat. `Kap_4` dan `Kap_5`
tetap tersedia untuk analisis sensor, bukan masukan model lima fitur yang sedang
diintegrasikan. Keputusan ini tidak membuktikan bahwa `Kap_7` adalah kanal
terbaik secara klinis.
