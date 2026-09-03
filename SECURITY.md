# Keamanan dan Data Sensitif

Repository publik ini tidak boleh berisi credential, konfigurasi `.env`, raw
data P001-P007, data yang dapat mengidentifikasi pengguna, atau artefak lain
yang belum memperoleh izin publikasi.

Jangan melaporkan kerentanan dengan melampirkan data sensitif pada GitHub Issue
publik. Laporkan secara privat kepada koordinator teknis tim OSTOSENSE melalui
kanal internal tim, dengan langkah reproduksi yang telah dibersihkan dari data
pribadi dan credential.

## Batas keamanan integrasi saat ini

- Payload `LIVE_EXPERIMENTAL` tidak boleh memicu notifikasi pasien atau tindakan
  klinis.
- Backend wajib memvalidasi schema dan mengikat payload ke perangkat, sesi,
  waktu penerimaan, serta pengguna yang benar.
- Transport produksi harus terenkripsi dan perangkat harus diautentikasi.
- Broker MQTT publik dan fallback data buatan tidak boleh digunakan untuk data
  pasien atau demonstrasi yang dipresentasikan sebagai data langsung.

Belum ada lisensi distribusi formal pada repository ini. Publikasi source tidak
dengan sendirinya memberikan persetujuan penggunaan klinis, redistribusi, atau
pengolahan data manusia.
