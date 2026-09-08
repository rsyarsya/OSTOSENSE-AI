# OSTOSENSE Engineering Model v0.3

Model ini tersedia langsung dalam repo dan paket Python. Ia dilatih dengan
fixture sintetis v0.3 menggunakan StandardScaler dan mord.LogisticAT. Data
development berisi 76 window; data validation tidak dipakai untuk fitting.

Model menerima lima fitur kapasitif dalam urutan yang tercantum dalam
`ordinal_model.json`, bukan lima kanal sensor, persentase volume, atau nilai
kelembapan. Keluaran ordinalnya adalah Safe, Monitor, Caution, dan Urgent.

## Penggunaan yang diperbolehkan

- Menguji instalasi, perhitungan, pertukaran pesan, database, dan tampilan.
- Simulasi internal dengan label TEST_ONLY.
- Sambungan sensor nyata yang telah dikonfirmasi operator dengan label
  AI Eksperimental/UNVALIDATED. Skala sensor nyata belum terbukti setara dengan
  skala data pelatihan; keluaran dapat terkonsentrasi pada satu kelas.

## Batasan

Ini bukan model yang dilatih pada P001-P007, bukan validasi risiko pasien, dan
bukan bukti performa klinis. Jangan membuat notifikasi pasien, persentase risiko,
perkiraan waktu kebocoran, atau tindakan medis dari keluaran ini. Konfirmasi
operator bukan pengganti kalibrasi atau pemeriksaan kualitas otomatis.

`examples.json` berisi empat fitur dari partisi validation sintetis dan keluaran
model yang benar-benar dihitung. Waktu akhir relatif berasal dari waktu window
dikurangi awal sesi sumber; identitas window dan sesi sumber disertakan.
`training_manifest.json` mencatat provenance pelatihan; `bundle.json` mencatat
hash file paket. Hash memeriksa konsistensi byte, bukan sertifikasi sensor/model.

Regenerasi dari root repo pada environment pipeline terpin:

```bash
./scripts/build_engineering_demo.sh /tmp/ostosense-model-rebuild
PYTHONPATH=ai/src ai/.venv/bin/python scripts/package_engineering_model.py /tmp/ostosense-model-rebuild /tmp/ostosense-model-bundle
```

Bandingkan hasil dengan bundle yang dilacak sebelum mengganti versi. Jangan
mengganti file model diam-diam ketika layanan sedang berjalan.
