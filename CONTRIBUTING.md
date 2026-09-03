# Kontribusi ke OSTOSENSE-AI

Repository ini menggunakan Python 3.11 atau lebih baru dan C++17. Mulai dari
salinan repository yang bersih:

```bash
cd ai
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[pipeline,quality]"
cd ..
./scripts/verify.sh
```

`scripts/verify.sh` menjalankan test Python, validasi JSON Schema, Ruff,
Pyright, build package, clean-wheel import, ketiga test C++ host-side, dan
pemeriksaan file terlarang. Perubahan hanya siap direview ketika seluruh gate
lulus.

Smoke test artefak simulasi untuk konsumen software dapat dibuat tanpa
menyimpan hasilnya di repository:

```bash
./scripts/build_engineering_demo.sh /tmp/ostosense-engineering-demo
```

## Aturan perubahan

- Pertahankan runtime contract lama; perubahan field atau arti state harus
  memakai versi schema baru.
- Tambahkan test untuk perubahan perilaku dan kasus penolakan input.
- Jangan memasukkan raw P001-P007, data pasien, credential, `.env`, atau model
  yang belum disetujui.
- Metrik sintetis hanya menguji mekanik pipeline dan tidak boleh ditulis sebagai
  kinerja OSTOSENSE.
- Jangan mengubah batas `B1/B2/B3` produksi tanpa keputusan governance dan data
  yang memenuhi protokol.

Nama distribusi Python saat ini adalah `ostosense-contract`, sedangkan import
yang tersedia adalah `ostosense_contract` dan `ostosense_ai`. Nama distribusi
dipertahankan pada versi ini agar instalasi lama tidak rusak.
