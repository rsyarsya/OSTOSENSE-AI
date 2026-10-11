# Readiness OSTOSENSE-AI v0.1

> Historical record. This document preserves earlier project assumptions or
> a dated audit. See the [current documentation](../README.md) for supported behavior.

Pembaruan 8 September 2026: aset model siap pakai, CLI deterministik, adapter
raw-window, dan layanan HTTP kini ditambahkan pada working tree repo AI.
Lihat [panduan layanan](../ai-service-integration-v0.1.md) serta
[temuan konsumen terbaru](software-must-fix-2026-09-08.md). Status tabel dan
hasil 3 September di bawah merupakan catatan historis, bukan klaim layanan
baru sudah di-deploy. Hasil batch baru ada pada bagian verifikasi 8 September
di bawah. Status lokal siap diajukan tidak sama dengan rilis/deployment terverifikasi.

Dokumen ini memisahkan kesiapan codebase dari validitas model. Status harus
diperbarui menggunakan bukti perintah, bukan berdasarkan keberadaan file saja.

| Tingkat | Status saat batch dibuat | Makna |
|---|---|---|
| Repository quality | `LOCAL_PASS_CI_PENDING` | Seluruh quality gate lokal lulus; workflow baru belum dapat dinilai di GitHub sebelum di-push. |
| Engineering integration | `CONTRACT_IMPLEMENTED_CONSUMER_PENDING` | Schema, fixture, type, dan reference emitter tersedia; tiga repo software belum menjalankan conformance test. |
| Live experimental | `HOST_REFERENCE_IMPLEMENTED_UNVALIDATED` | Reference emitter host dapat menerima lima fitur nyata `Kap_7`; loop ESP32 belum diterapkan. |
| Validated live | `NOT_READY` | Tidak ada data nyata berlabel per-window, evaluasi Final Test, deployment ESP32, atau bukti klinis. |

## Exit criteria repository

- Python 3.11 dan 3.12: tests, Ruff, Pyright, external JSON Schema validation,
  build wheel, dan clean-wheel import lulus.
- Ketiga C++17 host tests lulus dengan warning sebagai error.
- Tidak ada secret-like file, raw P001-P007, atau generated model header yang
  dilacak Git.
- `git diff --check` bersih dan dokumentasi tidak memiliki link lokal rusak.

## Blocker pada audit awal (historis)

- Backend masih mempunyai nilai fallback, prediksi 42 jam, ambang risiko
  numerik, dan koneksi MQTT publik tanpa enkripsi.
- Mobile masih mempertahankan fallback saat backend gagal.
- Web masih memakai risk numerik dan data pasien fallback.
- Belum ada conformance run lintas backend, mobile, dan web terhadap v0.2.
- Formula berskala pada dokumen fondasi belum menjadi implementasi model v0.1;
  model saat ini memakai `raw - baseline`. Kontrak input mengunci skala aktual,
  tetapi keputusan formula harus difinalkan sebelum pelatihan data nyata.
- Lisensi distribusi formal belum dipilih; integrasi internal tim dapat diuji,
  tetapi distribusi eksternal belum dinyatakan siap.

Status `Validated live` tidak boleh dinaikkan hanya karena integrasi software
berhasil.

## Verifikasi lokal 3 September 2026

Perintah `./scripts/verify.sh` lulus pada Python 3.12.3:

- 287 Python tests: `OK` tanpa skip pada environment `[pipeline,quality]`.
- System Python: 287 tests `OK`, 56 skip yang seluruhnya terkait dependency
  opsional yang tidak terpasang.
- Ruff: seluruh correctness gate lulus.
- Pyright: 0 error dan 0 warning.
- JSON Schema runtime v0.1/v0.2 dan feature-input v0.1: Draft 2020-12 valid;
  seluruh fixture, kedua mode prediksi, dan keempat kelas lulus validator
  eksternal.
- Source distribution dan wheel berhasil dibangun; wheel dapat di-import dari
  environment bersih.
- `data_contract`, `ordinal_inference`, dan `capacitive_features` C++17 lulus
  dengan `-Werror`.
- `pip check` melaporkan tidak ada dependency rusak.
- `build_engineering_demo.sh` berhasil menghasilkan pipeline sintetis 11 sesi,
  model development, dan payload `TEST_ONLY` tanpa menulis artefak ke repo.
- Raw P001-P007, secret-like filename, dan generated model/golden header tidak
  ditemukan pada kandidat file repository.

Python 3.11 belum tersedia pada mesin lokal. Dukungan 3.11 baru dapat dinyatakan
lulus setelah job GitHub Actions 3.11 selesai. Workflow GitHub juga belum dapat
dinyatakan hijau sebelum batch ini di-commit dan di-push.

## Verifikasi lokal 8 September 2026

Paket yang disiapkan: `ostosense-contract==0.2.0`; nama import tetap
`ostosense_ai` dan `ostosense_contract`. Hasil berikut berasal dari working
tree lokal, bukan job GitHub atau pengujian perangkat pasien.

| Pemeriksaan | Bukti hasil |
|---|---|
| `./scripts/verify.sh`, Python 3.12.3 | Exit 0; 317 tests `OK`, tanpa skip pada environment lengkap. |
| System Python, `-W error -m unittest discover -s ai/tests` dengan `PYTHONPATH=ai/src` | 317 tests `OK`, 67 skip karena dependensi opsional tidak terpasang. CLI/model dasar tetap diuji. |
| Ruff dan Pyright | Correctness gate lulus; Pyright 0 error/0 warning. Kedua script Python baru juga lulus Ruff. |
| Paket | Sdist dan wheel 0.2.0 berhasil dibangun. Instalasi wheel tanpa dependensi, import, pemeriksaan versi, dan empat kelas model lulus dari environment terisolasi. |
| HTTP lokal | Proses Uvicorn sungguhan diuji melalui loopback: empat kelas identik dengan acuan, input raw menghasilkan `UNVALIDATED`, akses tanpa token ditolak. Proses ditutup setelah tes. |
| Schema dan CLI | Referensi JSON Schema lokal, payload, determinisme byte, exit 0/1/2, penolakan overwrite, dan perlindungan output/input lulus. |
| C++17 | Ketiga tes `data_contract`, `ordinal_inference`, `capacitive_features` compile/run lulus dengan `-Werror`. |
| Pendamping software | Interface TypeScript baru lulus `tsc --noEmit --strict`; YAML workflow valid. Tidak mengubah atau mengompilasi ulang aplikasi software. |
| Dependensi dan hygiene | `pip check` bersih; `git diff --check`, tautan dokumentasi lokal, dan larangan file rahasia/raw/generated header lulus. |

Uji clean-wheel diperketat setelah ditemukan bahwa `PYTHONPATH=ai/src` dapat
membuat pip menganggap paket sudah terpasang. Baik instalasi maupun pemuatan
wheel kini memakai `python -I`; keberhasilan tidak lagi bergantung pada source
atau editable install di workspace.

Pipeline sintetis direproduksi ke `/tmp/ostosense-release-20260908-rebuild`,
kemudian dikemas dengan `scripts/package_engineering_model.py` ke
`/tmp/ostosense-release-20260908-bundle`. Model, training manifest, empat contoh,
dan bundle manifest identik byte-per-byte dengan aset paket; `diff` exit 0.
Hash model tetap
`ceb2922841431a38c274e0ebd5c350873a28a95f40a6f6aab2194a1603b93cef`.
Data mentah hasil reproduksi hanya berada di direktori sementara tersebut.

## Batas kesiapan batch 0.2.0

- **Siap diajukan/push setelah persetujuan:** source AI, model uji bawaan,
  kontrak, contoh, dokumentasi, dan quality gate lokal telah diverifikasi.
- **Belum terverifikasi:** Python 3.11 dan image Docker karena runtime tersebut
  tidak tersedia lokal. Job Python 3.11/3.12 serta build/probe container sudah
  disiapkan di workflow; hasil GitHub belum boleh diklaim lulus sebelum push.
- **Belum dikerjakan:** commit/push, rilis, deployment, perubahan ketiga repo
  software, migrasi database, dan integrasi ESP32/microSD. Tim software masih
  harus menjalankan checklist penerimaan pada dokumen MUST FIX.
- **Tetap bukan model klinis:** bundle ini dilatih pada fixture sintetis,
  bukan P001-P007 berlabel. Kelas dari sensor nyata wajib `UNVALIDATED` dan
  tidak boleh digunakan untuk notifikasi pasien atau tindakan medis.
- **Izin distribusi:** lisensi formal tetap memerlukan keputusan pemilik;
  batch ini tidak menetapkan lisensi atau izin atas repo pihak lain.

Saat publikasi disetujui, pilih hanya file batch AI/dokumentasinya; jangan
memakai `git add .`. Periksa staged diff dan hasil CI sebelum menandai rilis
siap digunakan lintas environment. File lokal yang tidak terkait, data mentah,
secret, backup, dan ketiga working tree software bukan bagian publikasi.

Backup awal file yang diubah tersedia secara lokal di
`.backup/2026-09-08-live-integration/`. Untuk rollback sebelum publikasi,
pulihkan hanya file terkait dari backup dan hapus hanya file baru batch ini
setelah memeriksa daftar perubahan. Jangan reset seluruh working tree atau
menghapus data pengukuran. Perubahan dependensi project-venv dapat dibatalkan
dengan membuat environment baru dari konfigurasi yang telah dipulihkan.
