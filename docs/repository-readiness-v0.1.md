# Readiness OSTOSENSE-AI v0.1

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

## Blocker di luar repo AI

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
