# Dokumentasi AI OSTOSENSE

Dokumentasi fondasi AI disusun dalam dua lapisan:

1. [Fondasi Arsitektur AI OSTOSENSE](./ostosense-ai-foundation.md)

   Penjelasan konseptual sistem, model, user flow, runtime, eksperimen,
   evaluasi, edge-cloud split, dan roadmap MVP.

2. [OSTOSENSE AI Data Contract v1.1](./ai-data-contract-v1.1.md)

   Definisi teknis tabel, field, enum quality, calibration events, timestamp,
   dan invariant dataset.

3. [OSTOSENSE AI Label Rulebook v0.3](./ai-label-rulebook-v0.3.md) —
   **BERLAKU. STRUCTURE LOCKED — NUMERIC BOUNDARIES PILOT_PENDING.**

   Aturan penurunan label ordinal `Safe`/`Monitor`/`Caution`/`Urgent` dari
   data eksperimen: konvensi window `(t - W, t]`, enam input pelabelan,
   larangan derivasi label dan fitur, aturan partisi dataset, dan aturan
   fail-closed untuk event wajib yang malformed. Struktur dikunci; nilai
   numerik `B1/B2/B3` tetap `PILOT_PENDING`.
   [v0.2](./ai-label-rulebook-v0.2.md) **digantikan (superseded)** dan
   dipertahankan sebagai bukti historis.

4. [Protokol Pengambilan Data Terintegrasi v0.1](./ai-data-collection-protocol-v0.1.md) —
   **DRAFT, belum dikunci.**

   Panduan praktis pengambilan data terintegrasi kapasitif + LIG: rig uji,
   alur sesi, definisi window, fase dry/baseline, protocol manifest,
   skenario eksperimen, ground truth, matriks pilot, dan QC.

5. [Runbook Shakedown Intake v0.1](./ai-shakedown-runbook-v0.1.md) —
   **DRAFT, belum dikunci.**

   Panduan operasional untuk 3–5 sesi shakedown rekayasa dua-kanal pertama:
   penyiapan protocol manifest, ID stabil, logger tersinkron, baseline/dry,
   event operator, ground truth, penyalinan mentah immutable, perintah
   `ostosense_ai.raw_qc`, dan interpretasi `PASS`/`FAIL`/`PARTIAL`. Template
   manifest ada di
   [`templates/protocol_manifest-shakedown-v0.1.example.csv`](./templates/protocol_manifest-shakedown-v0.1.example.csv)
   (baris `EXAMPLE_ONLY`, wajib diganti; bukan data nyata atau ambang tetap).

6. [Kontrak Integrasi Keluaran AI v0.2](./ai-software-integration-contract-v0.2.md) —
   **BERLAKU UNTUK INTEGRASI SOFTWARE BARU.**
   Menjelaskan keadaan unavailable, simulasi `TEST_ONLY`, dan keluaran sensor
   nyata `LIVE_EXPERIMENTAL`/`UNVALIDATED`, termasuk daftar `MUST FIX` untuk
   backend, aplikasi, serta web. [v0.1](./ai-software-integration-contract-v0.1.md)
   dipertahankan untuk kompatibilitas.

7. [Audit Data Pilot Nyata P001-P007 v0.1](./real-pilot-data-audit-v0.1.md) —
   Ringkasan kualitas, kegunaan, dan keterbatasan 11 sesi nyata tanpa
   mempublikasikan raw CSV.

8. [Readiness Repository v0.1](./repository-readiness-v0.1.md) —
   Pemisahan status repository, integrasi engineering, live eksperimental, dan
   model live tervalidasi.

Gunakan dokumen fondasi untuk diskusi tim, dosen, dan penyusunan laporan.
Gunakan data contract sebagai acuan saat logger dan firmware mulai dikerjakan.
Gunakan label rulebook v0.3 sebagai acuan skrip pelabelan. Struktur pelabelan
sudah dikunci, tetapi nilai boundary `B1/B2/B3` masih `PILOT_PENDING`: jangan
menetapkan boundary numerik atau melatih model pada data nyata sebelum pilot
Tier 1 tersedia dan boundary dibekukan sebelum Final Test.
Gunakan kontrak integrasi keluaran AI sebagai satu-satunya acuan kelas risiko
yang diterima software; jangan menurunkan persentase atau hitung mundur dari
kelas tersebut.
