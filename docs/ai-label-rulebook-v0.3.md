# OSTOSENSE AI Label Rulebook v0.3

**Status: STRUCTURE LOCKED — NUMERIC BOUNDARIES PILOT_PENDING**

Dokumen ini mendefinisikan bagaimana data eksperimen terkontrol OSTOSENSE
menerima empat label ordinal `Safe`, `Monitor`, `Caution`, `Urgent`. Seluruh
isi adalah aturan rekayasa untuk prototipe PKM-KC, bukan aturan klinis. Tidak
ada satu pun angka di dokumen ini yang boleh dikutip sebagai threshold klinis
tervalidasi.

**Arti status v0.3.** *Struktur* pelabelan — konvensi window, definisi kelas
ordinal, aturan per skenario, larangan fitur, aturan partisi, dan daftar input
pelabelan — **dikunci pengguna pada 2026-07-27**. Yang masih **PILOT_PENDING**
hanyalah *nilai numerik produksi* boundary `B1/B2/B3`, yang tetap kosong sampai
ada data pilot Tier 1 (Bagian 4, 14). Mengunci struktur tidak membuat
pelabelan atau training pada data nyata menjadi sah sebelum boundary numerik
ditetapkan dari pilot dan dibekukan sebelum Final Test. Pengujian mekanik
pipeline pada data sintetis tetap diperbolehkan dengan boundary fixture yang
secara eksplisit ditandai `ENGINEERING_TEST_ONLY` dan tidak dapat dipakai pada
data nyata.

v0.3 merevisi v0.2 dengan (a) merekonsiliasi Protokol Pengambilan Data
Terintegrasi v0.1, (b) mengunci keputusan struktural yang di v0.2 masih berupa
rekomendasi, dan (c) mengoreksi notasi interval window menjadi setengah-terbuka.
v0.2 dipertahankan sebagai bukti historis (`docs/ai-label-rulebook-v0.2.md`);
v0.3 adalah rulebook yang berlaku. Lihat Bagian 16 untuk riwayat revisi.

Acuan:

- [AI Data Contract v1.1](./ai-data-contract-v1.1.md) — skema `sessions.csv`,
  `samples.csv`, `events.csv`, enum quality, dan event. **Tidak diubah** oleh
  rulebook ini.
- [Protokol Pengambilan Data Terintegrasi v0.1](./ai-data-collection-protocol-v0.1.md)
  — definisi window deterministik (§5), fase dry/baseline (§6), protocol
  manifest dan enam input pelabelan (§7), skenario (§8), ground truth (§9),
  partisi (§11).
- [Fondasi Arsitektur AI](./ostosense-ai-foundation.md) — §6.3 (kelas), §12
  (label time-to-leak), §13 (arm eksperimen), §15 (evaluasi).

## Ringkasan Penguncian (2026-07-27)

Delapan keputusan berikut dikunci pengguna dan menjadi dasar seluruh dokumen:

1. Metrik agreement ordinal resmi = **quadratic weighted Cohen's kappa**
   (`AGENTS.md`, Locked evaluation targets).
2. Unit & konvensi window = **rolling window setengah-terbuka `(t - W, t]`**,
   `W` = 120 detik, stride 10 detik, 1 Hz (Bagian 2).
3. Boundary utama = **skema waktu-bench tetap** (Opsi A); Opsi B hanya analisis
   sensitivitas (Bagian 4).
4. `Urgent` = **pra-leak saja** (Opsi U1); window `t >= T_physical_leak`
   dikeluarkan dari dataset ordinal (Bagian 6).
5. Non-leaking fill → `Safe` **hanya jika** `planned_safe_horizon_s` ditetapkan
   di muka **dan** terpenuhi, di-anchor pada event **`INJECTION_END` terakhir**
   (Bagian 7).
6. Window yang memuat **>= 1 sample kapasitif non-`OK`** dibuang seluruhnya
   (`INVALID_CAP_QUALITY`, Bagian 8).
7. Arm **`LEAK_SUDDEN`** dan **`FIELD`** dikeluarkan dari training/evaluasi
   ordinal (Bagian 7).
8. Nilai numerik **produksi `B1/B2/B3` tetap PILOT_PENDING** dan konfigurasi
   produksi wajib kosong sampai pilot Tier 1. Fixture sintetis terbatas mengikuti
   pengecualian Bagian 4 (Bagian 4, 14).

## 1. Pemisahan Lima Konsep

Lima hal berikut sering tertukar dan wajib dipisahkan dalam seluruh kode,
laporan, dan diskusi:

| Konsep | Sumber | Peran | Bukan |
|---|---|---|---|
| `risk_label` | Diturunkan offline dari kondisi eksperimen (arm, event, timing) | Target training/evaluasi model ordinal 4 kelas | Bukan output model, bukan status notifikasi |
| Ground truth leak fisik | Event `PHYSICAL_LEAK_OBSERVED` (`T_physical_leak`), observasi/video operator | Endpoint independen; jangkar utama pelabelan arm gradual | Bukan deteksi LIG |
| LIG leak flag | `LEAK_FLAG_FIRST` (`T_flag`), `LEAK_FLAG_CONFIRMED` (`T_confirm`) | Kanal fail-safe/deteksi leak aktual; ukuran latensi deteksi | Bukan ground truth, bukan fitur ML, bukan satu-satunya sumber label |
| Quality state | `cap_quality`, `lig_quality`, `system_quality` | Menentukan validitas record untuk training/eval | Bukan kelas risiko |
| Output notifikasi | Kebijakan alert runtime (konfirmasi, hysteresis, mapping alert) | Dievaluasi pada jalur metrik sistem (akurasi notifikasi >= 80%, lead time, false alarm) | Bukan label, bukan metrik klasifier |

Konsekuensi: Macro F1 dan **quadratic weighted Cohen's kappa** dihitung terhadap
`risk_label`; akurasi notifikasi dihitung terhadap ground truth leak fisik.
Keduanya tidak boleh dicampur (lihat `AGENTS.md`, Locked evaluation targets).

## 2. Unit yang Dilabeli

Unit label **dikunci** sebagai rolling window yang identik dengan window fitur;
nilai labelnya diturunkan dari timeline/fase eksperimen (Bagian 3–4), bukan dari
isi sinyal. Pelabelan per sample (granularitas tak terpakai model) dan per fase
(membuang gradasi ordinal) ditolak.

Definisi teknis (selaras Protokol v0.1 §5):

- Window interval **setengah-terbuka `(t - W, t]`** — terbuka di kiri, tertutup
  di kanan: sample pada `t - W` **dikecualikan**, sample pada `t` **disertakan**.
  Dengan sampling tepat 1 Hz, satu window penuh memuat **tepat 120 sample**
  (`W` = 120 detik). Ini menyelaraskan rulebook dengan Protokol v0.1 §5.1 dan
  **menggantikan** notasi `[t - W, t]` pada v0.2 (koreksi off-by-one yang
  dicatat Protokol §15.6).
- Anchor stride deterministik: `t_ref` = timestamp sample pertama sesi pada
  `samples.csv`; kandidat akhir window `t_k = t_ref + (W + k*S) * 1000 ms` untuk
  `k = 0, 1, 2, ...` dengan stride `S` = 10 detik; keanggotaan sample:
  `t_k - W*1000 < ts <= t_k` (milidetik, epoch UTC).
- Window yang melewati batas fase mengikuti kondisi pada `t` (bukan mayoritas
  isi window) agar aturan tetap deterministik.
- **Hanya window penuh yang eligible**: window parsial di awal sesi tidak
  dilabeli dan tidak masuk dataset (`exclusion_reason = PARTIAL_WINDOW`). Aturan
  padding eksplisit boleh diusulkan kelak sebagai perubahan terdokumentasi,
  bukan default diam-diam.

## 3. Timeline Eksperimen Fisik

Jangkar timeline diambil dari `events.csv` dan berlaku per sesi:

```text
t_start            Sesi mulai; baseline kering (dry baseline)
T_inj_start        INJECTION_START     mulai injeksi saline
  (progresi cairan: injeksi bertahap/berulang, INJECTION_END per langkah)
T_physical_leak    PHYSICAL_LEAK_OBSERVED   leak fisik teramati (ground truth)
T_flag             LEAK_FLAG_FIRST          deteksi LIG pertama
T_confirm          LEAK_FLAG_CONFIRMED      deteksi LIG terkonfirmasi
  (periode post-leak: T_physical_leak sampai end_timestamp)
t_end              Sesi selesai (end_reason pada sessions.csv)
```

Catatan:

- Urutan `T_flag` relatif terhadap `T_physical_leak` **tidak boleh
  diasumsikan**. Observasi visual bisa terlambat, sehingga `T_flag` dapat
  mendahului `T_physical_leak`. Selisih keduanya adalah ukuran latensi deteksi
  LIG (bertanda), bukan bahan koreksi label. Setiap kasus urutan terbalik wajib
  dicatat dan diaudit penyebabnya (observasi lambat vs false flag) sebelum sesi
  dipakai untuk evaluasi latensi.
- Pada sesi tanpa leak (arm `SAFE`, non-leaking fill), `T_physical_leak` tidak
  ada; sesi bersifat right-censored.
- Fase dry baseline sebelum `T_inj_start` ada di semua arm dan menjadi sumber
  contoh `Safe` yang paling bersih.
- `INJECTION_END` **terakhir** sesi adalah jangkar `planned_safe_horizon_s` untuk
  non-leaking fill (Bagian 7).

## 4. Definisi Operasional Kelas

Basis pelabelan arm gradual mengikuti fondasi §12: untuk window yang berakhir
pada `t` sebelum leak,

```text
tau = T_physical_leak - t
```

lalu `tau` dipetakan ke kelas dengan tiga boundary berurutan `B1 < B2 < B3`:

| Kelas | Kondisi eksperimen (arm gradual) |
|---|---|
| `Safe` | `tau > B3`, atau fase dry baseline sebelum injeksi |
| `Monitor` | `B2 < tau <= B3` |
| `Caution` | `B1 < tau <= B2` |
| `Urgent` | `0 < tau <= B1` (pra-leak; Bagian 6 mengunci `t >= T_physical_leak` keluar) |

**Nilai produksi B1/B2/B3 tetap PILOT_PENDING dan wajib kosong.** Fondasi §6.3
menyebut boundary awal 2/6/12 jam sebagai definisi operasional v0 untuk
pemakaian riil. Boundary itu TIDAK bisa dipakai mentah pada data bench: sesi
bench terkompresi waktu (injeksi terkontrol, durasi kemungkinan puluhan menit),
sehingga dengan boundary jam-riil hampir semua window pre-leak akan berlabel
`Urgent` dan kelas `Monitor`/`Caution` nyaris kosong.

Skema boundary (dikunci):

- **Opsi A — boundary waktu bench tetap (skema utama, DIKUNCI):** B1/B2/B3
  diturunkan dari durasi protokol injeksi (proporsional terhadap durasi tipikal
  `T_inj_start` sampai `T_physical_leak` pada pilot). Konfigurasi
  machine-readable menyimpan nilai sebagai detik positif berurutan
  (`b1_s < b2_s < b3_s`); laporan boleh menampilkannya dalam menit. Kelas
  dilaporkan eksplisit sebagai "kelas risiko skala-bench". Nilai produksi hanya
  boleh ditetapkan dari data Pilot/Development, dituangkan dalam
  `boundary_config_version` (Bagian 9), dan **dibekukan sebelum Final Test
  dibuka**.
- **Opsi B — boundary fraksional (hanya analisis sensitivitas):** `tau`
  dinormalisasi terhadap durasi pre-leak sesi. Skema ini memakai durasi total
  sesi — informasi masa depan — sehingga kondisi fisik yang sama dapat menerima
  kelas berbeda pada sesi dengan kecepatan injeksi berbeda. Karena itu Opsi B
  **tidak** dipakai sebagai skema pelabelan utama; hanya boleh dipakai offline
  sebagai pembanding dan wajib dilaporkan demikian.

Konfigurasi boundary untuk data nyata tidak boleh berisi angka sebelum
dikonfirmasi dari pilot; nilai final Opsi A adalah keputusan pengguna
berdasarkan pilot Tier 1 (Bagian 14). Satu-satunya pengecualian adalah fixture
di bawah `ai/tests/fixtures/` untuk golden test pipeline sintetis. Fixture
tersebut wajib memakai `boundary_config_version` yang mengandung
`ENGINEERING_TEST_ONLY`, mendeklarasikan origin
`SYNTHETIC_PIPELINE_TEST_ONLY`, dan harus ditolak bila dipasangkan dengan data
berorigin lain.

## 5. Larangan Derivasi Label dan Larangan Fitur

Label TIDAK boleh diturunkan dari:

- Nilai kapasitansi, fitur turunan kapasitansi, atau `delta_C` — melabeli sinyal
  dengan sinyal itu sendiri membuat evaluasi sirkular dan angka metrik tidak
  bermakna.
- Prediksi model (versi mana pun) — self-labeling.
- `lig_raw` sebagai gradasi risiko — LIG adalah kanal fail-safe; wet/contact
  bukan satu-satunya label AI.

Label hanya boleh berasal dari kondisi eksperimen yang diketahui secara
independen: arm sesi, event tercatat (`events.csv`), dan timing terkontrol
protokol.

Arah sebaliknya juga dilarang. **Metadata pembentuk label tidak boleh masuk
feature vector**, karena model akan membaca jawaban dari soal:

- `arm` sesi;
- `tau`, `T_physical_leak`, dan turunannya;
- event injeksi (`INJECTION_START`/`INJECTION_END`) beserta timing-nya;
- nilai boundary B1/B2/B3;
- waktu menuju akhir sesi atau durasi total sesi;
- `risk_label` itu sendiri dan kolom audit dataset turunan (Bagian 11).

Feature vector hanya boleh berisi besaran yang tersedia bagi perangkat pada saat
runtime: fitur sinyal kapasitif dalam window, konteks yang memang dicatat
perangkat (misalnya `activity_state`, `orientation_position`), dan waktu relatif
yang dapat dihitung tanpa mengetahui masa depan sesi. Skrip pembentuk fitur
wajib memiliki daftar kolom terlarang ini sebagai guard eksplisit (sudah
diterapkan pada ekstraktor fitur, `FEATURE_COLUMNS` + `FORBIDDEN_FEATURE_NAMES`).

**Batas pemakaian `features.csv`/`feature_manifest.json` oleh labeler.** Skrip
pelabelan masa depan (ENGINEERING_TEST_ONLY, belum diimplementasikan pada batch
ini) boleh membaca `features.csv` dan `feature_manifest.json` hasil ekstraktor
**hanya** untuk dua tujuan mekanis: (a) menyelaraskan kandidat window pelabelan
dengan window fitur yang identik, dan (b) memverifikasi hash input mentah
(`input_sessions_sha256`/`input_samples_sha256` di manifest fitur) agar label
dan fitur berasal dari data yang sama. **Nilai fitur tidak pernah boleh
menentukan label** — larangan di atas tetap berlaku penuh.

## 6. Keputusan Terkunci: `Urgent` Pra-Leak Saja

**Dikunci: Opsi U1.** `Urgent` = pra-leak saja. Window dengan
`t >= T_physical_leak` **dikeluarkan** dari dataset ordinal
(`exclusion_reason = POST_LEAK`). Model murni prediktif; kondisi leak aktual
ditangani jalur fail-safe LIG dan, pada tampilan, kelas `Leak` terpisah (fondasi
§6.3). Konsekuensi yang diterima: model tidak pernah dilatih pada kondisi leak
berlangsung.

Alternatif yang **ditolak** (Opsi U2, `Urgent` mencakup leak aktual) akan
mencampur "risiko tertinggi sebelum leak" dengan "leak sedang terjadi", tumpang
tindih dengan fungsi fail-safe, dan membuat recall `Urgent` sulit ditafsirkan.

## 7. Aturan per Skenario Sesi

Nilai `arm` memakai enum persis Data Contract v1.1 (`SAFE`, `LEAK_GRADUAL`,
`LEAK_SUDDEN`, `FIELD`).

| Skenario | Sumber label | Aturan v0.3 (dikunci) |
|---|---|---|
| Sesi safe/no-leak (arm `SAFE`) | Kondisi arm | Semua window valid berlabel `Safe` **hanya jika** protokol arm `SAFE` menetapkan di muka kondisi non-leaking dan `planned_safe_horizon_s`, lalu observasi tanpa leak memenuhi horizon tersebut. Untuk arm `SAFE`, horizon dihitung dari sample pertama ketika `cap_quality == OK` dan `lig_quality == OK`. Jika horizon tidak dideklarasikan atau tidak terpenuhi, window dikeluarkan sebagai `CENSORED_NO_SAFE_HORIZON`. Right-censored; tidak ada `tau`. Juga sumber utama pengukuran false alarm (jalur metrik sistem, bukan label). |
| Gradual leak (arm `LEAK_GRADUAL`) | Timeline Bagian 3–4 | Dry baseline → `Safe`; setelah injeksi, mapping `tau` → kelas; `t >= T_physical_leak` dikeluarkan (`POST_LEAK`, Bagian 6). |
| Non-leaking fill / sub-threshold fill (dalam arm gradual) | Kondisi protokol | Right-censored. Window pasca-injeksi berlabel `Safe` **hanya jika** (i) `planned_safe_horizon_s` dideklarasikan di `protocol_manifest.csv` **sebelum** sesi, dan (ii) observasi berlanjut tanpa leak minimal `planned_safe_horizon_s` detik **dihitung dari `INJECTION_END` terakhir** (Bagian 3). Jika horizon tidak dideklarasikan atau tidak terpenuhi (sesi sekadar dihentikan lebih awal), window pasca-injeksi dikeluarkan sebagai `CENSORED_NO_SAFE_HORIZON`. Dry baseline tetap `Safe`. Jika trial yang direncanakan non-leaking ternyata menghasilkan `PHYSICAL_LEAK_OBSERVED` yang valid, sesi diperlakukan sebagai gradual leak berdasarkan endpoint aktual dan ditandai sebagai deviasi protokol; sesi tidak dibuang sebagai malformed. |
| Sudden leak (arm `LEAK_SUDDEN`) | Kondisi arm | **Dikeluarkan** dari training/evaluasi ordinal (`SUDDEN_ARM`; fondasi §13.3: tidak dapat diprediksi dari kapasitansi). Dipakai untuk latensi fail-safe dan evaluasi notifikasi level sistem. |
| Arm `FIELD` | Kondisi arm | **Dikeluarkan** dari training/evaluasi ordinal (`FIELD_ARM_EXCLUDED`). Tidak ada aturan bench yang berlaku; keputusan terpisah diperlukan bila kelak dibutuhkan. |
| Gangguan tekanan/gerakan tanpa cairan | Kondisi protokol + `activity_state`/`orientation_position` | Tidak ada cairan dan tidak ada leak (dalam protokol non-leaking yang ditetapkan di muka) → `Safe`. Hard negative penting; gangguan di tengah arm gradual tidak mengubah label (label tetap dari timeline). |
| Record kualitas invalid | `system_quality` / channel quality | Lihat Bagian 8. Tidak pernah dilabeli ulang. |
| Record post-leak | Timeline | Tidak masuk training ordinal (Bagian 6). Tetap disimpan untuk evaluasi sensor, latensi LIG, dan skin-exposure logging. Tidak pernah dilabeli `Safe`. |
| Event wajib malformed/kontradiktif | — | Sesi **gagal dilabeli** (Bagian 10), bukan dilabeli ulang diam-diam. |

## 8. Record dengan Kualitas Invalid

Aturan tegas (dikunci):

1. Window yang memuat **>= 1 sample `cap_quality != OK`** pada kanal kapasitif
   **dibuang seluruhnya** dari dataset ordinal
   (`exclusion_reason = INVALID_CAP_QUALITY`). Toleransi parsial (mis. < 5%
   sample non-OK) adalah alternatif yang boleh dipertimbangkan nanti dengan
   bukti, bukan default.
2. Record invalid **tidak dilabeli ulang** menjadi `Safe` maupun `Urgent`.
   Ketiadaan data valid bukan informasi risiko rendah ataupun tinggi.
3. Record invalid **tidak dibuang dari penyimpanan**. Ia dikeluarkan dari
   dataset ordinal tetapi dipertahankan untuk pengujian level sistem: verifikasi
   state machine `system_quality`, perilaku suppress prediksi (`ML_UNAVAILABLE`),
   dan ketersediaan fail-safe (`FAILSAFE_DEGRADED`).
4. Status `INITIALIZING` (warming up) diperlakukan sama dengan invalid untuk
   keperluan pelabelan: tidak ada label ordinal yang diterbitkan darinya.
5. `lig_quality` tidak memengaruhi label ordinal (LIG bukan sumber label
   gradasi), tetapi wajib dicatat karena menentukan validitas evaluasi fail-safe
   pada sesi yang sama. Pemakaiannya sebagai anchor awal horizon arm `SAFE`
   hanya menandai bahwa rig telah selesai melakukan inisialisasi sesuai protokol;
   ia bukan fitur, pembentuk kelas, atau penentu `tau`. Kasus khusus
   `ADC_SATURATED` saat wet contact masih
   menunggu konfirmasi hardware (fondasi §8) dan tidak boleh otomatis dianggap
   fault.

## 9. Input Pelabelan Lengkap

Skrip pelabelan deterministik menerima **enam input** (selaras Protokol v0.1
§7.3). Tiga adalah tabel contract; tiga adalah companion artifact offline yang
**tidak mengubah** Data Contract v1.1 dan tidak diwajibkan ada di perangkat.

| # | Input | Peran pelabelan |
|---|---|---|
| 1 | `sessions.csv` (contract) | Identitas sesi, baseline, `end_reason`, `arm` |
| 2 | `samples.csv` (contract) | Deret 1 Hz + quality per sample; wajib untuk pembentukan window dan cek kualitas |
| 3 | `events.csv` (contract) | Jangkar timeline: injeksi, leak fisik, kalibrasi, alert |
| 4 | `protocol_manifest.csv` (companion, Protokol §7.2) | Rencana yang ditetapkan di muka: `planned_arm`, `planned_safe_horizon_s`, identitas `bag_id`/`sensor_id`, metode observasi ground truth |
| 5 | Konfigurasi boundary ber-versi (`boundary_config_version`) | Nilai produksi `b1_s/b2_s/b3_s` (**PILOT_PENDING**; kosong sampai pilot), atau fixture `ENGINEERING_TEST_ONLY` yang hanya sah untuk data sintetis |
| 6 | `partition_manifest.csv` (companion, Protokol §11.3) | Penetapan `dataset_partition` per sesi/bag/sensor sebelum fitting |

Opsional, labeler boleh membaca `features.csv`/`feature_manifest.json` **dalam
batas Bagian 5** (hanya penyelarasan window + verifikasi hash input; bukan sumber
label). Output pelabelan adalah dataset turunan Bagian 11. Determinisme: input
identik → output identik.

## 10. Penanganan Event Wajib Malformed atau Kontradiktif

Skrip pelabelan bersifat **fail-closed dan atomic** terhadap event wajib yang
rusak atau kontradiktif. Jika event wajib untuk sebuah arm tidak ada, ganda,
tidak berurutan secara kausal, atau saling bertentangan, seluruh run pelabelan
**gagal sebelum output dibuat atau ditimpa** dengan error code
`MALFORMED_REQUIRED_EVENTS`. Data tidak dilabeli ulang diam-diam ke kelas apa
pun. Contoh kondisi gagal:

- arm `LEAK_GRADUAL` tanpa `PHYSICAL_LEAK_OBSERVED` padahal
  `end_reason = LEAK_CONFIRMED`;
- `INJECTION_END` tanpa `INJECTION_START` pendahulunya;
- lebih dari satu `PHYSICAL_LEAK_OBSERVED` pada satu sesi tanpa resolusi;
- timestamp event di luar `[start_timestamp, end_timestamp]` sesi;
- arm `SAFE` yang memuat `PHYSICAL_LEAK_OBSERVED` tanpa resolusi deviasi
  protokol dan perubahan arm yang terdokumentasi.

`MALFORMED_REQUIRED_EVENTS` adalah error validasi run, **bukan**
`exclusion_reason` window pada `labels.csv`. Implementasi boleh melaporkannya
melalui exception/diagnostic stderr, tetapi tidak boleh meninggalkan output
parsial. Data mentah tetap tidak diubah dan tersedia untuk audit. Prinsipnya:
ketidakpastian ground truth tidak boleh disembunyikan menjadi label yang
tampak valid.

## 11. Dataset Turunan Berlabel

Jika seluruh input lolos validasi run, output skrip pelabelan adalah **dataset
turunan offline** (artefak baru, tidak mengubah Data Contract v1.1) dengan satu
baris per window kandidat. Kandidat dan `window_id` dibentuk langsung dari
konvensi Bagian 2; `features.csv` tetap opsional dan, bila tersedia, dipakai
untuk verifikasi kecocokan satu-ke-satu. Field minimum wajib:

| Field | Isi |
|---|---|
| `window_id` | ID deterministik `{session_id}-win-{window_index:04d}`; jika `features.csv` diberikan, nilainya wajib cocok persis untuk join satu-ke-satu |
| `session_id` | ID sesi sumber (kunci join ke `sessions.csv`) |
| `window_index` | indeks kandidat window mulai dari 0, identik dengan konvensi feature extractor |
| `window_start` | Timestamp batas kiri window `t - W` (**dikecualikan**, interval `(t - W, t]`) |
| `window_end` | Timestamp akhir window `t` (**disertakan**) |
| `risk_label` | `Safe`/`Monitor`/`Caution`/`Urgent`, kosong jika tidak valid |
| `risk_label_index` | `0/1/2/3` untuk `Safe/Monitor/Caution/Urgent`, kosong jika tidak valid |
| `label_valid` | boolean; `false` untuk window yang dikeluarkan |
| `exclusion_reason` | kosong jika valid; enum alasan: `DUPLICATE_TIMESTAMP`, `PARTIAL_WINDOW`, `TIMING_OUT_OF_TOLERANCE`, `INVALID_CAP_QUALITY`, `CENSORED_NO_SAFE_HORIZON`, `POST_LEAK`, `SUDDEN_ARM`, `FIELD_ARM_EXCLUDED` |
| `rulebook_version` | versi dokumen ini yang dipakai skrip (`v0.3`) |
| `boundary_config_version` | versi konfigurasi B1/B2/B3 yang dipakai |
| `dataset_partition` | `development`/`validation`/`final_test` (Bagian 12) |
| `protocol_deviation` | boolean; `true` bila hasil aktual menyimpang dari rencana sesi |
| `protocol_deviation_reason` | kosong jika tidak ada deviasi; `UNPLANNED_PHYSICAL_LEAK` untuk trial non-leaking yang ternyata bocor |

Window yang dikeluarkan tetap ditulis dengan `label_valid = false` dan
`exclusion_reason` terisi, supaya audit cakupan dan alasan eksklusi dapat
direproduksi. Kolom audit tambahan (misalnya `arm`, `tau`) boleh disertakan
untuk keperluan audit, tetapi seluruh kolom dataset turunan ini tunduk pada
larangan fitur Bagian 5 — tidak satu pun boleh masuk feature vector kecuali
kolom kunci join yang dipakai sekadar untuk menyusun window sinyal.

## 12. Partisi Dataset dan Anti-Leakage

Window 120 detik dengan stride 10 detik saling tumpang tindih ~92%, sehingga
random split per-window hampir pasti menempatkan window yang nyaris identik di
partisi berbeda dan menghasilkan angka evaluasi yang bagus secara palsu. Aturan
wajib:

1. **Split per grup, bukan per window.** Seluruh window dari satu `session_id`
   berada pada partisi yang sama.
2. **Grup diperluas ke hardware fisik.** Sesi-sesi yang memakai baseplate/kantong
   yang sama atau spesimen sensor yang sama masuk partisi yang sama. Identitas
   `bag_id`/`sensor_id` per sesi wajib tercatat pada `protocol_manifest.csv`.
3. Partisi dicatat eksplisit pada `partition_manifest.csv` (Protokol §11.3) dan
   difinalkan pada field `dataset_partition` (Bagian 11) **sebelum** fitting apa
   pun.
4. Partisi `final_test` tunduk pada invariant `AGENTS.md`: tersegel dari fitting,
   preprocessing, dan tuning, termasuk dari penetapan boundary B1/B2/B3 (Bagian
   4) dan dari keputusan aturan eksklusi.

## 13. Skema Pelabelan Minimal yang Dikunci

Ringkasan satu halaman untuk implementasi labeler kelak:

1. Unit label = rolling window penuh `(t - W, t]` yang sama dengan window fitur;
   label dari kondisi pada waktu akhir window; window parsial tidak dilabeli.
2. Sumber label = arm + event + timing; tidak pernah dari kapasitansi, LIG raw,
   atau prediksi model. Sebaliknya, metadata pembentuk label tidak pernah menjadi
   fitur (Bagian 5).
3. Dry baseline dan gangguan-tanpa-cairan dalam protokol non-leaking yang
   ditetapkan di muka → `Safe`. Non-leaking fill → `Safe` hanya jika
   `planned_safe_horizon_s` (anchor `INJECTION_END` terakhir) ditetapkan di muka
   dan terpenuhi; selain itu `CENSORED_NO_SAFE_HORIZON`.
4. Arm `LEAK_GRADUAL` pasca injeksi → mapping `tau` dengan boundary B1/B2/B3
   Opsi A (waktu bench tetap), nilai PILOT_PENDING, dibekukan sebelum Final Test.
5. Window pasca `T_physical_leak` → `POST_LEAK`, keluar dari dataset ordinal.
6. Arm `LEAK_SUDDEN` → `SUDDEN_ARM`; arm `FIELD` → `FIELD_ARM_EXCLUDED`; keduanya
   bukan data ordinal.
7. Window dengan sample kapasitif non-`OK` → `INVALID_CAP_QUALITY`, dipertahankan
   untuk pengujian kualitas sistem; tidak pernah dilabeli ulang.
8. Event wajib malformed/kontradiktif → seluruh run gagal secara atomic dengan
   `MALFORMED_REQUIRED_EVENTS`; tidak ada output parsial (Bagian 10).
9. Enam input pelabelan (Bagian 9); output = dataset turunan Bagian 11, dengan
   partisi per sesi/bag/sensor (Bagian 12).
10. Skrip pelabelan harus deterministik: enam input + konfigurasi boundary →
    output identik setiap dijalankan.

## 14. Status Penguncian dan Syarat Boundary Numerik

**Struktur dikunci** (Ringkasan Penguncian). Yang tersisa:

- **Numerik produksi (PILOT_PENDING):** nilai `B1/B2/B3` — ditetapkan dari data
  Pilot/Development, dituangkan pada `boundary_config_version`, dan dibekukan
  sebelum Final Test dibuka. Sampai itu terjadi, konfigurasi untuk data nyata
  wajib kosong. Fixture angka di `ai/tests/fixtures/` hanya boleh digunakan
  untuk golden test sintetis sesuai Bagian 4 dan bukan boundary OSTOSENSE.
- **Item minor terbuka (bukan blocker struktur):** (a) pemakaian window pra-leak
  arm `LEAK_SUDDEN` sebagai contoh `Safe` tambahan pada evaluasi (bukan
  training) — default v0.3: tidak dipakai; (b) cara mengomunikasikan boundary
  bench vs framing 2/6/12 jam pemakaian riil pada laporan tanpa overclaiming.

Syarat sebelum boundary numerik boleh dikunci dan pelabelan produksi sah:

1. Data logger dua kanal tersinkronisasi dari rig terintegrasi sudah ada dan
   lolos QC (timestamp, gap, quality state) — saat ini belum tersedia.
2. Pilot arm `LEAK_GRADUAL` memberikan estimasi durasi pre-leak sehingga boundary
   B1/B2/B3 dapat ditetapkan dengan justifikasi tertulis.
3. Metode observasi `T_physical_leak` didefinisikan dan diuji (Bagian 15 item 6).
4. Skrip pelabelan deterministik tersedia dengan golden test cases yang mencakup
   semua skenario Bagian 7, semua `exclusion_reason` Bagian 11, aturan malformed
   Bagian 10, dan aturan partisi Bagian 12.
5. Review independen (Codex) atas v0.3 selesai dan direkonsiliasi bila ada
   perubahan substantif berikutnya.

Sampai boundary numerik produksi dikunci, setiap dataset nyata yang dilabeli
dengan dokumen ini harus mencantumkan **"labeled under STRUCTURE-LOCKED
rulebook v0.3 (numeric production boundaries PILOT_PENDING)"** pada
metadata/laporannya. Output golden test sintetis wajib mencantumkan origin
`SYNTHETIC_PIPELINE_TEST_ONLY` dan warning bahwa hasilnya bukan performa model.

## 15. Pertanyaan yang Membutuhkan Konfirmasi Hardware

1. Berapa durasi realistis `T_inj_start` → `T_physical_leak` pada rig bench
   dengan protokol injeksi yang direncanakan? (Menentukan skala boundary.)
2. Apakah wet contact dapat membuat kanal LIG `ADC_SATURATED`, dan bagaimana
   membedakannya dari fault? (Fondasi §8, kontrak v1.1.)
3. Durasi dan aturan validitas kalibrasi LIG (kontrak v1.1, state machine
   pending) — menentukan kapan `T_flag` sah dievaluasi.
4. Stabilitas sampling 1 Hz pada logger terintegrasi — pelabelan window
   mengasumsikan timestamp monoton dan gap terdeteksi (`DATA_GAP`).
5. Reproduksibilitas laju injeksi (pompa vs manual) — pilot kapasitif saat ini
   memakai tekanan manual dan sampling tidak teratur, sehingga belum memenuhi
   asumsi timeline dokumen ini.
6. Bagaimana `PHYSICAL_LEAK_OBSERVED` dioperasionalkan di bench (visual, kertas
   indikator, video + timestamp)? Presisi `T_physical_leak` menentukan presisi
   seluruh label arm gradual — termasuk berapa lag observasi visual yang
   realistis (Bagian 3, urutan `T_flag`).

## 16. Riwayat Revisi

- **v0.3 (2026-07-27)** — **STRUCTURE LOCKED — NUMERIC BOUNDARIES PILOT_PENDING.**
  Menggantikan v0.2 sebagai rulebook berlaku (v0.2 dipertahankan sebagai bukti
  historis). Perubahan utama: keputusan struktural v0.2 dikunci pengguna
  (Ringkasan Penguncian); interval window dikoreksi ke setengah-terbuka
  `(t - W, t]` selaras Protokol v0.1 §5 (Bagian 2); enam input pelabelan
  didokumentasikan eksplisit termasuk `protocol_manifest.csv` dan
  `partition_manifest.csv` (Bagian 9, 12); `planned_safe_horizon_s` di-anchor
  pada `INJECTION_END` terakhir (Bagian 3, 7); arm `FIELD` dinyatakan eksplisit
  dikeluarkan (`FIELD_ARM_EXCLUDED`, Bagian 7); aturan fail-closed untuk event
  wajib malformed/kontradiktif ditambahkan (Bagian 10, `MALFORMED_REQUIRED_EVENTS`);
  batas pemakaian `features.csv`/`feature_manifest.json` oleh labeler ditambahkan
  (Bagian 5); istilah metrik diperjelas menjadi "quadratic weighted Cohen's kappa"
  (Bagian 1). Nilai produksi B1/B2/B3 tetap kosong (PILOT_PENDING); fixture
  sintetis terbatas mengikuti Bagian 4.
- **v0.2 (2026-07-12)** — merekonsiliasi review independen Codex; menggantikan
  v0.1. Perubahan utama: `samples.csv` ditambahkan sebagai input pelabelan wajib;
  non-leaking fill tidak lagi otomatis `Safe`; aturan partisi anti-leakage per
  sesi/bag/sensor; larangan metadata label sebagai fitur; nilai enum arm
  disamakan persis dengan Data Contract v1.1; definisi dataset turunan berlabel;
  Opsi A dijadikan skema boundary utama dan Opsi B diturunkan menjadi analisis
  sensitivitas; asumsi urutan `T_flag` vs `T_physical_leak` dihapus; aturan
  window penuh; istilah "sub-lethal fill" diganti "non-leaking fill". Diarsipkan
  di `docs/ai-label-rulebook-v0.2.md`.
- **v0.1 (2026-07-11)** — draft awal; diarsipkan di
  `.backup/2026-07-12-rulebook-codex-review/ai-label-rulebook-v0.1.md`.
