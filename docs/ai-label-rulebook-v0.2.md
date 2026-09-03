# OSTOSENSE AI Label Rulebook v0.2

**Status: DRAFT — NOT LOCKED**

Dokumen ini mendefinisikan bagaimana data eksperimen terkontrol OSTOSENSE
kelak menerima empat label ordinal `Safe`, `Monitor`, `Caution`, `Urgent`.
Seluruh isi adalah usulan rekayasa untuk prototipe PKM-KC, bukan aturan
klinis. Tidak ada satu pun angka di dokumen ini yang boleh dikutip sebagai
threshold klinis tervalidasi.

Versi ini merevisi v0.1 berdasarkan hasil review independen Codex
(2026-07-12); lihat Bagian 15 untuk riwayat revisi. Versi sebelumnya
diarsipkan di `.backup/2026-07-12-rulebook-codex-review/`.

Acuan:

- [AI Data Contract v1.1](./ai-data-contract-v1.1.md) — skema
  `sessions.csv`, `samples.csv`, `events.csv`, enum quality, dan event.
- [Fondasi Arsitektur AI](./ostosense-ai-foundation.md) — khususnya §6.3
  (kelas), §12 (label time-to-leak), §13 (arm eksperimen), §15 (evaluasi).

Rulebook ini tidak mengubah Data Contract v1.1. Label `risk_label`
diturunkan **offline** dari `sessions.csv`, `samples.csv`, dan `events.csv`;
tidak ada kolom baru yang diwajibkan pada logger. `samples.csv` wajib
menjadi input karena pembentukan window (timestamp, gap) dan pengecekan
kualitas per sample (Bagian 8) tidak mungkin dilakukan hanya dari
`sessions.csv` dan `events.csv`.

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

Konsekuensi: Macro F1 dan weighted kappa dihitung terhadap `risk_label`;
akurasi notifikasi dihitung terhadap ground truth leak fisik. Keduanya tidak
boleh dicampur (lihat `AGENTS.md`, Locked evaluation targets).

## 2. Unit yang Dilabeli

Tiga opsi dipertimbangkan:

| Opsi | Deskripsi | Masalah utama |
|---|---|---|
| Per sample (1 Hz) | Setiap baris `samples.csv` diberi label | Tidak cocok dengan unit inferensi (window); noise label tinggi; volume label semu besar |
| Per rolling window | Window fitur (v0: 120 detik, stride 10 detik) yang berakhir pada waktu `t` diberi label dari kondisi eksperimen pada `t` | Butuh definisi window yang konsisten dengan pipeline fitur |
| Per fase eksperimen | Satu label per fase (baseline, injeksi, pre-leak, post-leak) | Terlalu kasar; menghilangkan gradasi ordinal dalam satu fase |

**Rekomendasi: label per rolling window.** Alasannya: model melakukan
inferensi per window, sehingga unit label harus sama dengan unit prediksi;
nilai labelnya sendiri tetap diturunkan dari timeline/fase eksperimen
(Bagian 3–4), bukan dari isi sinyal. Ini opsi paling sederhana yang masih
defensible: pelabelan per fase murni membuang informasi ordinal, pelabelan
per sample menciptakan granularitas yang tidak pernah dipakai model.

Definisi teknis:

- Window dengan rentang `[t - W, t]` menerima label dari kondisi eksperimen
  pada waktu akhir `t`. Window yang melewati batas fase mengikuti kondisi
  pada `t` (bukan mayoritas isi window) agar aturan tetap deterministik.
- **Hanya window penuh yang eligible**: window pertama yang sah pada sebuah
  sesi adalah window yang sudah mencakup durasi penuh `W` (v0: 120 detik).
  Window parsial di awal sesi tidak dilabeli dan tidak masuk dataset.
  Aturan padding eksplisit boleh diusulkan kelak sebagai perubahan
  terdokumentasi, bukan default diam-diam.

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
  mendahului `T_physical_leak`. Selisih keduanya adalah ukuran latensi
  deteksi LIG (bertanda), bukan bahan koreksi label. Setiap kasus urutan
  terbalik wajib dicatat dan diaudit penyebabnya (observasi lambat vs
  false flag) sebelum sesi dipakai untuk evaluasi latensi.
- Pada sesi tanpa leak (arm `SAFE`, non-leaking fill), `T_physical_leak`
  tidak ada; sesi bersifat right-censored.
- Fase dry baseline sebelum `T_inj_start` ada di semua arm dan menjadi
  sumber contoh `Safe` yang paling bersih.

## 4. Definisi Operasional Kelas (Usulan)

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
| `Urgent` | `0 < tau <= B1` (menjelang leak; lihat Bagian 6 untuk t >= T_physical_leak) |

**Nilai B1/B2/B3 belum ditetapkan.** Fondasi §6.3 menyebut boundary awal
2/6/12 jam sebagai definisi operasional v0 untuk pemakaian riil. Boundary itu
TIDAK bisa dipakai mentah pada data bench: sesi bench terkompresi waktu
(injeksi terkontrol, durasi kemungkinan puluhan menit), sehingga dengan
boundary jam-riil hampir semua window pre-leak akan berlabel `Urgent` dan
kelas `Monitor`/`Caution` nyaris kosong. Ini masalah feasibility nomor satu
rulebook ini (lihat Bagian 12).

Dua opsi penetapan boundary, keduanya **asumsi rekayasa prototipe, bukan
threshold klinis**:

- **Opsi A — boundary waktu bench tetap (skema utama yang
  direkomendasikan):** B1/B2/B3 dalam menit, diturunkan dari durasi
  protokol injeksi (misalnya proporsional terhadap durasi tipikal
  `T_inj_start` sampai `T_physical_leak` pada pilot). Kelas dilaporkan
  eksplisit sebagai "kelas risiko skala-bench". Nilai boundary hanya boleh
  ditetapkan dari data Pilot/Development, dituangkan dalam
  `boundary_config_version` (Bagian 9), dan **dibekukan sebelum Final Test
  dibuka**.
- **Opsi B — boundary fraksional (hanya analisis sensitivitas, bukan skema
  utama):** `tau` dinormalisasi terhadap durasi pre-leak sesi, boundary
  pada fraksi tetap (misalnya 0.15/0.40/0.70). Skema ini memakai durasi
  total sesi — informasi masa depan — sehingga kondisi fisik yang sama
  dapat menerima kelas berbeda pada sesi dengan kecepatan injeksi berbeda.
  Karena itu Opsi B tidak dipakai sebagai skema pelabelan utama; ia hanya
  boleh dipakai offline sebagai pembanding/analisis sensitivitas dan harus
  dilaporkan demikian.

Angka contoh pada kedua opsi adalah placeholder dan tidak boleh dipakai
sebelum dikonfirmasi dari data pilot. Angka final Opsi A adalah keputusan
pengguna berdasarkan pilot Tier 1 (Bagian 12).

## 5. Larangan Derivasi Label dan Larangan Fitur

Label TIDAK boleh diturunkan dari:

- Nilai kapasitansi, fitur turunan kapasitansi, atau `delta_C` — melabeli
  sinyal dengan sinyal itu sendiri membuat evaluasi sirkular dan angka
  metrik tidak bermakna.
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
- `risk_label` itu sendiri dan kolom audit dataset turunan (Bagian 9).

Feature vector hanya boleh berisi besaran yang tersedia bagi perangkat pada
saat runtime: fitur sinyal kapasitif dalam window, konteks yang memang
dicatat perangkat (misalnya `activity_state`, `orientation_position`), dan
waktu relatif yang dapat dihitung tanpa mengetahui masa depan sesi. Skrip
pembentuk fitur wajib memiliki daftar kolom terlarang ini sebagai guard
eksplisit.

## 6. Keputusan Eksplisit: Apakah `Urgent` Mencakup Leak Aktual

Ini keputusan desain yang belum dikunci dan tidak boleh diputuskan
diam-diam, karena deteksi leak aktual adalah tanggung jawab kanal fail-safe
LIG, bukan model ordinal.

- **Opsi U1 — `Urgent` = pre-leak saja (konsisten dengan fondasi §12):**
  window dengan `t >= T_physical_leak` dikeluarkan dari dataset ordinal.
  Model murni prediktif; kondisi leak aktual ditangani jalur LIG dan, pada
  tampilan, kelas `Leak` terpisah (fondasi §6.3). Konsekuensi: model tidak
  pernah dilatih pada kondisi leak berlangsung.
- **Opsi U2 — `Urgent` mencakup leak aktual:** window pasca
  `T_physical_leak` dilabeli `Urgent`. Konsekuensi: definisi `Urgent`
  bercampur antara "risiko tertinggi sebelum leak" dan "leak sedang
  terjadi", tumpang tindih dengan fungsi fail-safe, dan angka recall
  `Urgent` menjadi sulit ditafsirkan.

**Rekomendasi v0.2: Opsi U1.** Selaras dengan fondasi §12 (window pasca
leak tidak masuk training prediktif), menjaga pemisahan prediktif/reaktif,
dan dikonfirmasi review independen Codex (2026-07-12) sebagai opsi paling
konsisten dengan arsitektur kelas `Leak` terpisah dan jalur fail-safe LIG.
Tetap berstatus usulan — pengguna yang mengunci.

## 7. Aturan per Skenario Sesi

| Skenario | Sumber label | Aturan v0.2 |
|---|---|---|
| Sesi safe/no-leak (arm `SAFE`) | Kondisi arm | Semua window valid berlabel `Safe` **hanya jika** protokol arm `SAFE` menetapkan di muka kondisi non-leaking (level isi/beban di bawah ambang desain) dan durasi observasi minimum. Right-censored; tidak ada `tau`. Juga sumber utama pengukuran false alarm (jalur metrik sistem, bukan label). |
| Gradual leak (arm `LEAK_GRADUAL`) | Timeline Bagian 3–4 | Dry baseline → `Safe`; setelah injeksi, mapping `tau` → kelas; `t >= T_physical_leak` mengikuti keputusan Bagian 6. |
| Non-leaking fill / sub-threshold fill (berhenti sebelum leak, dalam arm gradual) | Kondisi protokol | Right-censored: tidak ada `T_physical_leak`, `tau` tidak terdefinisi. **Usulan v0.2 (merevisi v0.1):** window pasca-injeksi berlabel `Safe` **hanya jika** sesi dirancang di muka sebagai non-leaking fill — target level isi di bawah ambang leak protokol dan horizon observasi ditetapkan sebelum eksperimen. Sesi gradual yang sekadar dihentikan lebih awal **tidak otomatis `Safe`**: window pasca-injeksinya dikeluarkan dari dataset ordinal sebagai censored/ambiguous (`exclusion_reason = CENSORED_NO_SAFE_HORIZON`). Dry baseline tetap `Safe`. Tujuan contoh non-leaking fill terjaga (model tidak belajar "ada injeksi = pasti leak", fondasi §13.1) tanpa memalsukan `Safe` dari ketidaktahuan. |
| Sudden leak (arm `LEAK_SUDDEN`) | Kondisi arm | Dikeluarkan dari training/evaluasi ordinal (fondasi §13.3: tidak dapat diprediksi dari kapasitansi). Dipakai untuk latensi fail-safe dan evaluasi notifikasi level sistem. |
| Gangguan tekanan/gerakan tanpa cairan | Kondisi protokol + `activity_state`/`orientation_position` | Tidak ada cairan dan tidak ada leak (dalam protokol non-leaking yang ditetapkan di muka) → `Safe`. Ini hard negative yang penting; gangguan di tengah arm gradual tidak mengubah label (label tetap dari timeline). |
| Record kualitas invalid | `system_quality` / channel quality | Lihat Bagian 8. Tidak pernah dilabeli ulang. |
| Record post-leak | Timeline | Tidak masuk training ordinal (jika U1). Tetap disimpan untuk evaluasi sensor, latensi LIG, dan skin-exposure logging. Tidak pernah dilabeli `Safe`. |

Arm `FIELD` (ada di enum contract) belum tercakup rulebook ini dan tidak
boleh dilabeli dengan aturan bench di atas tanpa keputusan terpisah.

Nilai `arm` pada tabel ini adalah nilai enum persis dari Data Contract v1.1
(`SAFE`, `LEAK_GRADUAL`, `LEAK_SUDDEN`, `FIELD`); v0.1 sempat memakai
prefiks `ARM_*` yang tidak ada di contract dan telah dikoreksi.

Catatan istilah: v0.1 dan fondasi §13.1 memakai istilah "sub-lethal fill";
istilah itu diganti menjadi **non-leaking fill (sub-threshold fill)** karena
tidak ada konteks letalitas di sini. Penyelarasan istilah pada dokumen
fondasi menyusul dalam batch rekonsiliasi fondasi.

## 8. Record dengan Kualitas Invalid

Aturan tegas:

1. Sample dengan `cap_quality != OK` tidak boleh masuk window training/eval
   model ordinal. Aturan MVP (dikonfirmasi review 2026-07-12): window yang
   memuat >= 1 sample non-`OK` pada kanal kapasitif dibuang dari dataset
   ordinal (`exclusion_reason = INVALID_CAP_QUALITY`). Toleransi parsial
   (misalnya < 5% sample non-OK) adalah alternatif yang boleh
   dipertimbangkan nanti dengan bukti, bukan default.
2. Record invalid **tidak dilabeli ulang** menjadi `Safe` maupun `Urgent`.
   Ketiadaan data valid bukan informasi risiko rendah ataupun tinggi.
3. Record invalid **tidak dibuang dari penyimpanan**. Ia dikeluarkan dari
   dataset ordinal tetapi dipertahankan untuk pengujian level sistem:
   verifikasi state machine `system_quality`, perilaku suppress prediksi
   (`ML_UNAVAILABLE`), dan ketersediaan fail-safe (`FAILSAFE_DEGRADED`).
4. Status `INITIALIZING` (warming up) diperlakukan sama dengan invalid untuk
   keperluan pelabelan: tidak ada label ordinal yang diterbitkan darinya.
5. `lig_quality` tidak memengaruhi label ordinal (LIG bukan sumber label
   gradasi), tetapi wajib dicatat karena menentukan validitas evaluasi
   fail-safe pada sesi yang sama. Kasus khusus `ADC_SATURATED` saat wet
   contact masih menunggu konfirmasi hardware (fondasi §8) dan tidak boleh
   otomatis dianggap fault.

## 9. Dataset Turunan Berlabel

Output skrip pelabelan adalah **dataset turunan offline** (artefak baru,
tidak mengubah Data Contract v1.1) dengan satu baris per window kandidat.
Field minimum wajib:

| Field | Isi |
|---|---|
| `session_id` | ID sesi sumber (kunci join ke `sessions.csv`) |
| `window_start` | Timestamp awal window (`t - W`) |
| `window_end` | Timestamp akhir window (`t`) |
| `risk_label` | `Safe`/`Monitor`/`Caution`/`Urgent`, kosong jika tidak valid |
| `label_valid` | boolean; `false` untuk window yang dikeluarkan |
| `exclusion_reason` | kosong jika valid; enum alasan, minimal: `INVALID_CAP_QUALITY`, `CENSORED_NO_SAFE_HORIZON`, `POST_LEAK`, `SUDDEN_ARM`, `PARTIAL_WINDOW`, `FIELD_ARM_UNCOVERED` |
| `rulebook_version` | versi dokumen ini yang dipakai skrip (misalnya `v0.2`) |
| `boundary_config_version` | versi konfigurasi B1/B2/B3 yang dipakai |
| `dataset_partition` | `development`/`validation`/`final_test` (Bagian 10) |

Window yang dikeluarkan tetap ditulis dengan `label_valid = false` dan
`exclusion_reason` terisi, supaya audit cakupan dan alasan eksklusi dapat
direproduksi. Kolom audit tambahan (misalnya `arm`, `tau`) boleh disertakan
untuk keperluan audit, tetapi seluruh kolom dataset turunan ini tunduk pada
larangan fitur Bagian 5 — tidak satu pun boleh masuk feature vector kecuali
kolom kunci join yang dipakai sekadar untuk menyusun window sinyal.

## 10. Partisi Dataset dan Anti-Leakage

Window 120 detik dengan stride 10 detik saling tumpang tindih ~92%, sehingga
random split per-window hampir pasti menempatkan window yang nyaris identik
di partisi berbeda dan menghasilkan angka evaluasi yang bagus secara palsu.
Aturan wajib:

1. **Split per grup, bukan per window.** Seluruh window dari satu
   `session_id` berada pada partisi yang sama.
2. **Grup diperluas ke hardware fisik.** Sesi-sesi yang memakai baseplate/
   kantong yang sama atau spesimen sensor yang sama masuk partisi yang
   sama, karena karakteristik fisik spesimen membuat sesi-sesi tersebut
   berkorelasi. Identitas bag/sensor per sesi wajib tercatat pada metadata
   sesi/protokol.
3. Partisi dicatat eksplisit pada field `dataset_partition` (Bagian 9) dan
   ditetapkan sebelum fitting apa pun.
4. Partisi `final_test` tunduk pada invariant `AGENTS.md`: tersegel dari
   fitting, preprocessing, dan tuning, termasuk dari penetapan boundary
   B1/B2/B3 (Bagian 4) dan dari keputusan aturan eksklusi.

## 11. Skema Pelabelan Minimal yang Direkomendasikan (v0.2)

Ringkasan satu halaman untuk implementasi kelak:

1. Unit label = rolling window penuh yang sama dengan window fitur; label
   dari kondisi pada waktu akhir window; window parsial tidak dilabeli.
2. Sumber label = arm + event + timing; tidak pernah dari kapasitansi, LIG
   raw, atau prediksi model. Sebaliknya, metadata pembentuk label tidak
   pernah menjadi fitur (Bagian 5).
3. Dry baseline dan gangguan-tanpa-cairan dalam protokol non-leaking yang
   ditetapkan di muka → `Safe`. Non-leaking fill → `Safe` hanya jika
   horizon aman ditetapkan sebelum eksperimen; sesi yang sekadar berhenti
   lebih awal → censored, dikeluarkan.
4. Arm `LEAK_GRADUAL` pasca injeksi → mapping `tau` dengan boundary
   B1/B2/B3 Opsi A (waktu bench tetap) yang akan ditetapkan dari pilot
   Development dan dibekukan sebelum Final Test (Bagian 4).
5. Window pasca `T_physical_leak` → keluar dari dataset ordinal (usulan U1),
   disimpan untuk evaluasi sistem.
6. Arm `LEAK_SUDDEN` → bukan data ordinal; evaluasi fail-safe dan notifikasi.
7. Window dengan sample kapasitif non-`OK` → keluar dari dataset ordinal,
   dipertahankan untuk pengujian kualitas sistem; tidak pernah dilabeli
   ulang.
8. Output = dataset turunan Bagian 9, dengan partisi per sesi/bag/sensor
   (Bagian 10).
9. Skrip pelabelan harus deterministik: input `sessions.csv` +
   `samples.csv` + `events.csv` + konfigurasi boundary → output identik
   setiap dijalankan.

## 12. Keputusan yang Belum Terselesaikan

Semua item berikut menunggu keputusan pengguna (dan sebagian menunggu data
pilot); rulebook tidak boleh dikunci sebelum item 1–4 selesai. Rekomendasi
di bawah sudah merekonsiliasi review Codex 2026-07-12; keputusan tetap milik
pengguna.

1. **Skala boundary B1/B2/B3** — rekomendasi: Opsi A (waktu bench tetap),
   angka menunggu pilot Development, dibekukan sebelum Final Test. Opsi B
   hanya analisis sensitivitas.
2. **`Urgent` dan leak aktual** — rekomendasi: Opsi U1, pre-leak saja
   (Bagian 6).
3. **Non-leaking fill** — rekomendasi: `Safe` hanya jika horizon aman
   ditetapkan sebelum eksperimen; selain itu dikeluarkan sebagai censored
   (Bagian 7).
4. **Toleransi sample invalid dalam window** — rekomendasi: MVP buang
   seluruh window jika ada sample kapasitif non-`OK` (Bagian 8).
5. Pemakaian window pre-leak arm `LEAK_SUDDEN` sebagai contoh `Safe`
   tambahan pada evaluasi (bukan training) — perlu atau tidak.
6. Perlakuan arm `FIELD` — di luar cakupan versi ini.
7. Hubungan boundary bench dengan framing 2/6/12 jam pemakaian riil pada
   laporan — cara mengomunikasikannya tanpa overclaiming.

## 13. Pertanyaan yang Membutuhkan Konfirmasi Hardware

1. Berapa durasi realistis `T_inj_start` → `T_physical_leak` pada rig bench
   dengan protokol injeksi yang direncanakan? (Menentukan skala boundary.)
2. Apakah wet contact dapat membuat kanal LIG `ADC_SATURATED`, dan bagaimana
   membedakannya dari fault? (Fondasi §8, kontrak v1.1.)
3. Durasi dan aturan validitas kalibrasi LIG (kontrak v1.1, state machine
   pending) — menentukan kapan `T_flag` sah dievaluasi.
4. Stabilitas sampling 1 Hz pada logger terintegrasi — pelabelan window
   mengasumsikan timestamp monoton dan gap terdeteksi (`DATA_GAP`).
5. Reproduksibilitas laju injeksi (pompa vs manual) — pilot kapasitif saat
   ini memakai tekanan manual dan sampling tidak teratur, sehingga belum
   memenuhi asumsi timeline dokumen ini.
6. Bagaimana `PHYSICAL_LEAK_OBSERVED` dioperasionalkan di bench (visual,
   kertas indikator, video + timestamp)? Presisi `T_physical_leak`
   menentukan presisi seluruh label arm gradual — termasuk berapa lag
   observasi visual yang realistis (Bagian 3, urutan `T_flag`).

## 14. Syarat Sebelum Rulebook Boleh Dikunci

1. Data logger dua kanal tersinkronisasi dari rig terintegrasi sudah ada dan
   lolos QC (timestamp, gap, quality state) — saat ini belum tersedia.
2. Pilot arm `LEAK_GRADUAL` memberikan estimasi durasi pre-leak sehingga
   boundary B1/B2/B3 dapat ditetapkan dengan justifikasi tertulis.
3. Keputusan Bagian 12 item 1–4 dikunci eksplisit oleh pengguna.
4. Metode observasi `T_physical_leak` didefinisikan dan diuji (Bagian 13
   item 6).
5. Skrip pelabelan deterministik tersedia dengan golden test cases yang
   mencakup semua skenario Bagian 7, semua `exclusion_reason` Bagian 9, dan
   aturan partisi Bagian 10.
6. Review independen (Codex) selesai dan hasilnya direkonsiliasi — review
   pertama selesai 2026-07-12 dan direkonsiliasi dalam v0.2 ini; review
   ulang diperlukan bila ada perubahan substantif berikutnya.

Sampai seluruh syarat terpenuhi, setiap dataset yang dilabeli dengan
dokumen ini harus mencantumkan "labeled under DRAFT rulebook v0.2" pada
metadata/laporannya.

## 15. Riwayat Revisi

- **v0.2 (2026-07-12)** — merekonsiliasi review independen Codex
  (disampaikan pengguna 2026-07-12); menggantikan v0.1. Perubahan utama:
  `samples.csv` ditambahkan sebagai input pelabelan wajib; non-leaking fill
  tidak lagi otomatis `Safe` (butuh horizon aman yang ditetapkan di muka,
  selain itu censored); aturan partisi anti-leakage per sesi/bag/sensor
  (Bagian 10 baru); larangan metadata label sebagai fitur (Bagian 5);
  nilai enum arm disamakan persis dengan Data Contract v1.1; definisi
  dataset turunan berlabel (Bagian 9 baru); Opsi A dijadikan skema boundary
  utama dan Opsi B diturunkan menjadi analisis sensitivitas; asumsi urutan
  `T_flag` vs `T_physical_leak` dihapus; aturan window penuh; istilah
  "sub-lethal fill" diganti "non-leaking fill".
- **v0.1 (2026-07-11)** — draft awal; diarsipkan di
  `.backup/2026-07-12-rulebook-codex-review/ai-label-rulebook-v0.1.md`.
