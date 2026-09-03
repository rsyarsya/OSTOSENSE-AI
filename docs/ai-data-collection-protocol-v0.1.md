# Protokol Pengambilan Data Terintegrasi OSTOSENSE v0.1

**Status: DRAFT — NOT LOCKED**

Dokumen ini menjelaskan secara praktis bagaimana tim elektronis kelak
mengambil data terintegrasi dari sensor kapasitif dan LIG untuk pengembangan
AI OSTOSENSE: pemasangan rig, pemberian saline, perekaman ESP32, pencatatan
ground truth, skenario eksperimen, sampai data siap diaudit. Seluruh isi
adalah usulan rekayasa prototipe PKM-KC, bukan prosedur klinis. Tidak ada
angka di dokumen ini yang boleh dikutip sebagai spesifikasi final sebelum
ditandai terkunci; angka berlabel **proposed pilot setting** wajib
dikonfirmasi tim hardware terlebih dahulu.

Acuan:

- [AI Data Contract v1.1](./ai-data-contract-v1.1.md) — skema `sessions.csv`,
  `samples.csv`, `events.csv`, enum quality, dan event. Protokol ini TIDAK
  mengubah contract tersebut.
- [AI Label Rulebook v0.3](./ai-label-rulebook-v0.3.md) — aturan pelabelan
  offline dengan struktur terkunci; nilai boundary produksi tetap
  `PILOT_PENDING`. Hubungan dokumen diringkas di Bagian 15.6.
- [Fondasi Arsitektur AI](./ostosense-ai-foundation.md) — §7 (akuisisi),
  §9 (user flow), §13 (arm eksperimen), §14 (alur pengembangan).

## Cara Membaca Dokumen Ini

Dokumen ini ditujukan untuk anggota tim yang belum mendalami machine
learning. Istilah yang cukup diketahui:

- **Sesi**: satu kali eksperimen dari pemasangan kantong sampai sesi ditutup.
  Semua data satu sesi memakai satu `session_id`.
- **Window**: potongan data 120 detik yang kelak menjadi satu "contoh" bagi
  model AI. Model tidak melihat satu angka; ia melihat pola dalam satu window.
- **Label**: "jawaban benar" yang ditempelkan ke setiap window secara offline
  (bukan oleh perangkat), berdasarkan kondisi eksperimen yang kita kontrol —
  aturannya ada di Label Rulebook v0.3.
- **Ground truth**: fakta fisik yang dicatat operator (misalnya "cairan
  benar-benar bocor pada jam sekian"), independen dari pembacaan sensor.
- **Partisi**: pembagian sesi menjadi kelompok Development (boleh dipakai
  bebas) dan Final Test (disegel; hanya untuk evaluasi akhir).
- **QC (quality control)**: pemeriksaan bahwa data satu sesi lengkap dan
  konsisten sebelum boleh dipakai.

Prinsip utama: **yang dicatat rapi hari ini menentukan apa yang boleh
diklaim nanti.** Data yang tidak lengkap catatannya tidak bisa "diperbaiki"
belakangan tanpa merusak kejujuran evaluasi.

## 1. Batas Tahap Proyek

1. Data LIG manual dan pilot kapasitif yang sudah ada saat ini tetap
   berstatus **hardware characterization** — berguna untuk memahami sensor,
   tetapi bukan bagian dari dataset training AI dan tidak diproses dengan
   protokol ini.
2. Protokol ini baru digunakan setelah kanal kapasitif dan LIG dapat
   **direkam bersamaan oleh satu ESP32** ke tiga tabel contract dengan
   timestamp dari jam yang sama. Sebelum itu tercapai, tidak ada sesi
   protokol yang sah.
3. PCB final **tidak wajib** untuk pilot. Rangkaian breadboard/modul boleh,
   dengan syarat: stabil, terdokumentasi (foto + catatan revisi), dan
   **tidak berubah dalam satu batch eksperimen**. Perubahan rangkaian
   memulai batch baru dengan identitas `protocol_version` baru (Bagian 7).
4. Data dari rangkaian sementara **tidak otomatis menjadi Final Test**.
   Kelayakan suatu batch menjadi Final Test diputuskan pengguna berdasarkan
   stabilitas rangkaian dan hasil QC, dan harus diputuskan **sebelum** data
   batch tersebut dilihat untuk keperluan tuning apa pun.

## 2. Rig Uji Fisik (Meja Uji)

### 2.1 Diagram penampang samping

```text
                      syringe / sumber saline
                      (manual, pompa opsional)
                              |
                              |  selang injeksi
                              v
                    +--------------------+
                    | tube simulasi stoma|
                    | (port masuk cairan)|
        +-----------+--------------------+-----------+
        |                                            |
        |            KANTONG OSTOMI (bag_id)         |
        |    cairan masuk DARI SISI DALAM kantong    |
        |                                            |
        |   [sensor kapasitif menempel di dinding    |
        |    luar kantong, area akumulasi cairan]    |
        +---+------------------------------------+---+
            |        baseplate / adhesive        |
   =========+===•================================+=========
   permukaan     ^                                     ^
   simulasi      celah bocor buatan                [sensor LIG]
   kulit         (jalur cairan dari dalam          di permukaan kulit
   (mis. lembar  kantong ke tepi baseplate)        pada jalur celah
   silikon)                                        bocor, tepi baseplate
   =========================================================
                     meja uji / fixture rata
```

### 2.2 Komponen dan fungsinya

| Komponen | Fungsi | Catatan |
|---|---|---|
| Permukaan simulasi kulit | Alas tempat baseplate menempel; media rambat cairan bocor | Material (mis. lembar silikon) dicatat di manifest/notes; konsisten dalam satu batch |
| Kantong + baseplate ostomi (`bag_id`) | Objek yang dipantau; wadah cairan | Setiap unit fisik punya `bag_id` unik dan dicatat setiap sesi |
| Tube simulasi stoma | Port masuk saline dari sisi dalam kantong | Meniru arah masuk output stoma sungguhan |
| Sensor kapasitif (`sensor_id`) | Membaca dinamika isi kantong (input utama model) | Posisi di dinding luar kantong pada area akumulasi cairan; posisi persis = pertanyaan hardware (Bagian 15.4), difoto dan dijaga tetap dalam satu batch |
| Sensor LIG (`sensor_id`) | Fail-safe: mendeteksi cairan yang sudah keluar | Di permukaan kulit simulasi, pada jalur celah bocor di tepi baseplate |
| Syringe / sumber saline | Memberi cairan terkontrol | Manual default; pompa hanya jika tersedia (Bagian 10) |
| Celah bocor buatan | Jalur cairan dari dalam kantong menuju tepi baseplate → LIG | Cara pembuatan yang reprodusibel = pertanyaan hardware (Bagian 15.4) |
| ESP32 + logger (`device_id`) | Merekam kedua kanal + quality state ke MicroSD | Satu jam (RTC) untuk semua timestamp |

### 2.3 Aturan arah cairan

**Saline untuk data training dimasukkan dari sisi dalam kantong melalui tube
simulasi stoma**, sehingga sensor "melihat" urutan kejadian yang sama seperti
pemakaian nyata: kantong terisi → cairan menekan seal → bocor melalui celah →
mencapai LIG.

Saline **tidak boleh diteteskan langsung ke sensor** pada sesi protokol ini.
Meneteskan langsung hanya sah pada pengujian karakterisasi sensor yang
terpisah, di luar protokol ini, dan datanya tidak masuk dataset training.

## 3. Alur Satu Sesi Eksperimen

Urutan baku setiap sesi:

1. **Persiapan dan manifest.** Isi baris protocol manifest (Bagian 7) untuk
   sesi ini SEBELUM sesi dimulai: arm yang direncanakan, target volume,
   metode injeksi, safe horizon (jika arm non-leak), operator, dan ID.
2. **Pemasangan.** Pasang kantong + baseplate pada permukaan simulasi kulit,
   pasang sensor kapasitif dan LIG pada posisi baku, sambungkan ke ESP32.
3. **Pemeriksaan ID.** Cocokkan `session_id`, `bag_id`, `sensor_id`,
   `device_id` fisik dengan manifest. ID salah = sesi tidak sah.
4. **Mulai sesi dan kalibrasi.** Long-press tombol (sesi baru): baseline
   kapasitif di-reset dan kalibrasi LIG dimulai. Status `WARMING_UP` /
   `INITIALIZING` adalah normal pada fase ini. Tunggu sampai kedua kanal
   `OK` (perangkat/aplikasi menunjukkan `NORMAL`). Jika kalibrasi gagal
   (`BASELINE_INVALID`), catat dan hentikan; jangan memaksakan sesi.
5. **Fase pre-injection dry.** Biarkan rig diam tanpa cairan minimal
   **120 detik setelah kedua kanal `OK`** (Bagian 6). Tidak ada sentuhan,
   tekanan, atau injeksi pada fase ini.
6. **Injeksi saline.** Mulai injeksi sesuai profil di manifest. Catat event
   `INJECTION_START`; setiap langkah injeksi ditutup `INJECTION_END` dengan
   volume yang diberikan (Bagian 9–10).
7. **Progresi cairan.** Lanjutkan langkah injeksi sesuai profil sampai
   target manifest tercapai (leak untuk arm bocor; target volume/waktu untuk
   arm aman). Amati terus jalur celah bocor.
8. **Observasi kebocoran fisik.** Saat cairan pertama kali terlihat keluar
   (metode Bagian 9), catat event `PHYSICAL_LEAK_OBSERVED` saat itu juga.
9. **Deteksi LIG.** `LEAK_FLAG_FIRST` dan `LEAK_FLAG_CONFIRMED` dicatat
   otomatis oleh firmware dari kanal LIG. Operator tidak menunggu atau
   menyesuaikan apa pun terhadap LIG — biarkan sistem bekerja sendiri.
10. **Penghentian sesi.** Tutup sesi sesuai kondisi: `LEAK_CONFIRMED`
    (leak terkonfirmasi), `CEILING_REACHED` (target aman tercapai), atau
    `MANUAL_STOP` (dihentikan operator — alasannya wajib ditulis di catatan
    operator). `end_reason` harus jujur; sesi gagal tetap ditutup dan
    disimpan.
11. **Pembersihan.** Keringkan/bersihkan rig. Tentukan status unit: kantong
    dan sensor yang terkena cairan diberi catatan; keputusan pakai-ulang vs
    ganti mengikuti kebijakan batch (pertanyaan hardware, Bagian 15.4).
    Data mentah di MicroSD disalin ke penyimpanan kerja tanpa diedit.

### 3.1 Contoh timeline sesi gradual (ilustrasi urutan, bukan hasil ukur)

Seluruh waktu di bawah hanya menggambarkan **urutan dan orde besaran**,
bukan data pengukuran:

```text
T+00:00  Long-press: sesi baru; reset baseline kapasitif; kalibrasi LIG
         dimulai (WARMING_UP / INITIALIZING)
T+01:00  (ilustrasi) kedua kanal OK -> system_quality NORMAL;
         fase pre-injection dry dimulai
T+03:00  Fase dry >= 120 s terpenuhi; INJECTION_START (langkah 1,
         manual syringe)
T+03:30  INJECTION_END langkah 1 (volume langkah tercatat)
T+05:30  INJECTION_START langkah 2 ... (berulang sesuai profil manifest)
T+12:40  Cairan terlihat pada jalur celah bocor ->
         PHYSICAL_LEAK_OBSERVED dicatat operator
T+13:05  LEAK_FLAG_FIRST (otomatis dari firmware; urutan relatif terhadap
         observasi visual TIDAK dijamin — bisa lebih dulu)
T+13:25  LEAK_FLAG_CONFIRMED (otomatis)
T+15:00  Sesi ditutup, end_reason = LEAK_CONFIRMED
T+15:00+ Pembersihan, catatan operator, salin data mentah
```

## 4. Penyelesaian Tiga MUST FIX Review Codex

Tiga bagian berikut menyelesaikan tiga temuan MUST FIX review Codex atas
fondasi pelabelan: definisi window yang deterministik (Bagian 5), durasi
fase dry dan baseline (Bagian 6), serta protocol manifest dan input
pelabelan lengkap (Bagian 7).

## 5. Definisi Window dan Aturan Waktu

Aturan berikut membuat pembentukan window **deterministik**: siapa pun yang
menjalankan skrip pada data yang sama akan mendapat window yang sama.

### 5.1 Interval dan ukuran

- Window didefinisikan pada interval **`(t - W, t]`** — terbuka di kiri,
  tertutup di kanan — dengan `t` = waktu akhir window.
- `W` v0 = **120 detik**; sampling target = **1 Hz**.
- Dengan sampling tepat 1 Hz, satu window penuh memuat **tepat 120 sample**.
  Interval setengah-terbuka inilah yang membuat hitungan tepat 120 (interval
  tertutup dua sisi akan memuat 121 sample). Rulebook v0.3 telah mengadopsi
  konvensi yang sama sehingga tidak ada lagi perbedaan off-by-one.

### 5.2 Anchor stride yang deterministik

- `t_ref` = timestamp sample **pertama** sesi tersebut pada `samples.csv`.
- Kandidat akhir window: `t_k = t_ref + (W + k*S) * 1000 ms` untuk
  `k = 0, 1, 2, ...`, dengan stride `S` v0 = **10 detik**.
- Keanggotaan sample: sample ber-timestamp `ts` masuk window `k` jika
  `t_k - W*1000 < ts <= t_k` (milidetik, epoch UTC sesuai contract).
- Grid dihitung dari `t_ref`, bukan dari "sample ke-n", sehingga hasilnya
  tidak berubah walau ada sample hilang di tengah sesi.
- Window-window awal yang masih memuat sample `WARMING_UP` otomatis gugur
  oleh aturan kualitas rulebook (semua `cap_quality` harus `OK`); ini
  perilaku yang diharapkan, bukan error.

### 5.3 Aturan gangguan waktu

Semua angka toleransi di bawah adalah **proposed pilot setting** dan
dikonfirmasi ulang saat shakedown:

| Kondisi | Definisi operasional | Konsekuensi |
|---|---|---|
| Jitter normal | Selisih antar-sample dalam `1000 ± 200 ms` | Diterima |
| Missing sample | Ada slot nominal 1 detik tanpa sample (selisih antar-sample `> 1200 ms`) | Window yang memuat slot kosong itu tidak penuh → tidak dilabeli (`PARTIAL_WINDOW`/gugur); sample lain tetap dipakai window lain |
| Duplicate timestamp | Dua baris `samples.csv` dengan `session_id` dan `timestamp` ms sama | Window yang memuatnya gugur; sesi ditandai audit; data mentah TIDAK di-dedup diam-diam |
| `DATA_GAP` | Firmware menandai kanal `DATA_GAP` sesuai contract | Sample non-`OK` → window gugur oleh aturan rulebook. QC offline juga memeriksa: selisih antar-sample `>= 2000 ms` tanpa `DATA_GAP` dari firmware = temuan bug logger |
| `DEVICE_RESTART` di tengah sesi | Event `DEVICE_RESTART` muncul setelah sesi berjalan | Sesi ditandai audit; tidak boleh disambung diam-diam menjadi satu timeline mulus |

Aturan "window penuh = tepat 120 sample" bersifat struktural dan diperiksa
sebelum aturan kualitas. Window yang tidak penuh tidak pernah dilabeli,
apa pun isinya.

## 6. Fase Dry dan Baseline

### 6.1 Durasi dry minimum

Fase pre-injection dry berlangsung minimal **120 detik dihitung sejak kedua
kanal pertama kali `OK`** (keluar dari `WARMING_UP`), tanpa cairan, tanpa
sentuhan, tanpa manipulasi rig. Alasannya sederhana: model butuh minimal
satu window penuh 120 detik yang seluruh sample-nya `OK` dan benar-benar
kering sebagai contoh `Safe` paling bersih. Dry lebih lama lebih baik
(usulan praktis: 300 detik, **proposed pilot setting**) karena tiap 10 detik
ekstra menghasilkan satu window `Safe` tambahan.

`INJECTION_START` tidak boleh dicatat sebelum syarat dry minimum terpenuhi.

### 6.2 Baseline 60 detik

- Statistik baseline mengikuti contract: `baseline_value` = **median** dan
  `baseline_std` = **standard deviation** kapasitansi selama 60 detik.
- 60 detik yang digunakan adalah **60 detik pertama fase kalibrasi setelah
  sesi baru dimulai** (setelah long-press), yaitu periode yang sama dengan
  perhitungan baseline firmware. Alasan pemilihan:
  1. Kantong dijamin kosong dan kering — sesi baru hanya sah pada kantong
     baru/kosong (fondasi §9), sehingga inilah referensi "kosong" yang benar.
  2. Deterministik — tidak bergantung kapan operator mulai menginjeksi.
  3. Konsisten dengan makna `baseline_value` pada contract sebagai referensi
     isi sesi; window fitur kelak dinormalisasi terhadap nilai ini.
- Penempatan persis 60 detik ini di dalam urutan kalibrasi firmware
  (sebelum/sesudah/berimpit kalibrasi LIG) belum dikunci — menunggu
  konfirmasi durasi kalibrasi LIG (pertanyaan hardware, Bagian 15.4).
- QC offline menghitung ulang median dan standard deviation dari 60 detik
  tersebut pada `samples.csv` dan mencocokkannya dengan `baseline_value` /
  `baseline_std` di `sessions.csv`; selisih di luar toleransi kecil
  (**proposed**, ditetapkan saat shakedown) = temuan QC.

## 7. Protocol Manifest dan Input Pelabelan

### 7.1 Mengapa perlu manifest

Rulebook v0.3 mensyaratkan beberapa hal yang **harus ditetapkan sebelum
sesi berjalan** — misalnya safe horizon untuk sesi non-leaking fill, dan
identitas bag/sensor untuk aturan partisi. Fakta "direncanakan di muka"
itulah yang membuat label jujur; ia harus tercatat pada artefak tersendiri,
bukan diingat-ingat belakangan. Artefak itu adalah **protocol manifest**:
satu baris per sesi, diisi SEBELUM sesi dimulai.

### 7.2 Skema minimum `protocol_manifest.csv` (versioned)

| Field | Isi |
|---|---|
| `session_id` | Sama persis dengan `sessions.csv` |
| `protocol_version` | Versi dokumen protokol + revisi konfigurasi rig/rangkaian yang dipakai (mis. `v0.1-rigA`) |
| `planned_arm` | Enum contract: `SAFE`, `LEAK_GRADUAL`, `LEAK_SUDDEN`, `FIELD` |
| `planned_safe_horizon_s` | Untuk sesi non-leak: durasi observasi aman yang ditetapkan di muka (detik). Arm `SAFE` di-anchor pada sample pertama saat kedua kanal `OK`; non-leaking fill di-anchor pada `INJECTION_END` terakhir; kosong untuk arm bocor yang direncanakan |
| `target_fill_or_volume` | Target level isi / total volume yang direncanakan |
| `injection_profile` | Profil pemberian (mis. `stepwise`, `continuous`) |
| `injection_method` | `manual_syringe` (default) atau `pump` |
| `planned_flow_ml_min` | Target laju; kosong jika tidak berlaku |
| `physical_leak_observation_method` | Metode ground truth yang dipakai (Bagian 9) |
| `operator_id` | Operator sesi |
| `bag_id` | Unit kantong fisik |
| `sensor_id` | Unit rakitan sensor |
| `device_id` | Unit ESP32 |

Aturan manifest:

- Diisi sebelum sesi; perubahan setelah sesi hanya boleh berupa koreksi
  ber-tanggal yang ditambahkan (append) dengan alasan — tidak pernah
  mengedit nilai lama diam-diam.
- Nilai `planned_*` adalah rencana; realisasi tercatat di `events.csv` dan
  catatan operator. Keduanya sengaja dibedakan.
- Skema ini adalah **offline companion artifact**; ia tidak mengubah tiga
  CSV contract.

### 7.3 Input pelabelan lengkap

Skrip pelabelan kelak menerima **enam** input:

1. `sessions.csv` (contract),
2. `samples.csv` (contract),
3. `events.csv` (contract),
4. `protocol_manifest.csv` (Bagian 7.2),
5. konfigurasi boundary B1/B2/B3 yang versioned (`boundary_config_version`),
6. `partition_manifest.csv` (Bagian 11.3).

Rulebook v0.3 telah mengadopsi keenam input ini. Implementasi labeler wajib
memvalidasi konsistensinya sebelum menghasilkan output.

## 8. Skenario Eksperimen

Nilai `arm` memakai enum persis dari contract: `SAFE`, `LEAK_GRADUAL`,
`LEAK_SUDDEN`, `FIELD`.

| Skenario | `planned_arm` | Tujuan | Catatan pelaksanaan |
|---|---|---|---|
| Dry / no-liquid safe | `SAFE` | Contoh `Safe` bersih; ukur false alarm dan drift | Tanpa cairan sama sekali; `planned_safe_horizon_s` wajib diisi |
| Non-leaking fill (safe horizon di muka) | `LEAK_GRADUAL` | Agar model tidak belajar "ada injeksi = pasti leak" | Target isi di bawah ambang bocor protokol DAN `planned_safe_horizon_s` diisi sebelum sesi; tanpa itu, window pasca-injeksi berstatus censored (rulebook v0.3 Bagian 7). Jika physical leak tetap terjadi, pertahankan sebagai gradual-leak endpoint dan catat `UNPLANNED_PHYSICAL_LEAK` sebagai deviasi protokol |
| Gradual leak | `LEAK_GRADUAL` | **Data utama model ordinal**; seluruh gradasi `Safe`→`Urgent` | Injeksi bertahap sampai leak fisik teramati; ground truth Bagian 9 |
| Sudden leak | `LEAK_SUDDEN` | Uji fail-safe LIG dan latensi alarm | Bukan data training ordinal; kegagalan seal dibuat mendadak |
| Gangguan tanpa cairan: tekanan, gerakan, tekukan, perubahan orientasi | `SAFE` (atau di tengah arm lain sesuai protokol) | Hard negative; uji ketahanan terhadap gangguan mekanis | `activity_state` / `orientation_position` diisi; tanpa cairan = tetap `Safe`; gangguan di tengah arm gradual tidak mengubah label |
| Sensor invalid / kabel dilepas | `SAFE` (sesi QC khusus) | Uji state machine quality: `DISCONNECTED`, `ML_UNAVAILABLE`, `FAILSAFE_DEGRADED` | Sengaja lepas kanal di tengah sesi; datanya untuk pengujian sistem, bukan dataset ordinal |

`FIELD` tetap **di luar** training bench sampai ada keputusan terpisah
(konsisten rulebook v0.3); protokol ini tidak mendefinisikan sesi `FIELD`.

## 9. Ground Truth dan Pencatatan Event

### 9.1 Lima event kunci

| Event | Siapa yang mencatat | Kapan |
|---|---|---|
| `INJECTION_START` | Operator (melalui mekanisme pencatatan event ber-timestamp RTC perangkat) | Tepat saat langkah injeksi dimulai; metadata: `delivery_mode`, `target_flow_ml_min` |
| `INJECTION_END` | Operator | Tepat saat langkah injeksi selesai; metadata: `delivered_volume_ml`, `cumulative_volume_ml`, `measured_flow_ml_min` |
| `PHYSICAL_LEAK_OBSERVED` | Operator | Saat cairan pertama kali terlihat keluar pada jalur bocor; metadata: `observation_method`, `operator_id`, opsional `video_reference` |
| `LEAK_FLAG_FIRST` | Firmware (otomatis) | `leak_flag` LIG pertama aktif |
| `LEAK_FLAG_CONFIRMED` | Firmware (otomatis) | `leak_flag` aktif tiga sample berturut-turut |

Semua event harus ber-timestamp dari **jam yang sama dengan `samples.csv`**
(RTC perangkat). Mekanisme konkret pencatatan event operator (tombol,
perintah serial, atau companion script) belum dikunci dan merupakan
pertanyaan implementasi logger — yang dikunci adalah persyaratannya: satu
sumber jam, dicatat saat kejadian, bukan direkonstruksi setelah sesi.

### 9.2 Urutan LIG vs observasi visual

**Jangan mengasumsikan urutan.** `LEAK_FLAG_FIRST` bisa mendahului
`PHYSICAL_LEAK_OBSERVED` (operator terlambat melihat) atau sebaliknya.
Keduanya dicatat independen; selisihnya adalah data latensi (bertanda),
bukan sesuatu yang perlu "dirapikan". Kasus urutan terbalik ditandai untuk
audit penyebab (observasi lambat vs false flag) sesuai rulebook v0.3.

### 9.3 Metode observasi yang direkomendasikan (proposed)

Untuk prototipe, metode sederhana dan feasible:

- **Observasi visual operator** pada jalur celah bocor, didukung
- **kertas indikator** (berubah warna saat basah) yang dipasang pada jalur
  bocor sebelum sesi, dan
- **rekaman video** rig dengan tampilan jam yang dapat dicocokkan ke RTC
  (mis. layar yang menampilkan waktu perangkat), sebagai bukti audit.

Ketelitian timestamp yang realistis dengan metode ini adalah orde
**±2 detik** (**proposed pilot setting**; dikonfirmasi saat shakedown
dengan membandingkan catatan operator dan video).

### 9.4 Jika waktu observasi ambigu

- Catat event dengan perkiraan terbaik SAAT ITU, lalu tulis ketidakpastian
  di `event_metadata` (mis. rentang waktu) dan catatan operator.
- Jangan pernah menggeser timestamp setelah sesi agar "cocok" dengan sinyal
  sensor — itu merusak independensi ground truth.
- Sesi dengan `T_physical_leak` yang ambiguitasnya melebihi toleransi di
  atas ditandai pada QC; keputusan pakai/keluarkan diambil saat audit
  pelabelan, dan alasannya dicatat.

## 10. Aturan Pemberian Saline

1. **Yang wajib dicatat setiap sesi:** jenis/konsentrasi cairan
   (`fluid_type` di `sessions.csv`), volume per langkah dan kumulatif
   (metadata `INJECTION_END`), titik masuk (tube simulasi stoma), dan metode
   (`injection_method` di manifest).
2. **Syringe manual adalah default prototipe.** Pompa hanya opsional jika
   memang tersedia; protokol tidak boleh bergantung pada pompa.
3. **Reproduksibilitas laju.** Dengan syringe manual, gunakan profil
   bertahap: aliquot volume tetap per langkah dengan jeda waktu tetap
   (stepwise), dipandu timer. Ini lebih reprodusibel daripada dorongan
   kontinu manual dan menghasilkan event `INJECTION_START`/`INJECTION_END`
   yang jelas per langkah.
4. **Angka.** Konsentrasi, volume aliquot, jeda antar langkah, dan total
   volume final **tidak ditetapkan di dokumen ini** — belum ada dasar
   pengukurannya. Contoh nilai awal yang boleh dipakai shakedown, semuanya
   **proposed pilot setting** yang memerlukan konfirmasi hardware: saline
   fisiologis NaCl 0.9%; aliquot kecil berulang dengan jeda tetap. Nilai
   terkonfirmasi kelak ditulis sebagai revisi protokol ber-versi, bukan
   diubah diam-diam.

## 11. Matriks Pilot yang Feasible

Prinsip: **jumlah sesi independen lebih berharga daripada jumlah window**;
window dari satu sesi saling tumpang tindih dan tidak menjadi "data baru".
Prioritaskan beberapa unit bag/sensor dan pengulangan skenario. Tidak ada
target ribuan pengujian manual.

Semua jumlah di bawah adalah **usulan awal** yang disesuaikan dengan
ketersediaan unit dan waktu; perubahan dicatat di revisi protokol.

### 11.1 Tahap 1 — Engineering shakedown

| Item | Usulan |
|---|---|
| Tujuan | Membuktikan rig + logger + QC bekerja; menera durasi tipikal injeksi→leak; menguji metode ground truth |
| Jumlah | 3–5 sesi, arm bebas |
| Status data | **Tidak pernah** masuk training/evaluasi; boleh dilihat sebebasnya |

### 11.2 Tahap 2 — Pilot/Development

| Skenario | Usulan sesi |
|---|---|
| Gradual leak (`LEAK_GRADUAL`) | 12–18 |
| Non-leaking fill (safe horizon di muka) | 3–5 |
| Dry safe (`SAFE`), termasuk sesi gangguan mekanis | 5–8 |
| Sudden leak (`LEAK_SUDDEN`) | 3–5 |
| Sesi QC quality (kabel lepas, dsb.) | 2–3 |

Sebaran: minimal **3 kombinasi unit bag/sensor berbeda**, operator
bergantian bila mungkin. Data tahap ini boleh dipakai untuk eksplorasi,
penetapan boundary B1/B2/B3, preprocessing, dan tuning.

### 11.3 Tahap 3 — Final Test (disegel)

| Item | Usulan |
|---|---|
| Isi | 6–8 sesi `LEAK_GRADUAL`, 2–3 `SAFE`, 2–3 `LEAK_SUDDEN` |
| Unit | Bag/sensor yang **tidak pernah** dipakai Development |
| Segel | Ditetapkan di `partition_manifest.csv` SEBELUM fitting apa pun; disegel dari pemilihan boundary, preprocessing, aturan eksklusi, dan tuning (invariant `AGENTS.md` dan contract) |

`partition_manifest.csv` (offline companion artifact) minimum berisi:
`session_id`, `dataset_partition` (`development`/`validation`/`final_test`),
`partition_version`, plus `bag_id`/`sensor_id` sebagai kolom audit
pengelompokan. Seluruh window satu sesi/bag/sensor berada di partisi yang
sama (rulebook v0.3 Bagian 12).

## 12. QC dan Acceptance Criteria per Sesi

Sesi dinyatakan **lolos QC** bila semua terpenuhi:

1. Timestamp `samples.csv` monoton naik dalam sesi.
2. Tidak ada duplicate timestamp dalam sesi.
3. Sampling mendekati 1 Hz sesuai toleransi Bagian 5.3.
4. Setiap gap terdeteksi: selisih antar-sample melebihi ambang harus
   tercermin sebagai `DATA_GAP` oleh firmware; ketidakcocokan = temuan bug.
5. Quality state valid: nilai enum sesuai contract, dan `system_quality`
   konsisten diturunkan dari `cap_quality` + `lig_quality` (invariant).
6. Semua event wajib tersedia sesuai arm: arm bocor minimal
   `INJECTION_START`, `INJECTION_END`, `PHYSICAL_LEAK_OBSERVED`; kanal LIG
   sehat seharusnya menghasilkan `LEAK_FLAG_FIRST`/`LEAK_FLAG_CONFIRMED`
   (ketiadaannya dicatat sebagai temuan, bukan diedit).
7. Baris protocol manifest lengkap dan terisi sebelum sesi.
8. `session_id`, `bag_id`, `sensor_id`, `device_id` konsisten antara
   manifest, `sessions.csv`, dan label fisik unit.
9. Baseline: `baseline_value`/`baseline_std` di `sessions.csv` cocok dengan
   perhitungan ulang offline (Bagian 6.2).
10. Sesi gagal atau dikeluarkan **tetap dicatat** beserta alasannya
    (manifest/catatan operator); tidak ada sesi yang "hilang".
11. Data mentah tidak dihapus, tidak diedit, dan tidak "diperbaiki"
    diam-diam. Koreksi apa pun berbentuk catatan tambahan ber-tanggal.

Sesi yang gagal QC tidak masuk dataset ordinal tetapi tetap disimpan;
sebagian tetap berguna untuk pengujian sistem (mis. sesi QC quality).

## 13. Batas Fitur dan Label

Ringkasan operasional dari rulebook v0.3 Bagian 5, untuk dipegang tim saat
menyiapkan data:

- **Tidak boleh menjadi fitur model:** `arm`, event dan timing injeksi,
  `tau`, `T_physical_leak`, safe horizon, nilai boundary B1/B2/B3, partisi,
  dan waktu menuju akhir sesi. Semua itu adalah bahan pembentuk label atau
  metadata eksperimen — model yang membacanya berarti membaca kunci jawaban.
- `activity_state` / `orientation_position` hanya boleh menjadi fitur jika
  nilainya memang **tersedia pada perangkat saat runtime** (mis. dari IMU).
  Jika pada bench keduanya hanya anotasi operator, keduanya berstatus
  **audit metadata**, bukan fitur.
- **LIG tetap fail-safe.** `lig_raw`/`leak_flag` bukan input utama model
  ordinal (fondasi §3.2); perannya alarm langsung dan evaluasi latensi.

## 14. Output Data per Sesi

Satu sesi yang selesai menghasilkan:

| File | Status skema | Isi |
|---|---|---|
| `sessions.csv` | Contract v1.1 — TIDAK diubah | Satu baris identitas + baseline + `end_reason` |
| `samples.csv` | Contract v1.1 — TIDAK diubah | Data 1 Hz dua kanal + quality |
| `events.csv` | Contract v1.1 — TIDAK diubah | Event Bagian 9 + kalibrasi + lainnya |
| `protocol_manifest.csv` | Offline companion artifact | Satu baris rencana sesi (Bagian 7.2) |
| `partition_manifest.csv` | Offline companion artifact | Penetapan partisi (Bagian 11.3); diisi per batch, bukan per sesi |
| Boundary config | Offline companion artifact, versioned | Konfigurasi B1/B2/B3 ber-versi (`boundary_config_version`); format file ditetapkan saat skrip pelabelan dibuat |

Companion artifacts tidak menambah kolom apa pun pada tiga CSV contract dan
tidak diwajibkan ada di perangkat/logger.

## 15. Keputusan, Unknown, dan Hubungan dengan Rulebook

### 15.1 Rekomendasi default prototipe

- Injeksi: syringe manual, profil stepwise ber-timer, dari sisi dalam
  kantong melalui tube simulasi stoma.
- Ground truth: observasi visual + kertas indikator + video ber-jam RTC.
- Window: `(t-W, t]`, W=120 s, stride 10 s, anchor `t_ref` = sample pertama
  sesi, hanya window penuh 120 sample.
- Dry minimal 120 s pasca kedua kanal `OK`; baseline = 60 s pertama fase
  kalibrasi.
- Matriks pilot Bagian 11 sebagai kerangka jumlah sesi.

### 15.2 Keputusan yang dapat dikunci sekarang (menunggu keputusan pengguna)

1. Definisi window `(t-W, t]` + anchor stride deterministik (Bagian 5).
2. Dry minimum 120 detik pasca kanal `OK` (Bagian 6.1).
3. Skema minimum protocol manifest (Bagian 7.2) dan enam input pelabelan
   (Bagian 7.3).
4. Aturan arah cairan: injeksi dari sisi dalam via simulasi stoma; tidak
   meneteskan langsung ke sensor pada sesi protokol (Bagian 2.3).
5. Daftar QC/acceptance criteria (Bagian 12) dan batas fitur (Bagian 13).

### 15.3 Angka yang harus menunggu pilot/shakedown

- Toleransi jitter, ambang missing sample, dan toleransi pembanding baseline
  (Bagian 5.3, 6.2).
- Konsentrasi, volume aliquot, jeda langkah, total volume, dan laju injeksi
  (Bagian 10).
- Durasi tipikal `INJECTION_START` → `PHYSICAL_LEAK_OBSERVED` → dasar
  penetapan B1/B2/B3 (rulebook v0.3 Bagian 4).
- Nilai `planned_safe_horizon_s` yang wajar untuk sesi non-leaking fill.
- Jumlah final sesi tiap tahap matriks (Bagian 11).
- Ketelitian timestamp observasi yang tercapai di lapangan (Bagian 9.3).

### 15.4 Pertanyaan untuk tim hardware

1. Posisi dan cara pemasangan pasti sensor kapasitif pada dinding kantong,
   dan sensor LIG pada jalur bocor — geometri baku yang bisa direplikasi?
2. Cara membuat celah bocor buatan yang reprodusibel antar sesi?
3. Durasi dan kriteria validitas kalibrasi LIG; penempatan 60 detik baseline
   kapasitif dalam urutan kalibrasi (Bagian 6.2)?
4. Apakah wet contact dapat membuat LIG `ADC_SATURATED`, dan bagaimana
   membedakannya dari fault (fondasi §8)?
5. Stabilitas sampling 1 Hz dan perilaku `DATA_GAP` logger terintegrasi?
6. Kebijakan pakai-ulang: berapa kali bag/baseplate dan LIG boleh dipakai
   ulang setelah terkena saline, dan bagaimana degradasinya dicatat?
7. Mekanisme pencatatan event operator ber-timestamp RTC (tombol, serial,
   companion script) — mana yang paling andal untuk rig ini?
8. Sinkronisasi jam video dengan RTC perangkat untuk audit ground truth?

### 15.5 Syarat sebelum protokol boleh dikunci

1. Logger terintegrasi dua kanal menghasilkan tiga CSV contract yang lolos
   QC Bagian 12 pada sesi shakedown nyata — saat ini belum tersedia.
2. Pertanyaan hardware Bagian 15.4 butir 1–5 terjawab.
3. Angka Bagian 15.3 terisi dari shakedown dengan justifikasi tertulis.
4. Keputusan Bagian 15.2 dikunci eksplisit oleh pengguna.
5. Review independen (Codex) atas protokol ini selesai dan direkonsiliasi.

### 15.6 Hubungan dengan rulebook v0.3

Rulebook v0.3 telah menyelesaikan item penyelarasan yang sebelumnya terbuka:
interval `(t - W, t]`, enam input pelabelan, anchor
`planned_safe_horizon_s`, dan exclusion reason struktural dari feature
extractor. Protokol ini tetap berstatus DRAFT karena angka pelaksanaan dan
pertanyaan hardware pada Bagian 15.3–15.4 belum dikunci.

## 16. Riwayat Revisi

- **v0.1 reconciliation patch (2026-07-27)** — memperbarui cross-reference ke
  Rulebook v0.3 dan mencatat penyelesaian perbedaan window, enam input,
  safe-horizon anchor, serta exclusion reason; tidak mengunci angka protokol.
- **v0.1 (2026-07-12)** — draft awal protokol pengambilan data terintegrasi;
  menyelesaikan tiga MUST FIX review Codex (definisi window, durasi dry dan
  baseline, protocol manifest). Belum direview independen.
