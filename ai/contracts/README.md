# Kontrak Keluaran AI

Folder ini adalah antarmuka mesin antara pipeline AI dan software OSTOSENSE.

- `ai-raw-window-input-v0.1.schema.json`: input layanan raw eksperimental,
  dengan baseline kering, window, dan konfirmasi operator.
- `ai-service-response-v0.1.schema.json`: wrapper layanan. Objek `ai` di
  dalamnya tetap memakai runtime v0.2; diagnostik tidak ditambahkan ke runtime.
- `typescript/ai-service-v0.1.ts`: tipe pendamping permintaan/respons layanan.
- `examples/service-v0.1/`: permintaan dan respons ketika belum ada pembacaan.

- `ai-runtime-output-v0.2.schema.json`: kontrak aktif untuk integrasi baru.
- `ai-feature-input-v0.1.schema.json`: kontrak satu window lima fitur untuk
  reference emitter v0.2; mengunci urutan dan rumus fitur.
- `examples/v0.2/`: contoh keadaan tidak tersedia serta keempat kelas untuk
  simulasi sintetis dan sensor nyata yang secara eksplisit belum tervalidasi.
- `examples/feature-input-v0.1/`: contoh input fitur sintetis dan `Kap_7` nyata.
- `typescript/ai-runtime-output-v0.2.ts`: discriminated union untuk backend,
  aplikasi, dan web TypeScript.
- `typescript/ai-feature-input-v0.1.ts`: tuple fitur dan pasangan sumber-kanal
  untuk komponen yang menjalankan reference emitter.
- `ai-runtime-output-v0.1.schema.json` dan contoh pada `examples/`: kontrak lama
  yang dipertahankan agar integrasi v0.1 tidak rusak.

Perubahan field atau makna state harus menghasilkan versi kontrak baru. Jangan
mengubah file v0.1 secara diam-diam setelah software mulai menggunakannya.
`features` pada feature-input selalu berurutan sesuai `feature_order` dan memakai
`RAW_MINUS_SESSION_BASELINE`. Kolom `*_delta_norm` dari laporan pilot bersifat
deskriptif dan bukan masukan model ini.
Hash berulang `aaaa...` dan nilai fitur nol pada folder `examples/` adalah
placeholder uji format, bukan hash model atau pembacaan sensor nyata.
Empat contoh dengan **hash model sebenarnya** dan keluaran yang sudah dihitung
tersedia di [aset model](../src/ostosense_ai/model_assets/engineering_v03/examples.json)
dan dapat diekspor melalui `python -m ostosense_ai.integration_cli examples`.
Aturan tampilan dan daftar pekerjaan tim software dijelaskan di
`../../docs/ai-software-integration-contract-v0.2.md`.
