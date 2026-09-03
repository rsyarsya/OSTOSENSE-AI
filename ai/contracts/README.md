# Kontrak Keluaran AI

Folder ini adalah antarmuka mesin antara pipeline AI dan software OSTOSENSE.

- `ai-runtime-output-v0.1.schema.json`: JSON Schema Draft 2020-12 yang menjadi
  sumber validasi payload.
- `examples/live-unavailable.json`: keadaan LIVE yang berlaku selama belum ada
  model nyata berlabel yang disetujui.
- `examples/engineering-test-monitor.json`: contoh simulasi sintetis internal.

Perubahan field atau makna state harus menghasilkan versi kontrak baru. Jangan
mengubah file v0.1 secara diam-diam setelah software mulai menggunakannya.
Aturan tampilan dan daftar pekerjaan tim software dijelaskan di
`docs/ai-software-integration-contract-v0.1.md`.
