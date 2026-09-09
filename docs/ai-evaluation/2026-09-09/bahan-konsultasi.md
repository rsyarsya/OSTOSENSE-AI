# Bahan konsultasi AI OSTOSENSE

## Penjelasan pembuka

"Kami mengevaluasi tiga model menggunakan fixture sintetis berlabel dan menguji karakteristik 11 sesi sensor nyata.
Model OLR menghasilkan accuracy validation sintetis 94.74%, Macro F1 0.9529, dan kappa kuadratik 0.9733.
Kami belum menyatakan angka itu sebagai akurasi data nyata karena CSV nyata belum memiliki label risiko dan waktu kejadian pembanding yang terverifikasi."

## Pertanyaan yang perlu siap dijawab

1. **Dari mana label berasal?** Ini menentukan apa yang sebenarnya diukur oleh accuracy.
   Jawaban: label evaluasi berasal dari rulebook dan event fixture sintetis. Nama skenario P001–P007 tidak dijadikan label tiap window.
2. **Berapa data training dan validation?** Banyak window tidak sama dengan banyak percobaan independen.
   Jawaban: 76 window dari 4 sesi untuk fitting dan 38 window dari 2 sesi untuk validation; bag/sensor/sesi tidak silang.
   Window dalam sesi bertumpang tindih, dan validation sudah dikenal saat pengembangan pipeline; hasil bukan Final Test independen.
3. **Mengapa OLR dibandingkan dengan LR dan Tree?** Pemilihan pembanding perlu alasan yang jelas.
   Jawaban: OLR memodelkan urutan empat kelas; LR dan Tree memberi pembanding sederhana pada fitur, label, dan partisi yang sama.
   Parameternya tetap sejak awal. Model dengan skor fixture tertinggi tidak otomatis menjadi model terbaik pada alat nyata.
4. **Apakah sudah bebas overfitting?** Skor training tinggi saja tidak menjawab generalisasi.
   Jawaban: belum dapat disimpulkan. Selisih training/validation dilaporkan, tetapi fixture dan jumlah sesi sangat terbatas.
5. **Apa yang dibuktikan oleh data nyata?** Penting untuk membedakan hasil alat dan hasil klasifikasi.
   Jawaban: struktur/timing logger, respons raw lima kanal, statistik, korelasi deskriptif, dan pembentukan fitur tanpa label.
   Accuracy risiko, sensitivitas kebocoran, serta lead time belum dapat dihitung secara sah.
6. **Mengapa P006 tidak ikut korelasi?** Pengeluaran data harus terlihat dan beralasan.
   Jawaban: P006 adalah uji gangguan sensor, berbeda dari sesi operasional. P006 tetap ada pada QC, statistik raw, dan grafik tersendiri.
7. **Apakah semua data dipakai?** Denominator dan data parsial memengaruhi interpretasi.
   Jawaban: seluruh 21.887 raw masuk grafik/statistik; 47 raw akhir belum membentuk kelompok lengkap dan tidak masuk agregasi 1 Hz.
   Hasil agregasi berisi 2.184 kelompok lengkap dan 94 window nyata tanpa label. Window sintetis non-eligible dicatat beserta alasannya.
8. **Apa arti grafik optimisasi?** Jangan menyamakan proses model yang berbeda.
   Jawaban: objective dan skor validation sepanjang iterasi L-BFGS-B untuk OLR; bukan epoch neural network.
   Parameter, probabilitas, dan kelas hasil instrumentasi diuji paritas terhadap implementasi canonical.
9. **Apakah target Macro F1 ≥0,70 dan kappa ≥0,61 sudah tercapai?** Target memerlukan jenis bukti yang ditetapkan.
   Jawaban: belum dapat dinyatakan tercapai, karena target berlaku pada Final Test nyata, sedangkan angka paket ini sintetis.
10. **Apakah sistem sudah berguna sebagai prototipe?** Kesimpulan harus sesuai hal yang diuji.
   Jawaban: mekanik model dapat berjalan dan karakterisasi sensor tersedia. Ketepatan risiko pada kondisi nyata belum diketahui;
   penilaian kecukupan demonstrasi prototipe dibahas dengan pembimbing, tanpa menyatakan standar minimum PKM tertentu.

## Usulan kalimat untuk laporan/presentasi

"Pada evaluasi fixture sintetis dengan pemisahan Development dan Validation berdasarkan sesi, kantong, dan sensor,
OLR memperoleh accuracy 94.74%, Macro F1 0.9529, dan quadratic weighted kappa 0.9733.
Pengujian ini menunjukkan mekanik klasifikasi pada fixture tersebut. Sebanyak 11 sesi sensor nyata dianalisis secara deskriptif;
evaluasi klasifikasi pada data nyata belum dilakukan karena ground truth belum tersedia."

## Pertanyaan untuk pembimbing

"Dengan tidak memungkinkan pengambilan data tambahan, kami menyajikan evaluasi sintetis dan karakterisasi sensor nyata secara terpisah.
Apakah pemisahan hasil, keterbatasan, dan bentuk kesimpulan ini sudah tepat untuk mempertanggungjawabkan prototipe kami?"

Rujukan angka: [laporan](laporan.md) dan [metrik](tables/model_metrics.csv).
