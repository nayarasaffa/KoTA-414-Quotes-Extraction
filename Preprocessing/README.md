# KoTA-414-Quotes-Extraction/Preprocessing
Direktori ini berisi kode untuk membersihkan dan menyiapkan data sebelum digunakan dalam proses pelatihan model Named Entity Recognition (NER).

## Struktur Direktori
1. **dataset**
   Menyimpan file hasil ekspor dari Label Studio dalam format `.conll` serta file hasil preprocessing dalam format `.tsv`.
2. **idsentsegmenter**
   Berisi kode untuk melakukan segmentasi kalimat pada teks berita.
3. **split_dataset**
   Berisi kode untuk membagi dataset berdasarkan jumlah berita dalam satu file. File dalam format JSON akan dipecah berdasarkan jumlah berita, lalu hasilnya diimpor kembali ke Label Studio untuk diekspor ulang dalam format `.conll`.

## Struktur File
1. **main.py**
   Script utama untuk menjalankan preprocessing, token segmentation, serta memeriksa kondisi dataset sebelum digunakan dalam pelatihan model.
   **Daftar Argumen:**
   |Argumen|Deskripsi|
   | --- | --- |
   |`-i` `--input_file`|Path ke file input|
   |`-o` `--output_file`|Path ke file output|
   |`-c` `--check_dataset`|Memeriksa kondisi dataset|

   **Contoh Menjalankan:**
   ```
   python main.py -i example-test-dataset.conll -o example-test-dataset.tsv -c
   ```
3. **preprocessing.py**
   Berisi kelas untuk melakukan preprocessing pada dataset, termasuk:
   - Konversi format dari `.conll` ke `.tsv`
   - Mengubah skema anotasi dari BIO ke BILOU
   - Melakukan segmentasi kalimat pada token teks
4. **token_segmentation.py**
   Berisi kelas untuk membagi token berdasarkan struktur kalimatnya.
   
