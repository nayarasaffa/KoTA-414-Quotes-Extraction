# KoTA-414-Quotes-Extraction/Auto-Labeling

Berisi kode untuk melakukan anotasi otomatis pada teks berita menggunakan model NER yang telah dilatih. Hasil anotasi akan disimpan dalam format JSON yang kompatibel dengan Label Studio.

## Struktur Direktori
1. **data**
   Berisi data dalam format JSON yang akan diproses untuk anotasi. Hasil anotasi juga disimpan di dalam direktori ini. Selain itu, terdapat skrip `data_chunker.py` untuk membagi data berita dalam file JSON menjadi beberapa bagian yang lebih kecil.
2. **dataset**
   Berisi dataset yang digunakan sebagai input model BiLSTM.
3. **idsentsegmenter**
   Berisi kode untuk melakukan segmentasi kalimat pada teks berita.
4. **models**
   Berisi direktori dan kode yang digunakan untuk menjalankan model. Direktori ini memerlukan beberapa subdirektori tambahan:
   - **model**:  Berisi model BiLSTM yang telah dilatih. Model dapat diunduh melalui tautan berikut: [Download Model BiLSTM](https://drive.google.com/file/d/1--REEJK8Lb7KddHzUP_eYUFEHGDiDcGg/view?usp=drive_link)
   - **pretrain**: Berisi model Word2Vec yang telah dipra-latih. Model pretrain dapat diunduh melalui tautan berikut: [Download Model Pretrain Word2Vec](https://drive.google.com/drive/folders/1a5RwTTHxH_YdPjlKcbpIOD0Qihk_A10f?usp=drive_link)

## Struktur File
1. **auto_labeling.py**
   Script utama untuk melakukan anotasi otomatis pada teks berita dan menyimpannya dalam format JSON yang didukung oleh Label Studio.
   **Daftar Argumen:**
   |Argumen|Deskripsi|
   | --- | --- |
   |`-i` `--input_file`|Path ke file input|
   |`-o` `--output_file`|Path ke file output|

   **Contoh Menjalankan:**
   ```
   python auto_labeling.py -i data/nayara/news_nayara.json -o data/nayara/news_nayara_anotated.json
   ```
