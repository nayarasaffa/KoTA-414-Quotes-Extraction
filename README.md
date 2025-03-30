# KoTA-414-Quotes-Extraction

Repositori ini berisi kode untuk melakukan ekstraksi kutipan dari teks berita dalam Bahasa Indonesia menggunakan Named Entity Recognition (NER). Model ini dirancang untuk mengidentifikasi kutipan, pembicara, dan entitas terkait secara otomatis.

## **Persiapan environment:**
Sebelum menjalankan kode dalam repositori ini, pastikan untuk mengatur environment dengan benar.
1. **Buat Virtual Environment**
   ```
   python -m venv .venv
   ```
2. **Aktifkan Virtual Environment**
   ```
   .\.venv\Scripts\activate
   ```
3. **Instalasi Dependensi**
   ```
   python -m pip install -r requirements.txt
   ```

## **Struktur Direktori**
Repositori ini memiliki beberapa direktori utama yang digunakan dalam pipeline ekstraksi kutipan:

### **1. Auto-Labeling**
Berisi kode untuk melakukan anotasi otomatis pada teks berita menggunakan model NER yang telah dilatih. Proses ini membantu mempercepat pelabelan data untuk pelatihan model lebih lanjut.

### **2. Preprocessing**
Berisi kode untuk membersihkan dan menyiapkan data sebelum digunakan dalam proses pelatihan model. Langkah-langkah preprocessing meliputi normalisasi teks, pemisahan kalimat, splitting dataset, dan konversi ke format yang sesuai untuk NER.
