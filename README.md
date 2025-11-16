# Proyek Ujian Tengah Semester (UTS)
# Sistem Temu Kembali Informasi (STKI) - A11.4703

**Implementasi Boolean Retrieval dan Vector Space Model (VSM) pada Korpus Berita CNN**

---

Disusun Oleh:
* Nama	: Tosan Iqbal Kurniawan
* NIM	: A11.2023.15409
* Kelas	: A11.4703

##  deskripsi Proyek

Proyek ini merupakan implementasi sistem temu kembali informasi (STKI) mini untuk memenuhi tugas Ujian Tengah Semester Ganjil 2025/2026 mata kuliah STKI. 

Tujuan utama dari eksperimen ini adalah untuk membangun dan mengevaluasi dua model STKI klasik—**Boolean Retrieval Model** dan **Vector Space Model (VSM)** —pada korpus kecil yang terdiri dari 9 dokumen berita CNN.

Eksperimen ini mencakup seluruh alur STKI, mulai dari:
1. Document Preprocessing:** Tokenisasi, Case-Folding, Stopword Removal, Stemming .
2. Indexing: Pembuatan Incidence Matrix dan Inverted Index .
3. Term Weighting:Implementasi TF, IDF, dan TF-IDF .
4. Ranking:Perhitungan Cosine Similarity .
5. Evaluasi: Perhitungan Precision@k, Average Precision (AP), dan Mean Average Precision (MAP) .

Seluruh eksperimen dan implementasi dilakukan menggunakan Python di lingkungan Google Colab.

## Struktur Proyek

Berikut adalah susunan file dan folder yang digunakan dalam proyek ini, disesuaikan dari struktur yang disarankan:

stki/
│
├── Data/
│   ├── data1.txt
│   ├── data2.txt
│   ├── ...
│   └── data10.txt
│
├── Notebooks/
│   └── Proyek_UTS_STKI.ipynb
│
├── report/
│   └── Laporan_UTS_STKI.pdf
│
├── src/
│   ├── preprocessing.py
│   ├── boolean.py
│   └── vsm.py
│
├── README.md
