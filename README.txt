===================================================
 Panduan Menjalankan Aplikasi Deteksi Korosi Berbasis Model Yolov5
===================================================

Dokumen ini berisi panduan untuk mempersiapkan dan menjalankan aplikasi web deteksi korosi yang dibangun menggunakan Streamlit dan YOLOv5.

--------------------
1. Prasyarat
--------------------

Pastikan sistem Anda telah terpasang perangkat lunak berikut:

- Python 3.8 atau versi yang lebih baru. Anda dapat mengunduhnya dari https://www.python.org/
- pip (biasanya sudah terpasang bersama Python).
- membuat conda enviromet untuk aplikasi ini.

--------------------
2. Persiapan Lingkungan (Setup)
--------------------

Ikuti langkah-langkah berikut di terminal atau command prompt Anda.

a. Buat Direktori Proyek dan Masuk ke Dalamnya
   Jika Anda belum memiliki folder proyek, buatlah satu dan navigasikan ke dalamnya.

b. Buat dan Aktifkan Lingkungan Virtual  Menggunakan Conda(Sangat Direkomendasikan)
   Ini akan mengisolasi dependensi proyek Anda.

   # Buat lingkungan virtual bernama 'venv'
   python -m venv venv

   # Aktifkan lingkungan virtual
   # Di Windows:
   .\venv\Scripts\activate

   # Di macOS dan Linux:
   source venv/bin/activate

   Setelah aktif, Anda akan melihat `(venv)` di awal baris perintah.

c. Instal Semua Dependensi yang Diperlukan
   Pastikan file `requirements.txt` berada di dalam direktori proyek Anda. Kemudian jalankan:

   pip install -r requirements.txt

d. Siapkan File Model (PENTING!)
   Aplikasi ini memerlukan file bobot (weights) model YOLOv5 untuk dapat berfungsi.
   
   - Unduh atau pastikan Anda memiliki file bernama `yolov5-corrosion.pt`.
   - Letakkan file `yolov5-corrosion.pt` di dalam direktori utama proyek, di lokasi yang sama dengan file `web_ui.py`.

-----------------------------
3. Menjalankan Aplikasi Web
-----------------------------

Setelah semua persiapan selesai, Anda siap menjalankan aplikasi.

a. Jalankan Perintah Streamlit
   Pastikan Anda masih berada di dalam direktori proyek dan lingkungan virtual `(venv)` sudah aktif. Jalankan perintah berikut:

   streamlit run web_ui.py

b. Buka Aplikasi di Browser
   Setelah perintah di atas dijalankan, Streamlit akan secara otomatis membuka tab baru di browser Anda dengan alamat URL lokal (biasanya http://localhost:8501).

c. Gunakan Aplikasi
   - Gunakan sidebar untuk mengatur "Confidence Threshold" dan "IOU Threshold".
   - Unggah gambar yang ingin Anda analisis melalui tombol "Pilih sebuah gambar...".
   - Klik tombol "Deteksi Korosi" untuk melihat hasilnya.
   
d. Jika anda hanya ingin menjalankan aplikasi di lokal tanpa UI maka coba untuk menjalankan python app.py pada terminal

contoh hasil running