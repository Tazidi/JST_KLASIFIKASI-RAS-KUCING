**Penjelasan:**

* `conda create -n tf-gpu python=3.10`
  Membuat environment baru bernama `tf-gpu` dengan versi Python 3.10.
* `conda activate tf-gpu`
  Mengaktifkan environment tersebut. Semua perintah `python` dan `pip` setelah ini akan berjalan di env `tf-gpu`.

---

## 2. Instalasi Library yang Dibutuhkan

Di dalam environment `tf-gpu`, install semua dependencies:

```bash
pip install "tensorflow==2.10.0"
pip install "numpy<2"
pip install scikit-learn
pip install pillow
pip install matplotlib
```

**Penjelasan singkat tiap paket:**

* `tensorflow==2.10.0`
  Versi TensorFlow terakhir yang masih mendukung GPU di Windows (dengan CUDA).
* `numpy<2`
  TensorFlow 2.10 hanya kompatibel dengan NumPy 1.x, jadi versi NumPy harus di bawah 2.
* `scikit-learn`
  Dipakai untuk fungsi-fungsi seperti `confusion_matrix`.
* `pillow`
  Library untuk memproses gambar, diperlukan oleh `ImageDataGenerator`.
* `matplotlib`
  Opsional, dipakai jika ingin plotting grafik (akurasi, loss, dsb).

---

## 3. Menjalankan Program

Masih di dalam environment `tf-gpu`, jalankan script utama:

```bash
python "c:/Path/ke/file/programm.py"
```

**Catatan:**

* Pastikan path ke file `.py` sesuai dengan lokasi file di komputer kamu.
* Sebelum menjalankan, cek prompt terminal sudah bertuliskan `(tf-gpu)` di depan, contoh:

  ```text
  (tf-gpu) PS C:\Path\ke\Folder>
  ```

---

## 4. Menggunakan Environment `tf-gpu` di VS Code

Agar VS Code menjalankan kode dengan environment `tf-gpu`, ikuti langkah berikut:

1. Buka **VS Code** di folder project (misalnya di `C:\Path\ke\Folder\Pyhton`).

2. Lihat di **pojok kiri bawah** VS Code, klik tulisan `Python 3.x (...)` atau nama interpreter Python yang muncul.

3. Pilih interpreter yang namanya mengandung **`tf-gpu`** dan bertipe **Conda**
   (misalnya: `Python 3.10.x ('tf-gpu': conda)`).

4. Tutup **Integrated Terminal** yang sedang terbuka (kalau ada).

5. Buka **Terminal baru** lewat menu:

   * `Terminal` → `New Terminal`

6. VS Code akan otomatis menjalankan:

   ```bash
   conda activate tf-gpu
   ```

   sehingga prompt di terminal akan menjadi:

   ```text
   (tf-gpu) PS C:\Path\ke\Folder\Pyhton>
   ```

7. Sekarang jalankan file seperti biasa:

   * Via tombol **Run** / **F5**, atau
   * Via terminal:

     ```bash
     python programm.py
     ```

**Intinya:**
Setelah interpreter `tf-gpu` dipilih di VS Code dan terminal baru dibuka, seluruh eksekusi Python di project ini akan berjalan menggunakan environment `tf-gpu` (TensorFlow 2.10 + GPU).

---
