import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Memuat model yang sudah dilatih
model = load_model(r'C:\KULIAH\SEMESTER 5\JST\Tugas Besar\kucing_resnet_model_revisi1.h5')

# Mendefinisikan class_labels secara manual
class_labels = {
    0: 'Abyssinian',
    1: 'Bengal',
    2: 'Birman',
    3: 'Bombay',
    4: 'British Shorthair',
    5: 'Egyptian',
    6: 'Maine Coon',
    7: 'Persian',
    8: 'Ragdoll',
    9: 'Russian Blue',
    10: 'Siamese',
    11: 'Sphynx'
}

# Fungsi untuk memilih file dan melakukan prediksi
def predict_image():
    file_path = filedialog.askopenfilename(title="Pilih Gambar")
    if file_path:
        img = image.load_img(file_path, target_size=(224, 224))  # Ukuran input model
        img_array = image.img_to_array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
        
        # Melakukan prediksi
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        
        # Mendapatkan nama kelas berdasarkan indeks prediksi
        predicted_class_name = class_labels[predicted_class_idx]
        
        # Menampilkan hasil prediksi
        result_label.config(text=f"Predicted Class: {predicted_class_name}")

# Membuat window aplikasi Tkinter
window = tk.Tk()
window.title("Klasifikasi Gambar")

# Menambahkan tombol untuk memilih gambar
predict_button = tk.Button(window, text="Pilih Gambar", command=predict_image)
predict_button.pack()

# Menambahkan label untuk menampilkan hasil prediksi
result_label = tk.Label(window, text="Predicted Class: -")
result_label.pack()

# Menjalankan aplikasi
window.mainloop()
