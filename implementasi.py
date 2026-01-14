import os
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# Path ke folder train & validation
train_dir = r'D:\!Kuliah\JST\JST\Tugas Besar\12-ras-kucing-noval\train'
test_images_path = r'D:\!Kuliah\JST\JST\Tugas Besar\12-ras-kucing-noval\validation'

# Path model yang telah dilatih
model_path = r'D:\!Kuliah\JST\JST\Model\kucing_resnet_model_revisi2.h5'

# Ambil label kelas langsung dari folder train (SAMA seperti saat training)
class_labels = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])
print("Class labels (dari train):", class_labels)

# Memuat model yang telah dilatih
model = load_model(model_path)

# Fungsi untuk menghapus digit di akhir nama folder (mis: "Abyssunian1" -> "Abyssunian")
def strip_digits(name: str) -> str:
    return re.sub(r'\d+$', '', name)

# Kumpulkan (image_path, true_label_idx)
image_infos = []

for class_name in os.listdir(test_images_path):
    class_dir = os.path.join(test_images_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Nama kelas sebenarnya tanpa angka,
    # mis. "Abyssunian1" -> "Abyssunian"
    base_class_name = strip_digits(class_name)

    if base_class_name not in class_labels:
        print(f"PERINGATAN: '{base_class_name}' tidak ada di class_labels, dilewati.")
        continue

    true_idx = class_labels.index(base_class_name)

    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(class_dir, fname)
            image_infos.append((image_path, true_idx))

print("Total gambar yang ditemukan:", len(image_infos))

# Batas jumlah gambar yang akan ditampilkan
max_images_to_display = 10

total = 0
correct = 0

for idx, (image_path, true_idx) in enumerate(image_infos):
    img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    pred_label = class_labels[predicted_class]
    true_label = class_labels[true_idx]

    total += 1
    if predicted_class == true_idx:
        correct += 1

    if idx < max_images_to_display:
        original_img = Image.open(image_path)
        plt.figure(figsize=(4, 4))
        plt.imshow(original_img)
        plt.title(
            f"GT: {true_label}\nPred: {pred_label} ({confidence * 100:.0f}%)"
        )
        plt.axis('off')
        plt.show()
    else:
        print(
            f"Gambar: {os.path.basename(image_path)}, "
            f"GT: {true_label}, Pred: {pred_label} ({confidence * 100:.0f}%)"
        )

accuracy = correct / total if total > 0 else 0.0
print("\n===== HASIL AKHIR =====")
print(f"Total gambar   : {total}")
print(f"Benar          : {correct}")
print(f"Salah          : {total - correct}")
print(f"Akurasi total  : {accuracy * 100:.2f}%")
