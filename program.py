import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix

# Load ResNet50 pre-trained model as feature extractor (exclude top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze ResNet50 layers (we will not train the ResNet layers)
base_model.trainable = False

# Build the model
input_layer = Input(shape=(224, 224, 3))
x = base_model(input_layer)
x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce dimensionality
x = Dense(1024, activation='relu')(x)  # Fully connected layer
output_layer = Dense(12, activation='softmax')(x)  # Output layer with 12 classes for kucing ras

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with Adam optimizer and sparse categorical cross-entropy loss (for multi-class classification)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary to confirm the architecture
model.summary()

# Persiapkan data pelatihan dan validasi
train_dir = r'C:\Users\lenovo\Downloads\Tugas Besar\12-ras-kucing-noval\train'
val_dir = r'C:\Users\lenovo\Downloads\Tugas Besar\12-ras-kucing-noval\validation'

# Buat objek ImageDataGenerator untuk augmentasi gambar dan normalisasi
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi pixel menjadi range 0-1
    horizontal_flip=True,  # Augmentasi dengan flip horizontal
    rotation_range=30  # Augmentasi dengan rotasi acak
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Validasi hanya normalisasi

# Menghasilkan batch data dari direktori (pastikan data sudah terorganisir dalam folder)
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(224, 224),  # Ukuran input gambar
    batch_size=32,  # Ukuran batch
    class_mode='sparse'  # Mode klasifikasi multi-kelas (ras kucing)
)

val_generator = val_datagen.flow_from_directory(
    val_dir, 
    target_size=(224, 224),  # Ukuran input gambar
    batch_size=32,  # Ukuran batch
    class_mode='sparse'  # Mode klasifikasi multi-kelas (ras kucing)
)

# Melatih model
model.fit(
    train_generator,  # Data pelatihan
    validation_data=val_generator,  # Data validasi
    epochs=10  # Jumlah epoch untuk melatih model
)

# Evaluasi Model pada Data Validasi
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Menyimpan Model
model.save('kucing_resnet_model.h5')

# Memuat Model yang telah disimpan (Jika diperlukan)
# model = load_model('kucing_resnet_model.h5')

# Jika ingin menghitung confusion matrix
# Prediksi pada data validasi
val_labels = val_generator.classes  # Kelas yang sebenarnya
val_preds = model.predict(val_generator)
val_preds = np.argmax(val_preds, axis=1)  # Mengambil kelas dengan probabilitas tertinggi

# Hitung confusion matrix
cm = confusion_matrix(val_labels, val_preds)
print("Confusion Matrix:")
print(cm)
