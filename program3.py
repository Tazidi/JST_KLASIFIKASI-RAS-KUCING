import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix

# Fungsi Focal Loss (Jika ada ketidakseimbangan kelas)
from tensorflow.keras import backend as K
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss_fixed

# Load ResNet50 pre-trained model as feature extractor (exclude top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tune ResNet50, only train the last 20 layers
for layer in base_model.layers[:-20]:  # Membekukan lebih sedikit layer
    layer.trainable = False

# Build the model
input_layer = Input(shape=(224, 224, 3))
x = base_model(input_layer)
x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce dimensionality
x = Dense(1024, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout for regularization
output_layer = Dense(12, activation='softmax')(x)  # Output layer with 12 classes for kucing ras

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with Adam optimizer and focal loss for multi-class classification
model.compile(optimizer=Adam(learning_rate=0.0001), loss=focal_loss(), metrics=['accuracy'])

# Print model summary to confirm the architecture
model.summary()

# Persiapkan data pelatihan dan validasi
train_dir = r'D:\!Kuliah\JST\JST\Tugas Besar\12-ras-kucing-noval\train'
val_dir = r'D:\!Kuliah\JST\JST\Tugas Besar\12-ras-kucing-noval\validation'

# Buat objek ImageDataGenerator untuk augmentasi gambar dan normalisasi
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi pixel menjadi range 0-1
    horizontal_flip=True,  # Augmentasi dengan flip horizontal
    rotation_range=30,  # Augmentasi dengan rotasi acak
    zoom_range=0.2,  # Zoom pada gambar
    shear_range=0.2,  # Shearing atau pemotongan gambar
    width_shift_range=0.2,  # Geser gambar secara horizontal
    height_shift_range=0.2,  # Geser gambar secara vertikal
    brightness_range=[0.5, 1.5],  # Penyesuaian kecerahan
    channel_shift_range=50.0  # Penyesuaian channel warna
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

# Callback untuk EarlyStopping dan ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_revisi1.h5', save_best_only=True, monitor='val_loss')

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 10:
        return lr * 0.5  # Kurangi learning rate setiap 10 epoch
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Melatih model dengan callbacks
model.fit(
    train_generator,  # Data pelatihan
    validation_data=val_generator,  # Data validasi
    epochs=50,  # Jumlah epoch
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Evaluasi Model pada Data Validasi
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Menyimpan Model
model.save('kucing_resnet_model_revisi2.h5')

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