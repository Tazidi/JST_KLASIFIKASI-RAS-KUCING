import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

train_dir = r'D:\!Kuliah\JST\JST\Tugas Besar\12-ras-kucing-noval\train'

def load_images_from_directory(directory, image_size=(224, 224)):
    images = []
    labels = []
    class_names = []

    # Ekstensi file gambar yang akan diterima
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # Loop setiap folder kelas di dalam train (Anggora, Persia, dll.)
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)

        # Skip kalau bukan folder
        if not os.path.isdir(class_dir):
            continue

        class_idx = len(class_names)
        class_names.append(class_name)
        print(f"Memproses kelas: {class_name}")

        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)

            # Skip kalau bukan file biasa (mis. folder lagi)
            if not os.path.isfile(img_path):
                continue

            # Skip kalau ekstensi bukan gambar
            if not fname.lower().endswith(valid_exts):
                print("Lewati (bukan gambar):", img_path)
                continue

            try:
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                # Kalau ada gambar korup / rusak tetap diskip, biar script tidak crash
                print("Gagal load:", img_path, "| Error:", e)
                continue

    images = np.array(images)
    labels = np.array(labels)

    print("Total gambar terbaca:", images.shape[0])
    print("Kelas:", class_names)

    return images, labels, class_names

# Load the images
images, labels, class_names = load_images_from_directory(train_dir)

# Normalize images to [0, 1] range
images = images / 255.0

# Split into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# One-hot encode the labels (categorical)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(class_names))

# Check the shape of the data
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on the training data
datagen.fit(X_train)

num_classes = len(class_names)

base_model = tf.keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the base model
base_model.trainable = False

# Create the final model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout layer to reduce overfitting
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Print the model summary
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=3,  # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Restore the weights from the epoch with the best validation loss
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Use the augmented data generator
    epochs=25,  # Increase epochs to give the model more chances to train
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]  # Use early stopping callback
)

def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

save_path = r'D:\!Kuliah\JST\JST\Model\kucing_resnet_model_revisi2.h5'
model.save(save_path)

