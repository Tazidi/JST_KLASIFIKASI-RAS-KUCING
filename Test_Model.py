import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

test_dir = r'D:\!Kuliah\JST\JST\Tugas Besar\12-ras-kucing-noval\validation'

# === Definisi focal loss (kalau model aslinya pakai ini) ===
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss_fixed

# Function to load and preprocess test images
def load_test_images_from_directory(directory, image_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(class_idx)  # Using class index as the label

    return np.array(images), np.array(labels), class_names

# Load model yang sudah disimpan
model_path = r'D:\!Kuliah\JST\JST\Model\kucing_resnet_model_revisi2.h5'
model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})

# Load and preprocess test data
test_images, test_labels, class_names = load_test_images_from_directory(test_dir)

# Normalize the test images
test_images = test_images / 255.0

# One-hot encode the test labels
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(class_names))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# Print the test results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
