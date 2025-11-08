import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Paths
dataset_path = "../augmented_dataset"
model_path = "../saved_model/cnn_railway_model.h5"
class_indices_path = "../saved_model/class_indices.json"

# Image settings
img_height, img_width = 96, 96
batch_size = 32

# Load model
model = load_model(model_path)

# Load class indices (just in case you need mapping)
with open(class_indices_path) as f:
    class_indices = json.load(f)

# Prepare validation data (same split logic as before)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Evaluate
loss, accuracy = model.evaluate(val_generator)
error_percent = (1 - accuracy) * 100

print(f"[INFO] Validation Accuracy: {accuracy*100:.2f}%")
print(f"[INFO] Error Percentage: {error_percent:.2f}%")
