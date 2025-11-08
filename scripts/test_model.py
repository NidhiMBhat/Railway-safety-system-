import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import serial
import time

# --- CONFIGURATION ---

SERIAL_PORT = 'COM6'       # 🔧 Update this if your port changes
BAUD_RATE = 115200
model_path = "../saved_model/cnn_railway_model.h5"
class_indices_path = "../saved_model/class_indices.json"
img_height, img_width = 96, 96  # 🔁 Must match your training input size

# --- LOAD MODEL & CLASSES ---

model = load_model(model_path)
print("[INFO] Model loaded successfully.")

with open(class_indices_path) as f:
    class_dict = json.load(f)
class_indices = {v: k for k, v in class_dict.items()}

# --- GET IMAGE FROM COMMAND LINE ---

if len(sys.argv) < 2:
    print("Usage: python test_model.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
    sys.exit(1)

# --- PREPROCESS IMAGE ---

img = image.load_img(image_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# --- PREDICT ---

prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)
predicted_class = class_indices[predicted_class_index]
confidence = prediction[0][predicted_class_index]
error_percent = (1 - confidence) * 100

# --- DETERMINE SAFETY STATUS ---

if predicted_class in ['major crack', 'Obstacle']:
    result_msg = "danger"
elif predicted_class == 'normal':
    result_msg = "safe"
else:
    result_msg = "unknown"

# --- OUTPUT ---

print(f"[RESULT] Class: {predicted_class}")
print(f"[CONFIDENCE] {confidence * 100:.2f}%")
print(f"[ERROR MARGIN] {error_percent:.2f}%")
print(f"[ACTION] Send to ESP32: {result_msg}")

# --- SEND TO ESP32 VIA SERIAL ---

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)  # Allow ESP32 to reset
    ser.write((result_msg + "\n").encode())
    ser.flush()
    ser.close()
    print("[INFO] Sent to ESP32 via Serial.")
except Exception as e:
    print(f"[ERROR] Could not send to ESP32: {e}")
