from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io
from PIL import Image
import serial
import time

app = Flask(__name__)
CORS(app)


# Load ML model
model = load_model("../saved_model/cnn_railway_model.h5")
with open("../saved_model/class_indices.json") as f:
    class_dict = json.load(f)
class_indices = {v: k for k, v in class_dict.items()}

# Image size (same as training)
img_height, img_width = 96, 96

# Serial settings (optional)
SERIAL_PORT = 'COM6'  # Change to your ESP32 port
BAUD_RATE = 115200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read image from POST
        img_bytes = request.files['frame'].read()
        img = Image.open(io.BytesIO(img_bytes)).resize((img_height, img_width))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict with model
        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction)
        predicted_class = class_indices[pred_idx]
        confidence = prediction[0][pred_idx] * 100

        # Decide danger or safe
        if predicted_class == 'major crack' or predicted_class == 'Obstacle':
            result = "danger"
        else:
            result = "safe"

        status_msg = f"{result.capitalize()}: {predicted_class}"
        print(status_msg)

        # Optional: Send to ESP32 via Serial
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
            time.sleep(2)
            ser.write((result + "\n").encode())
            ser.flush()
            ser.close()
        except:
            pass  # Ignore if not connected

        return jsonify({
            "class": predicted_class,
            "confidence": f"{confidence:.2f}",
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="192.168.223.36", port=5000, debug=False)
