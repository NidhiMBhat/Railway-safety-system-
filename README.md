# 🚄 AI-Powered Railway Track Crack Detection System

### 🔍 Real-Time Crack Monitoring using Computer Vision and Deep Learning

This project aims to detect railway track cracks automatically using a **Convolutional Neural Network (CNN)** model and a **web-based live monitoring interface**.  
This project addresses two major railway safety issues:
1.Early detection of track cracks or obstacles that could cause derailments.
(Traditional inspection methods are **manual, time-consuming, and error-prone**)
2.Automated visual monitoring of railway conditions to reduce manual inspection delays and human error.

## ⚙️ Tech Stack

| Category | Technologies Used |
|-----------|------------------|
| **AI / ML Framework** | TensorFlow, Keras |
| **Programming Language** | Python |
| **Web Frontend** | HTML, CSS, JavaScript |
| **Backend Server** | Flask |
| **Model Type** | Convolutional Neural Network (CNN) |
| **Dataset** | Custom augmented railway track images |


## 🧠 Model Training

- The CNN model is trained on a **balanced dataset of cracked and non-cracked track images**.  
- The dataset is augmented to improve generalization under varying light and environmental conditions.  
- The model achieves high accuracy using layers of convolution, pooling, dropout, and ReLU activations.

**File:** `train_model.py`  
```python

Conv2D → MaxPooling2D → Flatten → Dense(128, relu) → Dropout(0.5) → Dense(Softmax)

**Model Features:**
Early stopping to prevent overfitting
Data augmentation with random brightness, zoom, and rotation
Trained for 30 epochs with validation split = 0.2
Model and class indices stored for deployment
Model output:
* Safe Track 
* Crack Detected (Danger) 

🌐 Web Interface Integration

A simple and responsive web interface is included in the web/ folder to visualize live monitoring results.

Features:
Displays ESP32-CAM live video feed
Automatically sends frames to the Flask AI backend
Shows real-time crack detection results with confidence
Highlights status in glowing color (Green = Safe, Red = Danger)
Web File: web/index.html

Workflow:
Frame captured by camera module
Frame sent to Flask server
AI model predicts safety status
Frontend updates result dynamically

🚀 How to Run Locally
1️⃣ Train the Model (optional)
python train_model.py

2️⃣ Start the Flask Server
python test_model.py

3️⃣ Open the Web Dashboard

Go to web/index.html in your browser
The frontend connects to http://<your-local-IP>:5000/predict
You’ll see live AI detection results updating every 5 seconds
