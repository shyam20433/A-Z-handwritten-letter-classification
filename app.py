from flask import Flask, render_template, request, jsonify
import numpy as np
import os

# TensorFlow optimization settings - MUST be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage (no GPU)

import tensorflow as tf

# Configure TensorFlow memory growth to prevent OOM
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model with optimizations
print("Loading model...")
model = load_model("kaggle_az_model.h5", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded successfully!")


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]  # 784 values
    image = np.array(data).reshape(1, 28, 28, 1)

    # Debug (optional)
    # print(image.reshape(28, 28))

    pred = model.predict(image)[0]
    idx = np.argmax(pred)
    confidence = float(pred[idx]) * 100

    return jsonify({
        "letter": LETTERS[idx],
        "confidence": confidence,
        "probabilities": pred.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
