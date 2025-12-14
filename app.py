from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("kaggle_az_model.h5")


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
