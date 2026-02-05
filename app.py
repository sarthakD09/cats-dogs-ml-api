from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)  # allow Next.js to call this API

model = keras.models.load_model("cats_dogs_model_fine_tuned.keras")


IMG_SIZE = (160, 160)

def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)   # 🔥 THIS LINE WAS MISSING
    image = np.expand_dims(image, axis=0)
    return image



@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    processed = preprocess_image(img)
    pred = model.predict(processed)[0][0]

    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    print(pred)

    return jsonify({"prediction": label})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
