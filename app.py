from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
app = Flask(__name__)
CORS(app)

IMG_SIZE = (160, 160)

# -------- Load TFLite model (very light) --------
interpreter = tf.lite.Interpreter(model_path="cats_dogs_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0   # 🔥 correct for MobileNetV2

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

    # -------- TFLite inference --------
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    print(pred)

    return jsonify({"prediction": label})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
