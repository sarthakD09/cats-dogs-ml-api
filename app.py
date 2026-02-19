from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

from services.smart_pipeline import run_smart_pipeline
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000"]}},
    supports_credentials=True
)



app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


@app.route("/smart", methods=["POST" ,"OPTIONS"])
def smart():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    result = run_smart_pipeline(img)
    return jsonify(result)


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    from models.yolo_model import detect_objects
    detections = detect_objects(img)

    return jsonify({"detections": detections})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    from models.classifier import classify_animal
    from services.explainer import generate_explanation

    label, confidence = classify_animal(img)
    explanation = generate_explanation(label, confidence)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4),
        "explanation": explanation
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
