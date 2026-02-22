# models/yolo_model.py
from gradio_client import Client, handle_file
import requests
import tempfile
import cv2


client = Client("Novion10/yolo_api")
def run_yolo(image):

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, image)

        with open(tmp.name, "rb") as f:
            result = client.predict(
            image=handle_file(tmp.name),   # MUST match input name
            api_name="/detect"             # MUST match function
        )

    print("HF RESULT:", result)

    return result.get("detections", [])

def detect_objects(detections):
    return detections

def find_animal_box(detections):
    best = None
    max_conf = 0

    for obj in detections:
        if obj["label"] in ["dog", "cat"] and obj["confidence"] > max_conf:
            best = obj
            max_conf = obj["confidence"]

    return best["box"] if best else None