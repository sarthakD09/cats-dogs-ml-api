# models/yolo_model.py

from ultralytics import YOLO
import os
import torch
torch.set_grad_enabled(False)
# Warmup inference (runs once at startup)
import numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)



# Load once globally
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

yolo_model = YOLO(YOLO_PATH)
yolo_model(dummy)

def run_yolo(image):
    # """
    # Run YOLO once and return raw results.
    # """
    results = yolo_model(
        image,
        imgsz=640,       # controlled resolution
        conf=0.25,       # ignore weak detections
        verbose=False    # no console spam
    )
    return results

def detect_objects(results):
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            conf = float(box.conf[0])

            detections.append({
                "label": label,
                "confidence": round(conf, 3)
            })

    return detections


def find_animal_box(results):
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            if label in ["dog", "cat"]:
                return box.xyxy[0].cpu().numpy().astype(int)

    return None
