# services/smart_pipeline.py

from models.yolo_model import find_animal_box, detect_objects, run_yolo
from models.classifier import classify_animal
from services.explainer import generate_explanation


def run_smart_pipeline(image):
    
    results = run_yolo(image)
    print("DETECTIONS:", results)
    
    animal_box = find_animal_box(results)

    if animal_box is not None:
        x1, y1, x2, y2 = map(int, animal_box)
        h, w, _ = image.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        cropped = image[y1:y2, x1:x2]

        label, confidence = classify_animal(cropped)
        explanation = generate_explanation(label, confidence)

        return {
            "mode": "classification",
            "prediction": label,
            "confidence": confidence,
            "explanation": explanation
        }

    detections = detect_objects(results)

    return {
        "mode": "detection",
        "detections": detections
    }
