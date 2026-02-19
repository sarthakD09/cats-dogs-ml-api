# models/classifier.py

import numpy as np
import cv2
import tensorflow as tf
import os

IMG_SIZE = (160, 160)

# Load TFLite once

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "cats_dogs_model.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Keep your working logic unchanged
    image = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])

    return image


def classify_animal(image):
    processed = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return label, confidence
