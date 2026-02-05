import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("cats_dogs_model_fine_tuned.keras")

# Convert to TFLite (no quantization = same accuracy)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open("cats_dogs_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")
