#!/usr/bin/env python3
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Fallback for TFLite interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = tf.lite

# Custom module imports
from beard_stream import BeardDetector
from live_vit_camera import HairTypeClassifier
from your_script_name import FashionAnalyzer

# Gemini import for clothing suggestions (no voice)
import google.generativeai as gemini

# Configuration
def load_gender_model(config_path, weights_path):
    with open(config_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

# Generate outfit suggestions based on fashion tags
def generate_clothing_suggestions(tags):
    # Hardcode your API key here
    gemini.configure(api_key="AIzaSyB6Rl2rSpetIKDmkUsOTO-2Vm6qCaDCb64")  # replace
    prompt = (
        "You are a fashion assistant. Given these detected outfit tags: "
        f"{tags}. "
        "Suggest 2–3 complementary clothing combinations or styling tips."
    )
    response = gemini.generate_text(
        model="models/chat-bison-001",
        prompt=prompt
    )
    return response.text.strip()

# Main inference loop
def main():
    # Load models
    gender_model     = load_gender_model('config.json', 'model.weights.h5')
    beard_detector   = BeardDetector('beard_model.tflite')
    hair_classifier  = HairTypeClassifier('preprocessor_config.json', 'model.safetensors')
    fashion_analyzer = FashionAnalyzer('metadata.json')

    # Pi camera stream config
    PI_IP_ADDRESS = "192.168.119.136"  # update as needed
    STREAM_PORT   = 8080
    STREAM_URL    = f"http://{PI_IP_ADDRESS}:{STREAM_PORT}"

    print(f"Connecting to Pi camera stream at {STREAM_URL}...")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("Error: Cannot open video stream. Exiting.")
        return

    print("Press 'q' to capture a frame and view analysis.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Inference on captured frame
            # Gender
            g_img = cv2.resize(frame, (128,128)).astype('float32') / 255.0
            g_pred = gender_model.predict(np.expand_dims(g_img, 0), verbose=0)[0][0]
            gender_label = 'Male' if g_pred > 0.5 else 'Female'

            # Beard
            beard_label, beard_conf = beard_detector.predict(frame)

            # Hair
            hair_label, hair_conf = hair_classifier.predict(frame)

            # Fashion tags
            tags = fashion_analyzer.analyze(frame)
            tags_text = ", ".join(tags)

            # Suggestions
            suggestions = generate_clothing_suggestions(tags_text)

            # Print results
            print("--- Analysis ---")
            print(f"Gender: {gender_label} ({g_pred:.2f})")
            print(f"Beard: {beard_label} ({beard_conf:.2f})")
            print(f"Hair: {hair_label} ({hair_conf:.2f})")
            print(f"Fashion Tags: {tags_text}")
            print("\n--- Gemini Suggestions ---")
            print(suggestions)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
