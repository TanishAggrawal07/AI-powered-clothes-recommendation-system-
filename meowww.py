#!/usr/bin/env python3
import sys
print("Executable:", sys.executable)
print("Paths:", sys.path)
# sys.exit(0)

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

# Import custom modules
from beard_stream import BeardDetector
from live_vit_camera import HairTypeClassifier
from your_script_name import FashionAnalyzer

# Gemini import for clothing suggestions
import google.generativeai as gemini

def generate_clothing_suggestions(tags):
    """
    Use Google Gemini to suggest outfit combinations or style tips based on fashion tags.
    """
    # Hardcode your Gemini API key here
    gemini.configure(api_key="AIzaSyB6Rl2rSpetIKDmkUsOTO-2Vm6qCaDCb64")
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

# Load models
def load_gender_model(config_path, weights_path):
    with open(config_path, 'r') as f:
        m = model_from_json(f.read())
    m.load_weights(weights_path)
    return m

# Configuration for Pi camera stream
def main():
    PI_IP_ADDRESS = "192.168.119.136"  # Your Pi's IP
    STREAM_PORT   = 8080
    STREAM_URL    = f"http://{PI_IP_ADDRESS}:{STREAM_PORT}"

    # 1) Instantiate models
    gender_model    = load_gender_model('config.json','model.weights.h5')
    beard_detector  = BeardDetector('beard_model.tflite')
    hair_classifier = HairTypeClassifier('preprocessor_config.json','model.safetensors')
    fashion_analyzer= FashionAnalyzer('metadata.json')

    # 2) Connect to Pi stream
    print(f"Connecting to Pi camera at {STREAM_URL}...")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("Error: Cannot open stream. Exiting.")
        return

    print("Press 'q' to capture frame and get clothing suggestions.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Single snapshot
            # 3) Gender
            img_g = cv2.resize(frame,(128,128)).astype('float32')/255.0
            pred_g = gender_model.predict(np.expand_dims(img_g,0),verbose=0)[0][0]
            gender = 'Male' if pred_g>0.5 else 'Female'

            # 4) Beard
            beard_label, beard_conf = beard_detector.predict(frame)

            # 5) Hair
            hair_label, hair_conf = hair_classifier.predict(frame)

            # 6) Fashion tags
            tags = fashion_analyzer.analyze(frame)
            tags_text = ", ".join(tags)

            # 7) Generate suggestions via Gemini
            suggestions = generate_clothing_suggestions(tags_text)

            # 8) Display results
            print("--- Analysis ---")
            print(f"Gender: {gender} ({pred_g:.2f})")
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
