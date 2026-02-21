import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Fallback for TFLite interpreter on environments without tflite_runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = tf.lite

# Import your existing modules (ensure these are in the same directory)
from beard_stream import BeardDetector
from live_vit_camera import HairTypeClassifier
from your_script_name import FashionAnalyzer

# Function to load gender classification model
def load_gender_model(config_path, weights_path):
    with open(config_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

# Main inference logic
def main():
    # 1) Load Gender Classification Model
    gender_model = load_gender_model('config.json', 'model.weights.h5')

    # 2) Load Beard Detector (TFLite)
    beard_detector = BeardDetector('beard_model.tflite')

    # 3) Load Hair-Type Classifier (Vision Transformer)
    hair_classifier = HairTypeClassifier('preprocessor_config.json', 'model.safetensors')

    # 4) Load Fashion Analyzer
    fashion_analyzer = FashionAnalyzer('metadata.json')

    # 5) Initialize local webcam (ignore any IP stream)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam. Exiting.")
        return

    print("Starting unified inference on local webcam. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Empty frame. Skipping...")
            continue

        # Gender prediction (resize to 128×128)
        gimg = cv2.resize(frame, (128,128)).astype('float32') / 255.0
        gpred = gender_model.predict(np.expand_dims(gimg,0), verbose=0)[0][0]
        g_label = 'Male' if gpred > 0.5 else 'Female'

        # Beard detection
        b_label, b_conf = beard_detector.predict(frame)

        # Hair type classification
        h_label, h_conf = hair_classifier.predict(frame)

        # Fashion analysis
        f_tags = fashion_analyzer.analyze(frame)

        # Annotate frame
        cv2.putText(frame, f'Gender: {g_label} ({gpred:.2f})', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f'Beard: {b_label} ({b_conf:.2f})', (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f'Hair: {h_label} ({h_conf:.2f})', (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, 'Fashion: ' + ', '.join(f_tags), (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Display
        cv2.imshow('Unified Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
