import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="beard_model.tflite")
interpreter.allocate_tensors()

# Get input and output detailspipz
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape (e.g., [1, 128, 128, 3])
input_shape = input_details[0]['shape'][1:3]

# Start webcam

print("Initializing video stream...")
pi_ip_address = "192.168.119.136"
stream_url = f"http://{pi_ip_address}:8080"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

print("📸 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Preprocess frame
    img = cv2.resize(frame, tuple(input_shape))
    img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction
    label = "Beard" if output[0][0] > 0.5 else "No Beard"

    # Display result
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Beard Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
