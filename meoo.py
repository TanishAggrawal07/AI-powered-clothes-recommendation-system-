import os, time, requests
import cv2, numpy as np, torch, tensorflow as tf
from tensorflow import keras
from safetensors.torch import load_file
from transformers import ViTForImageClassification, ViTImageProcessor
import threading # Added for threading

# --- 0) Verify the required files exist ---
# (Your existing code for file verification - good!)
for fn in ("gender_model.keras/gender_savedmodel.h5",
           "beard_model.tflite",
           "model.safetensors"):
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Missing required file: {fn}")

# --- 1) Gender Model (Keras .h5) ---
def build_gender_model():
    m = keras.Sequential([
        keras.layers.Input(shape=(128,128,3)),
        keras.layers.Conv2D(16,3,activation="relu",padding="same"),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(32,3,activation="relu",padding="same"),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(2, activation="softmax", name="gender_out")
    ])
    return m

gender_model = build_gender_model()
gender_model.load_weights("gender_model.keras/gender_savedmodel.h5",
                          by_name=True, skip_mismatch=True)
GENDER_LABELS = ["Male","Female"]

def predict_gender(frame):
    x = cv2.resize(frame,(128,128)).astype("float32")/255.0
    probs = gender_model.predict(x[None], verbose=0)[0]
    idx = int(np.argmax(probs))
    return GENDER_LABELS[idx], float(probs[idx])

# --- 2) Beard Model (TFLite) ---
beard_interp = tf.lite.Interpreter("beard_model.tflite")
beard_interp.allocate_tensors()
_in = beard_interp.get_input_details()[0]
_out = beard_interp.get_output_details()[0]

def predict_beard(frame):
    x = cv2.resize(frame, tuple(_in["shape"][1:3])).astype("float32")/255.0
    beard_interp.set_tensor(_in["index"], x[None])
    beard_interp.invoke()
    p = float(beard_interp.get_tensor(_out["index"])[0][0])
    return ("Beard",p) if p>0.5 else ("No Beard",1-p)

# --- 3) Hair Model (ViT + safetensors) ---
HAIR_LABELS = ["Straight","Wavy","Curly","Coily"] # Replace with your actual labels

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
hair_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(HAIR_LABELS),
    id2label={i:lbl for i,lbl in enumerate(HAIR_LABELS)},
    label2id={lbl:i for i,lbl in enumerate(HAIR_LABELS)}
)
state_dict = load_file("model.safetensors")
# It seems you might be loading weights only into vit.encoder.
# If your safetensors file contains the full model (vit + classifier),
# use hair_model.load_state_dict(state_dict, strict=False)
# If it only contains the ViT backbone weights:
hair_model.vit.load_state_dict(state_dict, strict=False)
# If it's just the classifier head weights (unlikely for 'model.safetensors'):
# hair_model.classifier.load_state_dict(state_dict, strict=False)
# Assuming it's the backbone or full model:
hair_model.eval()

def predict_hair(frame):
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        out = hair_model(**inputs)
        probs = out.logits.softmax(-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return HAIR_LABELS[idx], float(probs[idx])

# --- 4) Fashion Tags (MobileNetV2) ---
fashion_model = keras.applications.MobileNetV2(weights="imagenet", include_top=True)
decode_imnet = keras.applications.imagenet_utils.decode_predictions

def predict_fashion(frame, top_k=3):
    img = cv2.resize(frame,(224,224)).astype("float32")
    x = keras.applications.mobilenet_v2.preprocess_input(img[None])
    preds = fashion_model.predict(x, verbose=0)
    decoded = decode_imnet(preds, top=top_k)[0]
    return [(label,float(conf)) for (_,label,conf) in decoded]

# --- 5) Clothing Tips via Gemini ---
GEMINI_API_KEY = "AIzaSyB6Rl2rSpetIKDmkUsOTO-2Vm6qCaDCb64" # Consider moving to env variable
# Global variable to store the latest tips
latest_tips = "Fetching tips..."
tips_thread = None
last_tags_for_tips = [] # To avoid re-fetching for identical tags too quickly
API_CALL_TIMEOUT = 10 # seconds for the request
TIPS_UPDATE_INTERVAL = 15 # seconds between API calls for new tips
last_api_call_time = 0

def fetch_clothing_tips_worker(tags_for_api):
    global latest_tips, last_api_call_time
    if not tags_for_api:
        latest_tips = "No fashion tags detected to get tips."
        return

    prompt = f"I see you’re wearing: {', '.join(t for t,_ in tags_for_api)}. Suggest 1-2 short styling tips."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent" # Updated Gemini API endpoint
    # The old endpoint /v1/chat:complete might be deprecated or different.
    # Check the current Gemini API documentation for the correct endpoint and request format.
    # Assuming a newer typical format for gemini-pro:
    headers = {"Content-Type": "application/json"}
    # The API key is usually passed as a query parameter `key` for this endpoint
    query_params = {"key": GEMINI_API_KEY}

    # Body structure for gemini-pro:generateContent
    # This might vary based on the exact Gemini model and version.
    # This is a common structure for text generation.
    body = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 50, # Keep it short
            "temperature": 0.7 # Adjust for creativity
        }
    }
    try:
        print(f"Requesting tips for: {tags_for_api}")
        r = requests.post(url, params=query_params, json=body, headers=headers, timeout=API_CALL_TIMEOUT)
        r.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
        
        response_data = r.json()
        # Navigating the typical Gemini API response structure:
        if "candidates" in response_data and response_data["candidates"]:
            content_parts = response_data["candidates"][0].get("content", {}).get("parts", [])
            if content_parts:
                latest_tips = content_parts[0].get("text", "Could not parse tips.").strip()
            else:
                latest_tips = "No content in tips response."
        else:
            latest_tips = "No candidates in tips response."
        print(f"Tips received: {latest_tips[:50]}...")

    except requests.exceptions.Timeout:
        latest_tips = "Could not fetch tips (timeout)."
        print("Error: Gemini API request timed out.")
    except requests.exceptions.RequestException as e:
        latest_tips = f"Could not fetch tips (error: {type(e).__name__})."
        print(f"Error: Gemini API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
    finally:
        last_api_call_time = time.time()


# --- 6) Main Loop ---
PI_IP = "192.168.119.136" # Consider making this configurable
# Try different backends if one is problematic
# cap = cv2.VideoCapture(f"http://{PI_IP}:8080", cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(f"http://{PI_IP}:8080")

if not cap.isOpened():
    # Try a local webcam as a fallback for testing
    print(f"Cannot open Pi camera stream (http://{PI_IP}:8080). Trying local webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open Pi camera stream or local webcam.")
    print("Using local webcam.")

print("Press 'q' to quit")

# For processing every N frames (optional)
frame_counter = 0
PROCESS_EVERY_N_FRAMES = 1 # Process every frame by default. Increase to improve FPS.

# To store last valid predictions to display on non-processed frames
last_g_lbl, last_g_conf = "N/A", 0.0
last_b_lbl, last_b_conf = "N/A", 0.0
last_h_lbl, last_h_conf = "N/A", 0.0
last_tags = []


while True:
    ok, frame = cap.read()
    if not ok:
        print("End of stream or cannot read frame.")
        time.sleep(0.5) # Wait a bit before retrying or exiting
        # You might want to attempt to re-open the stream here if it's an IP camera
        # For now, we'll just break
        break

    frame_counter += 1
    current_time_main_loop = time.time()
    
    # Create a copy for drawing, so original frame can be used for processing if needed
    display_frame = frame.copy()
    
    # --- Perform predictions ---
    # Optionally, process only every Nth frame to save resources
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        t0 = time.time() # Start timing for this processing block
        
        last_g_lbl,last_g_conf = predict_gender(frame)
        last_b_lbl,last_b_conf = predict_beard(frame)
        last_h_lbl,last_h_conf = predict_hair(frame)
        new_fashion_tags = predict_fashion(frame) # Renamed to avoid conflict
        
        # Only update last_tags if new_fashion_tags is not empty (prediction was successful)
        if new_fashion_tags:
            last_tags = new_fashion_tags

        # Check if it's time to fetch new tips and if the thread is not already running
        if (tips_thread is None or not tips_thread.is_alive()) and \
           (current_time_main_loop - last_api_call_time > TIPS_UPDATE_INTERVAL):
            # Make a copy of tags to pass to the thread, in case last_tags changes
            tags_for_api_call = list(last_tags) 
            if tags_for_api_call: # Only start thread if there are tags
                print("Main: Starting tips fetching thread...")
                tips_thread = threading.Thread(target=fetch_clothing_tips_worker, args=(tags_for_api_call,), daemon=True)
                tips_thread.start()
            else:
                # If no tags, we can set tips directly or wait
                latest_tips = "No fashion tags to get tips."
                last_api_call_time = current_time_main_loop # Still update time to avoid spamming this check

        latency = int((time.time()-t0)*1000)
    else:
        # For frames not processed, latency is effectively 0 for the models
        latency = 0 

    # --- Display information ---
    y = 30
    cv2.putText(display_frame, f"Gender: {last_g_lbl} ({last_g_conf:.2f})", (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y += 25
    cv2.putText(display_frame, f"Beard: {last_b_lbl} ({last_b_conf:.2f})", (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y += 25
    cv2.putText(display_frame, f"Hair: {last_h_lbl} ({last_h_conf:.2f})", (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y += 25
    
    if last_tags:
        fashion_str = "Fashion: " + ", ".join(f"{t[0]}({t[1]:.2f})" for t in last_tags)
    else:
        fashion_str = "Fashion: N/A"
    cv2.putText(display_frame, fashion_str, (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y += 25
    
    # Wrap tips text if it's too long
    max_text_width = display_frame.shape[1] - 20 # Max width for text
    font_scale = 0.6
    font_thickness = 2
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    tips_lines = []
    if latest_tips:
        words = latest_tips.split(' ')
        current_line = "Tips: "
        for word in words:
            test_line = current_line + word + " "
            (text_width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, font_thickness)
            if text_width > max_text_width and current_line != "Tips: ":
                tips_lines.append(current_line.strip())
                current_line = word + " "
            else:
                current_line = test_line
        tips_lines.append(current_line.strip())
    else:
        tips_lines = ["Tips: Not available"]

    for line in tips_lines:
        cv2.putText(display_frame, line, (10,y),
                    font_face, font_scale, (0,255,0), font_thickness)
        y += 25


    cv2.putText(display_frame, f"Process Latency: {latency}ms (Frame {frame_counter})", (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y += 25

    cv2.imshow("Meow Detector", display_frame)
    
    key = cv2.waitKey(1) & 0xFF # crucial: waitKey also processes GUI events
    if key == ord('q'):
        break

if tips_thread and tips_thread.is_alive():
    print("Waiting for tips thread to finish...")
    tips_thread.join(timeout=2) # Wait for thread to finish with a timeout

cap.release()
cv2.destroyAllWindows()