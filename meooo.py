import os, time, requests, base64
from dotenv import load_dotenv
import cv2, numpy as np, torch, tensorflow as tf
from tensorflow import keras
from safetensors.torch import load_file
from transformers import ViTForImageClassification, ViTImageProcessor
import threading


for fn in ("gender_model.keras/gender_savedmodel.h5",
           "beard_model.tflite",
           "model.safetensors"):
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Missing required file: {fn}")


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

beard_interp = tf.lite.Interpreter("beard_model.tflite")
beard_interp.allocate_tensors()
_in_beard = beard_interp.get_input_details()[0] 
_out_beard = beard_interp.get_output_details()[0] 

def predict_beard(frame):
    x = cv2.resize(frame, tuple(_in_beard["shape"][1:3])).astype("float32")/255.0
    beard_interp.set_tensor(_in_beard["index"], x[None])
    beard_interp.invoke()
    p = float(beard_interp.get_tensor(_out_beard["index"])[0][0])
    return ("Beard",p) if p>0.5 else ("No Beard",1-p)

HAIR_LABELS = ["Straight","Wavy","Curly","Coily"] 

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
hair_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(HAIR_LABELS),
    id2label={i:lbl for i,lbl in enumerate(HAIR_LABELS)},
    label2id={lbl:i for i,lbl in enumerate(HAIR_LABELS)}
)
state_dict = load_file("model.safetensors")

hair_model.vit.load_state_dict(state_dict, strict=False)
hair_model.eval()

def predict_hair(frame):
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        out = hair_model(**inputs)
        probs = out.logits.softmax(-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return HAIR_LABELS[idx], float(probs[idx])

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    print("WARNING: Replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.")


CURRENT_GEMINI_MODEL = "gemini-3-flash-preview" 

latest_tips = "Fetching tips..."
tips_thread = None
API_CALL_TIMEOUT = 25 
TIPS_UPDATE_INTERVAL = 30 
last_api_call_time = 0

def fetch_image_analysis_and_tips_worker(image_frame):
    global latest_tips, last_api_call_time

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY:
        latest_tips = "Gemini API Key not set."
        print("Error: Gemini API Key not configured.")
        last_api_call_time = time.time() 
        return

    h, w = image_frame.shape[:2]
    target_w, target_h = 640, 480 
    if w > target_w or h > target_h:
        scale = min(target_w/w, target_h/h)
        resized_frame = cv2.resize(image_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        resized_frame = image_frame
    
    is_success, buffer = cv2.imencode(".jpg", resized_frame)
    if not is_success:
        latest_tips = "Could not encode image."
        print("Error: Could not encode image to JPEG.")
        last_api_call_time = time.time()
        return
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    prompt_text = (
        "Analyze the clothing worn by the person in this image. "
        "Describe the main items of clothing you see (e.g., 'blue t-shirt', 'black jeans', 'red scarf'). "
        "Then, provide one or two concise styling tips related to what they are wearing. "
        "If the image is unclear or no person/clothing is visible, state that. "
        "Keep the entire response under 100 words." 
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CURRENT_GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }
                }
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 150, 
            "temperature": 0.4     
        }
    }

    try:
        print(f"Requesting image analysis and tips from Gemini ({CURRENT_GEMINI_MODEL})...")
        r = requests.post(url, json=body, headers=headers, timeout=API_CALL_TIMEOUT)
        r.raise_for_status()
        
        response_data = r.json()
        
        if "candidates" in response_data and response_data["candidates"]:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                latest_tips = candidate["content"]["parts"][0].get("text", "Could not parse tips from Gemini.").strip()
            elif "finishReason" in candidate and candidate["finishReason"] != "STOP":
                latest_tips = f"Gemini generation stopped: {candidate['finishReason']}"
                if "safetyRatings" in candidate:
                    latest_tips += f" - Safety: {candidate['safetyRatings']}"
            else:
                latest_tips = "No content in Gemini's response."
        elif "promptFeedback" in response_data:
            feedback = response_data["promptFeedback"]
            if "blockReason" in feedback:
                latest_tips = f"Blocked by Gemini: {feedback['blockReason']}"
            elif "safetyRatings" in feedback and any(sr.get("blocked") for sr in feedback["safetyRatings"]):
                 latest_tips = f"Content blocked due to safety ratings: {feedback['safetyRatings']}"
            else:
                latest_tips = "Prompt feedback received, but no clear block."
                print(f"Gemini Prompt Feedback: {feedback}")
        else:
            latest_tips = "No candidates in Gemini's response or unexpected format."
            print(f"Unexpected Gemini response: {response_data}")

        print(f"Gemini Tips received: {latest_tips[:100]}...")

    except requests.exceptions.Timeout:
        latest_tips = f"Gemini API ({CURRENT_GEMINI_MODEL}) request timed out."
        print(f"Error: {latest_tips}")
    except requests.exceptions.HTTPError as e:
        latest_tips = f"Gemini API ({CURRENT_GEMINI_MODEL}) HTTP error: {e.response.status_code}."
        print(f"Error: {latest_tips}")
        try:
            error_details = e.response.json()
            print(f"Gemini Error Details: {error_details}")
            if "error" in error_details and "message" in error_details["error"]:
                error_message = error_details['error']['message']
                latest_tips += f" Message: {error_message[:150]}" 
                if "status" in error_details["error"]:
                    latest_tips += f" Status: {error_details['error']['status']}"

        except ValueError:
            print(f"Gemini Raw Error Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        latest_tips = f"Gemini API ({CURRENT_GEMINI_MODEL}) request failed: {type(e).__name__}."
        print(f"Error: {latest_tips} Details: {e}")
    finally:
        last_api_call_time = time.time()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open local webcam 0. Trying local webcam 1...")
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened():
        raise RuntimeError("Cannot open any local webcam.")
print("Using local webcam.")

print("Press 'q' to quit")

frame_counter = 0
PROCESS_EVERY_N_FRAMES = 1 


last_g_lbl, last_g_conf = "N/A", 0.0
last_b_lbl, last_b_conf = "N/A", 0.0
last_h_lbl, last_h_conf = "N/A", 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("End of stream or cannot read frame.")
        time.sleep(0.5)
        break 

    frame_counter += 1
    current_time_main_loop = time.time()
    display_frame = frame.copy()
    
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        t0_models = time.time()
        
        last_g_lbl,last_g_conf = predict_gender(frame)
        last_b_lbl,last_b_conf = predict_beard(frame)
        last_h_lbl,last_h_conf = predict_hair(frame)
        
        model_latency = int((time.time()-t0_models)*1000)


        if (tips_thread is None or not tips_thread.is_alive()) and \
           (current_time_main_loop - last_api_call_time > TIPS_UPDATE_INTERVAL):
            

            frame_for_api = frame.copy() 
            print("Main: Starting image analysis & tips thread...")
            tips_thread = threading.Thread(target=fetch_image_analysis_and_tips_worker, args=(frame_for_api,), daemon=True)
            tips_thread.start()
    else:
        model_latency = 0 


    y = 30
    def draw_text(text, y_pos):
        cv2.putText(display_frame, text, (10,y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return y_pos + 25

    y = draw_text(f"Gender: {last_g_lbl} ({last_g_conf:.2f})", y)
    y = draw_text(f"Beard: {last_b_lbl} ({last_b_conf:.2f})", y)
    y = draw_text(f"Hair: {last_h_lbl} ({last_h_conf:.2f})", y)
    
    max_text_width = display_frame.shape[1] - 20
    font_scale = 0.6
    font_thickness = 1 
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    tips_lines = []
    current_display_tips = latest_tips 
    if current_display_tips:
        initial_lines = current_display_tips.split('\n')
        for initial_line in initial_lines:
            words = initial_line.split(' ')
            current_line_segment = "Tips: " if not tips_lines else "" 
            for word in words:
                test_line = current_line_segment + word + " "
                (text_width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, font_thickness)
                if text_width > max_text_width and current_line_segment != ("Tips: " if not tips_lines else ""):
                    tips_lines.append(current_line_segment.strip())
                    current_line_segment = word + " "
                else:
                    current_line_segment = test_line
            tips_lines.append(current_line_segment.strip())
    else:
        tips_lines = ["Tips: Not available"]
    
    if not tips_lines: 
        tips_lines = ["Tips: ..."]
    elif not tips_lines[0].startswith("Tips:"): 
         tips_lines[0] = "Tips: " + tips_lines[0]


    for line in tips_lines:
        y = draw_text(line, y) 

    y = draw_text(f"Model Latency: {model_latency}ms (Frame {frame_counter})", y)

    cv2.imshow("Meow Detector", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

if tips_thread and tips_thread.is_alive():
    print("Waiting for tips thread to finish...")
    tips_thread.join(timeout=5) 

cap.release()
cv2.destroyAllWindows()