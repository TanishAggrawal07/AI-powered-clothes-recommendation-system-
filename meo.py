# meo.py
import os, time, requests
import cv2, numpy as np, torch, tensorflow as tf
from tensorflow import keras
from safetensors.torch import load_file
from transformers import ViTForImageClassification, ViTImageProcessor

# ─── 0) Verify the required files exist ────────────────────────────────────────
for fn in ("gender_model.keras/gender_savedmodel.h5",
           "beard_model.tflite",
           "model.safetensors"):
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"Missing required file: {fn}")

# ─── 1) Gender Model (Keras .h5) ───────────────────────────────────────────────
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

# ─── 2) Beard Model (TFLite) ──────────────────────────────────────────────────
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

# ─── 3) Hair Model (ViT + safetensors) ────────────────────────────────────────
# Replace these with your actual hair‐type labels:
HAIR_LABELS = ["Straight","Wavy","Curly","Coily"]

# Load ViT feature extractor & model shell
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
hair_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(HAIR_LABELS),
    id2label={i:lbl for i,lbl in enumerate(HAIR_LABELS)},
    label2id={lbl:i for i,lbl in enumerate(HAIR_LABELS)}
)

# Overwrite its weights with yours (strict=False to allow extra/missing keys)
state_dict = load_file("model.safetensors")
hair_model.vit.load_state_dict(state_dict, strict=False)
hair_model.eval()

def predict_hair(frame):
    # ViT expects PIL‐style: resize to 224×224, normalize
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        out = hair_model(**inputs)
        probs = out.logits.softmax(-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return HAIR_LABELS[idx], float(probs[idx])

# ─── 4) Fashion Tags (MobileNetV2) ─────────────────────────────────────────────
fashion_model = keras.applications.MobileNetV2(weights="imagenet", include_top=True)
decode_imnet = keras.applications.imagenet_utils.decode_predictions

def predict_fashion(frame, top_k=3):
    img = cv2.resize(frame,(224,224)).astype("float32")
    x = keras.applications.mobilenet_v2.preprocess_input(img[None])
    preds = fashion_model.predict(x, verbose=0)
    decoded = decode_imnet(preds, top=top_k)[0]
    return [(label,float(conf)) for (_,label,conf) in decoded]

# ─── 5) Clothing Tips via Gemini ───────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyB6Rl2rSpetIKDmkUsOTO-2Vm6qCaDCb64"

def get_clothing_tips(tags):
    prompt = f"I see you’re wearing: {', '.join(t for t,_ in tags)}. Suggest styling tips."
    url = "https://gemini.googleapis.com/v1/chat:complete"
    headers = {"Authorization":f"Bearer {GEMINI_API_KEY}"}
    body = {"model":"gemini-proto","prompt":prompt,"maxOutputTokens":30}
    r = requests.post(url,json=body,headers=headers)
    if r.status_code==200:
        return r.json()["candidates"][0]["content"].strip()
    return "Could not fetch tips."

# ─── 6) Main Loop ──────────────────────────────────────────────────────────────
PI_IP = "192.168.119.136"
cap = cv2.VideoCapture(f"http://{PI_IP}:8080")
if not cap.isOpened():
    raise RuntimeError("Cannot open Pi camera stream")
print("Press 'q' to quit")

while True:
    ok, frame = cap.read()
    if not ok: break

    t0 = time.time()
    g_lbl,g_conf = predict_gender(frame)
    b_lbl,b_conf = predict_beard(frame)
    h_lbl,h_conf = predict_hair(frame)
    tags = predict_fashion(frame)
    tips = get_clothing_tips(tags)
    latency = int((time.time()-t0)*1000)

    y = 30
    for line in [
        f"Gender: {g_lbl} ({g_conf:.2f})",
        f"Beard: {b_lbl} ({b_conf:.2f})",
        f"Hair: {h_lbl} ({h_conf:.2f})",
        "Fashion: " + ", ".join(f"{t[0]}({t[1]:.2f})" for t in tags),
        f"Tips: {tips}",
        f"Latency: {latency}ms"
    ]:
        cv2.putText(frame, line, (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 25

    cv2.imshow("Meow Detector", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
