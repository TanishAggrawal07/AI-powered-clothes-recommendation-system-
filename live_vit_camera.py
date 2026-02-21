import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import cv2

# 1) Load processor & model
processor = ViTImageProcessor.from_pretrained('./')
model = ViTForImageClassification.from_pretrained(
    './',
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).eval()

# 2) Label mapping
id2label = {
    0: "curly",
    1: "dreadlocks",
    2: "kinky",
    3: "straight",
    4: "wavy"
}

# 3) New, lower threshold
CONF_THRESHOLD = 0.40

# 4) Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 5) Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")

    # 6) Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # 7) Get top class & confidence
    top_idx = int(probs.argmax())
    top_conf = float(probs[top_idx])

    # 8) Debug print
    print(f"Top confidence: {top_conf:.3f} → {id2label[top_idx]}")

    # 9) Decide what to overlay
    if top_conf >= CONF_THRESHOLD:
        text = f"{id2label[top_idx]} ({top_conf*100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = f"Uncertain ({top_conf*100:.1f}%)"
        color = (0, 165, 255)

    # 10) Overlay + show
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.imshow("Hair-Type Live Prediction", frame)

    # exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
