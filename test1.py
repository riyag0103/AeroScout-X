import os
import cv2
import torch
import time
from datetime import datetime

from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# -------------------------------
# Logger
# -------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# -------------------------------
# INPUT
# -------------------------------
video_path = input("📂 Enter video path: ").strip()

while not os.path.exists(video_path):
    print("❌ Invalid path. Try again.")
    video_path = input("📂 Enter video path: ").strip()

output_path = "outputs/output.avi"

# -------------------------------
# LOAD MODELS
# -------------------------------
log("Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

log("Loading BLIP-2...")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

# -------------------------------
# CAPTION FUNCTION
# -------------------------------
def generate_caption(crop):
    try:
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        prompt = "Describe what is happening in detail."

        inputs = processor(images=image, text=prompt, return_tensors="pt")

        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7
            )

        return processor.decode(out[0], skip_special_tokens=True)

    except Exception as e:
        print("⚠️ Caption error:", e)
        return "a scene"

# -------------------------------
# BOX CHANGE DETECTION
# -------------------------------
def box_changed(prev, curr, threshold=50):
    if prev is None:
        return True

    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = curr

    diff = abs(px1-cx1) + abs(py1-cy1) + abs(px2-cx2) + abs(py2-cy2)

    return diff > threshold

# -------------------------------
# VIDEO SETUP
# -------------------------------
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = 1.5
new_w, new_h = int(w * scale), int(h * scale)

os.makedirs("outputs", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

log(f"Processing started... Total Frames: {total_frames}")

# -------------------------------
# STATE
# -------------------------------
prev_cls = None
prev_box = None
last_caption = "Analyzing..."

start_time = time.time()
frame_count = 0

# -------------------------------
# LOOP
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    results = yolo_model(frame)[0]

    best_box = None
    best_conf = 0

    for b in results.boxes:
        conf = float(b.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = b

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cls_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])

        # Clamp values (IMPORTANT FIX)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        label = f"{yolo_model.names[cls_id]} {conf*100:.1f}%"
        current_box = (x1, y1, x2, y2)

        if cls_id != prev_cls or box_changed(prev_box, current_box):
            crop = frame[y1:y2, x1:x2]

            if crop.size != 0:
                print("\n🧠 Updating description...")
                last_caption = generate_caption(crop)

            prev_cls = cls_id
            prev_box = current_box

        # Draw box
        color = (0, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        cv2.putText(display, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -------------------------------
    # CAPTION TEXT
    # -------------------------------
    caption = last_caption[:150]

    cv2.putText(display, caption, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)

    cv2.putText(display, caption, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    display = cv2.resize(display, (new_w, new_h))

    cv2.imshow("AI Vision System", display)
    out.write(display)

    # -------------------------------
    # PROGRESS
    # -------------------------------
    frame_count += 1
    elapsed = time.time() - start_time
    fps_live = frame_count / elapsed if elapsed > 0 else 0

    if frame_count % 10 == 0:
        print(f"\r🚀 {frame_count}/{total_frames} | FPS {fps_live:.2f}", end="")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print()
log(f"✅ Done. Output saved at {output_path}")