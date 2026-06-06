import os
import cv2
import torch
import time
from datetime import datetime

from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

torch.backends.cudnn.benchmark = True

# -------------------------------
# Logger
# -------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# -------------------------------
# TEXT WRAP (PIXEL BASED)
# -------------------------------
def wrap_text_pixel(text, font, font_scale, thickness, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "

    if current_line:
        lines.append(current_line)

    return lines

# -------------------------------
# INPUT
# -------------------------------
video_path = input("📂 Enter video path: ").strip()

while not os.path.exists(video_path):
    print("❌ Invalid path. Try again.")
    video_path = input("📂 Enter video path: ").strip()

output_path = "outputs/output.avi"

# -------------------------------
# LOAD YOLO
# -------------------------------
log("Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

# -------------------------------
# DEVICE SETUP
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Using device: {device}")

# -------------------------------
# LOAD BLIP-2 (GPU OPTIMIZED)
# -------------------------------
log("Loading BLIP-2...")

processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    use_fast=False,
    cache_dir="./hf_models"
)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    cache_dir="./hf_models"
)

if device == "cpu":
    model = model.to(device)

model.eval()

# -------------------------------
# CAPTION FUNCTION (FAST)
# -------------------------------
def generate_caption(crop):
    try:
        crop = cv2.resize(crop, (224, 224))
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model.generate(
                    **inputs,
                    max_new_tokens=25
                )

        return processor.decode(out[0], skip_special_tokens=True)

    except:
        return "scene description"

# -------------------------------
# BOX CHANGE DETECTION
# -------------------------------
def box_changed(prev, curr, threshold=50):
    if prev is None:
        return True

    diff = sum(abs(a - b) for a, b in zip(prev, curr))
    return diff > threshold

# -------------------------------
# VIDEO SETUP
# -------------------------------
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 25

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = 1.5
new_w, new_h = int(w * scale), int(h * scale)

os.makedirs("outputs", exist_ok=True)

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (new_w, new_h)
)

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

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cls_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])

        label = f"{yolo_model.names[cls_id]} {conf*100:.1f}%"
        current_box = (x1, y1, x2, y2)

        # 🔥 only caption every 30 frames OR when object changes
        if (frame_count % 30 == 0) or (cls_id != prev_cls or box_changed(prev_box, current_box)):
            pad = 20
            crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

            if crop.size != 0:
                print("\n🧠 Updating description...")
                last_caption = generate_caption(crop)

            prev_cls = cls_id
            prev_box = current_box

        # Draw box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(display, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # -------------------------------
    # SUBTITLE (WRAPPED)
    # -------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    max_width = int(w * 0.8)

    lines = wrap_text_pixel(last_caption, font, font_scale, thickness, max_width)

    y0 = h - 20

    for i, line in enumerate(reversed(lines)):
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x_text = (w - text_w) // 2
        y_text = y0 - i * (text_h + 8)

        cv2.putText(display, line, (x_text, y_text), font, font_scale, (0,0,0), 4)
        cv2.putText(display, line, (x_text, y_text), font, font_scale, (0,255,0), thickness)

    display = cv2.resize(display, (new_w, new_h))

    cv2.imshow("AI Vision System", display)
    out.write(display)

    # -------------------------------
    # TERMINAL PROGRESS
    # -------------------------------
    frame_count += 1
    elapsed = time.time() - start_time
    fps_live = frame_count / elapsed if elapsed > 0 else 0
    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
    eta = (total_frames - frame_count) / fps_live if fps_live > 0 else 0

    mins = int(eta // 60)
    secs = int(eta % 60)

    if frame_count % 5 == 0:
        print(
            f"\r🎬 {progress:.1f}% | Frame {frame_count}/{total_frames} | FPS {fps_live:.2f} | ETA {mins}m {secs}s",
            end=""
        )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print()
log(f"⏱ Done in {int(total_time//60)}m {int(total_time%60)}s")
log(f"✅ Output saved at {output_path}")