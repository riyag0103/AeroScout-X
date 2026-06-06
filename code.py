import os
import cv2
import torch
import time
from datetime import datetime
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

torch.backends.cudnn.benchmark = True

# ================= CONFIG =================
class Config:
    PROJECT_NAME = "vision_pipeline_v1"

    DETECTOR_WEIGHTS = "yolov8n.pt"
    CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

    CONF_THRESHOLD = 0.3
    CAPTION_INTERVAL = 60
    INPUT_SIZE = 224

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ================= TEXT WRAP =================
def wrap_text_pixel(text, font, font_scale, thickness, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + word + " "
        (w, _), _ = cv2.getTextSize(test, font, font_scale, thickness)

        if w <= max_width:
            current = test
        else:
            lines.append(current)
            current = word + " "

    if current:
        lines.append(current)

    return lines

# ================= INPUT =================
video_path = input("📂 Enter video path: ").strip()

while not os.path.exists(video_path):
    print("❌ Invalid path. Try again.")
    video_path = input("📂 Enter video path: ").strip()

output_path = "outputs/output.avi"

# ================= INIT =================
log(f"Project: {cfg.PROJECT_NAME}")
log(f"Using device: {cfg.DEVICE}")

detector = YOLO(cfg.DETECTOR_WEIGHTS)

log("Loading BLIP...")
processor = BlipProcessor.from_pretrained(cfg.CAPTION_MODEL)
model = BlipForConditionalGeneration.from_pretrained(cfg.CAPTION_MODEL).to(cfg.DEVICE)

model.eval()

# ================= PIPELINE =================
class VisionPipeline:
    def __init__(self, detector):
        self.detector = detector

    def detect(self, frame):
        return self.detector(frame)[0]

    def describe(self, crop):
        return generate_caption(crop)

pipeline = VisionPipeline(detector)

# ================= CAPTION =================
def generate_caption(crop):
    try:
        crop = cv2.resize(crop, (cfg.INPUT_SIZE, cfg.INPUT_SIZE))
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt").to(cfg.DEVICE)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=25)

        return processor.decode(out[0], skip_special_tokens=True)

    except:
        return "scene description"

# ================= HELPER =================
def box_changed(prev, curr, threshold=50):
    if prev is None:
        return True
    return sum(abs(a - b) for a, b in zip(prev, curr)) > threshold

# ================= VIDEO =================
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
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

prev_cls = None
prev_box = None
last_caption = "Analyzing..."

frame_count = 0

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    results = pipeline.detect(frame)

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

        label = f"{detector.names[cls_id]} {conf*100:.1f}%"
        current_box = (x1, y1, x2, y2)

        if (frame_count % cfg.CAPTION_INTERVAL == 0) or (
            cls_id != prev_cls or box_changed(prev_box, current_box)
        ):
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            if crop.size != 0:
                print("\n🧠 Updating description...")
                last_caption = pipeline.describe(crop)

            prev_cls = cls_id
            prev_box = current_box

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ===== Caption Display =====
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

        cv2.putText(display, line, (x_text, y_text), font, font_scale, (0, 0, 0), 4)
        cv2.putText(display, line, (x_text, y_text), font, font_scale, (0, 255, 0), thickness)

    display = cv2.resize(display, (new_w, new_h))

    cv2.imshow("AI Vision System", display)
    out.write(display)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
out.release()
cv2.destroyAllWindows()

print()
log("✅ Done")
log(f"📁 Output saved at {output_path}")