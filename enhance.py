import os
import cv2
import torch
import subprocess
import numpy as np

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

# =========================================================
# 📁 PATHS
# =========================================================
input_video = r"D:\aero\WhatsApp Video 2026-04-10 at 14.57.57.mp4"

video_24fps = "video_24fps.mp4"
frames_dir = "frames"
enhanced_dir = "enhanced_frames"
temp_video = "enhanced_video.mp4"
final_video = "final_output.mp4"

# 🔥 IMPORTANT: SET YOUR FFMPEG PATH HERE
ffmpeg_path = r"D:\aero\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

# =========================================================
# 🔍 CHECKS
# =========================================================
if not os.path.exists(input_video):
    print("❌ Input video not found!")
    exit()

if not os.path.exists(ffmpeg_path):
    print("❌ FFmpeg not found at:", ffmpeg_path)
    exit()

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(enhanced_dir, exist_ok=True)

# =========================================================
# 🎥 STEP 1: CONVERT TO 24 FPS
# =========================================================
print("Converting to 24 FPS...")

subprocess.run([
    ffmpeg_path,
    "-y",
    "-i", input_video,
    "-r", "24",
    video_24fps
], check=True)

# =========================================================
# 🎞️ STEP 2: EXTRACT FRAMES
# =========================================================
print("Extracting frames...")

cap = cv2.VideoCapture(video_24fps)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:05d}.png"), frame)
    frame_idx += 1

cap.release()
print(f"Frames extracted: {frame_idx}")

# =========================================================
# ⚙️ STEP 3: LOAD MODELS
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RRDBNet(3, 3, 64, 23, 32, 4)

bg_upsampler = RealESRGANer(
    scale=4,
    model_path="RealESRGAN_x4plus.pth",
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

face_enhancer = GFPGANer(
    model_path="GFPGANv1.4.pth",
    upscale=4,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=bg_upsampler
)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# =========================================================
# 🚀 STEP 4: ENHANCE FRAMES
# =========================================================
print("Enhancing frames...")

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

for file in frame_files:
    in_path = os.path.join(frames_dir, file)
    out_path = os.path.join(enhanced_dir, file)

    img = cv2.imread(in_path)
    if img is None:
        continue

    _, _, output = face_enhancer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    output = cv2.filter2D(output, -1, kernel)
    cv2.imwrite(out_path, output)

print("Frame enhancement done!")

# =========================================================
# 🎬 STEP 5: REBUILD VIDEO
# =========================================================
print("Rebuilding video...")

subprocess.run([
    ffmpeg_path,
    "-y",
    "-framerate", "24",
    "-i", os.path.join(enhanced_dir, "frame_%05d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    temp_video
], check=True)

# =========================================================
# 🔊 STEP 6: MERGE AUDIO
# =========================================================
print("Merging audio...")

subprocess.run([
    ffmpeg_path,
    "-y",
    "-i", temp_video,
    "-i", input_video,
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    final_video
], check=True)

print("✅ Final output ready:", final_video)