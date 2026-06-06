"""
Real-ESRGAN + GFPGAN Full Quality Pipeline
-----------------------------------------
• Enhances full image using Real-ESRGAN
• Enhances faces using GFPGAN
• Applies optional sharpening
"""

import cv2
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

# =========================================================
# 📁 PATHS
# =========================================================
input_image = "input.jpg"
output_image = "final_output.png"

# =========================================================
# ⚙️ DEVICE
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================
# ⚙️ ESRGAN MODEL (BACKGROUND UPSCALER)
# =========================================================
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

bg_upsampler = RealESRGANer(
    scale=4,
    model_path="RealESRGAN_x4plus.pth",
    model=model,
    tile=0,          # best quality
    tile_pad=10,
    pre_pad=0,
    half=False       # full precision
)

# =========================================================
# ⚙️ GFPGAN (FACE ENHANCEMENT)
# =========================================================
face_enhancer = GFPGANer(
    model_path="GFPGANv1.4.pth",
    upscale=4,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=bg_upsampler
)

# =========================================================
# 🖼️ LOAD IMAGE
# =========================================================
img = cv2.imread(input_image, cv2.IMREAD_COLOR)

# =========================================================
# 🚀 ENHANCE IMAGE + FACES
# =========================================================
_, _, output = face_enhancer.enhance(
    img,
    has_aligned=False,
    only_center_face=False,
    paste_back=True
)

# =========================================================
# 🔍 OPTIONAL SHARPENING (EXTRA QUALITY)
# =========================================================
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

output = cv2.filter2D(output, -1, kernel)

# =========================================================
# 💾 SAVE OUTPUT
# =========================================================
cv2.imwrite(output_image, output)

print("✅ Super-resolution + face enhancement completed!")