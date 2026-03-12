from ultralytics import YOLO
import os
# Image path
img_path = r"D:\cnn\pexels-photo-16151478.webp"
# Your trained model
model_path = r"D:\cnn\trained\drone_dataset\yolov12_detection4\weights\best.pt"
# Output folder
output_dir = r"D:\cnn\output"
os.makedirs(output_dir, exist_ok=True)
# Load trained model
model = YOLO(model_path)
# Run inference
results = model(img_path, conf=0.25)
# Show
results[0].show()
# Save
save_path = os.path.join(output_dir, "result.png")
results[0].save(save_path)
print("Saved at:", save_path)
 