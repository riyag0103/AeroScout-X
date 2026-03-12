from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # ---------------------------------------------------
    # 📌 Load YOLOv12 Instance Segmentation Model
    # ---------------------------------------------------
    model = YOLO("yolov8n.pt")

    # ---------------------------------------------------
    # 📌 Train the Model + Custom Save Location
    # ---------------------------------------------------
    results = model.train(
        data=r"D:\cnn\drone_dataset\data.yaml",
        epochs=200,
        imgsz=640,
        batch=2,
        device=0,
        single_cls=False,

        # 🚀 CUSTOM SAVE LOCATION
        project=r"D:\cnn\trained\drone_dataset",
        name="yolov12_detection",   # your custom folder name

        # Hyperparameters
        lr0=0.005,
        lrf=0.2,
        momentum=0.937,
        weight_decay=0.0005,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.0,
        box=7.5,
        cls=0.5,
        patience=50,
        verbose=True
    )

    print("✅ Training started successfully! 🚀")

    # ---------------------------------------------------
    # 📊 Show Training Graph (results.png)
    # ---------------------------------------------------
    # Correct way to get save directory
    save_dir = results.save_dir
    results_img = os.path.join(save_dir, "results.png")
    if os.path.exists(results_img):
        img = plt.imread(results_img)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("📊 YOLO Training Metrics", fontsize=16)
        plt.show()
    else:
        print(f"⚠️ Could not find results image at: {results_img}")