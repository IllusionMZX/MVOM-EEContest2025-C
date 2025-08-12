import os
import random
import json
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# --- Part 1: Dataset Generation (允许重叠，为每个正方形生成独立标签) ---
print("--- Step 1: Generating Synthetic Dataset ---")

def get_rotated_bbox(center_x, center_y, side_len, angle_rad):
    """Calculate the rotated bounding box points of a square."""
    half_size = side_len / 2
    points_local = np.array([
        [-half_size, -half_size], 
        [half_size, -half_size],  
        [half_size, half_size],   
        [-half_size, half_size]   
    ])
    rotation_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)]
    ])
    return np.dot(points_local, rotation_matrix.T) + np.array([center_x, center_y])


def generate_coco_dataset(num_images=400, 
                          img_aspect=(21.0, 29.7), 
                          img_scale_factor=30,
                          border_cm=2.0, 
                          square_side_range_cm=(6.0, 12.0)):
    output_dir = "synthetic_squares_dataset"
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    a4_width_px = int(img_aspect[0] * img_scale_factor)
    a4_height_px = int(img_aspect[1] * img_scale_factor)
    img_size = (a4_width_px, a4_height_px)
    px_per_cm = img_scale_factor
    
    border_px = border_cm * px_per_cm
    min_side_px = square_side_range_cm[0] * px_per_cm
    max_side_px = square_side_range_cm[1] * px_per_cm
    
    print(f"Image Resolution (A4 aspect): {img_size[0]}x{img_size[1]} pixels")
    print(f"Pixel per cm: {px_per_cm:.2f}")
    print(f"Border size: {border_px:.2f} pixels")
    print(f"Square size range: {min_side_px:.2f} to {max_side_px:.2f} pixels")

    for i in range(num_images):
        img = Image.new('RGB', img_size, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        inner_box = [border_px, border_px, a4_width_px - border_px, a4_height_px - border_px]
        draw.rectangle(inner_box, fill=(255, 255, 255))
        
        yolo_annotations = []
        num_squares = random.randint(2, 4)
        
        for _ in range(num_squares):
            side_len = random.uniform(min_side_px, max_side_px)
            canvas_min_x = border_px + side_len / 2
            canvas_max_x = a4_width_px - border_px - side_len / 2
            canvas_min_y = border_px + side_len / 2
            canvas_max_y = a4_height_px - border_px - side_len / 2
            
            center_x = random.uniform(canvas_min_x, canvas_max_x)
            center_y = random.uniform(canvas_min_y, canvas_max_y)
            angle_deg = random.uniform(0, 360)
            angle_rad = math.radians(angle_deg)
            
            rotated_points = get_rotated_bbox(center_x, center_y, side_len, angle_rad)
            
            draw.polygon([tuple(p) for p in rotated_points], fill=(0, 0, 0))
            
            normalized_poly = []
            for px, py in rotated_points:
                normalized_poly.append(px / img_size[0])
                normalized_poly.append(py / img_size[1])
            
            yolo_line = f"0 {' '.join(map(str, normalized_poly))}"
            yolo_annotations.append(yolo_line)

        img_filename = f"image_{i:04d}.png"
        label_filename = f"image_{i:04d}.txt"
        img.save(os.path.join(images_dir, img_filename))
        
        with open(os.path.join(labels_dir, label_filename), 'w') as f:
            f.write('\n'.join(yolo_annotations))

    dataset_yaml_content = f"""
path: ./{output_dir}
train: images
val: images
names:
  0: square
"""
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(dataset_yaml_content)

    print(f"Dataset generation complete. Created {num_images} images with A4 aspect ratio and allowed overlap.")
    return output_dir

def train_yolov8_model(dataset_path):
    print("\n--- Step 2: Training YOLOv8n-seg Model ---")
    model = YOLO("yolov8n-seg.pt")
    
    print("YOLOv8 training is starting. Below, you will see a progress bar and loss metrics for each epoch.")
    
    # 增加训练轮次以更好地处理重叠
    model.train(data=os.path.join(dataset_path, "dataset.yaml"), 
                epochs=30, # 增加到30个epoch
                imgsz=640,
                batch=8,
                workers=4,
                patience=10)
    
    print("Model training complete.")
    return model

def export_model_to_onnx(model):
    print("\n--- Step 3: Exporting Model to ONNX format ---")
    try:
        exported_path = model.export(format="onnx")
        
        if os.path.exists(exported_path) and os.path.getsize(exported_path) > 1000000:
            print(f"Model exported to ONNX format successfully at: {exported_path}")
        else:
            print("Warning: ONNX file may not have been generated correctly. Check for errors above.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("Please check your environment (e.g., install onnx and onnxruntime).")

def main():
    dataset_path = generate_coco_dataset(num_images=200)
    trained_model = train_yolov8_model(dataset_path)
    export_model_to_onnx(trained_model)
    print("\nPC-side script finished. Look for the ONNX model in 'runs/segment/train/weights/best.onnx'")

if __name__ == "__main__":
    main()