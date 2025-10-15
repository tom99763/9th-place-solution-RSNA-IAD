
import sys
import os
sys.path.insert(0, "ultralytics-timm")

from ultralytics import YOLO

# Load the YOLO11 model
model_path = "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_positive_only/cv_effnetv2s_v2_drop_path_25d_fold0/weights/best.pt"
model = YOLO(model_path)

# Export the model to TensorRT format
print("Exporting model to TensorRT format...")
export_path = model.export(
    format="engine",
    dynamic=True,
    batch=32,              # Max batch size
    half=True              # FP16 for faster inference on T4
)
print(f"Model exported to: {export_path}")

# Load the exported TensorRT model
if os.path.exists(export_path):
    tensorrt_model = YOLO(export_path)
    print("TensorRT model loaded successfully")
else:
    print(f"Error: TensorRT model not found at {export_path}")
    sys.exit(1)

# Run inference with correct image size (512x512 as model was trained)
print("Running inference with TensorRT model...")
test_image = "/home/sersasj/RSNA-IAD-Codebase/data/yolo_dataset_negative_slices_fold0/images/train/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647_1.2.826.0.1.3680043.8.498.12257165606802989041095974109605560481_slice_neg.png"

results_tensorrt = tensorrt_model(test_image, imgsz=512)
print("TensorRT results:", results_tensorrt)

print("\nRunning inference with original PyTorch model...")
results_pytorch = model(test_image, imgsz=512)
print("PyTorch results:", results_pytorch)