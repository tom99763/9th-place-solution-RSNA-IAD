
import sys
import os
sys.path.insert(0, "ultralytics-timm")

from ultralytics import YOLO

# Load the YOLO11 model
model_path = "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_more_negatives_fold02/weights/best.pt"
model = YOLO(model_path)

# Export the model to TensorRT format
print("Exporting model to TensorRT format...")
export_path = model.export(format="engine")  # This will create 'best.engine'
print(f"Model exported to: {export_path}")

# Load the exported TensorRT model
if os.path.exists(export_path):
    tensorrt_model = YOLO(export_path)
    print("TensorRT model loaded successfully")
else:
    print(f"Error: TensorRT model not found at {export_path}")
    sys.exit(1)

# Run inference
print("Running inference")
results = tensorrt_model("/home/sersasj/RSNA-IAD-Codebase/data/yolo_dataset_negative_slices_fold0/images/train/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647_1.2.826.0.1.3680043.8.498.12257165606802989041095974109605560481_slice_neg.png")
print(results)

results_old = model("/home/sersasj/RSNA-IAD-Codebase/data/yolo_dataset_negative_slices_fold0/images/train/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647_1.2.826.0.1.3680043.8.498.12257165606802989041095974109605560481_slice_neg.png")
print(results_old)