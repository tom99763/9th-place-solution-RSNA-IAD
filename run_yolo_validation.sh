#!/bin/bash

# YOLO Multiclass Validation Script
# This script runs validation for three different folds of the YOLO model

echo "Starting YOLO multiclass validation for folds 2, 3, and 4..."

# Validation for fold 2
echo "Running validation for fold 2..."
python3 /home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_validation_old.py \
    --weights /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold22/weights/best.pt \
    --val-fold 2 \
    --out-dir /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold22/series_validation \
    --batch-size 32 \
    --slice-step 1 \
    --img-size 512 \
    --rgb-mode

if [ $? -eq 0 ]; then
    echo "Fold 2 validation completed successfully"
else
    echo "Fold 2 validation failed with exit code $?"
    exit 1
fi

# Validation for fold 3
echo "Running validation for fold 3..."
python3 /home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_validation_old.py \
    --weights /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold3/weights/best.pt \
    --val-fold 3 \
    --out-dir /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold3/series_validation \
    --batch-size 32 \
    --slice-step 1 \
    --img-size 512 \
    --rgb-mode

if [ $? -eq 0 ]; then
    echo "Fold 3 validation completed successfully"
else
    echo "Fold 3 validation failed with exit code $?"
    exit 1
fi

# Validation for fold 4
echo "Running validation for fold 4..."
python3 /home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_validation_old.py \
    --weights /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold4/weights/best.pt \
    --val-fold 4 \
    --out-dir /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold4/series_validation \
    --batch-size 32 \
    --slice-step 1 \
    --img-size 512 \
    --rgb-mode

if [ $? -eq 0 ]; then
    echo "Fold 4 validation completed successfully"
else
    echo "Fold 4 validation failed with exit code $?"
    exit 1
fi

echo "All YOLO multiclass validations completed successfully!"
