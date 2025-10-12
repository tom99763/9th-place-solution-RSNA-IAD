#!/bin/bash

# YOLO Multiclass Validation Script
# This script runs validation for three different folds of the YOLO model

echo "Starting YOLO multiclass validation for folds 2, 3, and 4..."

# Validation for fold 2
echo "Running validation for fold 0..."
python3 /home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_validation_old.py \
    --weights /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold0/weights/best.pt \
    --val-fold 0 \
    --out-dir /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold0/series_validation \
    --batch-size 32 \
    --slice-step 1 \
    --img-size 512 \
    --rgb-mode

if [ $? -eq 0 ]; then
    echo "Fold 0 validation completed successfully"
else
    echo "Fold 0 validation failed with exit code $?"
    exit 1
fi

# Validation for fold 3
echo "Running validation for fold 1..."
python3 /home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_validation_old.py \
    --weights /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold1/weights/best.pt \
    --val-fold 1 \
    --out-dir /home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold1/series_validation \
    --batch-size 32 \
    --slice-step 1 \
    --img-size 512 \
    --rgb-mode

if [ $? -eq 0 ]; then
    echo "Fold 1 validation completed successfully"
else
    echo "Fold 1 validation failed with exit code $?"
    exit 1
fi
