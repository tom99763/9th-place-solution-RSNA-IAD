
# RSNA IAD Codebase

To reproduce 0.79 submission:

1. Download `ultralytics-timm` from kaggle using Download ultralytics-timm from Kaggle

```bash
    kaggle datasets download sersasj/ultralytcs-timm-rsna
    unzip ultralytcs-timm-rsna.zip
```

2. Prepare data:

```bash
python3 ./prepare_yolo_dataset.py --generate-all-folds --out-name yolo_dataset --img-size 512 --label-scheme locations --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite --rgb-mode
```

3. Train YOLO 8m:

```bash
python3 ./run_yolo_pipeline.py  --epochs 100 --img 512 --batch 32 --model yolov8m --project yolo_aneurysm_locations --name yolo_8m --data-fold-template configs/yolo_fold{fold}.yaml  --folds 0,1,2,3,4 --cls 1.0
```

4. Train Yolo EfficientNetB2:

```bash
python3 ./run_yolo_pipeline.py  --epochs 100 --img 512 --batch 32 --model yolo-11-effnetv2_s.yaml --project yolo_aneurysm_locations --name yolo_8m --data-fold-template configs/yolo_fold{fold}.yaml  --folds 0,1,2,3,4 --cls 1.0
```
