
# RSNA IAD Codebase

To reproduce 0.79 submission:

1. Download ultralytics-timm from Kaggle

2. Prepare data:
```bash
python3 -m src.prepare_yolo_dataset_v2 --generate-all-folds --neg-per-series 1 --out-name yolo_dataset --img-size 512 --label-scheme locations --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite
```

3. Train model:
```bash
python3 -m src.run_yolo_pipeline --model xxxxx --epochs 100 --img 512 --batch 16 --project yolo_aneurysm_locations --name cv_efficientnet_v2_b0-config2 --data-fold-template configs/yolo_fold{fold}.yaml --folds 0,1,2,3,4
