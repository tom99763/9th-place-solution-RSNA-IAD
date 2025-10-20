
# 9th-place-solution-RSNA Intracranial Aneurysm Detection

## Yolo 2.5D

### Environment 
| Hardware Setup | GPU | CPU | RAM | Time per Fold | Total (5 folds) |
| --- | --- | --- | --- | --- | --- |
| @sersasj | RTX 3090 | Intel Core i5-12400F (12) @ 4.4GHz | 32GB | 4-5 hours | ~24 hours |
| @iamparadox | RTX 4090 | AMD Ryzen 9 7950X (32) @ 5.883GHz | 64GB | ~2 hours | ~10 hours |

To reproduce yolo 2.5D

1. Get all the competition data and setup directory structure

```bash
    ./get-data.sh
```

2. Prepare data:

```bash
python3 ./prepare_yolo_dataset.py --generate-all-folds --out-name yolo_dataset --img-size 512 --label-scheme locations --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite --rgb-mode
```

3. Train YOLO 11m:

```bash
python3 ./run_yolo_pipeline.py  --epochs 80 --img 512 --batch 32 --model yolo11m.pt --project ./models --name yolo_11m --data-fold-template configs/yolo_fold{fold}.yaml  --folds 0,1,2,3,4 --cls 1.0
```

4. Train Yolo with tf_efficientnetv2_s.in21k_ft_in1k backbone:

```bash
python3 ./run_yolo_pipeline.py  --epochs 100 --img 512 --batch 32 --model yolo-11-effnetv2_s.yaml --project ./models/ --name yolo_effv2 --data-fold-template configs/yolo_fold{fold}.yaml  --folds 0 --cls 1.0
```

5. Strip best weights for trained yolo:

```
python3 ./get_weights.py
```

Now all the weights of the the trained YOLO(s) will be present under ./models with format: yolo_{model_version}_fold{fold_number}.pt


## EfficientV2s + 3D-CenterNet (Flayer)
