
# 9th-place-solution-RSNA Intracranial Aneurysm Detection

## Install Dependencies
```bash
pip3 install -r requirements.txt
```

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

  - After that `cd ./yolo25d`


Run the following commands inside `yolo25d` directory.

2. Prepare data:

```bash
python3 ./prepare_yolo_dataset.py --generate-all-folds --out-name yolo_dataset --img-size 512 --label-scheme locations --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite --rgb-mode
```

3. Train YOLO 11m:

```bash
python3 ./run_yolo_pipeline.py  --epochs 80 --img 512 --batch 32 --model yolo11m.pt --project yolo_aneurysm_locations --name yolo_11m --data-fold-template ./configs/yolo_fold{fold}.yaml  --folds 0,1,2,3,4 --cls 1.0
```

4. Train Yolo with tf_efficientnetv2_s.in21k_ft_in1k backbone:

```bash
python3 ./run_yolo_pipeline.py  --epochs 50 --img 512 --batch 32 --model yolo-11-effnetv2_s.yaml --project yolo_aneurysm_locations --name yolo_effnetv2 --data-fold-template ./configs/yolo_fold{fold}.yaml  --folds 0,1,2,3,4 --cls 1.0
```

After running the above training commands, the weights of the models will be available at: `./yolo_aneurysm_locations` (inside ./yolo25d).

```bash
ls ./yolo25d/yolo_aneurysm_locations/
```

This should folders of both `yolo_11m` and `yolo_effnetv2`, eg:

```
yolo_11m_fold0  
yolo_effnetv2_fold0
```

## EfficientV2s + 3D-CenterNet (Flayer)

### Environment 

| Hardware Setup | GPU | CPU | RAM | Time per Fold | Total (5 folds) |
| --- | --- | --- | --- | --- | --- |
| @fateplsf | pro 6000 | AMD Ryzen Threadripper PRO 9975WX 32-Cores | 512GB | 4-5 hours | ~24 hours |


## Meta Classifier

1. Generate trained weights of meta classifier:
```batch
python3 ./train_meta_classifier.py --data_path [data path of train.csv] --meta_cls_weight_path [the weights directory of meta classifiers] --yolo_weight_path [the weight directory of yolo] --flayer_weight_path [the weight directory of flayer]
```

2. Output weights
```
meta_classifiers/cat/meta_classifier_[class name]_fold_fold[fold id].pkl
meta_classifiers/lgb/meta_classifier_[class name]_fold_fold[fold id].pkl
meta_classifiers/xgb/meta_classifier_[class name]_fold_fold[fold id].pkl
meta_classifiers/label_encoder_sex.pkl
```


## Inference Pipeline

After training Yolo, Flayer and meta classifier models. We can run inference. To run it use the following command:

```bash
python3 ./inference.py --data_path ./data/kaggle_evaluation/series --meta_cls_weight_path ./meta_classifiers --yolo_weight_path ./yolo25d/yolo_aneurysm_locations --flayer_weight_path ./flayer/flayer_weights
```

Note: You would might need to update the weight paths of these models.

If you want to run the inference directly without training, then download the models from: `https://www.kaggle.com/models/tom99763/9th-place-models-rsna-iad/pyTorch/default`.
