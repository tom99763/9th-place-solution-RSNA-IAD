#!/usr/bin/bash

set -xe

python ./train_5_folds.py experiment="ch32_segcls_resnet18d_2.5D" fold_id=0
python ./train_5_folds.py experiment="ch32_segcls_resnet18d_2.5D" fold_id=1
python ./train_5_folds.py experiment="ch32_segcls_resnet18d_2.5D" fold_id=2
python ./train_5_folds.py experiment="ch32_segcls_resnet18d_2.5D" fold_id=3
python ./train_5_folds.py experiment="ch32_segcls_resnet18d_2.5D" fold_id=4
