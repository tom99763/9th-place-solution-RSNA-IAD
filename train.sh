#!/usr/bin/bash

set -xe

python ./train.py experiment="ch32_effb2" fold_id=0
python ./train.py experiment="ch32_effb2" fold_id=1
python ./train.py experiment="ch32_effb2" fold_id=2
python ./train.py experiment="ch32_effb2" fold_id=3
python ./train.py experiment="ch32_effb2" fold_id=4
