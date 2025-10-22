## Flayer Model Training

1. Prepare the data
Put the RSNA 2025 dataset in ./data.

2. Preprocessing step 1 to generate the required DataFrame.
```python
python ./01_02_03_make_flayer_df.py --data_folder ./data 
```
3. Segmentation data preprocessing.
```python
python ./04_get_seg_space07_axisz.py --seg_folder ./data/segmentations 
```
4. Train the segmentation model.
```python
python ./05_try_seg_dynunet.py --device cuda:0
```

5. Vessel segmentation inference, saving, and resized predicted masks.
```python
python ./06_07_save_seg_pred.py
```
6. Preprocessing step 2.
```python
python ./08_precompute_volumes_with_labels_v2.py --data_folder ./data
```
7. Main model training
```python
python ./09_train_centernet3d_448_flayer2_v2_segaux_accumulate.py --device cuda:0 --output_dir ./model/flayer
```


Get out-of-fold predictions
```python
python flayer_model_oof_pred.py --output_dir ./model/flayer
```
