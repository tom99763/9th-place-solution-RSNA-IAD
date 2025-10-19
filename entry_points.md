python3 ./prepare_yolo_dataset.py --generate-all-folds --out-name yolo_dataset --img-size 512 --label-scheme locations --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite --rgb-mode #Prepare data
python3 ./run_yolo_pipeline.py  --epochs 80 --img 512 --batch 32 --model yolo11m.pt --project yolo_aneurysm_locations --name yolo_11m --rgb-mode --data-fold-template configs/yolo_fold{fold}.yaml  --folds 0,1,2,3,4 --cls 1.0 #Train YOLO 11m
python3 ./run_yolo_pipeline.py  --epochs 50 --img 512 --batch 32 --model yolo-11-effnetv2_s.yaml --project yolo_aneurysm_locations --name yolo_effnetv2 --rgb-mode --data-fold-template configs/yolo_fold{fold}.yaml  --folds 0,1,2,3,4 --cls 1.0 #Train Yolo with tf_efficientnetv2_s.in21k_ft_in1k backbone

