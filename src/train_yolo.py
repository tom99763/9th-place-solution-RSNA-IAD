import ultralytics
import wandb
import os
import platform
import torch

if __name__ == "__main__":

    model = ultralytics.YOLO("yolo11s.pt")
    model.train(data="/home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_dataset_2.5d/dataset.yaml",
     epochs=100, 
     imgsz=640,
      batch=16,
      optimizer='AdamW',
      lr0=1e-4,
      lrf=0.1,
      warmup_epochs=0,
      dropout=0.0,
      exist_ok=True,
      patience=100,   
      val=True,   
      mosaic=0.0,
      close_mosaic=0,
      mixup=0.0,
      flipud=0.5,
      scale=0.25,
      degrees=45,
      seed=42,
      deterministic=True,
      augment=True,
      device=0,
      erasing=0.0,
      #val_period=1,
      project="rsna-iad-multiclass-detection",
      name="yolo11s-multiclass-detection-v0_25d",
    )
    
    # Finish wandb run
    wandb.finish()

    
