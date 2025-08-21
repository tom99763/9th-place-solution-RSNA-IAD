"""Train a YOLO (Ultralytics) model for aneurysm detection.

Requires: pip install ultralytics

Example:
  python src/train_yolo_aneurysm.py \
      --data configs/yolo_aneurysm.yaml \
      --model yolov8n.pt \
      --epochs 50 --img 512 --batch 16

Will create runs/detect/<exp*> inside project root by default.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description='Train YOLO aneurysm detector')
    ap.add_argument('--data', type=str, default='configs/yolo_aneurysm.yaml', help='Dataset YAML path')
    ap.add_argument('--model', type=str, default='yolo11l.pt', help='Pretrained checkpoint or model config')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--img', type=int, default=512, help='Image size (will auto-resize)')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--project', type=str, default='runs/yolo_aneurysm')
    ap.add_argument('--name', type=str, default='exp-yolon-new-data')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze (backbone)')
    ap.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    ap.add_argument('--exist-ok', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    # sleep for 30 min
    #time.sleep(1800)  # simulate long-running process
    model = YOLO(args.model)  # Load model from checkpoint or config

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        workers=args.workers,
        freeze=args.freeze,
        patience=args.patience,
        exist_ok=args.exist_ok,
        seed=args.seed,
        verbose=True,
        deterministic=True
    )
    print(results)

    # Validate explicitly (optional) - uses best.pt
    print("Running validation on best.pt ...")
    model = YOLO(Path(results.save_dir) / 'weights' / 'best.pt')
    val_metrics = model.val(data=args.data, imgsz=args.img, split='val')
    print(val_metrics)

if __name__ == '__main__':
    main()
#python3 src/train_yolo_aneurysm.py --data configs/yolo_aneurysm.yaml --model yolov8n.pt --epochs 50 --img 512 --batch 16