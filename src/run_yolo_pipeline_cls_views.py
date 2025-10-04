"""
Simple pipeline to train YOLO view classification models.

Trains a YOLO classification model to predict anatomical views (axial, sagittal, coronal).

Usage example:
    python -m src.run_yolo_pipeline_cls_views \
            --model yolo11n-cls.pt \
            --data data/yolo_view_cls/dataset.yaml \
            --epochs 100 --img 512 --batch 16 \
            --project runs/yolo_view_cls --name baseline
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description='Train and evaluate YOLO view classification model')
    # Dataset and model
    ap.add_argument('--data', type=str, default='data/yolo_view_cls', help='Dataset directory path')
    ap.add_argument('--model', type=str, default='yolo11n-cls.pt', help='Pretrained classification checkpoint or trained weights')
    
    # Mode
    ap.add_argument('--eval-only', action='store_true', help='Only evaluate model, skip training')
    ap.add_argument('--split', type=str, default='test', help='Split to evaluate (train/test/val)')
    
    # Training parameters
    ap.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    ap.add_argument('--img', type=int, default=512, help='Input image size')
    ap.add_argument('--batch', type=int, default=16, help='Batch size')
    ap.add_argument('--device', type=str, default='', help='Device (e.g., 0, 0,1, cpu)')
    ap.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    
    # Output settings
    ap.add_argument('--project', type=str, default='runs/yolo_view_cls', help='Project directory')
    ap.add_argument('--name', type=str, default='exp', help='Experiment name')
    ap.add_argument('--exist-ok', action='store_true', help='Overwrite existing experiment')
    
    # Training options
    ap.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze')
    ap.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Augmentation
    ap.add_argument('--mixup', type=float, default=0.1, help='Mixup augmentation ratio')
    ap.add_argument('--fliplr', type=float, default=0.5, help='Horizontal flip probability')
    ap.add_argument('--flipud', type=float, default=0.5, help='Vertical flip probability')
    ap.add_argument('--hsv_h', type=float, default=0.015, help='HSV hue augmentation')
    ap.add_argument('--hsv_s', type=float, default=0.7, help='HSV saturation augmentation')
    ap.add_argument('--hsv_v', type=float, default=0.4, help='HSV value augmentation')
    ap.add_argument('--auto-augment', type=str, default='randaugment', help='Auto augmentation policy')
    
    return ap.parse_args()


def run():
    """Train or evaluate YOLO view classification model."""
    args = parse_args()
    
    # Resolve data path
    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    
    if not data_path.exists():
        raise SystemExit(f"Dataset directory not found: {data_path}")
    
    # Load model
    model = YOLO(args.model)
    
    # Evaluation-only mode
    if args.eval_only:
        print(f"\n{'='*60}")
        print(f"Evaluating YOLO View Classification Model")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Data: {data_path}")
        print(f"Split: {args.split}")
        print(f"Image size: {args.img}")
        print(f"Batch size: {args.batch}")
        print(f"{'='*60}\n")
        
        # Run validation
        val_results = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.img,
            batch=args.batch,
            device=args.device if args.device else None,
        )
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({args.split} set)")
        print(f"{'='*60}")
        print(f"Top-1 Accuracy: {val_results.top1:.4f}")
        print(f"Top-5 Accuracy: {val_results.top5:.4f}")
        
        # Print per-class metrics if available
        if hasattr(val_results, 'results_dict'):
            print(f"\nDetailed Metrics:")
            for key, value in val_results.results_dict.items():
                if key not in ['fitness']:
                    print(f"  {key}: {value:.4f}")
        
        print(f"{'='*60}\n")
        return
    
    # Training mode
    print(f"\n{'='*60}")
    print(f"Training YOLO View Classification")
    print(f"{'='*60}")
    print(f"Data: {data_path}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.img}")
    print(f"Batch size: {args.batch}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print(f"{'='*60}\n")
    
    results = model.train(
        data=str(data_path),
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
        deterministic=True,
        # Augmentation
        mixup=args.mixup,
        fliplr=args.fliplr,
        flipud=args.flipud,
        dropout=args.dropout,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        auto_augment=None if args.auto_augment.lower() == 'none' else args.auto_augment,
    )
    
    # Print summary
    save_dir = Path(results.save_dir)
    weights_path = save_dir / 'weights' / 'best.pt'
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {save_dir}")
    print(f"Best weights: {weights_path}")
    print(f"{'='*60}\n")
    
    # Run validation on test set
    print(f"Running validation on test set...")
    val_results = model.val(data=str(data_path), split='test')
    
    print(f"\nTest Set Results:")
    print(f"  Top-1 Accuracy: {val_results.top1:.4f}")
    print(f"  Top-5 Accuracy: {val_results.top5:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run()


# Example usage:

# Training:
# python -m src.run_yolo_pipeline_cls_views \
#     --model yolo11n-cls.pt \
#     --data data/yolo_view_cls \
#     --epochs 100 --img 512 --batch 16 \
#     --project runs/yolo_view_cls --name baseline

# Evaluation only (test set):
# python -m src.run_yolo_pipeline_cls_views \
#     --model runs/yolo_view_cls/baseline/weights/best.pt \
#     --data data/yolo_view_cls \
#     --eval-only --split test

# Evaluation on train set:
# python -m src.run_yolo_pipeline_cls_views \
#     --model runs/yolo_view_cls/baseline/weights/best.pt \
#     --data data/yolo_view_cls \
#     --eval-only --split train


