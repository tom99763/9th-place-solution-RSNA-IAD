import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

#try:
#    from ultralytics import YOLO
#except ImportError:
#    print("Ultralytics not installed. Install with: pip install ultralytics")
#    sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description='Train YOLO aneurysm detector')
    ap.add_argument('--data', type=str, default='configs/yolo_aneurysm_locations.yaml', help='Dataset YAML path')
    ap.add_argument('--model', type=str, default='yolo11s.pt', help='Pretrained checkpoint or model config')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--img', type=int, default=512, help='Image size (will auto-resize)')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--project', type=str, default='runs/yolo_aneurysm_locations')
    ap.add_argument('--name', type=str, default='exp_yolo11s')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze (backbone)')
    ap.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    ap.add_argument('--exist-ok', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-train', action='store_true', help='Skip training and only run val on existing best.pt under project/name')
    ap.add_argument('--fold', type=int, default=0, help='Fold to validate after training')
    return ap.parse_args()


def main():
    args = parse_args()
    exp_name = args.name
    model = YOLO(args.model)  # Load model from checkpoint or config

    if args.skip_train:
        save_dir = Path(args.project) / exp_name
        weights_path = save_dir / 'weights' / 'best.pt'
        if not weights_path.exists():
            raise SystemExit(f"--skip-train set but weights not found at {weights_path}")
    else:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.img,
            batch=args.batch,
            device=args.device if args.device else None,
            project=args.project,
            name=exp_name,
            workers=args.workers,
            freeze=args.freeze,
            patience=args.patience,
            exist_ok=args.exist_ok,
            seed=args.seed,
            verbose=True,
            deterministic=True,
        )
        print(results)
        save_dir = Path(results.save_dir)
        weights_path = save_dir / 'weights' / 'best.pt'

    # Quick built-in val (dataset val split)
    print("Running Ultralytics val on best.pt ...")
    model = YOLO(str(weights_path))
    val_metrics = model.val(data=args.data, imgsz=args.img, split='val')
    print(val_metrics)

if __name__ == '__main__':
    main()
#python3 src/train_yolo_aneurysm.py --data configs/yolo_aneurysm.yaml --model yolov8n.pt --epochs 50 --img 512 --batch 16