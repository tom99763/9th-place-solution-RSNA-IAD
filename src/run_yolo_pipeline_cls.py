"""
One-command pipeline: per-fold train YOLO, then validate on the same fold with organized outputs.

Behavior:
- For each fold f in --folds, trains a separate model using --data-fold-template (with {fold}).
- Validates fold f with that fold's best.pt.
- Saves per-fold metrics.json, per_series_predictions.csv, per_location_auc.csv.

Usage example:
    python -m src.run_yolo_pipeline --model yolo11s.pt \
            --epochs 100 --img 512 --batch 16 \
            --project runs/yolo_aneurysm_locations --name cv_y11s \
            --data-fold-template configs/yolo_fold{fold}.yaml \
            --folds 0,1,2,3,4
"""
import argparse
from pathlib import Path
import sys
import json
import shutil
from typing import List

sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description='Train and validate YOLO aneurysm pipeline')
    ap.add_argument('--data', type=str, default='configs/yolo_aneurysm_locations.yaml', help='Dataset YAML path')
    ap.add_argument('--model', type=str, default='yolo11n-cls.pt', help='Pretrained classification checkpoint or model config')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--img', type=int, default=512)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--project', type=str, default='yolo_aneurysm_locations')
    ap.add_argument('--name', type=str, default='exp-2-folds')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--freeze', type=int, default=0)
    ap.add_argument('--patience', type=int, default=150)
    ap.add_argument('--exist-ok', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-fold-template', type=str, default='', help='YAML template with {fold} placeholder for per-fold datasets (required)')
    ap.add_argument('--mixup', type=float, default=0.0, help='Mixup augmentation ratio (0.0 = no mixup)')
    ap.add_argument('--mosaic', type=float, default=0.0, help='Mosaic augmentation ratio (0.0 = no mosaic)')
    ap.add_argument('--fliplr', type=float, default=0.5, help='Horizontal flip augmentation ratio (0.0 = no flip)')
    ap.add_argument('--flipud', type=float, default=0.5, help='Vertical flip augmentation ratio (0.0 = no flip)')
    ap.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (0.0 = no dropout)')
    ap.add_argument('--hsv_h', type=float, default=0.0, help='HSV hue shift (0.0 = no shift)')
    ap.add_argument('--hsv_s', type=float, default=0.0, help='HSV saturation shift (0.0 = no shift)')
    ap.add_argument('--hsv_v', type=float, default=0.0, help='HSV value shift (0.0 = no shift)')
    ap.add_argument('--close-mosaic', type=int, default=0, help='Close mosaic boxes (0=disabled, 10=remove_last_10_epochs)')
    ap.add_argument('--erasing', type=float, default=0.0, help='Erasing rate (0.0 = no erasing)')
    ap.add_argument('--auto-augment', type=str, default="randaugment", help='Auto-augmentation method (None = no auto-augmentation)')
    # Validation settings
    ap.add_argument('--folds', type=str, default='0', help='Comma-separated fold IDs to validate (e.g., 0 or 0,1,2,3,4)')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice (for per-slice and rgb modes)')
    ap.add_argument('--mip-window', type=int, default=0, help='Half-window for MIP; 0 = per-slice mode; use --rgb-mode for 3-slice stacking')
    ap.add_argument('--mip-img-size', type=int, default=0)
    ap.add_argument('--mip-no-overlap', action='store_true')
    ap.add_argument('--max-slices', type=int, default=0)
    ap.add_argument('--series-limit', type=int, default=0)
    ap.add_argument('--val-batch', type=int, default=16)
    ap.add_argument('--verbose-val', action='store_true')
    # Optional override W&B details for validation; if not provided, we'll try to reuse training run info
    ap.add_argument('--val-wandb', action='store_true', help='Force-enable W&B logging for validation')
    ap.add_argument('--wandb-project', type=str, default='', help='W&B project (if set, used for validation logging)')
    ap.add_argument('--wandb-entity', type=str, default='', help='W&B entity (team/user)')
    ap.add_argument('--wandb-group', type=str, default='', help='W&B group')
    ap.add_argument('--wandb-tags', type=str, default='', help='Comma-separated W&B tags')
    ap.add_argument('--rgb-mode', action='store_true', help='Use rgb mode (3-channel images from stacked slices); overrides --mip-window')
    return ap.parse_args()


def run():
    print("waiting")
    import time
    args = parse_args()

    folds: List[int] = [int(x) for x in args.folds.split(',') if x.strip() != '']

    # Call the validation script programmatically
    val_script = ROOT / 'yolo_multiclass_validation_with_resize_cls.py'
    if not val_script.exists():
        raise SystemExit(f"Validation script not found at {val_script}")

    # For classification, we use the pre-converted dataset directories

    import subprocess
    import json as _json

    def _get_wandb_resume_info(save_dir: Path):
        """Try to read W&B run metadata from Ultralytics training output folder.
        Returns dict with keys: id, project, entity, group, name; or None if not found.
        """
        wandb_dir = save_dir / 'wandb'
        if not wandb_dir.exists():
            return None
        run_dir = None
        latest = wandb_dir / 'latest-run'
        if latest.exists():
            try:
                rel = latest.read_text().strip()
                if rel:
                    cand = wandb_dir / rel
                    if cand.exists():
                        run_dir = cand
            except Exception:
                pass
        if run_dir is None:
            candidates = [p for p in wandb_dir.glob('run-*') if p.is_dir()]
            if not candidates:
                return None
            run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        # metadata file may be in run_dir or run_dir/files
        md_paths = [run_dir / 'wandb-metadata.json', run_dir / 'files' / 'wandb-metadata.json']
        meta = {}
        for mp in md_paths:
            if mp.exists():
                try:
                    meta = _json.loads(mp.read_text())
                    break
                except Exception:
                    pass
        if not meta:
            return None
        return {
            'id': meta.get('id') or meta.get('run_id') or '',
            'project': meta.get('project') or '',
            'entity': meta.get('entity') or '',
            'group': meta.get('group') or '',
            'name': meta.get('name') or '',
        }
    # Train per-fold and validate with the fold's best weights
    for f in folds:
        # For classification, pass YAML file path to YOLO
        if args.data_fold_template:
            # Extract the dataset directory name from the YAML template
            yaml_path = args.data_fold_template.format(fold=f)
            # Read the path from the YAML file
            yaml_file = ROOT / yaml_path
            if not yaml_file.exists():
                raise SystemExit(f'YAML file not found: {yaml_file}')
            
            import yaml
            with open(yaml_file, 'r') as f_yaml:
                yaml_content = yaml.safe_load(f_yaml)
                dataset_path = yaml_content['path']
                dataset_dir = ROOT / dataset_path
        else:
            # Fallback to default naming
            dataset_dir = ROOT / f"data/yolo_cls_rgb_fold{f}"
            yaml_file = None
            
        if not dataset_dir.exists():
            raise SystemExit(f'Dataset directory not found: {dataset_dir}')
        model = YOLO(args.model)
        results = model.train(
            data=dataset_dir,
            epochs=args.epochs,
            imgsz=args.img,
            batch=args.batch,
            device=args.device if args.device else None,
            project=args.project,
            name=f"{args.name}_fold{f}",
            workers=args.workers,
            freeze=args.freeze,
            patience=args.patience,
            exist_ok=args.exist_ok,
            seed=args.seed,
            verbose=True,
            deterministic=True,
            mixup=args.mixup,
            mosaic=args.mosaic,
            fliplr=args.fliplr,
            flipud=args.flipud,
            dropout=args.dropout,
            erasing=args.erasing,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            close_mosaic=args.close_mosaic,
            auto_augment=None if args.auto_augment == 'None' else args.auto_augment,
            save_period=5
        )
        save_dir = Path(results.save_dir)
        weights_path = save_dir / 'weights' / 'best.pt'
        if not weights_path.exists():
            raise SystemExit(f"best.pt not found at {weights_path}")
        out_dir = save_dir / 'series_validation'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Validate this fold with its own weights
        cmd = [
            sys.executable,
            str(val_script),
            '--weights', str(weights_path),
            '--val-fold', str(f),
            '--out-dir', str(out_dir),
            '--batch-size', str(args.val_batch),
            '--slice-step', str(args.slice_step),
            '--mip-window', str(args.mip_window),
            '--mip-img-size', str(args.mip_img_size),
        ]
        if args.mip_no_overlap:
            cmd.append('--mip-no-overlap')
        if args.max_slices:
            cmd += ['--max-slices', str(args.max_slices)]
        if args.series_limit:
            cmd += ['--series-limit', str(args.series_limit)]
        if args.rgb_mode:
            cmd.append('--rgb-mode')
        # Try to attach validation logging to the same W&B run as training
        wandb_info = _get_wandb_resume_info(save_dir)
        if wandb_info or args.val_wandb or args.wandb_project:
            cmd.append('--wandb')
            run_name = f"{args.name}_fold{f}_val"
            if wandb_info:
                if wandb_info.get('project'):
                    cmd += ['--wandb-project', wandb_info['project']]
                if wandb_info.get('entity'):
                    cmd += ['--wandb-entity', wandb_info['entity']]
                if wandb_info.get('group'):
                    cmd += ['--wandb-group', wandb_info['group']]
                if wandb_info.get('id'):
                    cmd += ['--wandb-resume-id', wandb_info['id']]
#
        print('Running validation:', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    run()

# python3 -m src.run_yolo_pipeline_cls   --model yolo11n.pt  --epochs 5 --img 512 --batch 16 --project yolo_aneurysm_classification --name cv_y11m_with_resize --data-fold-template configs/yolo_cls_rgb_fold{fold}.yaml  --folds 0

# python3 -m src.run_yolo_pipeline_cls   --model yolo11n.pt  --epochs 100 --img 512 --batch 16 --project yolo_aneurysm_classification --name cv_y11n_baseline_with_resize --data-fold-template configs/yolo_cls_rgb_v0_fold{fold}.yaml  --folds 0

# python3 -m src.run_yolo_pipeline_cls     --model yolo11n-cls.pt     --epochs 70
# --img 512 --batch 16     --project yolo_aneurysm_classification     --name cv_y11n_25d_baseline     --data-fold-template co
# nfigs/yolo_cls_bgr_fold{fold}.yaml     --folds 0     --val-batch 16