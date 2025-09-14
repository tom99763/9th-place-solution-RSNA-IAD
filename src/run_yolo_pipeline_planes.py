"""
Train and validate YOLO models for all three anatomical planes (axial, coronal, sagittal).

This script:
1. Trains separate YOLO models for each plane view
2. Validates each model individually
3. Provides ensemble evaluation combining all three planes

Usage:
    python -m src.run_yolo_pipeline_planes --model yolo11s.pt --epochs 100 --fold 0
"""
import argparse
from pathlib import Path
import sys
import subprocess
from typing import List, Dict
import json
import yaml

sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# Add repo src to path to reuse data config
sys.path.insert(0, str(ROOT / 'src'))
try:
    from configs.data_config import data_path  # type: ignore
except Exception:
    data_path = None

# Location label mapping (must match yolo_validation_planes.py)
LABELS_TO_IDX = {
    'Anterior Communicating Artery': 0,
    'Basilar Tip': 1,
    'Left Anterior Cerebral Artery': 2,
    'Left Infraclinoid Internal Carotid Artery': 3,
    'Left Middle Cerebral Artery': 4,
    'Left Posterior Communicating Artery': 5,
    'Left Supraclinoid Internal Carotid Artery': 6,
    'Other Posterior Circulation': 7,
    'Right Anterior Cerebral Artery': 8,
    'Right Infraclinoid Internal Carotid Artery': 9,
    'Right Middle Cerebral Artery': 10,
    'Right Posterior Communicating Artery': 11,
    'Right Supraclinoid Internal Carotid Artery': 12,
}
LOCATION_LABELS = sorted(list(LABELS_TO_IDX.keys()))
N_LOC = len(LOCATION_LABELS)


def parse_args():
    ap = argparse.ArgumentParser(description='Train and validate YOLO plane models')
    ap.add_argument('--model', type=str, default='yolo11m.pt', help='Pretrained checkpoint')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--img', type=int, default=512)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--project', type=str, default='yolo_planes')
    ap.add_argument('--name', type=str, default='exp-yolo11m-single-view-plane-increasedRegularization')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--freeze', type=int, default=0)
    ap.add_argument('--patience', type=int, default=150)
    ap.add_argument('--exist-ok', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fold', type=int, default=0, help='Fold to train and validate')
    ap.add_argument('--multi-view', action='store_true', help='Train a single model on all views and validate across all')
    ap.add_argument('--planes', type=str, default='axial,coronal,sagittal', help="Comma-separated planes to use: axial,coronal,sagittal")
    
    # Augmentation parameters
    ap.add_argument('--mixup', type=float, default=0.4)
    ap.add_argument('--mosaic', type=float, default=0.5)
    ap.add_argument('--fliplr', type=float, default=0.0)
    ap.add_argument('--flipud', type=float, default=0.0)
    ap.add_argument('--dropout', type=float, default=0.3)
    
    # Validation settings
    ap.add_argument('--val-batch', type=int, default=16)
    ap.add_argument('--verbose-val', action='store_true')
    ap.add_argument('--series-limit', type=int, default=0)
    ap.add_argument('--max-slices', type=int, default=0)
    
    return ap.parse_args()


def train_plane_model(plane: str, args, fold: int) -> Path:
    """Train a YOLO model for a specific anatomical plane."""
    print(f"\n{'='*60}")
    print(f"Training {plane.upper()} plane model for fold {fold}")
    print(f"{'='*60}")
    
    # Find the plane-specific YAML config; fallback to base plane YAML if fold-specific is missing
    yaml_path_fold = ROOT / "configs" / f"yolo_planes_{plane}_fold{fold}.yaml"
    yaml_path_base = ROOT / "configs" / f"yolo_planes_{plane}.yaml"
    if yaml_path_fold.exists():
        yaml_path = yaml_path_fold
    elif yaml_path_base.exists():
        print(f"Config not found: {yaml_path_fold}, falling back to base config: {yaml_path_base}")
        yaml_path = yaml_path_base
    else:
        raise FileNotFoundError(f"Config not found: {yaml_path_fold} or {yaml_path_base}")
    
    model = YOLO(args.model)
    
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=f"{args.name}_{plane}_fold{fold}",
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
    )
    
    save_dir = Path(results.save_dir)
    weights_path = save_dir / 'weights' / 'best.pt'
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Best weights not found at {weights_path}")
    
    print(f"✓ {plane.upper()} model trained successfully")
    print(f"  Weights saved to: {weights_path}")
    
    return weights_path


def _get_plane_yaml_path(plane: str, fold: int) -> Path:
    """Resolve plane YAML path with fold fallback."""
    yaml_path_fold = ROOT / "configs" / f"yolo_planes_{plane}_fold{fold}.yaml"
    yaml_path_base = ROOT / "configs" / f"yolo_planes_{plane}.yaml"
    if yaml_path_fold.exists():
        return yaml_path_fold
    if yaml_path_base.exists():
        return yaml_path_base
    raise FileNotFoundError(f"Config not found: {yaml_path_fold} or {yaml_path_base}")


def _build_multiview_data_dict(fold: int, planes: List[str]) -> Dict:
    """Construct a Ultralytics data dict that merges train/val from axial, coronal, sagittal YAMLs.

    Returns a dict with absolute paths for 'train' and 'val' lists and 'names'/'nc'.
    """
    trains: List[str] = []
    vals: List[str] = []
    names: List[str] | None = None

    for plane in planes:
        ypath = _get_plane_yaml_path(plane, fold)
        with open(ypath, 'r') as f:
            y = yaml.safe_load(f)
        base = Path(y.get('path', '.'))
        train_rel = y.get('train', 'images/train')
        val_rel = y.get('val', 'images/val')
        trains.append(str((ROOT / base / train_rel).resolve()))
        vals.append(str((ROOT / base / val_rel).resolve()))
        if names is None:
            # Prefer a consistent global ordering for class names
            # Fall back to YAML-provided names if needed
            names_from_yaml = y.get('names')
            if isinstance(names_from_yaml, list) and len(names_from_yaml) == N_LOC:
                names = names_from_yaml
            else:
                names = LOCATION_LABELS

    if names is None:
        names = LOCATION_LABELS

    return {
        'train': trains,
        'val': vals,
        'names': names,
        'nc': len(names),
    }


def train_all_views_model(args, fold: int, planes: List[str]) -> Path:
    """Train a single YOLO model using merged datasets from all three planes."""
    print(f"\n{'='*60}")
    print(f"Training SINGLE model on ALL planes for fold {fold}")
    print(f"{'='*60}")

    data_dict = _build_multiview_data_dict(fold, planes)
    # Write to a temporary/reproducible YAML to avoid passing a dict (some versions expect a path-like)
    multiview_yaml = ROOT / "configs" / f"yolo_planes_allviews_fold{fold}.yaml"
    try:
        with open(multiview_yaml, 'w') as f:
            yaml.safe_dump(data_dict, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create multiview YAML at {multiview_yaml}: {e}")

    model = YOLO(args.model)
    results = model.train(
        data=str(multiview_yaml),
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=f"{args.name}_allviews_fold{fold}_{'-'.join(planes)}",
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
    )

    save_dir = Path(results.save_dir)
    weights_path = save_dir / 'weights' / 'best.pt'
    if not weights_path.exists():
        raise FileNotFoundError(f"Best weights not found at {weights_path}")
    print(f"✓ ALL-VIEWS model trained successfully")
    print(f"  Weights saved to: {weights_path}")
    return weights_path


def validate_plane_model(plane: str, weights_path: Path, fold: int, args) -> Dict:
    """Validate a single plane model and return metrics."""
    print(f"\n{'='*50}")
    print(f"Validating {plane.upper()} plane model")
    print(f"{'='*50}")
    
    val_script = ROOT / 'yolo_validation_planes.py'
    if not val_script.exists():
        raise FileNotFoundError(f"Validation script not found: {val_script}")
    
    # Create output directory
    out_dir = weights_path.parent.parent / 'validation_results'
    out_dir.mkdir(exist_ok=True)
    
    # Build validation command
    cmd = [
        sys.executable,
        str(val_script),
        '--weights', str(weights_path),
        '--val-fold', str(fold),
        '--batch-size', str(args.val_batch),
        '--save-csv', str(out_dir / f'{plane}_predictions.csv'),
    ]
    # Ensure the validation script uses the desired plane for a shared model
    cmd += ['--plane', plane]
    
    if args.series_limit:
        cmd += ['--series-limit', str(args.series_limit)]
    if args.max_slices:
        cmd += ['--max-slices', str(args.max_slices)]
    if args.verbose_val:
        cmd.append('--verbose')
    
    print(f"Running validation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Validation failed for {plane}:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return {'plane': plane, 'status': 'failed', 'auc': 0.0}
    
    # Parse AUCs from output
    auc = 0.0
    loc_auc = float('nan')
    for line in result.stdout.split('\n'):
        if 'Classification AUC' in line:
            try:
                auc = float(line.split(':')[1].strip())
            except (IndexError, ValueError):
                pass
        elif 'Location macro AUC' in line:
            try:
                loc_auc = float(line.split(':')[-1].strip())
            except (IndexError, ValueError):
                pass

    if not (loc_auc != loc_auc):  # check for NaN without importing math
        print(f"✓ {plane.upper()} validation completed - AUC: {auc:.4f} | Loc macro AUC: {loc_auc:.4f}")
    else:
        print(f"✓ {plane.upper()} validation completed - AUC: {auc:.4f}")

    return {
        'plane': plane,
        'status': 'success',
        'auc': auc,
        'loc_macro_auc': loc_auc if not (loc_auc != loc_auc) else None,
        'weights_path': str(weights_path),
        'predictions_csv': str(out_dir / f'{plane}_predictions.csv')
    }


def ensemble_validation(plane_results: List[Dict], fold: int, args) -> Dict:
    """Perform ensemble validation combining all plane predictions."""
    print(f"\n{'='*50}")
    print("Running ENSEMBLE validation")
    print(f"{'='*50}")
    
    # Check we have all three planes
    successful_planes = [r for r in plane_results if r['status'] == 'success']
    if len(successful_planes) < 3:
        print(f"Warning: Only {len(successful_planes)}/3 planes succeeded")
        failed_planes = [r['plane'] for r in plane_results if r['status'] == 'failed']
        print(f"Failed planes: {failed_planes}")
    
    if len(successful_planes) == 0:
        return {'status': 'failed', 'ensemble_auc': 0.0}
    
    # Load prediction CSVs and ensemble
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    ensemble_probs: Dict[str, List[float]] = {}
    true_labels: Dict[str, int] = {}
    # Accumulate per-class location probabilities across planes
    ensemble_loc_probs: Dict[str, List[List[float]]] = {}
    
    for result in successful_planes:
        plane = result['plane']
        csv_path = result['predictions_csv']

        if not Path(csv_path).exists():
            print(f"Warning: Predictions CSV not found for {plane}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            series_id = row['SeriesInstanceUID']
            prob = row['aneurysm_prob']
            true_label = row['true_label']

            if series_id not in ensemble_probs:
                ensemble_probs[series_id] = []
                true_labels[series_id] = true_label

            ensemble_probs[series_id].append(prob)
            # Collect per-class location probabilities if present
            loc_vec = []
            has_loc = True
            for i in range(N_LOC):
                col = f'loc_prob_{i}'
                if col in df.columns:
                    loc_vec.append(float(row[col]))
                else:
                    has_loc = False
                    break
            if has_loc:
                ensemble_loc_probs.setdefault(series_id, []).append(loc_vec)
    
    # Average predictions across planes
    final_probs: List[float] = []
    final_labels: List[int] = []
    final_loc_probs: Dict[str, List[float]] = {}
    
    for series_id in ensemble_probs:
        if len(ensemble_probs[series_id]) > 0:  # At least one plane prediction
            avg_prob = float(np.mean(ensemble_probs[series_id]))
            final_probs.append(avg_prob)
            final_labels.append(true_labels[series_id])
            # Average location probs across planes for this series if available
            if series_id in ensemble_loc_probs and len(ensemble_loc_probs[series_id]) > 0:
                arr = np.asarray(ensemble_loc_probs[series_id], dtype=float)  # shape [num_planes, N_LOC]
                final_loc_probs[series_id] = list(np.nanmean(arr, axis=0))
    
    if len(final_probs) == 0:
        return {'status': 'failed', 'ensemble_auc': 0.0}
    
    # Calculate ensemble AUC
    try:
        ensemble_auc = roc_auc_score(final_labels, final_probs)
    except ValueError as e:
        print(f"Failed to calculate ensemble AUC: {e}")
        ensemble_auc = 0.0

    # Calculate ensemble location macro AUC and partial AUC if possible
    loc_macro_auc = float('nan')
    if final_loc_probs:
        # Build ground-truth location labels for these series
        try:
            import pandas as pd
            data_root = Path(data_path) if data_path else ROOT / 'data'
            train_csv = data_root / 'train_df.csv'
            if not train_csv.exists():
                train_csv = data_root / 'train.csv'
            train_df = pd.read_csv(train_csv)
            # Map series -> label vector
            gt_loc_map: Dict[str, List[float]] = {}
            df_idx = train_df.set_index('SeriesInstanceUID')
            for sid in final_loc_probs.keys():
                if sid not in df_idx.index:
                    continue
                row = df_idx.loc[sid]
                vec = []
                for name in LOCATION_LABELS:
                    try:
                        vec.append(float(row.get(name, 0.0)))
                    except Exception:
                        vec.append(0.0)
                gt_loc_map[sid] = vec
            # Align predictions and labels
            y_loc_true = []
            y_loc_pred = []
            for sid, pred_vec in final_loc_probs.items():
                if sid in gt_loc_map:
                    y_loc_true.append(gt_loc_map[sid])
                    y_loc_pred.append(pred_vec)
            if y_loc_true:
                y_loc_true = np.asarray(y_loc_true, dtype=float)
                y_loc_pred = np.asarray(y_loc_pred, dtype=float)
                per_loc_aucs = []
                for i in range(N_LOC):
                    try:
                        per_loc_aucs.append(roc_auc_score(y_loc_true[:, i], y_loc_pred[:, i]))
                    except ValueError:
                        per_loc_aucs.append(float('nan'))
                loc_macro_auc = float(np.nanmean(per_loc_aucs))
        except Exception as e:
            print(f"Failed to compute ensemble location AUCs: {e}")
    
    # Save ensemble predictions
    rows = []
    for sid, probs in ensemble_probs.items():
        row = {
            'SeriesInstanceUID': sid,
            'ensemble_prob': float(np.mean(probs)) if probs else 0.0,
            'true_label': true_labels.get(sid, None),
            'num_planes': len(probs)
        }
        # attach loc probs if available
        if sid in final_loc_probs:
            for i in range(N_LOC):
                row[f'loc_prob_{i}'] = float(final_loc_probs[sid][i])
        rows.append(row)
    ensemble_df = pd.DataFrame(rows)
    
    # Save to first successful model's directory
    out_path = Path(successful_planes[0]['weights_path']).parent.parent / 'validation_results' / 'ensemble_predictions.csv'
    ensemble_df.to_csv(out_path, index=False)
    
    print(f"✓ Ensemble validation completed - AUC: {ensemble_auc:.4f}")
    if not np.isnan(loc_macro_auc):
        print(f"  Ensemble Location macro AUC: {loc_macro_auc:.4f}")
    print(f"  Predictions saved to: {out_path}")
    print(f"  Used {len(successful_planes)} planes: {[r['plane'] for r in successful_planes]}")
    
    return {
        'status': 'success',
        'ensemble_auc': ensemble_auc,
    'ensemble_loc_macro_auc': loc_macro_auc,
        'num_planes': len(successful_planes),
        'planes_used': [r['plane'] for r in successful_planes],
        'predictions_csv': str(out_path)
    }


def main():
    args = parse_args()
    fold = args.fold
    planes = [p.strip().lower() for p in args.planes.split(',') if p.strip()]
    planes = [p for p in planes if p in ['axial', 'coronal', 'sagittal']]
    if not planes:
        planes = ['axial', 'coronal', 'sagittal']
    
    print(f"YOLO Plane Models Pipeline - Fold {fold}")
    print(f"Model: {args.model}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    
    trained_models: Dict[str, Path | None] = {}
    validation_results: List[Dict] = []

    if args.multi_view:
        # Train a single model on all planes, then validate that model across each plane
        try:
            shared_weights = train_all_views_model(args, fold, planes)
        except Exception as e:
            print(f"✗ Failed to train ALL-VIEWS model: {e}")
            shared_weights = None

        for plane in planes:
            if shared_weights is not None:
                try:
                    result = validate_plane_model(plane, shared_weights, fold, args)
                    validation_results.append(result)
                except Exception as e:
                    print(f"✗ Failed to validate {plane} with ALL-VIEWS model: {e}")
                    validation_results.append({'plane': plane, 'status': 'failed', 'auc': 0.0})
            else:
                validation_results.append({'plane': plane, 'status': 'failed', 'auc': 0.0})
    else:
        # Original behavior: train and validate per plane
        for plane in planes:
            try:
                weights_path = train_plane_model(plane, args, fold)
                trained_models[plane] = weights_path
            except Exception as e:
                print(f"✗ Failed to train {plane} model: {e}")
                trained_models[plane] = None

        for plane in planes:
            if trained_models.get(plane) is not None:
                try:
                    result = validate_plane_model(plane, trained_models[plane], fold, args)
                    validation_results.append(result)
                except Exception as e:
                    print(f"✗ Failed to validate {plane} model: {e}")
                    validation_results.append({'plane': plane, 'status': 'failed', 'auc': 0.0})
            else:
                validation_results.append({'plane': plane, 'status': 'failed', 'auc': 0.0})
    
    # Step 3: Ensemble validation
    try:
        ensemble_result = ensemble_validation(validation_results, fold, args)
    except Exception as e:
        print(f"✗ Failed ensemble validation: {e}")
        ensemble_result = {'status': 'failed', 'ensemble_auc': 0.0}
    
    # Step 4: Summary report
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    
    individual_aucs = []
    for result in validation_results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        auc = result.get('auc', 0.0)
        if result['status'] == 'success':
            individual_aucs.append(auc)
        print(f"{status_icon} {result['plane'].upper():8} AUC: {auc:.4f}")
    
    if individual_aucs:
        import numpy as np
        mean_individual = np.mean(individual_aucs)
        print(f"  Mean Individual AUC: {mean_individual:.4f}")
    
    ensemble_auc = ensemble_result.get('ensemble_auc', 0.0)
    status_icon = "✓" if ensemble_result['status'] == 'success' else "✗"
    print(f"{status_icon} ENSEMBLE AUC: {ensemble_auc:.4f}")
    
    # Save summary
    import numpy as np
    summary = {
        'fold': fold,
        'individual_results': validation_results,
        'ensemble_result': ensemble_result,
        'mean_individual_auc': np.mean(individual_aucs) if individual_aucs else 0.0,
        'successful_planes': len([r for r in validation_results if r['status'] == 'success'])
    }
    
    # Save to first successful model's directory, or current directory if none succeeded
    summary_path = Path.cwd() / f'planes_summary_fold{fold}_multi_view.json'
    if validation_results and any(r['status'] == 'success' for r in validation_results):
        first_success = next(r for r in validation_results if r['status'] == 'success')
        if 'weights_path' in first_success:
            summary_dir = Path(first_success['weights_path']).parent.parent
            summary_path = summary_dir / f'planes_summary_fold{fold}.json'
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print("Pipeline completed!")


if __name__ == '__main__':
    main()