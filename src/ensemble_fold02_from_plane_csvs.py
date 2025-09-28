from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

# Ensure project root is on sys.path for absolute imports when running as a file
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble_from_plane_csvs import (  # type: ignore
    load_csvs,
    ensemble_from_dfs,
    compute_csv_metrics,
    detect_plane_from_path,
)


def main() -> None:
    # Absolute input CSV paths provided by the user
    input_csvs = [
        Path(
            "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11_efficientnet_v2_b0_update_fold4/series_validation/fold_4/per_series_predictions.csv"
        ),
        Path(
            "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11_yolo_mobile_net_fold4/series_validation/fold_4/per_series_predictions.csv"
        ),
        Path(
            "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11_yolo11m_new_fit_fold4/series_validation/fold_4/per_series_predictions.csv"
        ),
    ]

    # Output locations (create if needed)
    out_dir = Path(
        "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/ensemble_fold02/series_validation/fold_0"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ensemble_predictions.csv"
    out_json = out_dir / "ensemble_summary.json"

    # Load available CSVs
    items = load_csvs(input_csvs)
    if len(items) == 0:
        print("No valid CSVs found. Exiting.")
        return

    # Per-input summary and metrics
    print("Per-input metrics:")
    per_input_summary = []
    for p, df in items:
        plane = detect_plane_from_path(p)
        cls_auc_i, loc_auc_i = compute_csv_metrics(df)
        n_series = int(df.shape[0])
        cls_str = (
            f"{cls_auc_i:.6f}" if isinstance(cls_auc_i, (int, float)) and not np.isnan(cls_auc_i) else "nan"
        )
        loc_str = (
            f"{loc_auc_i:.6f}" if isinstance(loc_auc_i, (int, float)) and not np.isnan(loc_auc_i) else "nan"
        )
        print(f"  - {plane}: n={n_series}, cls_auc={cls_str}, loc_macro_auc={loc_str}")
        per_input_summary.append(
            {
                "plane": plane,
                "path": str(p),
                "n_series": n_series,
                "cls_auc": float(cls_auc_i) if not np.isnan(cls_auc_i) else float("nan"),
                "loc_macro_auc": float(loc_auc_i) if not np.isnan(loc_auc_i) else float("nan"),
            }
        )

    # Ensemble
    dfs = [df for _, df in items]
    out_df, cls_auc, loc_macro_auc = ensemble_from_dfs(dfs)
    out_df.to_csv(out_csv, index=False)

    print(f"Saved ensemble predictions to: {out_csv}")
    if not np.isnan(cls_auc):
        print(f"Ensemble Classification AUC: {cls_auc:.6f}")
    else:
        print("Ensemble Classification AUC: nan")
    if not np.isnan(loc_macro_auc):
        print(f"Ensemble Location macro AUC: {loc_macro_auc:.6f}")
    else:
        print("Ensemble Location macro AUC: nan")

    # Save summary JSON
    summary = {
        "num_inputs": len(dfs),
        "out_csv": str(out_csv),
        "cls_auc": float(cls_auc) if not np.isnan(cls_auc) else float("nan"),
        "loc_macro_auc": float(loc_macro_auc) if not np.isnan(loc_macro_auc) else float("nan"),
        "num_series": int(out_df.shape[0]),
        "per_inputs": per_input_summary,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to: {out_json}")


if __name__ == "__main__":
    main()


