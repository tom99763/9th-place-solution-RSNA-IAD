import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import roc_auc_score
from typing import List, Optional
from scipy.optimize import minimize
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from lightgbm import early_stopping
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from xgboost.callback import EarlyStopping
import argparse

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation'
]

# YOLO label mappings
YOLO_LABELS_TO_IDX = {
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
    'Right Supraclinoid Internal Carotid Artery': 12
}

YOLO_INV_MAP = {v: k for k, v in YOLO_LABELS_TO_IDX.items()}

YOLO_LABELS = sorted(list(YOLO_LABELS_TO_IDX.keys()))

loc_cols = [f"loc_prob_{i}" for i in range(13)]

N = 14

xgb.set_config(verbosity=0)

def get_yolo_weight_path(args):
    all_folds_yolo11m = {
        "fold0": {
            "y11_yolo11m": f"{args.yolo_weight_path}/yolo-11m-2.5D_fold0",
        },
        "fold1": {
            "y11_yolo11m": f"{args.yolo_weight_path}yolo-11m-2.5D_fold1",
        },
        "fold2": {
            "y11_yolo11m": f"{args.yolo_weight_path}/yolo-11m-2.5D_fold22",
        },
        "fold3": {
            "y11_yolo11m": f"{args.yolo_weight_path}/yolo-11m-2.5D_fold3",
        },
        "fold4": {
            "y11_yolo11m": f"{args.yolo_weight_path}/yolo-11m-2.5D_fold4",
        },
    }

    all_folds_yolo_eff2s = {
        "fold0": {
            "y11_effnetv_25d": f"{args.yolo_weight_path}/cv_effnetv2_s_drop_path_25d_fold0",
        },
        "fold1": {
            "y11_effnetv_25d": f"{args.yolo_weight_path}/cv_effnetv2_s_drop_path_25d_fold1",
        },
        "fold2": {
            "y11_effnetv_25d": f"{args.yolo_weight_path}/cv_effnetv2_s_drop_path_25d_fold2",
        },
        "fold3": {
            "y11_effnetv_25d": f"{args.yolo_weight_path}/cv_effnetv2_s_drop_path_25d_fold3",
        },
        "fold4": {
            "y11_effnetv_25d": f"{args.yolo_weight_path}/cv_effnetv2_s_drop_path_25d_fold4",
        },
    }
    return all_folds_yolo11m, all_folds_yolo_eff2s


def get_yolo_oof_preds(all_folds):
    results = {}
    for fold, models in all_folds.items():
        fold_results = {}
        for model_name, base in models.items():
            matches = glob.glob(f"{base}/**/per_series_predictions.csv", recursive=True)
            fold_results[model_name] = matches[0] if matches else None
        results[fold] = fold_results

    dfs = []
    for fold, models in results.items():
        for model_name, csv_path in models.items():
            if csv_path:
                df = pd.read_csv(csv_path)
                df["fold_id"] = fold
                df["model"] = model_name
                dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def weighted_multilabel_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_weights: Optional[List[float]] = None,
) -> float:
    """Compute weighted AUC for multilabel classification.

    Parameters:
    -----------
    y_true : np.ndarray of shape (n_samples, n_classes)
        True binary labels (0 or 1) for each class
    y_scores : np.ndarray of shape (n_samples, n_classes)
        Target scores (probability estimates or decision values)
    class_weights : array-like of shape (n_classes,), optional
        Weights for each class. If None, uniform weights are used.
        Weights will be normalized to sum to 1.

    Returns:
    --------
    weighted_auc : float
        The weighted average AUC

    Raises:
    -------
    ValueError
        If any class does not have both positive and negative samples
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_true.shape[1]

    individual_aucs = roc_auc_score(y_true, y_scores, average=None)

    if class_weights is None:  # Uniform weights
        weights_array = np.ones(n_classes)
    else:
        weights_array = np.asarray(class_weights)

    if len(weights_array) != n_classes:
        raise ValueError(
            f'Number of weights ({len(weights_array)}) must match '
            f'number of classes ({n_classes})'
        )

    # Check for non-negative weights
    if np.any(weights_array < 0):
        raise ValueError('All class weights must be non-negative')

    # Check that at least one weight is positive
    if np.sum(weights_array) == 0:
        raise ValueError('At least one class weight must be positive')

    # Normalize weights to sum to 1
    weights_array = weights_array / np.sum(weights_array)

    # Compute weighted average
    return np.sum(individual_aucs * weights_array)

def train_meta_multilabel_lgb(df, df_main, args, feature_cols):
    """
    Multi-label LightGBM training: trains one binary classifier per label.
    Computes OOF predictions and per-label AUROC.
    """
    features = feature_cols
    labels = LABEL_COLS + ["Aneurysm Present"]
    X = df[features]
    folds = df["fold_id"]

    oof_preds = pd.DataFrame(index=df.index, columns=labels, dtype=float)
    feature_importances = pd.DataFrame()
    feature_importances["feature"] = features

    unique_folds = folds.unique()
    best_iterations = []

    for label in labels:
        print(f"\n=== Training for label: {label} ===")
        oof_preds_label = np.zeros(len(df))
        for fold in unique_folds:
            print(f"\n--- Fold {fold} ---")
            train_idx = df[df["fold_id"] != fold].index
            valid_idx = df[df["fold_id"] == fold].index

            X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = df_main.loc[train_idx, label], df_main.loc[valid_idx, label]

            model = lgb.LGBMClassifier(
                objective="binary",
                metric="auc",
                max_depth=2,
                learning_rate=0.02,
                reg_lambda=0.0047,
                n_estimators=2000,
                random_state=0,
                verbose=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                callbacks=[early_stopping(stopping_rounds=300, verbose=True)]
            )

            oof_preds_label[valid_idx] = model.predict_proba(X_valid)[:, 1]
            feature_importances[f"{label}_fold_{fold}"] = model.feature_importances_
            joblib.dump(model, f"{args.meta_cls_weight_path}/lgb/meta_classifier_{label}_fold_{fold}.pkl")
            best_iterations.append(model.best_iteration_)

        oof_preds[label] = oof_preds_label
        label_auroc = roc_auc_score(df_main[label[:]], oof_preds_label)
        print(f"=== OOF AUROC for {label}: {label_auroc:.4f} ===")
    cls_score = weighted_multilabel_auc(df_main["Aneurysm Present"].values[:, None], oof_preds["Aneurysm Present"].values[:, None])
    loc_score = weighted_multilabel_auc(df_main[LABEL_COLS].values, oof_preds[LABEL_COLS].values)
    score = 0.5 * cls_score + 0.5 * loc_score
    print(f'lgbm kaggle score: {score}')

    # Average feature importance across all folds/labels
    fi_cols = [col for col in feature_importances.columns if col != "feature"]
    feature_importances["average"] = feature_importances[fi_cols].mean(axis=1)
    feature_importances = feature_importances.sort_values(by="average", ascending=False)
    return oof_preds, feature_importances


def train_meta_multilabel_xgb(df, df_main, args, feature_cols):
    """
    Multi-label XGBoost training: trains one binary classifier per label.
    Computes OOF predictions and per-label AUROC.
    """
    features = feature_cols
    labels = LABEL_COLS + ["Aneurysm Present"]
    X = df[features]
    folds = df["fold_id"]

    oof_preds = pd.DataFrame(index=df.index, columns=labels, dtype=float)
    feature_importances = pd.DataFrame()
    feature_importances["feature"] = features

    unique_folds = folds.unique()
    best_iterations = []

    for label in labels:
        print(f"\n=== Training for label: {label} ===")
        oof_preds_label = np.zeros(len(df))

        for fold in unique_folds:
            print(f"\n--- Fold {fold} ---")
            train_idx = df[df["fold_id"] != fold].index
            valid_idx = df[df["fold_id"] == fold].index

            X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = df_main.loc[train_idx, label], df_main.loc[valid_idx, label]

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                max_depth=2,
                learning_rate=0.02,
                reg_lambda=0.0047,
                n_estimators=2000,
                random_state=0,
                #tree_method="hist",       # faster for large datasets
                n_jobs=-1,
                verbosity=0
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[EarlyStopping(rounds=300, save_best=True)],
                verbose = False
            )

            # predict using the best iteration
            oof_preds_label[valid_idx] = model.predict_proba(X_valid, iteration_range=(0, model.best_iteration + 1))[:, 1]

            # Save feature importance for each fold
            feature_importances[f"{label}_fold_{fold}"] = model.feature_importances_

            # Save model
            joblib.dump(model, f"{args.meta_cls_weight_path}/xgb/meta_classifier_{label}_fold_{fold}.pkl")
            best_iterations.append(model.best_iteration)

        oof_preds[label] = oof_preds_label
        label_auroc = roc_auc_score(df_main[label], oof_preds_label)
        print(f"=== OOF AUROC for {label}: {label_auroc:.4f} ===")

    # Compute Kaggle-like score
    cls_score = weighted_multilabel_auc(
        df_main["Aneurysm Present"].values[:, None],
        oof_preds["Aneurysm Present"].values[:, None]
    )
    loc_score = weighted_multilabel_auc(
        df_main[LABEL_COLS].values,
        oof_preds[LABEL_COLS].values
    )
    score = 0.5 * cls_score + 0.5 * loc_score
    print(f'xgboost kaggle score: {score:.6f}')

    # Average feature importance across all folds/labels
    fi_cols = [col for col in feature_importances.columns if col != "feature"]
    feature_importances["average"] = feature_importances[fi_cols].mean(axis=1)
    feature_importances = feature_importances.sort_values(by="average", ascending=False)

    return oof_preds, feature_importances

def train_meta_multilabel_catboost(df, df_main, args, feature_cols):
    """
    Multi-label CatBoost training: trains one binary classifier per label.
    Computes OOF predictions and per-label AUROC.
    """
    features = feature_cols
    labels = LABEL_COLS + ["Aneurysm Present"]
    X = df[features]
    folds = df["fold_id"]

    oof_preds = pd.DataFrame(index=df.index, columns=labels, dtype=float)
    feature_importances = pd.DataFrame()
    feature_importances["feature"] = features

    unique_folds = folds.unique()
    best_iterations = []

    # Shared CatBoost parameters
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'depth': 6,
        'learning_rate': 0.02,
        'l2_leaf_reg': 4.7,
        'iterations': 2000,
        'random_seed': 0,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'od_type': 'Iter',
        'od_wait': 300,
        'verbose': False
    }

    for label in labels:
        print(f"\n=== Training for label: {label} ===")
        oof_preds_label = np.zeros(len(df))

        for fold in unique_folds:
            print(f"\n--- Fold {fold} ---")
            train_idx = df[df["fold_id"] != fold].index
            valid_idx = df[df["fold_id"] == fold].index

            X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = df_main.loc[train_idx, label], df_main.loc[valid_idx, label]

            train_pool = Pool(X_train, y_train)
            valid_pool = Pool(X_valid, y_valid)

            model = CatBoostClassifier(**params)

            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True
            )

            oof_preds_label[valid_idx] = model.predict_proba(X_valid)[:, 1]
            feature_importances[f"{label}_fold_{fold}"] = model.get_feature_importance(train_pool)

            joblib.dump(model, f"{args.meta_cls_weight_path}/cat/meta_classifier_{label}_fold_{fold}.pkl")
            best_iterations.append(model.get_best_iteration())

        oof_preds[label] = oof_preds_label
        label_auroc = roc_auc_score(df_main[label], oof_preds_label)
        print(f"=== OOF AUROC for {label}: {label_auroc:.4f} ===")

    # Combine aneurysm/global score
    cls_score = weighted_multilabel_auc(
        df_main["Aneurysm Present"].values[:, None],
        oof_preds["Aneurysm Present"].values[:, None]
    )
    loc_score = weighted_multilabel_auc(
        df_main[LABEL_COLS].values,
        oof_preds[LABEL_COLS].values
    )
    score = 0.5 * cls_score + 0.5 * loc_score
    print(f'ccatboost kaggle score: {score:.6f}')

    # Average feature importance across all folds and labels
    fi_cols = [col for col in feature_importances.columns if col != "feature"]
    feature_importances["average"] = feature_importances[fi_cols].mean(axis=1)
    feature_importances = feature_importances.sort_values(by="average", ascending=False)

    return oof_preds, feature_importances

def compute_score(p_cls, p_loc):
    return 0.5 * (p_cls + p_loc)

def train():
    args = parse_args()
    if not os.path.exists(args.meta_cls_weight_path):
        os.mkdir(args.meta_cls_weight_path)

    for name in ['lgb', 'xgb', 'cat']:
        if not os.path.exists(f'{args.meta_cls_weight_path}/{name}'):
            os.mkdir(f'{args.meta_cls_weight_path}/{name}')

    df_main = pd.read_csv(f'{args.data_path}/train.csv')
    # Load data
    df_flayer = pd.read_csv(f'{args.flayer_weight_path}/oof_df_cv7722_seg_aux.csv')
    df_meta = df_main.copy()[['SeriesInstanceUID', 'PatientAge', 'PatientSex']]

    # Convert age to numeric
    df_meta['PatientAge'] = df_meta['PatientAge'].astype('float32')

    # Encode PatientSex (e.g. M → 1, F → 0)
    le = LabelEncoder()
    df_meta['PatientSex'] = le.fit_transform(df_meta['PatientSex'].astype(str))

    # Save the LabelEncoder
    with open(f'{args.meta_cls_weight_path}/label_encoder_sex.pkl', 'wb') as f:
        pickle.dump(le, f)

    all_folds_yolo11m, all_folds_yolo_eff2s = get_yolo_weight_path(args)

    all_df_yolo11m = get_yolo_oof_preds(all_folds_yolo11m)
    all_df_yolo11m.rename(columns={loc_col: f'yolo11m_{loc_col}' for loc_col in loc_cols}, inplace=True)
    all_df_yolo11m.rename(columns={'aneurysm_prob': 'yolo11m_aneurysm_prob'}, inplace=True)

    all_df_yolo_eff2s = get_yolo_oof_preds(all_folds_yolo_eff2s)
    all_df_yolo_eff2s.rename(columns={loc_col: f'yolo_eff2s_{loc_col}' for loc_col in loc_cols}, inplace=True)
    all_df_yolo_eff2s.rename(columns={'aneurysm_prob': 'yolo_eff2s_aneurysm_prob'}, inplace=True)

    label_to_yolo_idx = {label: YOLO_LABELS_TO_IDX[label] for label in LABEL_COLS}

    yolo11m_loc_cols = [f'yolo11m_{loc_col}' for loc_col in loc_cols]
    yolo_eff2s_loc_cols = [f'yolo_eff2s_{loc_col}' for loc_col in loc_cols]
    yolo11m_loc_col_map = [f"yolo11m_loc_prob_{label_to_yolo_idx[label]}" for label in LABEL_COLS]
    yolo_eff2s_loc_col_map = [f"yolo_eff2s_loc_prob_{label_to_yolo_idx[label]}" for label in LABEL_COLS]
    feature_cols = ['yolo11m_aneurysm_prob'] + yolo11m_loc_cols + [
        'yolo_eff2s_aneurysm_prob'] + yolo_eff2s_loc_cols + LABEL_COLS + ['Aneurysm Present'] + ['PatientAge',
                                                                                                 'PatientSex']

    merged = pd.merge(all_df_yolo11m, all_df_yolo_eff2s, on="SeriesInstanceUID", how="inner")
    merged = pd.merge(merged, df_flayer, on="SeriesInstanceUID", how="inner")
    merged = pd.merge(merged, df_meta, on="SeriesInstanceUID", how="inner")
    merged['fold_id'] = all_df_yolo11m['fold_id'].copy()
    df_main = df_main.set_index('SeriesInstanceUID').loc[merged['SeriesInstanceUID']].reset_index()

    oof_preds_lgb, feature_importances_lgb = train_meta_multilabel_lgb(merged, df_main, args, feature_cols )
    oof_preds_xgb, feature_importances_xgb = train_meta_multilabel_xgb(merged, df_main, args, feature_cols )
    oof_preds_cat, feature_importances_cat = train_meta_multilabel_catboost(merged, df_main, args, feature_cols )

    #compute cv
    gt_loc = df_main[LABEL_COLS]
    pred_loc = oof_preds_lgb[LABEL_COLS] + oof_preds_xgb[LABEL_COLS] + oof_preds_cat[LABEL_COLS] + merged[LABEL_COLS] + \
               merged[yolo11m_loc_col_map].values + merged[yolo_eff2s_loc_col_map].values
    score_loc = weighted_multilabel_auc(gt_loc.values, pred_loc.values)

    gt_cls = df_main['Aneurysm Present']
    pred_cls = oof_preds_lgb['Aneurysm Present'] + oof_preds_xgb['Aneurysm Present'] + oof_preds_cat[
        'Aneurysm Present'] + merged['Aneurysm Present'] + merged['yolo11m_aneurysm_prob'] + merged[
                   'yolo_eff2s_aneurysm_prob']
    score_cls = weighted_multilabel_auc(gt_cls.values[:, None], pred_cls.values[:, None])
    score = compute_score(score_cls, score_loc)
    print('official score of ensemble:', score)

def parse_args():
    ap = argparse.ArgumentParser(description='Train and validate meta classifier pipeline')
    ap.add_argument('--data_path', type=str, default='./', help='path to acces train.csv')
    ap.add_argument('--meta_cls_weight_path', type=str, default='./meta_classifiers')
    ap.add_argument('--yolo_weight_path', type=str, default='./yolo25d/yolo_aneurysm_locations')
    ap.add_argument('--flayer_weight_path', type=str, default='./flayer/flayer_weights')
    return ap.parse_args()

if __name__ == '__main__':
    train()


