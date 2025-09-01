import os.path
import sys

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from parse_preprocess import *
import torch
from torch_cluster import knn_graph
from typing import Dict, List, Tuple, Set

# Optimization settings
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

root = Path('./src/data')

def set_seed(seed: int = 42):
    # Python built-in RNG
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_models():
    """Load all models on single GPU (cuda:0)"""
    models = []
    for config in MODEL_CONFIGS:
        print(f"Loading model: {config['name']} on cuda:0")

        model = YOLO(config["path"])
        model.to("cuda:0")

        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"]
        }
        models.append(model_dict)
    return models




@torch.no_grad()
def eval_one_series(slices, loc, models, uid):
    set_seed()
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    ensemble_locations = []
    all_features = []
    total_weight = 0.0

    for model_idx, model_dict in enumerate(models):
        model = model_dict["model"]
        weight = model_dict["weight"]

        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(LABELS), dtype=np.float32)

            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i + BATCH_SIZE]
                z_idxes = [i + batch_idx for batch_idx in range(len(batch_slices))]

                features = get_feature_map(model)
                results = model.predict(
                    batch_slices,
                    verbose=False,
                    batch=len(batch_slices),
                    device="cuda:0",
                    conf=conf_yolo
                )
                c3k2_feat = features['C3K2'].cpu().numpy()
                all_features.append(c3k2_feat)

                for z_idx, r in enumerate(results):
                    if r is None or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                        continue
                    try:
                        confs = r.boxes.conf
                        clses = r.boxes.cls
                        for j in range(len(confs)):
                            c = float(confs[j].item())
                            k = int(clses[j].item())
                            if c > max_conf_all:
                                max_conf_all = c
                            if 0 <= k < len(LABELS) and c > per_class_max[k]:
                                per_class_max[k] = c
                            x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2

                            ensemble_locations.append([round(z_idxes[z_idx]),  # depth
                                                       round(y_center),  # height
                                                       round(x_center),  # width
                                                       float(c),  # confidence
                                                       k,  # class
                                                       model_idx,
                                                       ])

                    except Exception as e:
                        print(e)
                        try:
                            batch_max = float(r.boxes.conf.max().item())
                            if batch_max > max_conf_all:
                                max_conf_all = batch_max
                        except Exception:
                            pass

            ensemble_cls_preds.append(max_conf_all * weight)
            ensemble_loc_preds.append(per_class_max * weight)
            total_weight += weight

        except Exception as e:
            print(f"Error in model {model_dict['name']}: {e}")
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(LABELS)) * 0.1 * weight)
            total_weight += weight
    if len(ensemble_locations) == 0:
        return

    all_detections = np.array(ensemble_locations)
    all_locations = all_detections[:, :3]
    #all_preds = all_detections[:, 3:-1]
    #all_model_idx = all_detections[:, -1][:, None]
    all_features = np.concatenate(all_features, axis=0)
    vol_size = (len(slices), slices[0].shape[0], slices[0].shape[1])
    points, extract_feat, dist_label = extract_tomo(all_locations, all_features, vol_size, loc)

    # knn graph
    edge_index_k5 = knn_graph(torch.from_numpy(points), k=5, loop=False)
    edge_index_k10 = knn_graph(torch.from_numpy(points), k=10, loop=False)
    edge_index_k15 = knn_graph(torch.from_numpy(points), k=15, loop=False)

    np.save(root/f'extract_data/{uid}/{uid}_edge_index_k5.npy', edge_index_k5)
    np.save(root/f'extract_data/{uid}/{uid}_edge_index_k10.npy', edge_index_k10)
    np.save(root/f'extract_data/{uid}/{uid}_edge_index_k15.npy', edge_index_k15)

    # delaunay_graph
    edge_index_del = delaunay_graph(torch.from_numpy(points))
    np.save(root/f'extract_data/{uid}/{uid}_edge_index_delaunay.npy', edge_index_del)

    print('del edges:', edge_index_del.shape)
    print(points.shape, extract_feat.shape, dist_label.shape)
    print('label sum:', dist_label.sum())

    np.save(root/f'extract_data/{uid}/{uid}_points.npy', points)
    np.save(root/f'extract_data/{uid}/{uid}_extract_feat.npy', extract_feat)
    np.save(root/f'extract_data/{uid}/{uid}_label.npy', dist_label)

    del all_features
    del all_detections
    del ensemble_locations
    del ensemble_cls_preds
    del ensemble_loc_preds

    gc.collect()


def load_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in label_df.columns or "y" not in label_df.columns:
        label_df["x"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        label_df["y"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    # Standardize dtypes
    label_df["SeriesInstanceUID"] = label_df["SeriesInstanceUID"].astype(str)
    label_df["SOPInstanceUID"] = label_df["SOPInstanceUID"].astype(str)
    return label_df


def load_slices(series_path: Path):
    """Load and consistently sort DICOM slices and filenames by orientation+position"""
    series_path = Path(series_path)
    dicom_files = collect_series_slices(series_path)

    # Sort DICOM files by orientation+position before processing
    dicom_files.sort(key=slice_sort_key)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_dicom_file, dicom_files))

    # Flatten into (loc, img)
    all_slices_with_loc = [item for sublist in results for item in sublist]

    # Already sorted by dicom_files order, but double-check (safe)
    all_slices_with_loc.sort(key=lambda x: x[0])

    # Extract just the images
    all_slices = [img for _, img in all_slices_with_loc]

    # Now dicom_files matches the sorted slices
    dcm_list = [f.stem for f in dicom_files]
    return all_slices, dcm_list


def main():
    ignore_uids: Set[str] = set(
        [
            "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
            "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
            "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242",
        ]
    )
    mf_path = root / "multiframe_dicoms.csv"
    if mf_path.exists():
        try:
            mf_df = pd.read_csv(mf_path)
            if "SeriesInstanceUID" in mf_df.columns:
                ignore_uids.update(mf_df["SeriesInstanceUID"].astype(str).tolist())
        except Exception:
            pass

    models = load_models()
    print(f"Loaded {len(models)} models on single GPU")

    label_df = load_labels(root)
    train_df = pd.read_csv(root/'train_df.csv')

    pos_uids = label_df[~label_df.SeriesInstanceUID.isin(ignore_uids)].SeriesInstanceUID.unique().tolist()
    neg_uids = train_df[train_df['Aneurysm Present']==0].SeriesInstanceUID.unique().tolist()
    uids = pos_uids + neg_uids
    print(f'#pos: {len(pos_uids)}--#neg:{len(neg_uids)}--#total:{len(uids)}')

    if not os.path.exists(root/'extract_data'):
        os.makedirs(root/'extract_data')

    for uid in tqdm(uids):
        print(uid)
        if not os.path.exists(root /f'extract_data/{uid}'):
            os.makedirs(root / f'extract_data/{uid}')
        else:
            continue
        all_slices, dcm_list = load_slices(root / f'series/{uid}')
        loc = label_df[label_df.SeriesInstanceUID == uid][['y', 'x']].values
        if len(loc)!=0:
            z = label_df[label_df.SeriesInstanceUID == uid].SOPInstanceUID.map(lambda x: dcm_list.index(x)).values
            loc = np.concatenate([z[:, None], loc], axis=-1)
        eval_one_series(all_slices, loc, models, uid)

if __name__ == '__main__':
    main()