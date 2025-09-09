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
    total_weight = 0.0
    vol_size = (len(slices), slices[0].shape[0], slices[0].shape[1])

    for model_idx, model_dict in enumerate(models):
        all_detections, all_feat_maps = [], []
        model, weight = model_dict["model"], model_dict["weight"]

        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(LABELS), dtype=np.float32)

            # Process slices in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i + BATCH_SIZE]
                z_idxes = [i + batch_idx for batch_idx in range(len(batch_slices))]

                features = get_feature_map(model)
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    results = model.predict(
                        batch_slices,
                        verbose=False,
                        batch=len(batch_slices),
                        device="cuda:0",
                        conf=conf_yolo
                    )

                c3k2_feat = features['C3K2']
                all_feat_maps.append(c3k2_feat.cpu())

                # Collect all valid detections first
                for z_idx, r in enumerate(results):
                    if r is None or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                        continue
                    try:
                        confs, clses = r.boxes.conf, r.boxes.cls
                        for j in range(len(confs)):
                            c, k = float(confs[j].item()), int(clses[j].item())
                            if c > max_conf_all:
                                max_conf_all = c
                            if 0 <= k < len(LABELS) and c > per_class_max[k]:
                                per_class_max[k] = c
                            x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
                            point = np.array([round(z_idxes[z_idx]), round(y_center), round(x_center)])
                            all_detections.append([*point, float(c), k, model_idx])
                    except Exception:
                        try:
                            batch_max = float(r.boxes.conf.max().item())
                            if batch_max > max_conf_all:
                                max_conf_all = batch_max
                        except Exception:
                            pass
            if len(all_detections) != 0:
                all_detections = torch.tensor(all_detections)
                all_locations = all_detections[:, :3]
                all_feat_maps = torch.cat(all_feat_maps, dim=0)

                # Assign labels for each sampled point
                points, extract_feat, dist_label = extract_tomo(all_locations.numpy(), all_feat_maps.numpy(),
                                                           vol_size, loc)

                # Save points, features, labels
                np.save(root / f'extract_data/fold{model_idx}/{uid}/{uid}_points_fold.npy', points)
                np.save(root / f'extract_data/fold{model_idx}/{uid}/{uid}_extract_feat_fold.npy', extract_feat)
                np.save(root / f'extract_data/fold{model_idx}/{uid}/{uid}_label_fold.npy', dist_label)

                # KNN graphs
                for k in [5, 10, 15]:
                    edge_index = knn_graph(torch.from_numpy(points), k=k, loop=False)
                    np.save(root / f'extract_data/fold{model_idx}/{uid}/{uid}_edge_index_k{k}_fold.npy', edge_index)

                # Delaunay graph
                edge_index_del = delaunay_graph(torch.from_numpy(points))
                np.save(root / f'extract_data/fold{model_idx}/{uid}/{uid}_edge_index_delaunay_fold.npy', edge_index_del)

                print('Delaunay edges:', edge_index_del.shape)
                print('Points, features, labels:', points.shape, extract_feat.shape, dist_label.shape)
                print('Label sum:', dist_label.sum())

            else:
                print("No points were detected for any model!")

            del c3k2_feat
            del all_feat_maps, all_locations
            features.clear()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(e)

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

    for i in range(3):
        if not os.path.exists(root / f'extract_data/fold{i}'):
            os.makedirs(root / f'extract_data/fold{i}')


    for uid in tqdm(uids):
        check_exist = 0
        for i in range(len(MODEL_CONFIGS)):
            if not os.path.exists(root / f'extract_data/fold{i}/{uid}'):
                os.makedirs(root / f'extract_data/fold{i}/{uid}')
            else:
                check_exist+=1
        if check_exist == 3:
            continue
        all_slices, dcm_list = load_slices(root / f'series/{uid}')
        loc = label_df[label_df.SeriesInstanceUID == uid][['y', 'x']].values
        if len(loc)!=0:
            z = label_df[label_df.SeriesInstanceUID == uid].SOPInstanceUID.map(lambda x: dcm_list.index(x)).values
            loc = np.concatenate([z[:, None], loc], axis=-1)
        eval_one_series(all_slices, loc, models, uid)

if __name__ == '__main__':
    main()