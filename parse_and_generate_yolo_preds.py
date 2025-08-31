import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from parse_preprocess import *
import torch
from torch_cluster import knn_graph

# Optimization settings
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

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
            for i in tqdm(range(0, len(slices), BATCH_SIZE)):
                batch_slices = slices[i:i + BATCH_SIZE]
                z_idxes = [i + batch_idx for batch_idx in range(len(batch_slices))]

                features = get_feature_map(model)
                results = model.predict(
                    batch_slices,
                    verbose=False,
                    batch=len(batch_slices),
                    device="cuda:0",
                    conf=0.01
                )
                c3k2_feat = features['C3K2'].cpu().numpy()
                all_features.append(c3k2_feat)

                for r in results:
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

                            ensemble_locations.append([round(z_idxes[j]),  # depth
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

    np.save(f'./extract_data/{uid}/{uid}_edge_index_k5.npy', edge_index_k5)
    np.save(f'./extract_data/{uid}/{uid}_edge_index_k10.npy', edge_index_k10)
    np.save(f'./extract_data/{uid}/{uid}_edge_index_k15.npy', edge_index_k15)

    # delaunay_graph
    edge_index_del = delaunay_graph(torch.from_numpy(points))
    np.save(f'./extract_data/{uid}/{uid}_edge_index_delaunay.npy', edge_index_del)

    print('del edges:', edge_index_del.shape)
    print(points.shape, extract_feat.shape, dist_label.shape)
    print('label sum:', dist_label.sum())

    np.save(f'./extract_data/{uid}/{uid}_points.npy', points)
    np.save(f'./extract_data/{uid}/{uid}_extract_feat.npy', extract_feat)
    np.save(f'./extract_data/{uid}/{uid}_label.npy', dist_label)

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


def main():
    label_df = load_labels(Path('./src/data'))
    models = load_models()
    print(f"Loaded {len(models)} models on single GPU")


if __name__ == '__main__':
    main()