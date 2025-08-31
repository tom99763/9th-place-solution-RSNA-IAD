import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from parse_preprocess import *
import torch

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
def eval_one_series(slices, models):
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
    all_preds = all_detections[:, 3:-1]
    all_model_idx = all_detections[:, -1][:, None]
    all_features = np.concatenate(all_features, axis=0)
    vol_size = (len(slices), slices[0].shape[0], slices[0].shape[1])
    data1, data2 = extract_tomo(all_locations, all_features, vol_size)
    return data1, data2


def main():
    # Load all models
    models = load_models()
    print(f"Loaded {len(models)} models on single GPU")


if __name__ == '__main__':
    main()