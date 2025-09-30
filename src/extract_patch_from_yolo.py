import os.path

from procs.proc_read_series import *
from procs.proc_patch_exraction import *
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimization settings
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ====================================================
# Configuration
# ====================================================
patch_size = 128
patch_depth = 31
iou_thresh = 5.0
k_candi = 2

IMG_SIZE = 512
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4

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

YOLO_LABELS = sorted(list(YOLO_LABELS_TO_IDX.keys()))

YOLO_MODEL_CONFIGS = [
    {
        "path": "../models/yolo_more_negs/fold0.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    {
        "path": "../models/yolo_more_negs/fold3.pt",
        "fold": "1",
        "weight": 1.0,
        "name": "YOLOv11n_fold1"
    },
    {
        "path": "../models/yolo_more_negs/fold4.pt",
        "fold": "1",
        "weight": 1.0,
        "name": "YOLOv11n_fold1"
    }
]

def load_yolo_models():
    """Load all YOLO models"""
    models = []
    for config in YOLO_MODEL_CONFIGS:
        model = YOLO(config["path"])
        model.to(device)

        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"]
        }
        models.append(model_dict)
    return models

def nms_3d_points(points, iou_thresh=2.0):
    """
    Simple 3D NMS for point detections.
    points: np.array of shape [N, 5] or [N,6], columns=[z,y,x,prob,class,...]
    iou_thresh: minimum distance (in voxels) to suppress a point
    Returns: indices of points to keep
    """
    if len(points) == 0:
        return []

    keep = []
    pts = points[:, :3]  # z,y,x
    scores = points[:, 3]
    order = scores.argsort()[::-1]

    suppressed = np.zeros(len(points), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        dists = np.linalg.norm(pts - pts[idx], axis=1)
        suppressed[dists < iou_thresh] = True
        suppressed[idx] = False  # keep current
    return keep


@torch.no_grad()
def predict_yolo_ensemble(slices, conf_yolo, YOLO_MODELS, iou_thresh=2.0, k = 3):
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1

    location_preds = {f'MODEL{i}': [] for i in range(len(YOLO_MODELS))}

    for model_idx, model_dict in enumerate(YOLO_MODELS):
        model = model_dict["model"]
        weight = model_dict["weight"]

        for i in range(0, len(slices), BATCH_SIZE):
            batch_slices = slices[i:i+BATCH_SIZE]
            z_idxes = [i + batch_idx for batch_idx in range(len(batch_slices))]
            results = model.predict(
                batch_slices,
                verbose=False,
                batch=len(batch_slices),
                device="cuda:0",
                conf=conf_yolo
            )

            for z_idx, r in enumerate(results):
                if r is None or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                    continue
                confs = r.boxes.conf.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()

                for j in range(len(confs)):
                    x1, y1, x2, y2 = xyxy[j]
                    x_center = (x1 + x2)/2
                    y_center = (y1 + y2)/2
                    point = np.array([z_idxes[z_idx], y_center, x_center, confs[j], clses[j], model_idx])
                    location_preds[f'MODEL{model_idx}'].append(point)

    # Apply NMS per model and per class
    final_preds = {f'MODEL{i}': [] for i in range(len(YOLO_MODELS))}
    for model_key, points in location_preds.items():
        points = np.array(points)
        if points.shape[0] < 3:
            continue
        for cls in np.unique(points[:,4]):
            cls_points = points[points[:,4]==cls]
            keep_idx = nms_3d_points(cls_points, iou_thresh=iou_thresh)
            final_preds[model_key].extend(cls_points[keep_idx])
        model_preds = torch.tensor(final_preds[model_key])
        values, indices = model_preds[:, -3].topk(k)   # get top-k values & indices
        final_preds[model_key] = model_preds[indices].numpy()
    return final_preds


def main():
    data_path = Path('./data')
    df = pd.read_csv(data_path/'train_df.csv')
    df_loc = pd.read_csv(data_path/'train_localizers.csv')
    df_multi_frame = pd.read_csv(data_path/'multiframe_dicoms.csv')
    YOLO_MODELS = load_yolo_models()
    aneurysm_proc = AneurysmVolumeProcessor3Planes(N=patch_size,
                 K_axial=patch_depth, K_sagittal=patch_depth, K_coronal=patch_depth,
                 Nr=patch_size, Ntheta=patch_size, augment=False, device='cpu')
    uids = df[~df.SeriesInstanceUID.isin(df_multi_frame.SeriesInstanceUID.unique())].SeriesInstanceUID.unique()

    if not os.path.exists(data_path/'patch_data'):
        os.makedirs(data_path/'patch_data')

    for i in range(len(YOLO_MODELS)):
        if not os.path.exists(data_path/f'patch_data/fold{i}'):
            os.makedirs(data_path/f'patch_data/fold{i}')

    for uid in tqdm(uids):
        series_path = data_path / f'series/{uid}'
        if not os.path.exists(series_path):
            continue
        count_next = 0
        for i in range(len(YOLO_MODELS)):
            if not os.path.exists(data_path / f'patch_data/fold{i}/{uid}'):
                os.makedirs(data_path / f'patch_data/fold{i}/{uid}')
            else:
                count_next += 1
                continue
        if count_next == 2:
            continue
        all_slices = process_dicom_for_yolo(series_path)
        if len(all_slices) == 0:
            continue
        vol = load_dicom_series(series_path)
        vol_norm = normalize_vol(vol)
        try:
            location_preds = predict_yolo_ensemble(all_slices, conf_yolo=0.01,
                                                   YOLO_MODELS= YOLO_MODELS,
                                                   iou_thresh=iou_thresh, k=k_candi)
        except Exception as e:
            print(uid)
            print(e)
            continue

        if isinstance(location_preds, tuple):
            continue

        for idx, (key, value) in enumerate(location_preds.items()):
            if len(value) == 0:
                continue
            yolo_points = location_preds[key][:, [2, 1, 0]].astype('int32')
            outputs = aneurysm_proc(vol_norm, yolo_points) #list:[patch0, ...]
            for patch_id, output in enumerate(outputs):
                cartesian = output['cartesian'].numpy()
                logpolar = output['logpolar'].numpy()
                axial = output['axial'].numpy()
                sagittal = output['sagittal'].numpy()
                coronal = output['coronal'].numpy()

                # Save to .npz file
                npz_path = data_path / f'patch_data/fold{idx}/{uid}/patch_{patch_id}.npz'
                np.savez_compressed(
                    npz_path,
                    cartesian=cartesian, #(3, 2, patch_size, patch_size);3:axial, sagittal, cornoal; 2: center slice, mip
                    logpolar=logpolar, #(2, patch_size, patch_size);3:axial, sagittal, cornoal; 2: center slice, mip
                    axial=axial, #(patch_depth, patch_size, patch_size)
                    sagittal=sagittal, #(patch_depth, patch_size, patch_size)
                    coronal=coronal #(patch_depth, patch_size, patch_size)
                )


if __name__ == '__main__':
    main()