from procs.proc_read_series import *
from procs.proc_patch_exraction import *
from ultralytics import YOLO
import pandas as pd
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
# YOLO Configuration
# ====================================================
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
        "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold0/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold1/weights/best.pt",
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
        if points.shape[0] == 0:
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
    YOLO_MODELS = load_yolo_models()



if __name__ == '__main__':
    pass