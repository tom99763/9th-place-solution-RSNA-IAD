import polars as pl
from tqdm import tqdm
from ultralytics import YOLO
from parse_preprocess import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import gc

# Optimization settings
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

# Constants
IMG_SIZE = 512
FACTOR = 1
# Batch size for batched YOLO inference
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4  # For parallel DICOM reading

# Label mappings
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
    'Right Supraclinoid Internal Carotid Artery': 12
}

LABELS = sorted(list(LABELS_TO_IDX.keys()))
LABEL_COLS = LABELS + ['Aneurysm Present']

# Model configurations - Add your models here
MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold02/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    # {
    #    "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold12/weights/best.pt",
    #    "fold": "1",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold1"
    # },
    # {
    #    "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold22/weights/best.pt",
    #    "fold": "2",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold2"
    # },
    # {
    #    "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold32/weights/best.pt",
    #    "fold": "3",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold3"
    # },
    # {
    #    "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold42/weights/best.pt",
    #    "fold": "4",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold4"
    # }
]


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
def eval_one_series_ensemble(slices: List[np.ndarray]):
    print('total slice num:', len(slices))
    print('process batch size:', BATCH_SIZE)
    """Run inference using all models on single GPU"""
    if not slices:
        return 0.1, np.ones(len(LABELS)) * 0.1

    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0

    for model_dict in models:
        model = model_dict["model"]
        weight = model_dict["weight"]

        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(LABELS), dtype=np.float32)

            # Process in batches
            for i in tqdm(range(0, len(slices), BATCH_SIZE)):
                batch_slices = slices[i:i + BATCH_SIZE]

                results = model.predict(
                    batch_slices,
                    verbose=False,
                    batch=len(batch_slices),
                    device="cuda:0",
                    conf=0.01
                )

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
                    except Exception:
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

    if total_weight > 0:
        final_cls_pred = sum(ensemble_cls_preds) / total_weight
        final_loc_preds = sum(ensemble_loc_preds) / total_weight
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(LABELS)) * 0.1

    return final_cls_pred, final_loc_preds


def _predict_inner(series_path):
    """Internal prediction logic with parallel preprocessing and single GPU inference"""
    series_path = Path(series_path)

    dicom_files = collect_series_slices(series_path)

    # Parallel DICOM processing
    all_slices: List[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_dicom_file, dcm_path): dcm_path
                          for dcm_path in dicom_files}

        for future in as_completed(future_to_file):
            try:
                slices = future.result()
                all_slices.extend(slices)
            except Exception as e:
                dcm_path = future_to_file[future]
                print(f"Failed processing {dcm_path.name}: {e}")

    # If no valid images were read, return a safe fallback row
    if not all_slices:
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions

    cls_prob, loc_probs = eval_one_series_ensemble(all_slices)

    # Ensure we have the right number of location probabilities
    if len(loc_probs) != len(LABELS):
        loc_probs = np.ones(len(LABELS)) * 0.1

    loc_probs = list(loc_probs)
    values = loc_probs + [cls_prob]

    predictions = pl.DataFrame(
        data=[values],
        schema=LABEL_COLS,
        orient='row'
    )
    predictions = predictions.with_columns(
        pl.when(pl.col("Aneurysm Present") + 0.2 > 1)
        .then(1)
        .otherwise(pl.col("Aneurysm Present") + 0.2)
        .alias("Aneurysm Present")
    )
    return predictions


def predict(series_path: str):
    """
    Top-level prediction function passed to the server.
    """
    try:
        return _predict_inner(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        # Cleanup
        if os.path.exists('/kaggle'):
            shared_dir = '/kaggle/shared'
        else:
            shared_dir = os.path.join(os.getcwd(), 'kaggle_shared')
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)

        # Memory cleanup for single GPU
        torch.cuda.empty_cache()
        gc.collect()


# Load all models
models = load_models()
print(f"Loaded {len(models)} models on single GPU")

def main():
    pass


if __name__ == '__main__':
    main()


