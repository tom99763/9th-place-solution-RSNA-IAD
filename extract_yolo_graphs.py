import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from configs.data_config import data_path, N_FOLDS, SEED  # type: ignore

rng = random.Random(SEED)


def read_dicom_frames_hu(path: Path) -> Tuple[List[np.ndarray], Optional[float]]:
    """Read DICOM frames and return HU values plus SliceThickness if available."""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    
    # Get SliceThickness if available
    slice_thickness = None
    if hasattr(ds, "SliceThickness"):
        try:
            slice_thickness = float(ds.SliceThickness)
        except:
            pass
    
    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    return frames, slice_thickness


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def ordered_dcm_paths(series_dir: Path) -> List[Path]:
    return sorted(series_dir.glob("*.dcm"), key=lambda p: p.name)


def load_series_slices(paths: List[Path]) -> Tuple[List[np.ndarray], Optional[float]]:
    """Load series slices and extract consistent SliceThickness."""
    imgs: List[np.ndarray] = []
    base_shape = None
    slice_thicknesses = []
    
    for p in paths:
        frames, thickness = read_dicom_frames_hu(p)
        if thickness is not None:
            slice_thicknesses.append(thickness)
        
        for s in frames:
            if base_shape is None:
                base_shape = s.shape
            if s.shape != base_shape:
                s = cv2.resize(s, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_LINEAR)
            imgs.append(min_max_normalize(s))
    
    # Use median SliceThickness if available
    avg_thickness = None
    if slice_thicknesses:
        avg_thickness = float(np.median(slice_thicknesses))
    
    return imgs, avg_thickness


def load_folds(root: Path) -> Dict[str, int]:
    df = pd.read_csv(root / "train_df.csv")
    series_df = df[["SeriesInstanceUID", "Aneurysm Present"]].drop_duplicates().reset_index(drop=True)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_map: Dict[str, int] = {}
    for i, (_, test_idx) in enumerate(skf.split(series_df["SeriesInstanceUID"], series_df["Aneurysm Present"])):
        for uid in series_df.loc[test_idx, "SeriesInstanceUID"].tolist():
            fold_map[uid] = i
    return fold_map


def knn_edges_weighted(centers: np.ndarray, k: int, radius: float | None, sigma: float = 0.1) -> List[Tuple[int, int, float]]:
    """Build kNN edges with Gaussian distance-based weights."""
    n = centers.shape[0]
    if n <= 1:
        return []
    
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        order = np.argsort(d)
        cnt = 0
        for j in order[1:]:  # skip self
            if radius is not None and d[j] > radius:
                break
            
            # Gaussian kernel weight
            weight = float(np.exp(-d[j] / sigma))
            edge_key = (min(i, j), max(i, j))
            
            # Keep maximum weight if edge already exists
            if edge_key in edges:
                edges[edge_key] = max(edges[edge_key], weight)
            else:
                edges[edge_key] = weight
            
            cnt += 1
            if cnt >= k:
                break
    
    return [(u, v, w) for (u, v), w in sorted(edges.items())]


def parse_args():
    ap = argparse.ArgumentParser(description="Extract YOLO detections per tomogram and build weighted graphs")
    ap.add_argument("--model", type=str, default="yolo11s.pt")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--knn", type=int, default=4, help="Intra-slice kNN")
    ap.add_argument("--inter", type=int, default=4, help="Inter-slice kNN per offset")
    ap.add_argument("--radius", type=float, default=0.25, help="Max normalized xy distance for edges; <=0 disables")
    ap.add_argument("--sigma", type=float, default=0.1, help="Sigma for Gaussian distance kernel")
    ap.add_argument("--out-dir", type=str, default="outputs/graphs_v2")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    import time 
    args = parse_args()
    root = Path(data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ignore known bad/multiframe series
    ignore: set[str] = {
        "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
        "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
        "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242",
    }
    mf_path = root / "multiframe_dicoms.csv"
    if mf_path.exists():
        try:
            mf_df = pd.read_csv(mf_path)
            if "SeriesInstanceUID" in mf_df.columns:
                ignore.update(mf_df["SeriesInstanceUID"].astype(str).tolist())
        except Exception:
            pass

    folds = load_folds(root)
    df = pd.read_csv(root / "train_df.csv")
    series_list = df["SeriesInstanceUID"].astype(str).unique().tolist()

    model = YOLO("/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11n_with_mix_up_mosaic_fold0/weights/best.pt")
    processed = 0
    for uid in series_list:
        if uid in ignore:
            continue
        series_dir = root / "series" / uid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] {uid}")
            continue

        paths = ordered_dcm_paths(series_dir)
        slices, slice_thickness = load_series_slices(paths)
        if len(slices) == 0:
            if args.verbose:
                print(f"[EMPTY] {uid}")
            continue

        H0, W0 = slices[0].shape
        z_den = max(1, len(slices) - 1)
        
        # Normalize slice thickness for z-distance calculations
        # If no thickness info, assume 1.0mm
        thickness_norm = slice_thickness if slice_thickness is not None else 1.0

        # Run YOLO per-slice in small batches
        batch = []
        batch_meta = []  # (slice_idx)
        all_nodes = []
        slice_to_node_idxs: List[List[int]] = [[] for _ in range(len(slices))]

        def flush_batch():
            nonlocal all_nodes, batch, batch_meta
            if not batch:
                return
            results = model.predict(source=batch, imgsz=args.img_size, conf=args.conf, iou=args.iou, verbose=False)
            for (res, sidx) in zip(results, batch_meta):
                boxes = res.boxes  # type: ignore
                if boxes is None or boxes.xywhn is None or len(boxes) == 0:
                    continue
                xywhn = boxes.xywhn.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                for i, (xc, yc, w, h) in enumerate(xywhn):
                    nid = len(all_nodes)
                    # Calculate actual z position considering slice thickness
                    z_mm = sidx * thickness_norm
                    node = {
                        "id": int(nid),
                        "slice": int(sidx),
                        "z": float(sidx / z_den),  # Normalized z [0,1]
                        "z_mm": float(z_mm),  # Actual z in mm
                        "xc": float(xc),
                        "yc": float(yc),
                        "conf": float(conf[i]),
                        "cls": int(cls[i]),
                    }
                    all_nodes.append(node)
                    slice_to_node_idxs[sidx].append(nid)
            batch = []
            batch_meta = []

        for sidx, img in enumerate(slices):
            im = img
            if args.img_size > 0 and (im.shape[0] != args.img_size or im.shape[1] != args.img_size):
                im = cv2.resize(im, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            batch.append(im)
            batch_meta.append(sidx)
            if len(batch) >= args.batch:
                flush_batch()
        flush_batch()

        # Build weighted edges
        edges: Dict[Tuple[int, int], float] = {}
        radius = args.radius if args.radius and args.radius > 0 else None
        
        # Intra-slice edges with Gaussian weights
        for sidx, idxs in enumerate(slice_to_node_idxs):
            if len(idxs) <= 1:
                continue
            centers = np.array([[all_nodes[i]["xc"], all_nodes[i]["yc"]] for i in idxs], dtype=np.float32)
            e_local = knn_edges_weighted(centers, k=max(0, args.knn), radius=radius, sigma=args.sigma)
            for u, v, w in e_local:
                a, b = idxs[u], idxs[v]
                if a != b:
                    edge_key = (min(a, b), max(a, b))
                    if edge_key in edges:
                        edges[edge_key] = max(edges[edge_key], w)
                    else:
                        edges[edge_key] = w
        
        # Inter-slice edges (multiple adjacent slices with distance-based weights)
        for sidx in range(len(slices)):
            idxs_src = slice_to_node_idxs[sidx]
            if not idxs_src:
                continue
            
            # Consider multiple adjacent slices: -2, -1, +1, +2
            for offset in [-2, -1, 1, 2]:
                sdst = sidx + offset
                if sdst < 0 or sdst >= len(slices):
                    continue
                
                idxs_dst = slice_to_node_idxs[sdst]
                if not idxs_dst:
                    continue
                
                # Weight by slice distance (considering actual thickness)
                slice_weight = 1.0 / (1.0 + abs(offset))
                
                # Use z_mm for actual physical distance if thickness is known
                if slice_thickness is not None:
                    src_cent = np.array([
                        [all_nodes[i]["xc"], all_nodes[i]["yc"], all_nodes[i]["z_mm"] / (thickness_norm * len(slices))] 
                        for i in idxs_src
                    ], dtype=np.float32)
                    dst_cent = np.array([
                        [all_nodes[i]["xc"], all_nodes[i]["yc"], all_nodes[i]["z_mm"] / (thickness_norm * len(slices))] 
                        for i in idxs_dst
                    ], dtype=np.float32)
                else:
                    src_cent = np.array([[all_nodes[i]["xc"], all_nodes[i]["yc"], all_nodes[i]["z"]] for i in idxs_src], dtype=np.float32)
                    dst_cent = np.array([[all_nodes[i]["xc"], all_nodes[i]["yc"], all_nodes[i]["z"]] for i in idxs_dst], dtype=np.float32)
                
                # Pairwise distances
                dists = np.linalg.norm(src_cent[:, None, :] - dst_cent[None, :, :], axis=2)
                
                for ii in range(dists.shape[0]):
                    order = np.argsort(dists[ii])
                    added = 0
                    for jj in order:
                        if radius is not None and dists[ii, jj] > radius:
                            break
                        
                        # Combined weight: Gaussian distance * slice distance weight
                        dist_weight = float(np.exp(-dists[ii, jj] / args.sigma))
                        combined_weight = dist_weight * slice_weight
                        
                        a, b = idxs_src[ii], idxs_dst[jj]
                        if a != b:
                            edge_key = (min(a, b), max(a, b))
                            if edge_key in edges:
                                edges[edge_key] = max(edges[edge_key], combined_weight)
                            else:
                                edges[edge_key] = combined_weight
                            added += 1
                        if added >= args.inter:
                            break

        # Convert edges dict to list format with weights
        edge_list = [(u, v, float(w)) for (u, v), w in sorted(edges.items())]
        
        graph = {
            "series_id": uid,
            "fold": folds.get(uid, -1),
            "image_size": [int(W0), int(H0)],
            "num_slices": len(slices),
            "slice_thickness_mm": float(thickness_norm) if slice_thickness is not None else None,
            "num_nodes": len(all_nodes),
            "nodes": all_nodes,
            "edges": edge_list,  # Now includes weights: [(u, v, weight), ...]
            "params": {
                "knn_intra": args.knn,
                "knn_inter": args.inter,
                "radius": args.radius,
                "sigma": args.sigma,
                "conf": args.conf,
                "iou": args.iou,
                "img_size": args.img_size,
            },
        }
        with open(out_dir / f"{uid}.json", "w") as f:
            json.dump(graph, f, indent=2)
        processed += 1
        if args.verbose and processed % 25 == 0:
            print(f"Saved graphs: {processed}")

    print(f"Done. Graphs saved: {processed} -> {out_dir}")


if __name__ == "__main__":
    main()