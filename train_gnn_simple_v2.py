import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Use the binary GNN we just created
from src.gnn_yolo_v2 import SimpleGNNBinary


# Constants (no CLI/kwargs as requested)
GRAPHS_DIR = Path("/home/sersasj/RSNA-IAD-Codebase/outputs/graphs_v2")
LABELS_CSV = Path("/home/sersasj/RSNA-IAD-Codebase/data/train.csv")
NUM_CLASSES = 13  # Still need this for node features (class one-hot)
EPOCHS = 20
LR = 1e-3
HIDDEN_DIM = 64
TRAIN_FOLDS = [1, 2, 3, 4]  # Train on folds 1-4
VAL_FOLD = 0  # Validate on fold 0
SEED = 42
CHECKPOINT_PATH = Path("/home/sersasj/RSNA-IAD-Codebase/outputs/gnn_simple_v2.pth")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_binary_labels(csv_path: Path) -> Dict[str, float]:
    """Load binary labels: 1 if aneurysm present anywhere, 0 otherwise."""
    df = pd.read_csv(csv_path)
    # Ensure SeriesInstanceUID is string
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)

    # Location columns in the competition
    loc_cols = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
    ]

    # Some CSVs may have NaNs; fill with 0
    df[loc_cols] = df[loc_cols].fillna(0).astype(int)
    
    # Create binary labels: 1 if aneurysm present in any location, 0 otherwise
    labels = {}
    for _, row in df.iterrows():
        uid = row["SeriesInstanceUID"]
        has_aneurysm = float(row[loc_cols].sum() > 0)  # 1.0 if any location positive
        labels[uid] = has_aneurysm
    
    return labels


def load_graph(json_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Returns:
        node_feats [N, 1+4+NUM_CLASSES]: conf, x, y, z_norm, z_mm, class one-hot
        edge_index [2, E]
        edge_weights [E]
        fold [int]: fold assignment for this graph
    """
    with open(json_path, "r") as f:
        g = json.load(f)

    nodes = g.get("nodes", [])
    edges = g.get("edges", [])
    fold = g.get("fold", -1)
    
    # Handle graphs without fold info (fallback to filename-based lookup)
    if fold == -1:
        series_id = g.get("series_id", json_path.stem)
        # Load folds mapping to get correct fold
        try:
            from pathlib import Path
            import pandas as pd
            from sklearn.model_selection import StratifiedKFold
            
            root = Path("/home/sersasj/RSNA-IAD-Codebase/data")
            df = pd.read_csv(root / "train_df.csv")
            series_df = df[["SeriesInstanceUID", "Aneurysm Present"]].drop_duplicates().reset_index(drop=True)
            series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_map = {}
            for i, (_, test_idx) in enumerate(skf.split(series_df["SeriesInstanceUID"], series_df["Aneurysm Present"])):
                for uid in series_df.loc[test_idx, "SeriesInstanceUID"].tolist():
                    fold_map[uid] = i
            fold = fold_map.get(series_id, -1)
        except:
            pass

    if len(nodes) == 0:
        # Return empty tensors; caller will skip
        return (
            torch.empty(0, 1 + 4 + NUM_CLASSES, dtype=torch.float32),
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
            fold
        )

    # Build node features (conf, x, y, z_norm, z_mm_norm, class_onehot)
    feats: List[torch.Tensor] = []
    for n in nodes:
        conf = float(n.get("conf", 0.0))
        xc = float(n.get("xc", 0.0))
        yc = float(n.get("yc", 0.0))
        # z in JSON is normalized [0,1] across slices if available; else 0
        z_norm = float(n.get("z", 0.0))
        # z_mm is absolute mm; normalize by (thickness_norm * num_slices) if graph provides params
        # Our extractor stored z_mm and used thickness_norm * len(slices) to normalize distances
        # For node features, we'll divide by (max z_mm) across nodes to keep in [0,1]
        z_mm = float(n.get("z_mm", 0.0))
        cls_idx = int(n.get("cls", 0))
        onehot = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if 0 <= cls_idx < NUM_CLASSES:
            onehot[cls_idx] = 1.0
        feat = torch.tensor([conf, xc, yc, z_norm, z_mm], dtype=torch.float32)
        feats.append(torch.cat([feat, onehot], dim=0))

    node_feats = torch.stack(feats, dim=0)
    # Normalize z_mm column to [0,1] per-graph to avoid scale issues if not already normalized
    if node_feats.shape[0] > 0:
        z_mm_col = 4  # [conf, x, y, z_norm, z_mm, ...]
        z_mm_vals = node_feats[:, z_mm_col]
        z_max = torch.max(z_mm_vals)
        if z_max > 0:
            node_feats[:, z_mm_col] = z_mm_vals / z_max

    # Build undirected edges with weights
    if len(edges) == 0:
        # Self-loops if no edges
        n = node_feats.shape[0]
        edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
        edge_weights = torch.ones(n, dtype=torch.float32)
        return node_feats, edge_index, edge_weights, fold

    e_list: List[List[int]] = []
    w_list: List[float] = []
    for u, v, w in edges:
        u = int(u)
        v = int(v)
        w = float(w)
        e_list.append([u, v])
        e_list.append([v, u])
        w_list.append(w)
        w_list.append(w)

    edge_index = torch.tensor(e_list, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(w_list, dtype=torch.float32)
    return node_feats, edge_index, edge_weights, fold


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load binary labels and find available graphs
    labels_map = load_binary_labels(LABELS_CSV)
    graph_files = sorted(GRAPHS_DIR.glob("*.json"))

    # Build dataset list of (json_path, target, fold)
    train_samples: List[Tuple[Path, torch.Tensor]] = []
    val_samples: List[Tuple[Path, torch.Tensor]] = []
    skipped_no_nodes = 0
    for jf in graph_files:
        uid = jf.stem
        if uid not in labels_map:
            continue
        node_feats, edge_index, edge_weights, fold = load_graph(jf)
        if node_feats.numel() == 0:
            skipped_no_nodes += 1
            continue
        target = torch.tensor(labels_map[uid], dtype=torch.float32)  # scalar binary label
        
        # Split by fold
        if fold in TRAIN_FOLDS:
            train_samples.append((jf, target))
        elif fold == VAL_FOLD:
            val_samples.append((jf, target))

    if len(train_samples) == 0:
        print("No training samples found. Check GRAPHS_DIR and TRAIN_FOLDS.")
        return
        
    if len(val_samples) == 0:
        print("No validation samples found. Check GRAPHS_DIR and VAL_FOLD.")
        return

    if skipped_no_nodes > 0:
        print(f"Skipped {skipped_no_nodes} graphs with no nodes.")
        
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    
    # Print label distribution
    train_labels = [float(target.item()) for _, target in train_samples]
    val_labels = [float(target.item()) for _, target in val_samples]
    print(f"Train positive ratio: {np.mean(train_labels):.3f}")
    print(f"Val positive ratio: {np.mean(val_labels):.3f}")

    # Create indices for iteration
    train_idx = list(range(len(train_samples)))
    val_idx = list(range(len(val_samples)))

    # Model, opt, loss
    model = SimpleGNNBinary(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    def forward_series(json_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        node_feats, edge_index, edge_weights, _ = load_graph(json_path)
        node_feats = node_feats.to(device)
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device) if edge_weights is not None else None
        out = model(node_feats, edge_index, edge_weights)
        # Get binary logit directly from GNN
        binary_logit = out["binary_logit"]  # scalar
        binary_prob = torch.sigmoid(binary_logit)  # scalar
        return binary_logit, binary_prob

    # Training loop with best model tracking
    best_val_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for i in train_idx:
            jf, target = train_samples[i]
            target = target.to(device)

            logit, _ = forward_series(jf)
            loss = criterion(logit, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss /= max(1, len(train_idx))

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_tgts = []
        with torch.no_grad():
            for i in val_idx:
                jf, target = val_samples[i]
                target = target.to(device)
                logit, prob = forward_series(jf)
                loss = criterion(logit, target)
                val_loss += float(loss.item())
                all_preds.append(float(prob.cpu().item()))
                all_tgts.append(float(target.cpu().item()))
        val_loss /= max(1, len(val_idx))

        # Calculate binary classification metrics
        if all_preds and len(set(all_tgts)) > 1:  # Need both classes for AUC
            try:
                auc = roc_auc_score(all_tgts, all_preds)
            except ValueError:
                auc = 0.0
            
            # Binary accuracy at 0.5 threshold
            pred_binary = [1.0 if p >= 0.5 else 0.0 for p in all_preds]
            acc = accuracy_score(all_tgts, pred_binary)
            
            # Print some predictions for debugging
            if epoch == 1:
                print(f"Sample predictions (first 10):")
                for i in range(min(10, len(all_preds))):
                    print(f"  Pred: {all_preds[i]:.3f}, True: {int(all_tgts[i])}")
        else:
            auc = acc = 0.0

        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
        print(f"  val_auc: {auc:.4f}  val_acc@0.5: {acc:.4f}")

        # Save checkpoint only if validation AUC improved
        if auc > best_val_auc:
            best_val_auc = auc
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "hidden_dim": HIDDEN_DIM,
                "num_classes": NUM_CLASSES,
                "best_val_auc": best_val_auc,
                "epoch": epoch,
            }, CHECKPOINT_PATH)
            print(f"  New best val_auc: {best_val_auc:.4f} - saved checkpoint to {CHECKPOINT_PATH}")

    print(f"Training complete. Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()