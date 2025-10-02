
def aggregate_top3(confs: List[float]) -> float:
    """Aggregate confidences using top3 strategy (mean of top 3)"""
    if not confs:
        return 0.0
    arr = np.array(confs)
    if len(arr) < 3:
        return float(np.mean(arr))
    top_3 = np.partition(arr, -3)[-3:]
    return float(np.mean(top_3))


@torch.no_grad()
def predict_yolo_ensemble(slices: List[np.ndarray]):
    """Run YOLO inference using all models"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1
    
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    for model_dict in YOLO_MODELS:
        model = model_dict["model"]
        weight = model_dict["weight"]
        
        try:
            all_confs = []
            per_class_confs = [[] for _ in range(len(YOLO_LABELS))]
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                
                results = model.predict(
                    batch_slices, 
                    verbose=False, 
                    batch=len(batch_slices), 
                    device=device, 
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
                            all_confs.append(c)
                            if 0 <= k < len(YOLO_LABELS):
                                per_class_confs[k].append(c)
                    except Exception:
                        try:
                            for c in r.boxes.conf:
                                all_confs.append(float(c.item()))
                        except Exception:
                            pass
            
            # Aggregate using top3 strategy
            agg_conf = aggregate_top3(all_confs) if all_confs else 0.1
            per_class_agg = np.array([aggregate_top3(confs) if confs else 0.0 
                                      for confs in per_class_confs], dtype=np.float32)
            
            ensemble_cls_preds.append(agg_conf * weight)
            ensemble_loc_preds.append(per_class_agg * weight)
            total_weight += weight
            
        except Exception as e:
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            total_weight += weight
    
    if total_weight > 0:
        final_cls_pred = sum(ensemble_cls_preds) / total_weight
        final_loc_preds = sum(ensemble_loc_preds) / total_weight
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1
    
    return final_cls_pred, final_loc_preds