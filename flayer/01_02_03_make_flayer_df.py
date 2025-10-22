import argparse
import os
import numpy as np
import pydicom
from pathlib import Path
import pandas as pd
import ast
import cv2
import gc
import glob
from tqdm.auto import tqdm
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ENABLE_PREVIEW = os.environ.get("RSNA_PREVIEW_LABEL", "0") == "1"


def visualize_label_on_slice(image: np.ndarray, x_val, y_val, series_uid: str, sop_uid: str):
    if not ENABLE_PREVIEW or plt is None:
        return
    #if image is None or x_val is None or y_val is None:
    if image is None :    
        return
    if x_val is None or y_val is None:
        x_val = 0
        y_val = 0

    if image.ndim == 3 and image.shape[-1] != 1:
        vis_img = image.copy()
    else:
        norm = image.astype(np.float32)
        norm -= norm.min()
        max_val = norm.max() or 1.0
        norm = (norm / max_val * 255.0).astype(np.uint8)
        vis_img = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    h, w = vis_img.shape[:2]
    px = int(round(x_val))
    py = int(round(y_val))
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(vis_img, (px, py), radius=max(2, min(h, w)//100), color=(255, 0, 0), thickness=2)
    title = f"Series: {series_uid}\nSOP: {sop_uid} | ({px}, {py})"
    plt.figure(figsize=(5, 5))
    plt.imshow(vis_img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)

class DICOMPreprocessorKaggle:
    def __init__(self, target_shape=(32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_dicom_series(self, series_path: str, df_locale=None, df_train=None):
        """
        列舉並讀取該 series 下的所有 DICOM，並在枚舉時：
          - file 名稱視為該 slice 的 UID (SOPInstanceUID)
          - series_path 名稱視為 SeriesInstanceUID (SUID)
          - 若 df_locale (train_localizers.csv) 中存在與該 UID 對應的記錄，則讀出 coordinates
          - 以 extract-like 的方式計算當前 slice 的 z_position (frame index 的排序依據)
          - 若有找到對應 slice 的 label，更新 df_train 內該 SUID 的 z_position 及 coordinates
        """
        series_path = Path(series_path)
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))

        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {series_path}")

        datasets = []
        slice_records = []
        series_uid = series_path.name  # SUID
        print(f"Loading series {series_uid} with {len(dicom_files)} DICOM files")
        # 將 localizers 以 dict/set 加速查找
        has_locale = df_locale is not None and len(df_locale) > 0
        locale_mask = None
        locale_index = {}
        if has_locale:
            locale_mask = df_locale["SeriesInstanceUID"].astype(str) == str(series_uid)
            has_locale = bool(locale_mask.any())
            if has_locale:
                locale_subset = df_locale.loc[locale_mask]
                locale_index = dict(zip(locale_subset["SOPInstanceUID"].astype(str), locale_subset["coordinates"]))
            else:
                print("No locale data found for this series")
                return None, None #only parse label data

        for filepath in dicom_files:
            try:
                instance_uid = Path(filepath).stem  # file 名即為 UID (SOPInstanceUID)
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)


                #print(f"Processing {instance_uid}")
                # 取得 z_position（當作 frame index 排序依據）
                try:
                    position = getattr(ds, "ImagePositionPatient", None)
                    #print(f"  z_position (ImagePositionPatient): {position}")
                    if position is not None and len(position) >= 3:
                        z_position = float(position[2])
                        #print(f"  z_position (InstanceNumber): {z_position}")
                    else:
                        z_position = float(getattr(ds, "InstanceNumber", 0) or 0)
                        #print(f"  z_position (InstanceNumber <3): {z_position}")
                except Exception:
                    z_position = 0.0

                slice_info = {
                    "SeriesInstanceUID": series_uid,  # SUID
                    "SOPInstanceUID": instance_uid,   # 該 slice 的 UID
                    "z_position": z_position,
                    "instance_number": getattr(ds, "InstanceNumber", None),
                }

                # 若 localizers 有對應到這個 "slice 的 UID"（雖然欄位名叫 SeriesInstanceUID，但依照你的說明用來放 UID）
                if has_locale and instance_uid in locale_index:
                    print("  Found label for this slice, z_position:", z_position)
                    coords = locale_index[instance_uid]
                    slice_info["coordinates"] = coords

                    x_val = y_val = f_val = None
                    try:
                        #print("hihihi--------------------")
                        #print(coords)
                        c = ast.literal_eval(str(coords))
                        #print("hihihi2--------------------")
                        #print(c)
                        if isinstance(c, dict):
                            x_val = c.get("x", None)
                            y_val = c.get("y", None)
                            f_val = c.get("f", None)
                            print(f"  Parsed coordinates: x={x_val}, y={y_val}, f={f_val}")
                        raise
                    except Exception:
                        pass


                    # 找到 label 的 slice → 更新 df_train 內此 SUID 的欄位
                    if df_train is not None:
                        print(f"  Update df_train with z_position and coordinates")
                        sel = (df_train["SeriesInstanceUID"].astype(str) == str(series_uid))
                        df_train.loc[sel, "z_position"] = z_position
                        df_train.loc[sel, "coordinates"] = coords

                        if x_val is not None:
                            df_train.loc[sel, "label_x"] = float(x_val)
                        if y_val is not None:
                            df_train.loc[sel, "label_y"] = float(y_val)
                        # 若 3D 有 f，則直接指定 label_frame_num（優先於 2D 的推算）
                        if f_val is not None:
                            try:
                                df_train.loc[sel, "label_frame_num"] = int(f_val)
                                #print("hihiok")
                            except Exception:
                                pass

                    if has_locale:
                        sop_mask = locale_mask & (df_locale["SOPInstanceUID"].astype(str) == instance_uid)
                        if sop_mask.any():
                            if x_val is not None:
                                df_locale.loc[sop_mask, "label_x"] = float(x_val)
                            if y_val is not None:
                                df_locale.loc[sop_mask, "label_y"] = float(y_val)
                            if f_val is not None:
                                try:
                                    df_locale.loc[sop_mask, "label_frame_num"] = int(f_val)
                                    #print("hihiok2")
                                except Exception:
                                    pass


                    #if ENABLE_PREVIEW and x_val is not None and y_val is not None:
                    #    img_preview = self.extract_pixel_array(ds)
                    #    visualize_label_on_slice(img_preview, x_val, y_val, series_uid, instance_uid)
                #put here to see all slices
                if ENABLE_PREVIEW :
                    img_preview = self.extract_pixel_array(ds)
                    visualize_label_on_slice(img_preview, 0, 0, series_uid, instance_uid)


                slice_records.append(slice_info)

            except Exception as e:
                print(f"Skip {filepath}: {e}")
                continue
        #print(len(slice_records), "slices with records")
        if slice_records and len(slice_records) != 1:
            sorted_slice_records = sorted(
                slice_records,
                key=lambda x: (x.get("z_position", 0.0), x.get("instance_number", 0) or 0)
            )
            sop_to_frame = {}
            for frame_idx, rec in enumerate(sorted_slice_records):
                rec["frame_index"] = frame_idx
                sop_to_frame[rec["SOPInstanceUID"]] = frame_idx

            if has_locale:
                sop_series = df_locale.loc[locale_mask, "SOPInstanceUID"].astype(str)
                df_locale.loc[locale_mask, "label_frame_num"] = sop_series.map(sop_to_frame)

                if df_train is not None:
                    series_mask = (df_train["SeriesInstanceUID"].astype(str) == str(series_uid))
                    if series_mask.any():
                        first_valid = df_locale.loc[locale_mask, "label_frame_num"].dropna()
                        if not first_valid.empty:
                            df_train.loc[series_mask, "label_frame_num"] = int(first_valid.iloc[0])

        if not datasets:
            raise FileNotFoundError(f"No valid DICOM files in {series_path}")

        return datasets, slice_records

    def extract_slice_info(self, datasets):
        slice_info = []
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    info['z_position'] = float(info['instance_number'])
            except Exception:
                info['z_position'] = float(i)
            slice_info.append(info)
        return slice_info

    def sort_slices_by_position(self, slice_info):
        return sorted(slice_info, key=lambda x: x['z_position'])

    def apply_percentile_normalization(self, img):
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            normalized = np.clip(img, p1, p99)
            normalized = (normalized - p1) / (p99 - p1)
            result = (normalized * 255).astype(np.uint8)
            return result
        else:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                normalized = (img - img_min) / (img_max - img_min)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                return np.zeros_like(img, dtype=np.uint8)

    def extract_pixel_array(self, ds):
        img = ds.pixel_array.astype(np.float32)
        if img.ndim == 3:
            # 若是多 frame，取中間那一張做 2D 正規化（3D volume 情境另處理）
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
        if img.ndim == 3 and img.shape[-1] == 3:
            import cv2
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
        return img



    def process_series_with_orig(self, datasets, series_uid: str, df_train=None, df_locale=None):
        """
        以 datasets 建 volume；同時把「resize 前的原始影像 shape」(height, width, depth) 記到 df_train：
          - orig_height, orig_width 取自於第一張 slice 的 img.shape
          - orig_depth 為排序後的 slice 數量（或 3D frame 數）csv_path = os.path.join(args.data_folder, "train.csv")
            localize_csv_path = os.path.join(args.data_folder, "train_localizers.csv")
            series_root = os.path.join(args.data_folder, "series")
        """
        first_ds = datasets[0]
        first_img = first_ds.pixel_array

        # 3D multi-frame (單一 DICOM 內含多 frame)
        if len(datasets) == 1 and first_img.ndim == 3:
            volume = first_img.astype(np.float32)  # shape: (depth, height, width)
            orig_depth, orig_height, orig_width = volume.shape

   
            del volume #ytt

        else:
            # 多個 2D 檔案構成一個 series
            slice_info = self.extract_slice_info(datasets)
            sorted_slices = self.sort_slices_by_position(slice_info)

            # 取得原始 H, W 與深度
            first_img_2d = self.extract_pixel_array(sorted_slices[0]["dataset"])
            orig_height, orig_width = first_img_2d.shape
            orig_depth = len(sorted_slices)
            #ytt
            del first_img_2d
 

        # 更新 df_train 的原始 shape
        if df_train is not None:
            sel = (df_train["SeriesInstanceUID"].astype(str) == str(series_uid))
            df_train.loc[sel, "orig_height"] = int(orig_height)
            df_train.loc[sel, "orig_width"]  = int(orig_width)
            df_train.loc[sel, "orig_depth"]  = int(orig_depth)

        if df_locale is not None:
            loc_sel = (df_locale["SeriesInstanceUID"].astype(str) == str(series_uid))
            if loc_sel.any():
                df_locale.loc[loc_sel, "orig_height"] = int(orig_height)
                df_locale.loc[loc_sel, "orig_width"] = int(orig_width)
                df_locale.loc[loc_sel, "orig_depth"] = int(orig_depth)

        #del volume
        #import gc
        #gc.collect()
        #eturn volume
        return None

def main(args):
    

    
    # 從 args 讀取參數
    
    
    csv_path = os.path.join(args.data_folder, "train.csv")
    localize_csv_path = os.path.join(args.data_folder, "train_localizers.csv")
    series_root = os.path.join(args.data_folder, "series")
    output_dir = args.output_dir

    print(f"csv_path: {csv_path}")
    print(f"localize_csv_path: {localize_csv_path}")
    print(f"series_root: {series_root}")
    print(f"output_dir: {output_dir}")
    
    
    
    os.makedirs(output_dir, exist_ok=True)

    # 0. load train.csv to df_train, load train_localizers.csv to df_locale
    df_train = pd.read_csv(csv_path)
    
    #0.1 make axis_df
    print("check_axis")
    sid_list=df_train["SeriesInstanceUID"].values
    
    axis_list=[]
    for sid in tqdm(sid_list):
        files = glob.glob(f"{series_root}/{sid}/*.dcm")
        slices = [pydicom.dcmread(f) for f in files]
        if len(files)==1:
            ds=slices[0]
            pos_list = []
            # Per-frame functional group sequence
            for frame in ds.PerFrameFunctionalGroupsSequence:
                # Plane position sequence
                pos = frame.PlanePositionSequence[0].ImagePositionPatient
                pos_list.append([float(x) for x in pos])    
        else:
            pos_list=[]
            for i in range(len(slices)):
                pos_list.append(slices[i].ImagePositionPatient)

        
        arr = np.array(pos_list)
        ranges = [
            np.ptp(arr[:, 0]),  # x 範圍
            np.ptp(arr[:, 1]),  # y 範圍
            np.ptp(arr[:, 2])   # z 範圍
        ]
        
        labels = ["x", "y", "z"]
        
        # 找最大值對應的標籤
        result = labels[np.argmax(ranges)]
        axis_list.append(result)
    df_train["axis"]=axis_list
    df_train.loc[df_train["SeriesInstanceUID"]=="1.2.826.0.1.3680043.8.498.99804081131933373817667779922320327920","axis"]="y"
    print(df_train["axis"].value_counts())
    
    df_train[["SeriesInstanceUID","axis"]].to_csv(f"{output_dir}/axis_df.csv",index=False)
    
    
    
    
    for col in ["label_x", "label_y", "label_frame_num"]:
        if col not in df_train.columns:
            df_train[col] = np.nan

    df_locale = pd.read_csv(localize_csv_path)
    for col in ["label_frame_num", "orig_height", "orig_width", "orig_depth", "label_x", "label_y"]:
        if col not in df_locale.columns:
            df_locale[col] = np.nan

    sid_list = df_train["SeriesInstanceUID"].astype(str).unique()
    print(f"Total series: {len(sid_list)}")

    preprocessor = DICOMPreprocessorKaggle()
  

    for sid in sid_list: #[:3]:
        #print(sid)
        #if(sid!="1.2.826.0.1.3680043.8.498.17677548211553545296698864792051352427"):
        #    continue
        series_dir = os.path.join(series_root, str(sid))
        try:
            # 1~3. 載入 series，於列舉時比對 UID 與 localizers，並寫入 z_position/coordinates 到 df_train
            datasets, slice_records = preprocessor.load_dicom_series(series_dir, df_locale=df_locale, df_train=df_train)

            if datasets is None or slice_records is None:
                print(f"Skipped {sid} due to no datasets or slice_records")
                continue

            # 進一步將 datasets 組成 volume，並且記錄「resize 前影像的原始 shape」到 df_train / df_locale
            volume = preprocessor.process_series_with_orig(
                datasets,
                series_uid=str(sid),
                df_train=df_train,
                df_locale=df_locale,
            )

            # 儲存 volume
            #npy_path = os.path.join(output_dir, f"{sid}.npy")
            #np.save(npy_path, volume)

            # 額外輸出此 series 的 slice 細節（包含有無對應 coordinates）
            #if slice_records:
            #    info_path = os.path.join(output_dir, f"{sid}_slices.csv")
            #    pd.DataFrame(slice_records).to_csv(info_path, index=False)
          
            del volume
            del datasets
            del slice_records
            gc.collect()            

            #print(f"Saved {npy_path} shape={volume.shape}")

        except Exception as e:
            print(f"Failed {sid}: {e}")

    # 4. 輸出帶有 frame number 的 localizer 樣本
    try:
        df_locale["label_frame_num"] = df_locale["label_frame_num"].astype("Int64")
    except Exception:
        pass
    locale_out_csv = os.path.join(output_dir, "train_locale_fnum.csv")
    df_locale.to_csv(locale_out_csv, index=False)
    print(f"Wrote localizer CSV -> {locale_out_csv}")

    ###
    prob_sid="1.2.826.0.1.3680043.8.498.75294325392457179365040684378207706807" #all black
    df_train = df_train[df_train["SeriesInstanceUID"]!=prob_sid].reset_index(drop=True)
    ###
    #5.split fold
    from sklearn.model_selection import StratifiedKFold
    n_splits = 5
    random_state = 42
    # 這個鍵結合了最重要的兩個維度：影像類型和是否有血管瘤
    df_train['stratify_group'] = df_train['Modality'].astype(str) + '_' + df_train['Aneurysm Present'].astype(str)
    #  執行分層 K-Fold 切分
    df_train['fold'] = -1  # 初始化 fold 欄位
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # 使用新的 'stratify_group' 欄位作為分層依據
    for fold, (train_index, val_index) in enumerate(skf.split(df_train, df_train['stratify_group'])):
        df_train.loc[val_index, 'fold'] = fold
    #  移除臨時輔助欄位
    df_train = df_train.drop(columns=['stratify_group'])
    
    # #6. add axis
    # axis_df=pd.read_csv(axis_df_path)
    # df_train = df_train.merge(axis_df,on="SeriesInstanceUID",how="left")
    
    #6. 將更新後欄位（z_position、coordinates、orig_height、orig_width、orig_depth）寫回新的 train.csv
    out_csv = os.path.join(output_dir, "train_with_folds_optimized_axis_v1.csv")
    df_train.to_csv(out_csv, index=False)
    print(f"Wrote updated CSV -> {out_csv}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RSNA 2025 preprocessing pipeline")
    
    parser.add_argument(
        "--data_folder", type=str,
        default="./data",
        help="Root folder of RSNA Intracranial Aneurysm Detection dataset"
    )

    # parser.add_argument("--csv_path", type=str,
    #                     default="data/train.csv",
    #                     help="Path to main training CSV file")
    # parser.add_argument("--localize_csv_path", type=str,
    #                     default="data/train_localizers.csv",
    #                     help="Path to localizer CSV file")
    # parser.add_argument("--series_root", type=str,
    #                     default="data/series",
    #                     help="Root directory of DICOM series")
    parser.add_argument("--output_dir", type=str,
                        default="output",
                        help="Output directory for generated data")

    args = parser.parse_args()
    main(args)
