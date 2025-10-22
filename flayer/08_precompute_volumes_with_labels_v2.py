import argparse
import os
import numpy as np
import pydicom
from pathlib import Path
import pandas as pd
import ast

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
        locale_index = None
        if has_locale:
            # 這裡依照需求：將「檔名的 UID」對照 train_localizers.csv 的 SeriesInstanceUID 欄位
            # 因此把該欄位抽成一個 dict: {SeriesInstanceUID(=slice UID) -> coordinates}
            locale_index = dict(zip(df_locale["SOPInstanceUID"].astype(str), df_locale["coordinates"]))

        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)

                instance_uid = Path(filepath).stem  # file 名即為 UID (SOPInstanceUID)
                print(f"Processing {instance_uid}")
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
                    "z_position": z_position
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
                            except Exception:
                                pass



                slice_records.append(slice_info)

            except Exception as e:
                print(f"Skip {filepath}: {e}")
                continue

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

    def resize_volume_3d(self, volume):
        from scipy import ndimage
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        if current_shape == target_shape:
            return volume
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        return resized_volume.astype(np.uint8)

    def process_series_with_orig(self, datasets, series_uid: str, df_train=None):
        """
        以 datasets 建 volume；同時把「resize 前的原始影像 shape」(height, width, depth) 記到 df_train：
          - orig_height, orig_width 取自於第一張 slice 的 img.shape
          - orig_depth 為排序後的 slice 數量（或 3D frame 數）
        """
        first_ds = datasets[0]
        first_img = first_ds.pixel_array

        # 3D multi-frame (單一 DICOM 內含多 frame)
        if len(datasets) == 1 and first_img.ndim == 3:
            volume = first_img.astype(np.float32)  # shape: (depth, height, width)
            orig_depth, orig_height, orig_width = volume.shape

            # 處理每一 slice 的 normalization
            processed_slices = []
            for i in range(orig_depth):
                slice_img = volume[i]
                processed_img = self.apply_percentile_normalization(slice_img)
                processed_slices.append(processed_img)
            volume = np.stack(processed_slices, axis=0)
            volume = self.resize_volume_3d(volume)

        else:
            # 多個 2D 檔案構成一個 series
            slice_info = self.extract_slice_info(datasets)
            sorted_slices = self.sort_slices_by_position(slice_info)

            # 取得原始 H, W 與深度
            first_img_2d = self.extract_pixel_array(sorted_slices[0]["dataset"])
            orig_height, orig_width = first_img_2d.shape
            orig_depth = len(sorted_slices)

            # >>> NEW: 計算 label_frame_num (只處理 multiple-slice；3D 先不管)
            if df_train is not None:
                sel_series = (df_train["SeriesInstanceUID"].astype(str) == str(series_uid))
                if sel_series.any() and "z_position" in df_train.columns:
                    # 先拿到 df_train 內之前記錄的 label z_position
                    label_z = df_train.loc[sel_series, "z_position"].iloc[0]
                    label_frame_num = -1  # 預設找不到時為 -1

                    # 以 sort_slices_by_position 後的順序，找出與 label_z 相同之 slice 的索引（0-based）
                    z_list = [s["z_position"] for s in sorted_slices]
                    # 若浮點誤差可加個容忍
                    tol = 1e-6
                    for idx, z in enumerate(z_list):
                        if abs(float(z) - float(label_z)) <= tol:
                            label_frame_num = idx
                            break

                    # 寫回 df_train
                    df_train.loc[sel_series, "label_frame_num"] = int(label_frame_num)
            # <<< NEW


            processed_slices = []
            import cv2
            for slice_data in sorted_slices:
                ds = slice_data['dataset']
                img = self.extract_pixel_array(ds)  # <-- 這裡就是 resize 前的 img
                processed_img = self.apply_percentile_normalization(img)
                resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
                processed_slices.append(resized_img)
                #processed_slices.append(processed_img)
            volume = np.stack(processed_slices, axis=0)
            volume = self.resize_volume_3d(volume)

        # 更新 df_train 的原始 shape
        if df_train is not None:
            sel = (df_train["SeriesInstanceUID"].astype(str) == str(series_uid))
            df_train.loc[sel, "orig_height"] = int(orig_height)
            df_train.loc[sel, "orig_width"]  = int(orig_width)
            df_train.loc[sel, "orig_depth"]  = int(orig_depth)

        return volume

def main(args):



    csv_path = os.path.join(args.data_folder, "train.csv")
    localize_csv_path = os.path.join(args.data_folder, "train_localizers.csv")
    series_root = os.path.join(args.data_folder, "series")
    
    # csv_path = args.csv_path
    # localize_csv_path = args.localize_csv_path
    # series_root = args.series_root
    output_dir = args.output_dir

    print(f"CSV path           : {csv_path}")
    print(f"Localizer CSV path : {localize_csv_path}")
    print(f"Series root        : {series_root}")
    print(f"Output directory   : {output_dir}")
    
    
    os.makedirs(output_dir, exist_ok=True)

    # 0. load train.csv to df_train, load train_localizers.csv to df_locale
    df_train = pd.read_csv(csv_path)
    for col in ["label_x", "label_y", "label_frame_num"]:
        if col not in df_train.columns:
            df_train[col] = np.nan

    df_locale = pd.read_csv(localize_csv_path)

    sid_list = df_train["SeriesInstanceUID"].astype(str).unique()
    print(f"Total series: {len(sid_list)}")

    #preprocessor = DICOMPreprocessorKaggle(target_shape=(32, 384, 384))
    #preprocessor = DICOMPreprocessorKaggle(target_shape=(32, 448, 448))
    preprocessor = DICOMPreprocessorKaggle(target_shape=(64, 448, 448))

    for sid in sid_list: #[:200]: #[:3]:
        series_dir = os.path.join(series_root, str(sid))
        try:
            # 1~3. 載入 series，於列舉時比對 UID 與 localizers，並寫入 z_position/coordinates 到 df_train
            datasets, slice_records = preprocessor.load_dicom_series(series_dir, df_locale=df_locale, df_train=df_train)

            # 進一步將 datasets 組成 volume，並且記錄「resize 前影像的原始 shape」到 df_train
            volume = preprocessor.process_series_with_orig(datasets, series_uid=str(sid), df_train=df_train)

            # 儲存 volume
            npy_path = os.path.join(output_dir, f"{sid}.npy")
            np.save(npy_path, volume)

            # 額外輸出此 series 的 slice 細節（包含有無對應 coordinates）
            #if slice_records:
            #    info_path = os.path.join(output_dir, f"{sid}_slices.csv")
            #    pd.DataFrame(slice_records).to_csv(info_path, index=False)

            print(f"Saved {npy_path} shape={volume.shape}")

        except Exception as e:
            print(f"Failed {sid}: {e}")

    # # 4. 將更新後欄位（z_position、coordinates、orig_height、orig_width、orig_depth）寫回新的 train.csv
    # out_csv = os.path.join(output_dir, "train_with_labels.csv")
    # df_train.to_csv(out_csv, index=False)
    # print(f"Wrote updated CSV -> {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model preprocessing pipeline for RSNA 2025"
    )



    parser.add_argument(
        "--data_folder", type=str,
        default="./data",
        help="Root folder of RSNA Intracranial Aneurysm Detection dataset"
    )
    

    # parser.add_argument(
    #     "--csv_path", type=str,
    #     default="data/train.csv",
    #     help="Path to the main training CSV file containing case metadata."
    # )

    # parser.add_argument(
    #     "--localize_csv_path", type=str,
    #     default="data/train_localizers.csv",
    #     help="Path to the CSV file containing localizer metadata for alignment."
    # )

    # parser.add_argument(
    #     "--series_root", type=str,
    #     default="data/series",
    #     help="Root directory containing DICOM or NIfTI image series."
    # )

    parser.add_argument(
        "--output_dir", type=str,
        default="output/pre_volumes_withlabel_448_64",
        help="Output directory for generated preprocessed volumes with labels."
    )

    args = parser.parse_args()
    main(args)
