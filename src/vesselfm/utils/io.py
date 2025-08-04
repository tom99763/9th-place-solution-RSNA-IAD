"""Reader/writer classes; we follow https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/imageio"""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class BaseReaderWriter(ABC):
    supported_file_formats = []
    
    @staticmethod
    def _check_all_same(lst: List) -> bool:
        """
        Check if all elements in a list are the same
        Args:
            lst: List of elements
        Returns:
            Boolean indicating if all elements are the same
        """
        return all(x == lst[0] for x in lst)
    
    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Read images from disk
        Args:
            image_fnames: List of image filenames
        Returns:
            Tuple of numpy array and dictionary
        """
        pass

    @abstractmethod
    def read_segs(self, seg_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Read segmentations from disk
        Args:
            seg_fnames: List of segmentation filenames
        Returns:
            Tuple of numpy array and dictionary
        """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, seg_fname: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Write segmentation to disk
        Args:
            seg: Segmentation array
            seg_fname: Segmentation filename
            metadata: Metadata dictionary
        """
        pass


class NumpyReaderWriter(BaseReaderWriter):
    supported_file_formats = ["npy", "npz"]

    def __init__(self):
        super().__init__()

    def read_images(
        self, image_fnames: Union[str, list[str]], metdata_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read images from disk

        Args:
            image_fnames: List of image filenames
        Returns:
            Tuple of numpy array and dictionary
        """
        image_data = []
        if type(image_fnames) is str:
            image_fnames = [image_fnames]

        for image_fname in image_fnames:

            file_extension = os.path.basename(image_fname).split(".")[-1]
            if file_extension not in self.supported_file_formats:
                raise RuntimeError(f"File format not supported for {image_fname}")

            if file_extension == "npy":
                image_data.append(self._load_npy(image_fname))
                if image_data[-1].ndim != 3:
                    raise RuntimeError(
                        f"Image {image_fname} has dimension {image_data[-1].ndim}, expected 3"
                    )
            elif file_extension == "npz":
                new_images = self._load_npz(image_fname)
                for image in new_images:
                    if image.ndim != 3:
                        raise RuntimeError(
                            f"Image in {image_fname} has dimension {image.ndim}, expected 3"
                        )
                image_data.extend(new_images)

        if len(image_data) > 1:
            image_data = np.vstack(image_data)
        else:
            image_data = image_data[0]

        if metdata_path is not None:
            with open(metdata_path, "r") as f:
                metadata = json.load(f)
            spacing = metadata["spacing"]
            return image_data, {"spacing": spacing, "other": metadata}
        else:
            spacing = [1, 1, 1]
            return image_data, {"spacing": spacing}

    def _load_npy(self, fname: str) -> np.ndarray:
        return np.load(fname)

    def _load_npz(self, fname: str) -> list[np.ndarray]:
        file = np.load(fname)
        array = []
        for key in file.files:
            array.append(file[key])
        return array

    def read_segs(
        self, seg_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.read_images(seg_fnames)

    def write_seg(
        self,
        seg: np.ndarray,
        seg_fname: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        file_extension = os.path.basename(seg_fname).split(".")[-1]
        if file_extension != ".npy":
            raise RuntimeError(
                f"File format {file_extension} not supported,  saving {seg_fname} failed!"
            )
        if metadata is not None:
            spacing = metadata["spacing"]
            with open(seg_fname, "wb") as f:
                np.save(f, seg)
            with open(seg_fname.replace(".npy", ".json"), "w") as f:
                json.dump(metadata, f)
        else:
            with open(seg_fname, "wb") as f:
                np.save(f, seg)


class NumpySeriesReaderWriter(BaseReaderWriter):
    supported_file_formats = ["npy_series", "npz_series"]
    slice_formats = ["npy", "npz"]

    def __init__(self):
        super().__init__()

    def read_images(
        self,
        image_folder: str,
        metdata_path: Optional[str] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        """
        Read a series of images from disk given a the folder and optionally a start and end index.

        Args:
            image_folder: Folder containing the images
            start_idx: Start index of the images to read
            end_idx: End index of the images to read
        Returns:
            Tuple of numpy array and dictionary
        """
        image_files = os.listdir(image_folder)
        image_files = [
            f for f in image_files if f.endswith(".npy") or f.endswith(".npz")
        ]
        if start_idx is not None:
            image_files = [f for f in image_files if int(f.split(".")[0]) >= start_idx]
        if end_idx is not None:
            image_files = [f for f in image_files if int(f.split(".")[0]) < end_idx]
        image_files.sort()
        image_files = [os.path.join(image_folder, f) for f in image_files]
        image_data = []
        for image_fname in image_files:
            file_extension = os.path.basename(image_fname).split(".")[-1].lower()
            if file_extension not in self.slice_formats:
                raise RuntimeError(f"File format not supported for {image_fname}")
            if file_extension == "npy":
                image = self._load_npy(image_fname)
                image_data.append(image)
                if image_data[-1].ndim != 2:
                    raise RuntimeError(
                        f"Image {image_fname} has dimension {image_data[-1].ndim}, expected 2"
                    )
            elif file_extension == "npz":
                new_images = self._load_npz(image_fname)
                for image in new_images:
                    if image.ndim != 2:
                        raise RuntimeError(
                            f"Image in {image_fname} has dimension {image.ndim}, expected 2"
                        )
                image_data.extend(new_images)
        if metdata_path is not None:
            with open(metdata_path, "r") as f:
                metadata = json.load(f)
            return np.stack(image_data, axis=0), metadata
        else:
            spacing = [1, 1, 1]
            return np.stack(image_data, axis=0), {"spacing": spacing}

    def _load_npy(self, fname: str) -> np.ndarray:
        return np.load(fname)

    def _load_npz(self, fname: str) -> list[np.ndarray]:
        file = np.load(fname)
        array = []
        for key in file.files:
            array.append(file[key])
        return array

    def read_segs(
        self,
        seg_fnames: Union[str, list[str]],
        metadata: Optional[Dict[str, Any]] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.read_images(seg_fnames, metadata, start_idx, end_idx)

    def _save_npy_series(self, array: np.ndarray, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        for i in range(array.shape[0]):
            np.save(os.path.join(output_folder, f"{i}.npy"), array[i])

    def write_seg(
        self,
        seg: np.ndarray,
        seg_fname: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if metadata is not None:
            self._save_npy_series(seg, seg_fname)
            json.dump(metadata, os.path.join(seg_fname, "metadata.json"))
        else:
            self._save_npy_series(seg, seg_fname)


class SimpleITKReaderWriter(BaseReaderWriter):
    supported_file_formats = ["nii", "nii.gz", "mha", "mhd", "nrrd", "gz"]

    def __init__(self):
        super().__init__()

    def read_images(
        self, image_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read images from disk

        Args:
            image_fnames: List of image filenames
        Returns:
            Tuple of numpy array and dictionary
        """
        if type(image_fnames) is not list:
            image_fnames = [image_fnames]
        image_data = []
        image_metadata = {"spacing": [], "origin": [], "direction": []}
        for image_fname in image_fnames:

            if (
                os.path.basename(image_fname).split(".")[-1]
                not in self.supported_file_formats
            ):
                raise RuntimeError(f"File format not supported for {image_fname}")

            image = sitk.ReadImage(image_fname)
            logger.debug(f"Image {image_fname} has shape {image.GetSize()}")
            image_data.append(sitk.GetArrayFromImage(image))

            if image_data[-1].ndim != 3:
                raise RuntimeError(
                    f"Image {image_fname} has dimension {image_data[-1].ndim}, expected 3"
                )

            image_metadata["spacing"].append(image.GetSpacing())
            image_metadata["origin"].append(image.GetOrigin())
            image_metadata["direction"].append(image.GetDirection())

        if not self._check_all_same(image_metadata["spacing"]):
            logger.error("Spacing is not the same for all images")
            raise RuntimeError("Spacing is not the same for all images")
        if not self._check_all_same(image_metadata["origin"]):
            logger.warning("Origin is not the same for all images")
            logger.warning("Please check if this is expected behavior")
        if not self._check_all_same(image_metadata["direction"]):
            logger.warning("Direction is not the same for all images")
            logger.warning("Please check if this is expected behavior")

        sitk_metadata = {}
        for key in image_metadata.keys():
            sitk_metadata[key] = image_metadata[key][0]
        spacing = [
            sitk_metadata["spacing"][2],
            sitk_metadata["spacing"][0],
            sitk_metadata["spacing"][1],
        ]
        meta_data = {"spacing": spacing, "other": sitk_metadata}
        logger.debug(f"Spacing: {spacing}")
        logger.debug(f"Final shape: {np.vstack(image_data).shape}")
        return np.vstack(image_data), meta_data

    def read_segs(
        self, seg_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read segmentations from disk
        Args:
            seg_fnames: List of segmentation filenames
        Returns:
            Tuple of numpy array and dictionary
        """
        return self.read_images(seg_fnames)

    def write_seg(
        self,
        seg: np.ndarray,
        seg_fname: str,
        metadata: Optional[Dict[str, Any]] = None,
        compression: bool = True,
    ):
        """
        Write segmentation to disk
        Args:
            seg: Segmentation array
            seg_fname: Segmentation filename
            metadata: Metadata dictionary
        """
        seg = sitk.GetImageFromArray(seg)
        if metadata is not None:
            seg.SetSpacing(metadata["spacing"])
            seg.SetOrigin(metadata["origin"])
            seg.SetDirection(metadata["direction"])

        sitk.WriteImage(seg, seg_fname, compression)


def determine_reader_writer(file_ending: str):
    LIST_OF_READERS_WRITERS = [
        NumpyReaderWriter,
        SimpleITKReaderWriter,
        NumpySeriesReaderWriter,
    ]

    for reader_writer in LIST_OF_READERS_WRITERS:
        if file_ending.lower() in reader_writer.supported_file_formats:
            logger.debug(
                f"Automatically determined reader_writer: {reader_writer.__name__} for file ending: {file_ending}"
            )
            return reader_writer

    raise ValueError(f"No reader_writer found for file ending: {file_ending}")
