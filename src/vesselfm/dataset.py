import logging
from typing import Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.io import determine_reader_writer
from utils.data import generate_transforms

logger = logging.getLogger(__name__)

class RSNASegDataset(Dataset):
    def __init__(self, uids, dataset_config, mode):
        super().__init__()
        # init datasets
        self.data_path = dataset_config.path
        self.uids = uids
        self.reader = determine_reader_writer(dataset_config.file_format)()
        self.transforms = generate_transforms(dataset_config.transforms[mode])

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        vol_path = f'{self.data_path}/{uid}/{uid}.nii'
        mask_path = f'{self.data_path}/{uid}/{uid}_cowseg.nii'
        vol = self.reader.read_images(vol_path)[0].astype(np.float32)
        mask = self.reader.read_images(mask_path)[0].astype(int)
        transformed = self.transforms({'Image': vol, 'Mask': mask})
        return transformed['Image'], transformed['Mask'] > 0



class UnionDataset(Dataset):
    """
    Dataset that accumulates all given datasets.
    """

    def __init__(self, dataset_configs, mode, finetune=False):
        super().__init__()
        # init datasets
        self.finetune = finetune
        self.datasets, probs = [], []
        self.len = 0
        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            paths = sorted(list(data_dir.iterdir()))  # ensures that we use same 1-shot sample

            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "reader": determine_reader_writer(dataset_config.file_format)(),
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs
                }
            )
            probs.append(dataset_config.sample_prop)

        # ensure that probs sum up to 1
        probs = torch.tensor(probs)
        self.probs = probs / probs.sum()

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # sample dataset
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        # sample data sample
        while True:
            data_idx = idx if self.finetune else torch.randint(0, len(dataset["paths"]), (1,)).item()
            sample_id = dataset["paths"][data_idx]

            img_path = [path for path in sample_id.iterdir() if 'img' in path.name][0]
            mask_path = [path for path in sample_id.iterdir() if 'mask' in path.name][0]

            if dataset['filter_dataset_IDs'] is not None:
                if int(img_path.stem.split("_")[-1]) in dataset['filter_dataset_IDs']:
                    continue

            img = dataset['reader'].read_images(str(img_path))[0].astype(np.float32)
            mask = dataset['reader'].read_images(str(mask_path))[0].astype(bool)

            transformed = dataset['transforms']({'Image': img, 'Mask': mask})
            return transformed['Image'], transformed['Mask'] > 0
