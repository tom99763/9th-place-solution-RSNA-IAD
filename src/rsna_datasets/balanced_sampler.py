import torch
import numpy as np
from torch.utils.data import Sampler
import random
import pandas as pd


class SimpleImbalanceSampler(Sampler):
    """Simple sampler for imbalanced binary labels with a target positive ratio.

    Designed to be flexible across different dataset structures:
    - Tries to locate a pandas DataFrame on the dataset among common attribute
      names (slice_df, df, meta, items, data_df).
    - Tries to find a binary label column among common names (has_aneurysm,
      series_has_aneurysm, aneurysm, target, label, y).
    - You can override both the dataframe attribute name and the label column.

    It avoids calling dataset.__getitem__ to prevent expensive I/O (e.g. loading
    .npz or .npy files) just to read labels.
    """

    DATAFRAME_ATTR_CANDIDATES = ["slice_df", "df", "meta", "items", "data_df"]
    LABEL_COL_CANDIDATES = [
        "has_aneurysm",
        "series_has_aneurysm",
        "aneurysm",
        "target",
        "label",
        "y",
    ]

    def __init__(
        self,
        dataset,
        pos_ratio: float = 0.8,
        samples_per_epoch: int | None = None,
        *,
        dataframe_attr: str | None = None,
        label_column: str | None = None,
        positive_values=(1, True, "1", "true", "True"),
        seed: int | None = None,
    ):
        """Args:
        dataset: The dataset instance (must expose a pandas DataFrame with labels).
        pos_ratio: Desired fraction of positive samples in each epoch (0-1).
        samples_per_epoch: Total samples per epoch (defaults to len(dataset)).
        dataframe_attr: Explicit attribute name for underlying DataFrame.
        label_column: Explicit column name containing binary labels.
        positive_values: Values interpreted as positive (for non-numeric labels).
        seed: Optional RNG seed for reproducibility of sampling.
        """
        super().__init__(data_source=None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if not (0 < pos_ratio < 1):
            raise ValueError(f"pos_ratio must be in (0,1), got {pos_ratio}")

        self.dataset = dataset
        self.pos_ratio = float(pos_ratio)

        # Resolve dataframe
        df_attr = dataframe_attr
        df_obj = None
        candidates_checked = []
        if df_attr is not None:
            if not hasattr(dataset, df_attr):
                raise AttributeError(f"Dataset has no attribute '{df_attr}' for dataframe override")
            df_obj = getattr(dataset, df_attr)
        else:
            for cand in self.DATAFRAME_ATTR_CANDIDATES:
                candidates_checked.append(cand)
                if hasattr(dataset, cand):
                    potential = getattr(dataset, cand)
                    if isinstance(potential, pd.DataFrame):
                        df_obj = potential
                        df_attr = cand
                        break
        if df_obj is None:
            raise ValueError(
                "Could not locate a pandas DataFrame on dataset. Tried: "
                + ", ".join(candidates_checked)
            )
        if not isinstance(df_obj, pd.DataFrame):
            raise TypeError(f"Resolved attribute '{df_attr}' is not a pandas DataFrame")
        self.df = df_obj.reset_index(drop=True)

        # Resolve label column
        label_col = label_column
        if label_col is None:
            for cand in self.LABEL_COL_CANDIDATES:
                if cand in self.df.columns:
                    label_col = cand
                    break
        if label_col is None:
            raise ValueError(
                "Could not find a label column. Provide 'label_column' explicitly. "
                f"Looked for: {self.LABEL_COL_CANDIDATES}. Columns: {list(self.df.columns)}"
            )
        self.label_column = label_col

        # Convert labels to binary 0/1
        col_data = self.df[self.label_column]
        # If dtype numeric: >0 considered positive. Else check membership in positive_values.
        if np.issubdtype(col_data.dtype, np.number):
            labels = (col_data.astype(float) > 0).astype(int).to_numpy()
        else:
            labels = col_data.apply(lambda v: int(v in positive_values)).to_numpy()

        # Build index lists
        self.positive_indices = np.where(labels == 1)[0].tolist()
        self.negative_indices = np.where(labels == 0)[0].tolist()
        self.n_pos = len(self.positive_indices)
        self.n_neg = len(self.negative_indices)

        if self.n_pos == 0 or self.n_neg == 0:
            raise ValueError(
                f"Dataset must contain both positive and negative samples. Got {self.n_pos} pos / {self.n_neg} neg"
            )

        # Determine epoch size
        if samples_per_epoch is None:
            samples_per_epoch = len(self.df)
        self.total_samples = int(samples_per_epoch)
        self.pos_samples = int(round(self.total_samples * self.pos_ratio))
        self.neg_samples = self.total_samples - self.pos_samples

        # Logging
        print(
            f"SimpleImbalanceSampler: dataframe_attr='{df_attr}', label_column='{self.label_column}'"  # noqa: E501
        )
        print(
            f"Targeting {self.pos_samples} pos + {self.neg_samples} neg = {self.total_samples} (pos_ratio={self.pos_ratio:.3f})"
        )
        print(f"Available: {self.n_pos} pos, {self.n_neg} neg")
        if self.pos_samples > self.n_pos:
            print(
                f"Will oversample positives: need {self.pos_samples}, have {self.n_pos} (replacement=True)"
            )
        if self.neg_samples > self.n_neg:
            print(
                f"Will oversample negatives: need {self.neg_samples}, have {self.n_neg} (replacement=True)"
            )
    
    def __iter__(self):
        # Sample positives & negatives (with replacement if required)
        pos_sampled = np.random.choice(
            self.positive_indices,
            size=self.pos_samples,
            replace=self.pos_samples > self.n_pos,
        )
        neg_sampled = np.random.choice(
            self.negative_indices,
            size=self.neg_samples,
            replace=self.neg_samples > self.n_neg,
        )
        all_indices = np.concatenate([pos_sampled, neg_sampled])
        np.random.shuffle(all_indices)
        return iter(all_indices.tolist())
    
    def __len__(self):
        return self.total_samples