import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from .dataset import TabularDataset

logger = logging.getLogger(__name__)

class DataLoaderManager:
    """
    Manages data loading for DeepHeal.
    """
    def __init__(self,
                 input_data,
                 meta_data=None,
                 domain_key=None,
                 class_key=None,
                 covariate_keys=None,
                 batch_size=256,
                 train_frac=1.0,
                 device=None,
                 **kwargs):
        """
        Args:
            input_data (pd.DataFrame): Input features.
            meta_data (pd.DataFrame): Meta data.
            domain_key (str): Key for domain column.
            class_key (str): Key for class column.
            covariate_keys (list): Keys for covariate columns.
            batch_size (int): Batch size.
            train_frac (float): Fraction of data for training.
            device (torch.device): Device.
        """
        self.input_data = input_data
        self.meta_data = meta_data
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys or []
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.device = device

        self.data_structure = ['input', 'idx']
        if self.domain_key: self.data_structure.append('domain')
        if self.class_key: self.data_structure.append('class')
        if self.covariate_keys: self.data_structure.extend(self.covariate_keys)

    def _prepare_tensors(self):
        # 1. Input Features
        if hasattr(self.input_data, 'values'):
            X = self.input_data.values.astype(np.float32)
        else:
            X = self.input_data.astype(np.float32)
        data_tensor = torch.from_numpy(X)

        # 2. Metadata
        domain_tensor = None
        class_tensor = None
        covariate_tensors = {}

        if self.meta_data is not None:
            if self.domain_key and self.domain_key in self.meta_data:
                # Assume already categorical or mapped?
                # Better to map to codes again to be safe/consistent
                codes = self.meta_data[self.domain_key].astype('category').cat.codes.values
                domain_tensor = torch.tensor(codes, dtype=torch.long)

            if self.class_key and self.class_key in self.meta_data:
                codes = self.meta_data[self.class_key].astype('category').cat.codes.values
                class_tensor = torch.tensor(codes, dtype=torch.long)

            for key in self.covariate_keys:
                if key in self.meta_data:
                    codes = self.meta_data[key].astype('category').cat.codes.values
                    covariate_tensors[key] = torch.tensor(codes, dtype=torch.long)

        return data_tensor, domain_tensor, class_tensor, covariate_tensors

    def get_dataloaders(self):
        data_tensor, domain_tensor, class_tensor, covariate_tensors = self._prepare_tensors()

        dataset = TabularDataset(data_tensor, domain_tensor, class_tensor, covariate_tensors)

        if self.train_frac == 1.0:
            train_dataset = dataset
            val_dataset = None
        else:
            total_len = len(dataset)
            train_len = int(total_len * self.train_frac)
            perm = torch.randperm(total_len)
            train_idx = perm[:train_len]
            val_idx = perm[train_len:]

            train_dataset = dataset.subset(train_idx)
            val_dataset = dataset.subset(val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0, # Simple for now
            pin_memory=True if self.device and self.device.type == 'cuda' else False
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device and self.device.type == 'cuda' else False
            )

        return train_loader, val_loader, self.data_structure
