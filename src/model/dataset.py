import torch
from torch.utils.data import Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TabularDataset(Dataset):
    """
    A PyTorch Dataset for tabular data (features + metadata).
    """
    def __init__(self, data_tensor, domain_tensor=None, class_tensor=None, covariate_tensors=None):
        """
        Args:
            data_tensor (torch.Tensor): Input features (N x F).
            domain_tensor (torch.Tensor, optional): Domain labels (N).
            class_tensor (torch.Tensor, optional): Class labels (N).
            covariate_tensors (dict, optional): Dict of tensors for other covariates.
        """
        self.data = data_tensor
        self.domain_labels = domain_tensor
        self.class_labels = class_tensor
        self.covariate_tensors = covariate_tensors if covariate_tensors is not None else {}
        self.indices = torch.arange(len(self.data), dtype=torch.long)

        # Validation
        if self.domain_labels is not None:
            assert len(self.domain_labels) == len(self.data)
        if self.class_labels is not None:
            assert len(self.class_labels) == len(self.data)
        for k, v in self.covariate_tensors.items():
            assert len(v) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a dictionary compatible with the model's forward pass
        batch = {
            'input': self.data[idx],
            'idx': self.indices[idx]
        }

        if self.domain_labels is not None:
            batch['domain'] = self.domain_labels[idx]
        else:
            # If no domain provided, return None or handle in collate/model
            # Model expects domain_labels. If num_domains=1, maybe we should provide dummy?
            # Or model handles None? Model handles None if domain_embedding_dim=0.
            pass

        if self.class_labels is not None:
            batch['class'] = self.class_labels[idx]

        for key, tensor in self.covariate_tensors.items():
            batch[key] = tensor[idx]

        return batch

    def subset(self, indices):
        return TabularDataset(
            self.data[indices],
            self.domain_labels[indices] if self.domain_labels is not None else None,
            self.class_labels[indices] if self.class_labels is not None else None,
            {k: v[indices] for k, v in self.covariate_tensors.items()}
        )
