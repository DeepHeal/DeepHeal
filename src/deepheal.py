from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import json
import torch.nn as nn
import logging

from .model.model import DeepHealModel
from .model.dataloader import DataLoaderManager
from .model.trainer import Trainer
from .model.augment import MaskNonZerosAugment, FeatureDropAugment
from .utils.other_util import add_file_handler, set_seed

# Initialize logger
logger = logging.getLogger(__name__)

def set_verbose_mode(verbose: bool = True):
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self):
        def serialize(value):
            if isinstance(value, torch.device):
                return str(value)
            if isinstance(value, (pd.Index, pd.CategoricalIndex)):
                return value.tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value
        return {k: serialize(getattr(self, k))
                for k in dir(self)
                if not k.startswith('__') and not callable(getattr(self, k))}

class DeepHeal:
    """
    DeepHeal: Self-supervised representations of drug-response proteomics.
    """
    def __init__(self, save_dir='save/', verbose=False, **kwargs):
        set_verbose_mode(verbose)

        self.config = None
        self.loader = None
        self.model = None
        self.trainer = None
        self.data_manager = None

        # Placeholders for data
        self.input_data = None # DataFrame or Numpy array
        self.meta_data = None  # DataFrame

        if save_dir is not None:
            self.save_dir = Path(save_dir)
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True, exist_ok=True)
            add_file_handler(logger, self.save_dir / "run.log")
        else:
            self.save_dir = None
            logger.warning("save_dir is None. Model and log files will not be saved.")

        self.default_params = dict(
            seed=0,
            input_dim=None,
            batch_size=256,
            n_epochs=15,
            lr=1e-2,
            schedule_ratio=0.97,
            train_frac=1.0,
            latent_dim=32,
            encoder_dims=[1000],
            decoder_dims=[1000],
            element_mask_prob=0.4,
            feature_mask_prob=0.3,
            domain_key=None,
            class_key=None,
            domain_embedding_dim=0, # Default to 0 if not using domain embeddings
            covariate_embedding_dims={},
            use_decoder=False,
            decoder_final_activation='relu',
            decoder_weight=1.0,
            clr_temperature=0.4,
            clr_beta=1.0,
            clr_weight=1.0,
            use_classifier=False,
            classifier_weight=1.0,
            unlabeled_class=None,
            use_importance_mask=False,
            importance_penalty_weight=0,
            importance_penalty_type='L1',
            dropout_prob=0.0,
            norm_type="layer_norm",
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

            # Legacy/Unused params kept for compatibility or future extension if needed,
            # but set to safe defaults for "no batch effect" mode.
            p_intra_knn=0.0,
            p_intra_domain=1.0,
            sampler_knn=None,
            sampler_emb=None,
            sampler_domain_minibatch_strategy='proportional',
            domain_coverage=None,
            dist_metric='euclidean',
            use_faiss=False, # Disable complex dependencies
            use_ivf=False,
            ivf_nprobe=10,
            preload_dense=True,
            num_workers=0
        )

        self.setup_config(**kwargs)
        set_seed(self.config.seed)

    def setup_config(self, **kwargs):
        initial_params = self.default_params.copy()
        # Allow extra kwargs but log warning? Or just ignore/store them.
        initial_params.update(kwargs)
        self.config = Config(initial_params)

    def set_data(self, input_data, meta_data=None):
        """
        Sets the input data for the model.

        Args:
            input_data (pd.DataFrame): The input features (samples x features).
            meta_data (pd.DataFrame, optional): Meta data containing domain/class info.
        """
        self.input_data = input_data
        self.meta_data = meta_data

        # Update input_dim in config
        self.config.input_dim = self.input_data.shape[1]

        # Check domain settings
        if self.config.domain_key:
            if self.meta_data is None or self.config.domain_key not in self.meta_data.columns:
                 # Fallback to single domain if key missing
                logger.warning(f"Domain key {self.config.domain_key} not found in meta data. Treating as single domain.")
                self.config.num_domains = 1
            else:
                self.config.unique_domains = self.meta_data[self.config.domain_key].astype('category').cat.categories.tolist()
                self.config.num_domains = len(self.config.unique_domains)
        else:
            self.config.num_domains = 1

        # Check class settings
        if self.config.use_classifier and self.config.class_key:
             if self.meta_data is not None and self.config.class_key in self.meta_data.columns:
                cats = self.meta_data[self.config.class_key].astype('category').cat.categories.tolist()
                self.config.unique_classes = pd.Index(cats)
                self.config.unique_classes_code = np.arange(len(cats), dtype=int)
                self.config.num_classes = len(cats)
             else:
                raise ValueError("Class key not found in meta data.")
        else:
            self.config.num_classes = None
            self.config.unique_classes_code = None


    def init_model(self):
        if self.config.input_dim is None:
             raise ValueError("Input dimension unknown. Call set_data() first.")

        self.model = DeepHealModel(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.latent_dim,
            num_domains=self.config.num_domains,
            num_classes=self.config.num_classes,
            domain_embedding_dim=self.config.domain_embedding_dim,
            covariate_embedding_dims=self.config.covariate_embedding_dims,
            # covariate_num_categories handled in dataset/loader or ignored if empty
            covariate_num_categories={},
            encoder_dims=self.config.encoder_dims,
            decoder_dims=self.config.decoder_dims,
            decoder_final_activation=self.config.decoder_final_activation,
            dropout_prob=self.config.dropout_prob,
            norm_type=self.config.norm_type,
            use_decoder=self.config.use_decoder,
            use_classifier=self.config.use_classifier,
            use_importance_mask=self.config.use_importance_mask
        ).to(self.config.device)

        logger.info(f'Model initialized on {self.config.device}. Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def init_trainer(self):
        augment = nn.Sequential(
            MaskNonZerosAugment(p=self.config.element_mask_prob),
            FeatureDropAugment(p=self.config.feature_mask_prob)
        )

        self.trainer = Trainer(
            model=self.model,
            data_structure=self.data_structure, # set by data manager
            device=self.config.device,
            logger=logger,
            lr=self.config.lr,
            schedule_ratio=self.config.schedule_ratio,
            augment=augment,
            use_classifier=self.config.use_classifier,
            classifier_weight=self.config.classifier_weight,
            unique_classes=self.config.unique_classes_code,
            unlabeled_class=self.config.unlabeled_class,
            use_decoder=self.config.use_decoder,
            decoder_weight=self.config.decoder_weight,
            clr_temperature=self.config.clr_temperature,
            clr_beta=self.config.clr_beta,
            clr_weight=self.config.clr_weight,
            importance_penalty_weight=self.config.importance_penalty_weight,
            importance_penalty_type=self.config.importance_penalty_type
        )

    def init_dataloader(self, train_frac=1.0):
        self.data_manager = DataLoaderManager(
            input_data=self.input_data,
            meta_data=self.meta_data,
            domain_key=self.config.domain_key,
            class_key=self.config.class_key,
            batch_size=self.config.batch_size,
            train_frac=train_frac,
            device=self.config.device,
            # Pass other config params as needed by the simplified DataLoaderManager
        )
        self.train_loader, self.val_loader, self.data_structure = self.data_manager.get_dataloaders()


    def train(self, save_model=True):
        self.init_model()
        self.init_dataloader(train_frac=self.config.train_frac)
        self.init_trainer()

        for epoch in range(self.config.n_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.n_epochs}")
            self.trainer.train_epoch(epoch, self.train_loader)
            if self.val_loader:
                self.trainer.validate_epoch(epoch, self.val_loader)
            self.trainer.scheduler.step()

        if save_model and self.save_dir:
            self.save_model(self.model, self.save_dir / "model.pt")

    def predict(self, input_data):
        # Helper to predict on new data (or same data)
        # For simplicity, we can use a temporary DataLoaderManager or just manual batching
        self.model.eval()
        data_tensor = torch.tensor(input_data.values, dtype=torch.float32).to(self.config.device)
        dataset = torch.utils.data.TensorDataset(data_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0]
                encoded = self.model.encode(x)
                embeddings.append(encoded.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
