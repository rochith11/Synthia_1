"""TVAE wrapper for Synthia - compatible with SDV 1.x API."""

import os
from typing import List
import pandas as pd
import numpy as np


class TVAEEngine:
    """Wrapper around SDV 1.x TVAESynthesizer for variant generation."""

    def __init__(self, embedding_dim: int = 128, compress_dims: tuple = (128, 128),
                 decompress_dims: tuple = (128, 128), latent_dim: int = 128,
                 epochs: int = 300, verbose: bool = False):
        """Initialize TVAE engine.

        Args:
            embedding_dim: Dimension of the embedding space
            compress_dims: Dimensions of encoder hidden layers
            decompress_dims: Dimensions of decoder hidden layers
            latent_dim: Dimension of latent space (not used in SDV 1.x TVAE, kept for config)
            epochs: Number of training epochs
            verbose: Print progress during training
        """
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.verbose = verbose
        self.model = None
        self.metadata_obj = None
        self.discrete_columns = []

    def _build_metadata(self, real_data: pd.DataFrame, discrete_columns: List[str]):
        """Build SDV Metadata from data.

        Args:
            real_data: DataFrame with real variant data
            discrete_columns: List of categorical column names
        """
        from sdv.metadata import Metadata

        self.metadata_obj = Metadata()
        self.metadata_obj.detect_table_from_dataframe(table_name='variants', data=real_data)

        # Explicitly set categorical columns
        for col in discrete_columns:
            if col in real_data.columns:
                self.metadata_obj.update_column(
                    table_name='variants', column_name=col, sdtype='categorical'
                )

        # Set numerical columns
        numerical_cols = [c for c in real_data.columns if c not in discrete_columns]
        for col in numerical_cols:
            if col in real_data.columns:
                self.metadata_obj.update_column(
                    table_name='variants', column_name=col, sdtype='numerical'
                )

    def fit(self, real_data: pd.DataFrame, discrete_columns: List[str]) -> None:
        """Train TVAE on real variant data.

        Args:
            real_data: DataFrame with real variant data
            discrete_columns: List of column names that are discrete/categorical
        """
        from sdv.single_table import TVAESynthesizer

        self.discrete_columns = discrete_columns

        print(f"[i] Training TVAE on {len(real_data)} records...")
        print(f"    - Discrete columns: {discrete_columns}")

        # Build metadata
        self._build_metadata(real_data, discrete_columns)

        # Create and train TVAE model with SDV 1.x API
        self.model = TVAESynthesizer(
            metadata=self.metadata_obj,
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            epochs=self.epochs,
            verbose=self.verbose
        )

        # Fit the model (SDV 1.x: only takes data)
        self.model.fit(real_data)
        print("[+] TVAE training complete")

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic variant records.

        Args:
            n_samples: Number of synthetic records to generate

        Returns:
            DataFrame with synthetic variant records
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        print(f"[i] Generating {n_samples} synthetic records via TVAE...")

        # SDV 1.x: sample uses num_rows parameter
        synthetic = self.model.sample(num_rows=n_samples)

        print(f"[+] Generated {len(synthetic)} records")
        return synthetic

    def save_model(self, filepath: str) -> None:
        """Save trained model.

        Args:
            filepath: Path to save model file
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        # SDV 1.x has built-in save method
        self.model.save(filepath)
        print(f"[+] Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model.

        Args:
            filepath: Path to model file
        """
        from sdv.single_table import TVAESynthesizer

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # SDV 1.x has built-in load class method
        self.model = TVAESynthesizer.load(filepath)
        print(f"[+] Model loaded from {filepath}")

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            'model_type': 'TVAE',
            'embedding_dim': self.embedding_dim,
            'compress_dims': self.compress_dims,
            'decompress_dims': self.decompress_dims,
            'latent_dim': self.latent_dim,
            'epochs': self.epochs,
            'discrete_columns': self.discrete_columns
        }
