"""CTGAN wrapper for Synthia - compatible with SDV 1.x API."""

import os
from typing import Optional, List
import pandas as pd
import numpy as np


class CTGANEngine:
    """Wrapper around SDV 1.x CTGANSynthesizer for variant generation."""

    def __init__(self, embedding_dim: int = 128, generator_dim: tuple = (256, 256),
                 discriminator_dim: tuple = (256, 256), batch_size: int = 500,
                 epochs: int = 300, verbose: bool = False):
        """Initialize CTGAN engine.

        Args:
            embedding_dim: Dimension of the embedding space
            generator_dim: Dimensions of generator hidden layers
            discriminator_dim: Dimensions of discriminator hidden layers
            batch_size: Batch size for training
            epochs: Number of training epochs
            verbose: Print progress during training
        """
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.batch_size = batch_size
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

    def fit(self, real_data: pd.DataFrame, discrete_columns: List[str],
            conditional_column: str = 'disease') -> None:
        """Train CTGAN on real variant data.

        Args:
            real_data: DataFrame with real variant data
            discrete_columns: List of column names that are discrete/categorical
            conditional_column: Column to use for conditional generation (stored for reference)
        """
        from sdv.single_table import CTGANSynthesizer

        self.discrete_columns = discrete_columns

        print(f"[i] Training CTGAN on {len(real_data)} records...")
        print(f"    - Discrete columns: {discrete_columns}")

        # Build metadata
        self._build_metadata(real_data, discrete_columns)

        # Create and train CTGAN model with SDV 1.x API
        self.model = CTGANSynthesizer(
            metadata=self.metadata_obj,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose
        )

        # Fit the model (SDV 1.x: only takes data, no discrete_columns)
        self.model.fit(real_data)
        print("[+] CTGAN training complete")

    def sample(self, n_samples: int, condition: Optional[str] = None,
               condition_column: str = 'disease') -> pd.DataFrame:
        """Generate synthetic variant records.

        Args:
            n_samples: Number of synthetic records to generate
            condition: Value for conditional generation (unused in SDV 1.x basic sample)
            condition_column: Column name for conditioning (unused in SDV 1.x basic sample)

        Returns:
            DataFrame with synthetic variant records
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        print(f"[i] Generating {n_samples} synthetic records via CTGAN...")

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
        from sdv.single_table import CTGANSynthesizer

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # SDV 1.x has built-in load class method
        self.model = CTGANSynthesizer.load(filepath)
        print(f"[+] Model loaded from {filepath}")

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            'model_type': 'CTGAN',
            'embedding_dim': self.embedding_dim,
            'generator_dim': self.generator_dim,
            'discriminator_dim': self.discriminator_dim,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'discrete_columns': self.discrete_columns
        }
