"""Main orchestrator for synthetic data generation."""

import os
import hashlib
from datetime import datetime
from typing import Optional, List
import pandas as pd
import numpy as np

from src.engines.ctgan_engine import CTGANEngine
from src.engines.tvae_engine import TVAEEngine
from src.models.generation_config import GenerationConfig


class SyntheticDataGenerator:
    """Orchestrates synthetic data generation using CTGAN or TVAE."""

    # List of discrete columns in variant metadata
    DISCRETE_COLUMNS = [
        'gene_symbol',
        'chromosome',
        'variant_type',
        'clinical_significance',
        'disease',
        'inheritance_pattern'
    ]

    def __init__(self, config: Optional[GenerationConfig] = None, random_seed: int = 42):
        """Initialize data generator.

        Args:
            config: GenerationConfig object (optional)
            random_seed: Random seed for reproducibility
        """
        self.config = config or GenerationConfig(
            model_type='CTGAN',
            n_samples=1000,
            random_seed=random_seed
        )
        self.model = None
        self.model_type = self.config.model_type
        self.training_data = None
        self.metadata = {}

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

    def train(self, real_data: pd.DataFrame, config: Optional[GenerationConfig] = None) -> None:
        """Train the generative model on real data.

        Args:
            real_data: DataFrame with real variant records
            config: Optional GenerationConfig to override default
        """
        if config:
            self.config = config
            self.model_type = config.model_type

        self.training_data = real_data.copy()

        # Compute data hash for reproducibility tracking
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(real_data, index=True).values.tobytes()
        ).hexdigest()

        self.metadata = {
            'model_type': self.model_type,
            'random_seed': self.config.random_seed,
            'training_data_hash': data_hash,
            'timestamp': datetime.now().isoformat(),
            'algorithm_version': '1.0.0',
            'hyperparameters': self.config.hyperparameters,
            'discrete_columns': self.DISCRETE_COLUMNS,
            'training_records': len(real_data)
        }

        print(f"[i] Training model: {self.model_type}")
        print(f"[i] Training data: {len(real_data)} records")
        print(f"[i] Random seed: {self.config.random_seed}")

        # Initialize and train appropriate model
        if self.model_type == 'CTGAN':
            self.model = CTGANEngine(
                embedding_dim=self.config.hyperparameters.get('embedding_dim', 128),
                generator_dim=tuple(self.config.hyperparameters.get('generator_dim', [256, 256])),
                discriminator_dim=tuple(self.config.hyperparameters.get('discriminator_dim', [256, 256])),
                batch_size=self.config.hyperparameters.get('batch_size', 500),
                epochs=self.config.hyperparameters.get('epochs', 300),
                verbose=False
            )
            self.model.fit(
                real_data,
                discrete_columns=self.DISCRETE_COLUMNS,
                conditional_column=self.config.disease_condition or 'disease'
            )
        elif self.model_type == 'TVAE':
            self.model = TVAEEngine(
                embedding_dim=self.config.hyperparameters.get('embedding_dim', 128),
                compress_dims=tuple(self.config.hyperparameters.get('compress_dims', [128, 128])),
                decompress_dims=tuple(self.config.hyperparameters.get('decompress_dims', [128, 128])),
                latent_dim=self.config.hyperparameters.get('latent_dim', 128),
                epochs=self.config.hyperparameters.get('epochs', 300),
                verbose=False
            )
            self.model.fit(real_data, discrete_columns=self.DISCRETE_COLUMNS)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print("[+] Model training complete")

    def generate(self, n_samples: Optional[int] = None,
                 condition: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic variant records.

        Args:
            n_samples: Number of samples to generate. If None, uses config.n_samples
            condition: Disease condition for CTGAN generation (optional)

        Returns:
            DataFrame with synthetic variant records
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        n_samples = n_samples or self.config.n_samples

        print(f"[i] Generating {n_samples} synthetic records...")

        # Generate data based on model type
        if self.model_type == 'CTGAN':
            synthetic = self.model.sample(n_samples, condition=condition)
        elif self.model_type == 'TVAE':
            synthetic = self.model.sample(n_samples)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print(f"[+] Generation complete: {len(synthetic)} records")

        # Validate output
        self._validate_synthetic_data(synthetic)

        return synthetic

    def _validate_synthetic_data(self, synthetic: pd.DataFrame) -> None:
        """Validate synthetic data has correct schema.

        Args:
            synthetic: DataFrame to validate
        """
        required_fields = [
            'gene_symbol', 'chromosome', 'variant_type',
            'clinical_significance', 'disease',
            'allele_frequency', 'inheritance_pattern'
        ]

        # Check all fields present
        missing_fields = set(required_fields) - set(synthetic.columns)
        if missing_fields:
            raise ValueError(f"Synthetic data missing fields: {missing_fields}")

        # Check no null values
        null_values = synthetic.isnull().sum().sum()
        if null_values > 0:
            print(f"[!] Warning: {null_values} null values in synthetic data")

        # Validate allele_frequency range
        af_col = synthetic['allele_frequency']
        out_of_range = ((af_col < 0) | (af_col > 1)).sum()
        if out_of_range > 0:
            print(f"[!] Warning: {out_of_range} allele frequencies out of range [0, 1]")

    def save_model(self, filepath: str) -> None:
        """Save trained model.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        os.makedirs(os.path.dirname(filepath) or 'models', exist_ok=True)
        self.model.save_model(filepath)
        print(f"[+] Model saved to {filepath}")

    def get_metadata(self) -> dict:
        """Get generation metadata.

        Returns:
            Dictionary with generation and model metadata
        """
        return self.metadata.copy()

    def get_progress_estimate(self, n_samples: int) -> str:
        """Estimate generation time.

        Args:
            n_samples: Number of samples

        Returns:
            Estimated time string
        """
        # Rough heuristic: ~1 second per 100 records for CTGAN, faster for TVAE
        if self.model_type == 'CTGAN':
            est_seconds = max(5, (n_samples / 100))
        else:  # TVAE
            est_seconds = max(3, (n_samples / 150))

        if est_seconds < 60:
            return f"~{int(est_seconds)} seconds"
        else:
            return f"~{int(est_seconds / 60)}-{int((est_seconds + 60) / 60)} minutes"
