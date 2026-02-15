"""Generation configuration model for Synthia."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Supported generative models."""
    CTGAN = 'CTGAN'
    TVAE = 'TVAE'


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation.

    Attributes:
        model_type: Type of generative model ('CTGAN' or 'TVAE')
        n_samples: Number of synthetic records to generate
        disease_condition: Disease to condition generation on (optional)
        random_seed: Random seed for reproducibility
        hyperparameters: Model-specific hyperparameters
        training_data_hash: Hash of training data for tracking (optional)
        timestamp: Generation timestamp (auto-set if not provided)
    """

    model_type: str
    n_samples: int
    random_seed: int
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    disease_condition: Optional[str] = None
    training_data_hash: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Validate configuration and set timestamp."""
        self.validate()
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def validate(self):
        """Validate configuration values."""
        errors = []

        # Validate model_type
        if self.model_type not in [m.value for m in ModelType]:
            errors.append(f'model_type must be one of {[m.value for m in ModelType]}')

        # Validate n_samples
        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            errors.append('n_samples must be a positive integer')

        # Validate random_seed
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            errors.append('random_seed must be a non-negative integer')

        # Validate hyperparameters
        if not isinstance(self.hyperparameters, dict):
            errors.append('hyperparameters must be a dictionary')

        if errors:
            raise ValueError(f'GenerationConfig validation failed: {"; ".join(errors)}')

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type,
            'n_samples': self.n_samples,
            'random_seed': self.random_seed,
            'disease_condition': self.disease_condition,
            'hyperparameters': self.hyperparameters,
            'training_data_hash': self.training_data_hash,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GenerationConfig':
        """Create from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'GenerationConfig(model={self.model_type}'
            f', n_samples={self.n_samples}'
            f', seed={self.random_seed}'
            f', disease={self.disease_condition}'
            f', timestamp={self.timestamp})'
        )
