"""Data loader utilities for Synthia."""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

def load_sample_data(data_dir: str = 'data') -> pd.DataFrame:
    """Load the full sample variant dataset.

    Args:
        data_dir: Directory containing data files

    Returns:
        DataFrame with all sample variants
    """
    sample_file = Path(data_dir) / 'sample_real_variants.csv'
    if sample_file.exists():
        return pd.read_csv(sample_file)
    else:
        raise FileNotFoundError(f"Sample data not found at {sample_file}")


def load_training_data(data_dir: str = 'data') -> pd.DataFrame:
    """Load the already-created training dataset."""
    train_file = Path(data_dir) / 'sample_real_variants_train.csv'

    if train_file.exists():
        return pd.read_csv(train_file)

    raise FileNotFoundError(
        f"Training data not found at {train_file}"
    )


def load_test_data(data_dir: str = 'data') -> pd.DataFrame:
    """Load the already-created test dataset."""
    test_file = Path(data_dir) / 'sample_real_variants_test.csv'

    if test_file.exists():
        return pd.read_csv(test_file)

    raise FileNotFoundError(
        f"Test data not found at {test_file}"
    )

