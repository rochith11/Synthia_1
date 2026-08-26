"""Data loader utilities for Synthia."""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(file_path: str) -> Path:
    """Resolve relative file paths from the project root."""
    path = Path(file_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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

def create_train_test_split(
    input_file="data/real_variants_preprocessed.csv",
    train_file="data/sample_real_variants_train.csv",
    test_file="data/sample_real_variants_test.csv",
    test_size=0.30,
    random_seed=42
):
    """Create and save train/test datasets."""
    input_path = _resolve_project_path(input_file)
    train_path = _resolve_project_path(train_file)
    test_path = _resolve_project_path(test_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found at {input_path}")

    df = pd.read_csv(input_path)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Total records: {len(df)}")
    print(f"[+] Training records: {len(train_df)}")
    print(f"[+] Test records: {len(test_df)}")

    return train_df, test_df

if __name__ == "__main__":
    create_train_test_split()