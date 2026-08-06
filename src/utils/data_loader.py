"""Data loader utilities for Synthia."""

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml


def _resolve_data_file(data_dir: str, candidates: list[str]) -> Optional[Path]:
    """Resolve the first existing data file from a list of candidates."""
    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)
        if path.is_absolute():
            if path.exists():
                return path
            continue

        for base in (Path.cwd(), Path(data_dir)):
            resolved = (base / path).resolve()
            if resolved.exists():
                return resolved

        if path.exists():
            return path.resolve()

    return None


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


def load_training_data(data_dir: str = 'data', training_file: Optional[str] = None) -> pd.DataFrame:
    """Load training data for model training.

    Args:
        data_dir: Directory containing data files
        training_file: Optional explicit path to a training CSV file

    Returns:
        Training DataFrame
    """
    candidates = []

    if training_file:
        candidates.append(training_file)

    env_path = os.getenv('SYNTHIA_TRAINING_DATA')
    if env_path:
        candidates.append(env_path)

    config_path = Path(__file__).resolve().parents[2] / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle) or {}
        configured_path = config.get('data', {}).get('sample_dir')
        if configured_path:
            candidates.append(configured_path)

    candidates.extend([
        str(Path(data_dir) / 'training.csv'),
        str(Path(data_dir) / 'sample_real_variants_train.csv'),
    ])

    resolved_path = _resolve_data_file(data_dir, candidates)
    if resolved_path is None:
        raise FileNotFoundError(f"Training data not found. Tried: {candidates}")

    return pd.read_csv(resolved_path)


def load_test_data(data_dir: str = 'data') -> pd.DataFrame:
    """Load test split (30% of data).

    Args:
        data_dir: Directory containing data files

    Returns:
        Test DataFrame
    """
    test_file = Path(data_dir) / 'sample_real_variants_test.csv'
    if test_file.exists():
        return pd.read_csv(test_file)
    else:
        raise FileNotFoundError(f"Test data not found at {test_file}")


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create and save sample variant data with train/test split.

    Returns:
        Tuple of (full_data, train_data, test_data)
    """
    import numpy as np

    # Random seed for reproducibility
    np.random.seed(42)

    # Sample data configuration
    genes = ['CFTR', 'DMD', 'HBB', 'F8', 'HEXA']
    chromosomes = ['chr7', 'chrX', 'chr11', 'chrX', 'chr15']
    variant_types = ['SNV', 'Insertion', 'Deletion', 'Duplication']
    clinical_sigs = ['Pathogenic', 'Likely Pathogenic', 'VUS', 'Benign']
    diseases = ['Cystic Fibrosis', 'Duchenne Muscular Dystrophy', 'Sickle Cell Disease']
    inheritance_patterns = ['Autosomal Dominant', 'Autosomal Recessive', 'X-linked']

    # Create 100 sample records
    n_records = 100

    data = {
        'gene_symbol': np.random.choice(genes, n_records),
        'chromosome': np.random.choice(chromosomes, n_records),
        'variant_type': np.random.choice(variant_types, n_records),
        'clinical_significance': np.random.choice(clinical_sigs, n_records),
        'disease': np.random.choice(diseases, n_records),
        'allele_frequency': np.random.uniform(0.001, 0.5, n_records),
        'inheritance_pattern': np.random.choice(inheritance_patterns, n_records)
    }

    df = pd.DataFrame(data)

    # Create 70/30 split
    split_index = int(0.7 * len(df))
    train_df = df[:split_index].reset_index(drop=True)
    test_df = df[split_index:].reset_index(drop=True)

    # Save to CSV files
    Path('data').mkdir(exist_ok=True)
    Path('data/datasets').mkdir(exist_ok=True)

    df.to_csv('data/sample_real_variants.csv', index=False)
    train_df.to_csv('data/sample_real_variants_train.csv', index=False)
    test_df.to_csv('data/sample_real_variants_test.csv', index=False)

    print(f"[+] Created sample data with {len(df)} records")
    print(f"[+] Training set: {len(train_df)} records (70%)")
    print(f"[+] Test set: {len(test_df)} records (30%)")

    return df, train_df, test_df
