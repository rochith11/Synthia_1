"""Data loader utilities for Synthia."""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


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
    """Load training split (70% of data).

    Args:
        data_dir: Directory containing data files

    Returns:
        Training DataFrame
    """
    train_file = Path(data_dir) / 'sample_real_variants_train.csv'
    if train_file.exists():
        return pd.read_csv(train_file)
    else:
        raise FileNotFoundError(f"Training data not found at {train_file}")


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
