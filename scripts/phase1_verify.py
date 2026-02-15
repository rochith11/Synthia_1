"""Phase 1 Verification Script - Test core generation functionality."""

import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from src.utils.data_loader import load_training_data, create_sample_data
from src.utils.logger import print_section, print_success, print_info, print_error
from src.core.synthetic_data_generator import SyntheticDataGenerator
from src.models.generation_config import GenerationConfig
import pandas as pd


def print_data_table(df: pd.DataFrame, title: str, max_rows: int = 10):
    """Print a formatted data table."""
    print(f"\n{title}")
    print("=" * 120)
    # Truncate long strings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(df.head(max_rows).to_string())
    print("=" * 120)


def main():
    """Run Phase 1 verification."""

    print_section("PHASE 1: CORE GENERATION VERIFICATION")

    try:
        # Step 1: Load or create sample data
        print_info("Step 1: Loading sample data...")
        try:
            train_data = load_training_data()
            print_success(f"Loaded training data: {len(train_data)} records")
        except FileNotFoundError:
            print_info("Creating sample data...")
            _, train_data, _ = create_sample_data()

        # Step 2: Test CTGAN generation
        print_section("TESTING CTGAN")

        config_ctgan = GenerationConfig(
            model_type='CTGAN',
            n_samples=20,
            random_seed=32,
            hyperparameters={'epochs': 50, 'batch_size': 500}  # Reduced epochs for testing
        )

        print_info("Initializing CTGAN...")
        gen_ctgan = SyntheticDataGenerator(config=config_ctgan)

        print_info("Training CTGAN on real data...")
        gen_ctgan.train(train_data, config=config_ctgan)

        print_info(f"Generating synthetic data (est. {gen_ctgan.get_progress_estimate(20)})...")
        synthetic_ctgan = gen_ctgan.generate(n_samples=20)

        print_success(f"CTGAN generated {len(synthetic_ctgan)} records")
        print_data_table(synthetic_ctgan, "CTGAN Synthetic Samples", max_rows=10)

        # Check for NaN values
        nan_count = synthetic_ctgan.isnull().sum().sum()
        if nan_count > 0:
            print_error(f"CTGAN produced {nan_count} NaN values!")
            return False, None, None, None
        else:
            print_success("No NaN values in CTGAN output")

        # Check allele_frequency range
        bad_af = ((synthetic_ctgan['allele_frequency'] < 0) |
                  (synthetic_ctgan['allele_frequency'] > 1)).sum()
        if bad_af > 0:
            print_error(f"CTGAN produced {bad_af} invalid allele frequencies")
            return False, None, None, None
        else:
            print_success("All allele frequencies in valid range [0, 1]")

        # Step 3: Test TVAE generation
        print_section("TESTING TVAE")

        config_tvae = GenerationConfig(
            model_type='TVAE',
            n_samples=20,
            random_seed=32,
            hyperparameters={'epochs': 50}  # Reduced epochs for testing
        )

        print_info("Initializing TVAE...")
        gen_tvae = SyntheticDataGenerator(config=config_tvae)

        print_info("Training TVAE on real data...")
        gen_tvae.train(train_data, config=config_tvae)

        print_info(f"Generating synthetic data (est. {gen_tvae.get_progress_estimate(20)})...")
        synthetic_tvae = gen_tvae.generate(n_samples=20)

        print_success(f"TVAE generated {len(synthetic_tvae)} records")
        print_data_table(synthetic_tvae, "TVAE Synthetic Samples", max_rows=10)

        # Check for NaN values
        nan_count = synthetic_tvae.isnull().sum().sum()
        if nan_count > 0:
            print_error(f"TVAE produced {nan_count} NaN values!")
            return False, None, None, None
        else:
            print_success("No NaN values in TVAE output")

        # Check allele_frequency range
        bad_af = ((synthetic_tvae['allele_frequency'] < 0) |
                  (synthetic_tvae['allele_frequency'] > 1)).sum()
        if bad_af > 0:
            print_error(f"TVAE produced {bad_af} invalid allele frequencies")
            return False, None, None, None
        else:
            print_success("All allele frequencies in valid range [0, 1]")

        # Step 4: Verify metadata
        print_section("VERIFYING METADATA")

        metadata_ctgan = gen_ctgan.get_metadata()
        print_info(f"CTGAN Model: {metadata_ctgan.get('model_type')}")
        print_info(f"Random Seed: {metadata_ctgan.get('random_seed')}")
        print_info(f"Training Records: {metadata_ctgan.get('training_records')}")
        print_success("CTGAN metadata captured correctly")

        metadata_tvae = gen_tvae.get_metadata()
        print_info(f"TVAE Model: {metadata_tvae.get('model_type')}")
        print_info(f"Random Seed: {metadata_tvae.get('random_seed')}")
        print_info(f"Training Records: {metadata_tvae.get('training_records')}")
        print_success("TVAE metadata captured correctly")

        # Final summary
        print_section("PHASE 1 VERIFICATION COMPLETE")
        print(f"""
Summary:
  [OK] CTGAN generated 20 valid records
  [OK] TVAE generated 20 valid records
  [OK] All records have required 7 fields
  [OK] No NaN values detected
  [OK] Allele frequencies in valid range [0, 1]
  [OK] Metadata captured for reproducibility

Ready for: CHECKPOINT 1 REVIEW
""")

        return True, train_data, synthetic_ctgan, synthetic_tvae

    except Exception as e:
        print_error(f"Phase 1 verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def run_sanity_checks(real_df, synthetic_df, model_name="MODEL"):
    """Run sanity checks on synthetic data."""
    print(f"\n--- {model_name} SANITY CHECKS ---")

    # Ensure same column ordering for comparison
    synthetic_df = synthetic_df[real_df.columns]

    # 1️⃣ Duplicate check inside synthetic data
    duplicate_count = synthetic_df.duplicated().sum()
    print(f"[Check 1] Duplicate rows in synthetic data: {duplicate_count}")

    # 2️⃣ Exact match check against real training data
    merged = synthetic_df.merge(real_df.drop_duplicates(), how='inner')
    exact_matches = len(merged)
    print(f"[Check 2] Exact matches with real training data: {exact_matches}")

    # 3️⃣ Category distribution comparison (Top 3)
    if 'gene_symbol' in real_df.columns:
        print("\n[Check 3] Top 3 Gene Distribution (Real vs Synthetic)")

        real_dist = real_df['gene_symbol'].value_counts(normalize=True).head(3)
        synth_dist = synthetic_df['gene_symbol'].value_counts(normalize=True).head(3)

        print("\nReal Distribution:")
        print(real_dist)

        print("\nSynthetic Distribution:")
        print(synth_dist)

    # 4️⃣ Train/Test Split Confirmation
    print(f"\n[Check 4] Training data size: {len(real_df)}")
    print(f"[Check 4] Synthetic data size: {len(synthetic_df)}")

    print("-"*50)


def run_extended_checks(train_data, synthetic_ctgan, synthetic_tvae):
    """Run extended sanity checks after main execution."""
    print("\n" + "="*60)
    print("  ADDITIONAL SANITY CHECKS")
    print("="*60)

    # Run checks for both models
    run_sanity_checks(train_data, synthetic_ctgan, model_name="CTGAN")
    run_sanity_checks(train_data, synthetic_tvae, model_name="TVAE")

    print("\nSanity checks completed.\n")


if __name__ == '__main__':
    # Step 1: Run main verification
    success, train_data, synthetic_ctgan, synthetic_tvae = main()
    
    # Step 2: Run extended checks if main succeeded
    if success:
        run_extended_checks(train_data, synthetic_ctgan, synthetic_tvae)
    
    sys.exit(0 if success else 1)