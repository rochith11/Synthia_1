"""Phase 3 Verification Script - Test privacy analysis functionality."""

import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from src.utils.data_loader import load_training_data, load_test_data, create_sample_data
from src.utils.logger import print_section, print_success, print_info, print_error
from src.core.synthetic_data_generator import SyntheticDataGenerator
from src.models.generation_config import GenerationConfig
from src.analysis.privacy_analyzer import PrivacyAnalyzer


def main():
    """Run Phase 3 verification."""

    print_section("PHASE 3: PRIVACY ANALYSIS VERIFICATION")

    try:
        # Step 1: Load data
        print_info("Step 1: Loading data...")
        try:
            train_data = load_training_data()
            test_data = load_test_data()
        except FileNotFoundError:
            print_info("Creating sample data...")
            _, train_data, test_data = create_sample_data()

        print_success(f"Training data: {len(train_data)} records")
        print_success(f"Test data: {len(test_data)} records")

        # Step 2: Generate synthetic data with CTGAN
        print_section("GENERATING SYNTHETIC DATA (CTGAN)")

        config = GenerationConfig(
            model_type='CTGAN',
            n_samples=50,
            random_seed=42,
            hyperparameters={'epochs': 100, 'batch_size': 500}
        )

        generator = SyntheticDataGenerator(config=config)
        generator.train(train_data, config=config)
        synthetic_data = generator.generate(n_samples=50)

        print_success(f"Generated {len(synthetic_data)} synthetic records")

        # Step 3: Run privacy analysis
        print_section("RUNNING PRIVACY ANALYSIS")

        analyzer = PrivacyAnalyzer()
        report = analyzer.analyze_privacy(
            synthetic=synthetic_data,
            real=test_data,
            percentile=90,
            dataset_id='phase3-ctgan-test'
        )

        # Step 4: Print detailed results
        print_section("DETAILED RESULTS")

        print("Nearest-Neighbor Distance Statistics:")
        nnd = report.nearest_neighbor_distances
        for key, val in nnd.items():
            print(f"  {key:10s}: {val:.4f}")

        print()
        print("Re-identification Risk:")
        risk = report.reidentification_risk
        print(f"  Percentile:          {risk['percentile']}")
        print(f"  Threshold distance:  {risk['threshold']:.4f}")
        print(f"  High-risk records:   {risk['high_risk_count']}")
        print(f"  High-risk %:         {risk['high_risk_percentage']:.2%}")

        print()
        print(f"Privacy Score: {report.privacy_score:.4f}")

        # ==========================================================
        # STRESS TESTS
        # ==========================================================
        print_section("PRIVACY STRESS TESTS")

        # TEST A: Exact copy should have high risk
        print("\n[TEST A] Exact Copy of Real Data")
        exact_copy = test_data.copy()
        copy_report = analyzer.analyze_privacy(
            synthetic=exact_copy,
            real=test_data,
            percentile=90,
            dataset_id='copy-test'
        )
        print(f"  Copy Mean NND: {copy_report.nearest_neighbor_distances['mean']:.4f}")
        print(f"  Copy Min NND:  {copy_report.nearest_neighbor_distances['min']:.4f}")
        print(f"  Copy Privacy Score: {copy_report.privacy_score:.4f}")

        assert copy_report.nearest_neighbor_distances['min'] == 0.0, \
            "Exact copy should have min NND of 0"
        print("  [OK] Exact copy correctly has min NND = 0")

        # TEST B: Random data should have large distances
        print("\n[TEST B] Random Data (should have larger distances)")
        random_data = test_data.copy()
        for col in random_data.select_dtypes(include=['float64', 'int64']).columns:
            random_data[col] = np.random.rand(len(random_data))
        for col in random_data.select_dtypes(include=['object', 'category']).columns:
            unique_vals = test_data[col].unique()
            random_data[col] = np.random.choice(unique_vals, size=len(random_data))

        random_report = analyzer.analyze_privacy(
            synthetic=random_data,
            real=test_data,
            percentile=90,
            dataset_id='random-test'
        )
        print(f"  Random Mean NND: {random_report.nearest_neighbor_distances['mean']:.4f}")
        print(f"  Random Privacy Score: {random_report.privacy_score:.4f}")

        # TEST C: Score bounds
        print("\n[TEST C] Score Bounds")
        assert 0 <= report.privacy_score <= 1, "Privacy score out of bounds"
        assert 0 <= copy_report.privacy_score <= 1, "Copy privacy score out of bounds"
        assert 0 <= random_report.privacy_score <= 1, "Random privacy score out of bounds"
        print("  All privacy scores in [0, 1]")

        # Summary
        print_section("PHASE 3 VERIFICATION SUMMARY")
        print(f"""
Results:
  Synthetic Mean NND:       {report.nearest_neighbor_distances['mean']:.4f}
  Synthetic Privacy Score:  {report.privacy_score:.4f}
  Exact Copy Privacy Score: {copy_report.privacy_score:.4f}
  Random Data Privacy Score:{random_report.privacy_score:.4f}
  High-Risk %:              {risk['high_risk_percentage']:.2%}

  [OK] NND computed successfully (vectorized)
  [OK] Re-identification risk computed
  [OK] Exact copy detected as zero distance
  [OK] Privacy scores in valid range [0, 1]

Ready for: CHECKPOINT 3a REVIEW
""")

        return True

    except Exception as e:
        print_error(f"Phase 3 verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
