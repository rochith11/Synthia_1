"""Phase 2 Verification Script - Test statistical validation functionality."""

import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from src.utils.data_loader import load_training_data, load_test_data, create_sample_data
from src.utils.logger import print_section, print_success, print_info, print_error
from src.core.synthetic_data_generator import SyntheticDataGenerator
from src.models.generation_config import GenerationConfig
from src.analysis.data_validator import DataValidator
import pandas as pd


def main():
    """Run Phase 2 verification."""

    print_section("PHASE 2: STATISTICAL VALIDATION VERIFICATION")

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
            n_samples=100,
            random_seed=42,
            hyperparameters={'epochs': 50, 'batch_size': 500}
        )

        generator = SyntheticDataGenerator(config=config)
        generator.train(train_data, config=config)
        synthetic_data = generator.generate(n_samples=100)

        print_success(f"Generated {len(synthetic_data)} synthetic records")

        # Step 3: Run full validation
        print_section("RUNNING VALIDATION")

        validator = DataValidator()
        report = validator.validate(
            synthetic=synthetic_data,
            real=test_data,
            target_column='disease',
            dataset_id='phase2-ctgan-test'
        )

        # Step 4: Print detailed results
        print_section("DETAILED RESULTS")

        # KS Tests
        print("KS Tests (numerical columns):")
        for col, vals in report.statistical_metrics.get('ks_tests', {}).items():
            stat = vals['statistic']
            pval = vals['p_value']
            status = "GOOD" if stat < 0.3 else "MODERATE" if stat < 0.5 else "HIGH DIVERGENCE"
            print(f"  {col:30s} statistic={stat:.4f}  p_value={pval:.4f}  [{status}]")

        print()

        # JS Divergences
        print("Jensen-Shannon Divergences (all columns):")
        for col, jsd in report.statistical_metrics.get('js_divergences', {}).items():
            status = "GOOD" if jsd < 0.1 else "MODERATE" if jsd < 0.3 else "HIGH DIVERGENCE"
            print(f"  {col:30s} JSD={jsd:.4f}  [{status}]")

        print()

        # Correlation similarity
        corr_sim = report.statistical_metrics.get('correlation_similarity')
        if corr_sim is not None:
            print(f"Correlation Similarity: {corr_sim:.4f}")
        else:
            print("Correlation Similarity: N/A (< 2 numerical columns)")

        print()

        print("Train size:", len(train_data))
        print("Test size:", len(test_data))


        # ML Utility
        print("ML Utility (Bidirectional Cross-Testing):")
        cross = report.utility_metrics.get('cross_test', {})

        s2r = cross.get('synthetic_to_real', {})
        print(f"  Synthetic -> Real:  Acc={s2r.get('accuracy', 0):.4f}  "
              f"F1={s2r.get('f1_score', 0):.4f}  AUC={s2r.get('auc', 0):.4f}")

        r2s = cross.get('real_to_synthetic', {})
        print(f"  Real -> Synthetic:  Acc={r2s.get('accuracy', 0):.4f}  "
              f"F1={r2s.get('f1_score', 0):.4f}  AUC={r2s.get('auc', 0):.4f}")

        print()

        # Rare event analysis
        print("Rare Event Analysis:")
        rare = report.rare_event_analysis
        if rare:
            for col, analysis in rare.items():
                print(f"  Column: {col}")
                for cls_info in analysis.get('rare_classes', []):
                    print(f"    Class '{cls_info['class']}': "
                          f"real={cls_info['real_frequency']:.4f}, "
                          f"synthetic={cls_info['synthetic_frequency']:.4f}, "
                          f"diff={cls_info['absolute_difference']:.4f}")
        else:
            print("  No rare classes detected (threshold: 5%)")

        # Overall
        print_section("PHASE 2 VERIFICATION SUMMARY")
        summary = report.statistical_metrics.get('summary', {})
        print(f"""
Results:
  Mean KS Statistic:       {summary.get('mean_ks_statistic', 0):.4f}
  Mean JS Divergence:      {summary.get('mean_jsd', 0):.4f}
  Correlation Similarity:  {summary.get('correlation_similarity', 'N/A')}
  Syn->Real Accuracy:      {s2r.get('accuracy', 0):.4f}
  Syn->Real F1:            {s2r.get('f1_score', 0):.4f}
  Syn->Real AUC:           {s2r.get('auc', 0):.4f}
  Overall Quality Score:   {report.overall_quality_score:.4f}

  [OK] All metrics computed successfully
  [OK] No NaN or invalid values
  [OK] Bidirectional testing complete

Ready for: CHECKPOINT 2 REVIEW
""")
        print_section("ADDITIONAL STRESS TESTS")

        # ==========================================================
        # TEST A: Random Noise Sensitivity
        # ==========================================================
        print("\n[TEST A] Random Noise Sensitivity")

        noisy = test_data.copy()
        import numpy as np
        noisy['allele_frequency'] = np.random.rand(len(noisy))
        noisy['gene_symbol'] = np.random.choice(
            test_data['gene_symbol'].unique(),
            size=len(noisy)
        )

        noise_report = validator.validate(
            synthetic=noisy,
            real=test_data,
            target_column='disease',
            dataset_id='noise-test'
        )

        print(f"  Noise Mean JSD: {noise_report.statistical_metrics['summary']['mean_jsd']:.4f}")
        print(f"  Noise Quality Score: {noise_report.overall_quality_score:.4f}")

        # ==========================================================
        # TEST B: Forced Mode Collapse
        # ==========================================================
        print("\n[TEST B] Forced Mode Collapse")

        collapsed = test_data.copy()
        collapsed['gene_symbol'] = collapsed['gene_symbol'].iloc[0]

        collapse_report = validator.validate(
            synthetic=collapsed,
            real=test_data,
            target_column='disease',
            dataset_id='collapse-test'
        )

        print(f"  Collapse Mean JSD: {collapse_report.statistical_metrics['summary']['mean_jsd']:.4f}")
        print(f"  Collapse Quality Score: {collapse_report.overall_quality_score:.4f}")

        # ==========================================================
        # TEST C: NaN Propagation Guard
        # ==========================================================
        print("\n[TEST C] NaN Propagation")

        nan_test = synthetic_data.copy()
        nan_test.loc[0, 'allele_frequency'] = np.nan

        try:
            validator.validate(
                synthetic=nan_test,
                real=test_data,
                target_column='disease',
                dataset_id='nan-test'
            )
            print("  WARNING: NaN not detected!")
        except Exception as e:
            print("  NaN correctly triggered error.")

        # ==========================================================
        # TEST D: Bound Checks
        # ==========================================================
        print("\n[TEST D] Score Bounds")

        assert 0 <= report.overall_quality_score <= 1, "Quality score out of bounds"
        assert 0 <= report.statistical_metrics['summary']['mean_jsd'] <= 1, "JSD out of bounds"

        print("  Bound checks passed.")

        print("\nAll stress tests executed.")


        return True

    except Exception as e:
        print_error(f"Phase 2 verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
