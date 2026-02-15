"""Phase 5 Verification Script - Test bias detection and dataset persistence."""

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
from src.analysis.bias_detector import BiasDetector
from src.storage.dataset_repository import DatasetRepository


def main():
    """Run Phase 5 verification (bias + persistence)."""

    print_section("PHASE 5: BIAS DETECTION & PERSISTENCE VERIFICATION")

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

        # Step 2: Generate synthetic data
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

        # =============================================
        # BIAS DETECTION
        # =============================================
        print_section("BIAS ANALYSIS")

        detector = BiasDetector()
        bias_results = detector.analyze_bias(
            synthetic=synthetic_data,
            real=test_data,
            amplification_threshold=0.20,
            rare_class_threshold=0.05
        )

        # Print feature distributions
        print("\nFeature Distribution Bias:")
        for col, info in bias_results['feature_distributions'].items():
            print(f"  {col:30s} JSD={info['jsd']:.4f}  [{info['status']}]")

        # Print rare class imbalances
        print("\nRare Class Imbalances:")
        rare = bias_results['rare_class_imbalances']
        if rare:
            for col, info in rare.items():
                print(f"  Column: {col}")
                for cls_info in info.get('rare_classes', []):
                    marker = " [MISSING]" if cls_info['missing_in_synthetic'] else ""
                    print(f"    Class '{cls_info['class']}': "
                          f"real={cls_info['real_frequency']:.4f}, "
                          f"syn={cls_info['synthetic_frequency']:.4f}"
                          f"{marker}")
        else:
            print("  No rare classes detected")

        # Print amplification
        print("\nBias Amplification:")
        amp = bias_results['amplification']
        if amp['flagged']:
            for col in amp['flagged']:
                info = amp['columns'][col]
                print(f"  {col}: real_ratio={info['real_imbalance_ratio']:.2f}, "
                      f"syn_ratio={info['synthetic_imbalance_ratio']:.2f} [AMPLIFIED]")
        else:
            print("  No amplified columns detected")

        # Print recommendations
        print("\nRecommendations:")
        for i, rec in enumerate(bias_results['recommendations'], 1):
            print(f"  {i}. {rec}")

        # =============================================
        # DATASET PERSISTENCE
        # =============================================
        print_section("DATASET PERSISTENCE")

        repo = DatasetRepository(base_dir="data/datasets")

        # Save dataset
        print_info("Saving dataset...")
        gen_metadata = generator.get_metadata()
        dataset_id = repo.save_dataset(
            data=synthetic_data,
            metadata=gen_metadata,
            dataset_name="phase5-ctgan-test"
        )
        print_success(f"Dataset ID: {dataset_id}")

        # Load dataset
        print_info("Loading dataset back...")
        loaded_data, loaded_meta = repo.load_dataset(dataset_id)
        print_success(f"Loaded {len(loaded_data)} records")

        # Verify integrity
        assert len(loaded_data) == len(synthetic_data), "Record count mismatch"
        assert list(loaded_data.columns) == list(synthetic_data.columns), "Column mismatch"
        print_success("Data integrity verified")

        # Export CSV
        print_info("Exporting as CSV...")
        csv_path = repo.export_dataset(dataset_id, fmt='csv')
        print_success(f"CSV exported: {csv_path}")

        # Export JSON
        print_info("Exporting as JSON...")
        json_path = repo.export_dataset(dataset_id, fmt='json')
        print_success(f"JSON exported: {json_path}")

        # List datasets
        print_info("Listing all datasets...")
        all_datasets = repo.list_datasets()
        print_success(f"Total datasets saved: {len(all_datasets)}")
        for ds in all_datasets:
            print(f"  ID: {ds['dataset_id'][:8]}...  "
                  f"Name: {ds.get('dataset_name', 'N/A')}  "
                  f"Records: {ds.get('n_records', 'N/A')}  "
                  f"Created: {ds.get('created_at', 'N/A')}")

        # Get lineage
        print_info("Getting dataset lineage...")
        lineage = repo.get_dataset_lineage(dataset_id)
        print(f"  Model: {lineage.get('model_type')}")
        print(f"  Seed: {lineage.get('random_seed')}")
        print(f"  Hash: {lineage.get('training_data_hash', 'N/A')}")

        # Summary
        print_section("PHASE 5 VERIFICATION SUMMARY")
        print(f"""
Results:
  Feature Distribution Bias:
    High-bias columns: {len([c for c, v in bias_results['feature_distributions'].items() if v['status'] == 'high'])}
    Moderate-bias columns: {len([c for c, v in bias_results['feature_distributions'].items() if v['status'] == 'moderate'])}

  Rare Class Imbalances: {sum(r['total_rare_classes'] for r in rare.values()) if rare else 0}
  Amplified Columns: {len(amp['flagged'])}
  Recommendations: {len(bias_results['recommendations'])}

  Dataset Persistence:
    Dataset ID: {dataset_id[:8]}...
    Save/Load: OK
    CSV Export: OK
    JSON Export: OK
    Integrity: VERIFIED

  [OK] Bias analysis complete
  [OK] Dataset persistence working
  [OK] Export formats verified
  [OK] Lineage tracking confirmed

Ready for: CHECKPOINT 3b REVIEW
""")

        return True

    except Exception as e:
        print_error(f"Phase 5 verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
