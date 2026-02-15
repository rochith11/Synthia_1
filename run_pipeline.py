"""Synthia — End-to-End CLI Pipeline.

Usage:
    python run_pipeline.py [options]

Options:
    --model       CTGAN or TVAE (default: from config.yaml)
    --samples     Number of synthetic records (default: from config.yaml)
    --seed        Random seed (default: from config.yaml)
    --epochs      Training epochs (default: from config.yaml)
    --disease     Filter training data by disease (optional)
    --config      Path to config.yaml (default: config.yaml)
    --user        Username for audit log (default: default_user)
"""

import sys
import os
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from src.utils.config_manager import ConfigManager
from src.utils.logger import print_section, print_success, print_info, print_error
from src.utils.data_loader import load_training_data, load_test_data, create_sample_data
from src.utils.metadata_manager import MetadataManager
from src.utils.user_manager import UserManager
from src.utils.audit_logger import AuditLogger
from src.models.generation_config import GenerationConfig
from src.core.synthetic_data_generator import SyntheticDataGenerator
from src.analysis.data_validator import DataValidator
from src.analysis.privacy_analyzer import PrivacyAnalyzer
from src.analysis.bias_detector import BiasDetector
from src.storage.dataset_repository import DatasetRepository


def parse_args():
    parser = argparse.ArgumentParser(description="Synthia — Synthetic Rare Disease Data Generator")
    parser.add_argument("--model", type=str, help="Model type: CTGAN or TVAE")
    parser.add_argument("--samples", type=int, help="Number of synthetic records")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--disease", type=str, help="Filter by disease name")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--user", type=str, default="default_user", help="Username")
    return parser.parse_args()


def main():
    args = parse_args()

    print_section("SYNTHIA - SYNTHETIC RARE DISEASE DATA GENERATOR")

    # ── 1. Configuration ──────────────────────────────────────────────
    print_info("Loading configuration...")
    cfg = ConfigManager(args.config)
    defaults = cfg.get_model_defaults()

    model_type = args.model or defaults.get("model_type", "CTGAN")
    n_samples = args.samples or defaults.get("n_samples", 1000)
    seed = args.seed or defaults.get("random_seed", 42)
    epochs = args.epochs or defaults.get("epochs", 300)
    batch_size = defaults.get("batch_size", 500)
    risk_percentile = cfg.get("privacy.risk_percentile", 90)
    amp_threshold = cfg.get("bias.amplification_threshold", 0.20)
    rare_threshold = cfg.get("bias.rare_class_threshold", 0.05)

    print_success(f"Model: {model_type}  Samples: {n_samples}  Seed: {seed}  Epochs: {epochs}")

    user_mgr = UserManager(args.user)
    audit = AuditLogger()
    meta_mgr = MetadataManager()
    repo = DatasetRepository()

    # ── 2. Load data ──────────────────────────────────────────────────
    print_section("LOADING DATA")
    try:
        train_data = load_training_data()
        test_data = load_test_data()
    except FileNotFoundError:
        print_info("Sample data not found — creating...")
        _, train_data, test_data = create_sample_data()

    if args.disease:
        train_data = train_data[train_data['disease'] == args.disease].reset_index(drop=True)
        test_data = test_data[test_data['disease'] == args.disease].reset_index(drop=True)
        if len(train_data) == 0:
            print_error(f"No training records for disease: {args.disease}")
            return False
        print_info(f"Filtered to disease: {args.disease}")

    print_success(f"Training records: {len(train_data)}")
    print_success(f"Test records:     {len(test_data)}")

    # ── 3. Generate synthetic data ────────────────────────────────────
    print_section("GENERATION")

    hyper = {'epochs': epochs, 'batch_size': batch_size}
    config = GenerationConfig(
        model_type=model_type,
        n_samples=n_samples,
        random_seed=seed,
        hyperparameters=hyper,
        disease_condition=args.disease,
    )

    generator = SyntheticDataGenerator(config=config)
    generator.train(train_data, config=config)
    synthetic_data = generator.generate(n_samples=n_samples)

    print_success(f"Generated {len(synthetic_data)} synthetic records")
    audit.log("generate", username=user_mgr.get_current_user(),
              details={"model": model_type, "n_samples": n_samples})

    # ── 4. Validation ─────────────────────────────────────────────────
    print_section("VALIDATION")

    validator = DataValidator()
    target_col = 'disease'

    val_report = validator.validate(
        synthetic=synthetic_data,
        real=test_data,
        target_column=target_col,
        dataset_id='pipeline-run',
    )

    # ── 5. Privacy analysis ───────────────────────────────────────────
    print_section("PRIVACY ANALYSIS")

    privacy = PrivacyAnalyzer()
    priv_report = privacy.analyze_privacy(
        synthetic=synthetic_data,
        real=test_data,
        percentile=risk_percentile,
        dataset_id='pipeline-run',
    )

    # ── 6. Bias analysis ─────────────────────────────────────────────
    print_section("BIAS ANALYSIS")

    bias = BiasDetector()
    bias_results = bias.analyze_bias(
        synthetic=synthetic_data,
        real=test_data,
        amplification_threshold=amp_threshold,
        rare_class_threshold=rare_threshold,
    )

    # ── 7. Save dataset ──────────────────────────────────────────────
    print_section("SAVING DATASET")

    gen_meta = generator.get_metadata()
    gen_meta['quality_score'] = val_report.overall_quality_score
    gen_meta['privacy_score'] = priv_report.privacy_score
    gen_meta['user'] = user_mgr.get_current_user()

    dataset_id = repo.save_dataset(
        data=synthetic_data,
        metadata=gen_meta,
        dataset_name=f"{model_type}-{args.disease or 'all'}-{n_samples}",
    )

    audit.log("save", resource_id=dataset_id,
              username=user_mgr.get_current_user(),
              details={"dataset_name": f"{model_type}-{n_samples}"})

    # ── 8. Summary report ─────────────────────────────────────────────
    print_section("PIPELINE SUMMARY")

    summary = val_report.statistical_metrics.get('summary', {})
    cross = val_report.utility_metrics.get('cross_test', {})
    s2r = cross.get('synthetic_to_real', {})
    nnd = priv_report.nearest_neighbor_distances
    risk = priv_report.reidentification_risk

    corr_val = summary.get('correlation_similarity')
    corr_str = f"{corr_val:.4f}" if corr_val is not None else "N/A"

    print(f"""
=== GENERATION ===
  Model:             {model_type}
  Samples Generated: {len(synthetic_data)}
  Random Seed:       {seed}
  Epochs:            {epochs}

=== VALIDATION ===
  Mean KS Statistic:      {summary.get('mean_ks_statistic', 0):.4f}
  Mean JS Divergence:     {summary.get('mean_jsd', 0):.4f}
  Max JS Divergence:      {summary.get('max_jsd', 0):.4f}
  Correlation Similarity: {corr_str}
  Syn->Real Accuracy:     {s2r.get('accuracy', 0):.4f}
  Syn->Real F1:           {s2r.get('f1_score', 0):.4f}
  Syn->Real AUC:          {s2r.get('auc', 0):.4f}
  Overall Quality Score:  {val_report.overall_quality_score:.4f}

=== PRIVACY ===
  Mean NND:               {nnd.get('mean', 0):.4f}
  Min NND:                {nnd.get('min', 0):.4f}
  High-Risk Records:      {risk.get('high_risk_count', 0)} / {len(synthetic_data)}
  High-Risk %:            {risk.get('high_risk_percentage', 0):.2%}
  Privacy Score:          {priv_report.privacy_score:.4f}

=== BIAS ===
  High-Bias Columns:      {len([c for c, v in bias_results['feature_distributions'].items() if v['status'] == 'high'])}
  Amplified Columns:      {len(bias_results['amplification'].get('flagged', []))}
  Recommendations:        {len(bias_results['recommendations'])}

=== DATASET ===
  Dataset ID:             {dataset_id}
  User:                   {user_mgr.get_current_user()}
""")

    print_success("Pipeline complete.")
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
