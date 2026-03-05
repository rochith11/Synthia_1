"""Synthia AI Diagnostic Agent — CLI Interface.

Usage:
    python run_diagnostic.py [options]

Commands:
    run         Run a full diagnostic cycle (generate + diagnose)
    analyze     Analyze the latest generation results
    report      Generate diagnostic report from latest cycle
    experiments List all tracked experiments
    benchmarks  Show benchmark status dashboard
    optimize    Run multi-model training and hyperparameter optimization
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
from src.models.generation_config import GenerationConfig
from src.core.synthetic_data_generator import SyntheticDataGenerator
from src.analysis.data_validator import DataValidator
from src.analysis.privacy_analyzer import PrivacyAnalyzer
from src.analysis.bias_detector import BiasDetector
from src.storage.dataset_repository import DatasetRepository
from src.utils.audit_logger import AuditLogger

from src.ai_diagnostic_agent.diagnostic_agent import DiagnosticAgent
from src.ai_diagnostic_agent.report_generator import ReportGenerator
from src.ai_diagnostic_agent.optimization.model_orchestrator import ModelOrchestrator
from src.ai_diagnostic_agent.profiling.data_profiler import DataProfiler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthia AI Diagnostic Agent — Automated Quality Optimization"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run full diagnostic cycle')
    run_parser.add_argument("--model", type=str, default=None, help="Model type: CTGAN, TVAE, or CopulaGAN")
    run_parser.add_argument("--samples", type=int, default=None, help="Number of synthetic records")
    run_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    run_parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    run_parser.add_argument("--disease", type=str, default=None, help="Filter by disease name")
    run_parser.add_argument("--format", type=str, default='text', choices=['text', 'html', 'json'],
                           help="Report output format")

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze latest results')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate diagnostic report')
    report_parser.add_argument("--format", type=str, default='text', choices=['text', 'html', 'json'])

    # Experiments command
    exp_parser = subparsers.add_parser('experiments', help='List tracked experiments')
    exp_parser.add_argument("--limit", type=int, default=10, help="Max experiments to show")

    # Benchmarks command
    bench_parser = subparsers.add_parser('benchmarks', help='Show benchmark dashboard')

    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run multi-model optimization')
    opt_parser.add_argument("--models", type=str, nargs='+', default=['CTGAN', 'TVAE'],
                           help="Models to train")
    opt_parser.add_argument("--samples", type=int, default=None)
    opt_parser.add_argument("--disease", type=str, default=None)

    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile the training dataset')

    return parser.parse_args()


def load_data(disease=None):
    """Load training and test data."""
    try:
        train = load_training_data()
        test = load_test_data()
    except FileNotFoundError:
        print_info("Sample data not found — creating...")
        _, train, test = create_sample_data()

    if disease:
        train = train[train['disease'] == disease].reset_index(drop=True)
        test = test[test['disease'] == disease].reset_index(drop=True)
        if len(train) == 0:
            print_error(f"No training records for disease: {disease}")
            sys.exit(1)

    return train, test


def cmd_run(args):
    """Run a full diagnostic cycle."""
    print_section("AI DIAGNOSTIC AGENT — FULL DIAGNOSTIC CYCLE")

    cfg = ConfigManager()
    defaults = cfg.get_model_defaults()

    model_type = args.model or defaults.get("model_type", "CTGAN")
    n_samples = args.samples or defaults.get("n_samples", 1000)
    seed = args.seed or defaults.get("random_seed", 42)
    epochs = args.epochs or defaults.get("epochs", 300)
    batch_size = defaults.get("batch_size", 500)

    print_info(f"Config: model={model_type}, samples={n_samples}, epochs={epochs}, seed={seed}")

    # Load data
    train_data, test_data = load_data(args.disease)
    print_success(f"Training: {len(train_data)} rows, Test: {len(test_data)} rows")

    # Generate
    print_section("GENERATING SYNTHETIC DATA")
    config = GenerationConfig(
        model_type=model_type, n_samples=n_samples,
        random_seed=seed, hyperparameters={'epochs': epochs, 'batch_size': batch_size},
        disease_condition=args.disease,
    )
    generator = SyntheticDataGenerator(config=config)
    generator.train(train_data, config=config)
    synthetic = generator.generate(n_samples=n_samples)
    print_success(f"Generated {len(synthetic)} records")

    # Validate
    print_section("RUNNING EVALUATION PIPELINE")
    validator = DataValidator()
    val_report = validator.validate(synthetic, test_data, target_column='disease', dataset_id='diagnostic-run')
    privacy = PrivacyAnalyzer()
    priv_report = privacy.analyze_privacy(synthetic, test_data, percentile=90, dataset_id='diagnostic-run')
    bias_detector = BiasDetector()
    bias_results = bias_detector.analyze_bias(synthetic, test_data)

    # Run diagnostic cycle
    print_section("RUNNING DIAGNOSTIC ANALYSIS")
    agent = DiagnosticAgent()
    report_gen = ReportGenerator()

    current_config = {
        'model_type': model_type, 'epochs': epochs,
        'batch_size': batch_size, 'n_samples': n_samples,
        'embedding_dim': 128, 'random_seed': seed,
    }

    diag_report = agent.run_diagnostic_cycle(
        synthetic_data=synthetic,
        real_data=test_data,
        validation_report=val_report.to_dict(),
        privacy_report=priv_report.to_dict(),
        bias_report=bias_results,
        training_config=current_config,
        current_config=current_config,
    )

    # Generate report
    print_section("GENERATING REPORT")
    report_content = report_gen.generate_diagnostic_report(diag_report, format=args.format)
    print(report_content)

    # Print summary
    print_section("OPTIMIZATION SUMMARY")
    summary = agent.get_optimization_summary()
    print_info(f"Benchmarks Met: {summary['benchmarks_met']}")
    print_info(f"Latest Status: {summary['latest_status']}")
    print_info(f"Stalled: {summary['optimization_stalled']}")

    if diag_report.benchmarks_met:
        print_success("ALL ENTERPRISE BENCHMARKS ACHIEVED!")
    else:
        print_info(f"Recommendations: {len(diag_report.recommendations)}")
        for rec in diag_report.recommendations[:5]:
            print_info(f"  [P{rec.priority}] {rec.title}")

    print_success("Diagnostic cycle complete.")


def cmd_profile(args):
    """Profile the training dataset."""
    print_section("AI DIAGNOSTIC AGENT — DATASET PROFILING")

    train_data, _ = load_data()
    profiler = DataProfiler()
    profile = profiler.profile_dataset(train_data)

    print_success(f"Dataset: {profile.n_rows} rows x {profile.n_columns} columns")
    print("")
    print("Column Types:")
    for col, ctype in profile.column_types.items():
        stats = profile.statistics.get(col)
        extra = ""
        if stats and stats.skewness is not None:
            extra = f" (skewness: {stats.skewness:.2f})"
        elif stats:
            extra = f" (unique: {stats.unique_count})"
        print(f"  {col:30s} {ctype:12s}{extra}")

    print("")
    if profile.missing_values:
        missing = {k: v for k, v in profile.missing_values.items() if v > 0}
        if missing:
            print("Missing Values:")
            for col, pct in missing.items():
                print(f"  {col}: {pct:.1f}%")
            print("")

    if profile.recommended_transformations:
        print("Recommended Transformations:")
        for t in profile.recommended_transformations:
            print(f"  {t.column_name}: {t.transformation_type} — {t.reason}")

    print_success("Profiling complete.")


def cmd_optimize(args):
    """Run multi-model training and selection."""
    print_section("AI DIAGNOSTIC AGENT — MULTI-MODEL OPTIMIZATION")

    train_data, test_data = load_data(args.disease)
    print_success(f"Training: {len(train_data)} rows")

    orchestrator = ModelOrchestrator(models=args.models)

    print_section("TRAINING MODELS")
    orchestrator.train_all_models(train_data)

    n_samples = args.samples or 1000
    print_section("GENERATING FROM ALL MODELS")
    datasets = orchestrator.generate_from_all(n_samples=n_samples)

    print_section("EVALUATING ALL MODELS")
    results = orchestrator.evaluate_all_models(datasets, test_data)

    print_section("MODEL RANKINGS")
    rankings = orchestrator.get_model_rankings()
    for rank, (name, score) in enumerate(rankings, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  #{rank} {name:15s} composite_score = {score:.4f}{marker}")

    if rankings:
        best_name, _ = orchestrator.select_best_model()
        print_success(f"Best model: {best_name}")

    print_success("Multi-model optimization complete.")


def cmd_experiments(args):
    """List tracked experiments."""
    print_section("EXPERIMENT HISTORY")

    from src.ai_diagnostic_agent.tracking.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments()

    if not experiments:
        print_info("No experiments tracked yet.")
        return

    for exp in experiments[:args.limit]:
        print(f"  [{exp.status:9s}] {exp.experiment_id[:12]}... "
              f"model={exp.model_type:6s} "
              f"quality={exp.metrics.get('quality_score', 0):.4f} "
              f"privacy={exp.metrics.get('privacy_score', 0):.4f} "
              f"@ {exp.timestamp}")

    print_info(f"Total: {tracker.get_experiment_count()} experiments")


def cmd_benchmarks(args):
    """Show benchmark status dashboard."""
    print_section("BENCHMARK STATUS DASHBOARD")

    from src.ai_diagnostic_agent.tracking.experiment_tracker import ExperimentTracker
    from src.ai_diagnostic_agent.tracking.benchmark_tracker import BenchmarkTracker

    tracker = ExperimentTracker()
    best = tracker.get_best_experiment('quality_score')

    if not best:
        print_info("No experiments found. Run a diagnostic cycle first.")
        return

    benchmark = BenchmarkTracker()
    dashboard = benchmark.generate_dashboard(best.metrics)
    print(dashboard)


def main():
    args = parse_args()
    if args.command is None:
        print_info("Usage: python run_diagnostic.py <command> [options]")
        print_info("Commands: run, analyze, report, experiments, benchmarks, optimize, profile")
        print_info("Use --help for detailed options.")
        return

    command_map = {
        'run': cmd_run,
        'profile': cmd_profile,
        'optimize': cmd_optimize,
        'experiments': cmd_experiments,
        'benchmarks': cmd_benchmarks,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        print_error(f"Unknown command: {args.command}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print_error(f"Diagnostic agent failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
