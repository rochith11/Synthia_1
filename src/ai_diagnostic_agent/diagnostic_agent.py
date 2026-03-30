"""Diagnostic Agent Main Orchestrator - coordinates all diagnostic and optimization components."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.ai_diagnostic_agent.models import (
    DiagnosticReport, MetricAnalysis, DataProfile, ExperimentPlan,
    Experiment, ExperimentStatus, QualityLevel, DiversityReport
)
from src.ai_diagnostic_agent.config import STALL_DETECTION_RUNS, ENTERPRISE_BENCHMARKS
from src.ai_diagnostic_agent.profiling.data_profiler import DataProfiler
from src.ai_diagnostic_agent.diagnostic.metric_analyzer import MetricAnalyzer
from src.ai_diagnostic_agent.diagnostic.root_cause_analyzer import RootCauseAnalyzer
from src.ai_diagnostic_agent.diagnostic.feature_diagnostics import FeatureDiagnostics
from src.ai_diagnostic_agent.diagnostic.constraint_validator import ConstraintValidator
from src.ai_diagnostic_agent.diagnostic.diversity_monitor import DiversityMonitor
from src.ai_diagnostic_agent.diagnostic.dataset_size_diagnostics import DatasetSizeDiagnostics
from src.ai_diagnostic_agent.optimization.pipeline_optimizer import PipelineOptimizer
from src.ai_diagnostic_agent.optimization.experiment_planner import ExperimentPlanner
from src.ai_diagnostic_agent.tracking.experiment_tracker import ExperimentTracker
from src.ai_diagnostic_agent.tracking.benchmark_tracker import BenchmarkTracker


class DiagnosticAgent:
    """Main orchestrator that coordinates all diagnostic and optimization components.

    Runs the full optimization loop:
    Profile -> Train -> Generate -> Validate -> Diagnose -> Optimize -> Plan -> Track
    """

    def __init__(self, storage_path: str = "data/experiments"):
        self.logger = logging.getLogger(__name__)

        # Initialize all components
        self.data_profiler = DataProfiler()
        self.metric_analyzer = MetricAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.feature_diagnostics = FeatureDiagnostics()
        self.constraint_validator = ConstraintValidator()
        self.diversity_monitor = DiversityMonitor()
        self.dataset_size_diagnostics = DatasetSizeDiagnostics()
        self.pipeline_optimizer = PipelineOptimizer()
        self.experiment_planner = ExperimentPlanner()
        self.experiment_tracker = ExperimentTracker(storage_path=storage_path)
        self.benchmark_tracker = BenchmarkTracker()

        # Optimization state
        self.optimization_history: List[DiagnosticReport] = []
        self.cycle_count = 0
        self.is_complete = False

    def run_diagnostic_cycle(
        self,
        synthetic_data,
        real_data,
        validation_report: dict,
        privacy_report: dict,
        bias_report: dict,
        training_config: dict = None,
        current_config: dict = None,
    ) -> DiagnosticReport:
        """Run a complete diagnostic and optimization cycle.

        Parameters
        ----------
        synthetic_data : pd.DataFrame
            Generated synthetic dataset.
        real_data : pd.DataFrame
            Original real dataset.
        validation_report : dict
            Output from DataValidator.validate().to_dict()
        privacy_report : dict
            Output from PrivacyAnalyzer.analyze_privacy().to_dict()
        bias_report : dict
            Output from BiasDetector.analyze_bias()
        training_config : dict, optional
            Current training configuration.
        current_config : dict, optional
            Full pipeline configuration.

        Returns
        -------
        DiagnosticReport
            Comprehensive diagnostic report for this cycle.
        """
        self.cycle_count += 1
        self.logger.info("=" * 60)
        self.logger.info("Starting Diagnostic Cycle #%d", self.cycle_count)
        self.logger.info("=" * 60)

        report = DiagnosticReport(cycle_number=self.cycle_count)

        # Step 1: Profile the real data
        self.logger.info("Step 1: Profiling real dataset...")
        data_profile = self.data_profiler.profile_dataset(real_data)
        self.logger.info(
            "Profile: %d rows, %d columns, %d transformations recommended",
            data_profile.n_rows, data_profile.n_columns,
            len(data_profile.recommended_transformations),
        )

        # Step 2: Dataset size diagnostics
        self.logger.info("Step 2: Analyzing dataset size...")
        n_categorical = sum(
            1 for t in data_profile.column_types.values() if t == 'categorical'
        )
        size_analysis = self.dataset_size_diagnostics.analyze_dataset_size(
            data_profile.n_rows, data_profile.n_columns, n_categorical
        )
        self.logger.info(
            "Dataset size: %s (severity: %s)",
            size_analysis.get('size_category', 'unknown'),
            size_analysis.get('severity', QualityLevel.ACCEPTABLE).value,
        )

        # Step 3: Analyze metrics
        self.logger.info("Step 3: Analyzing evaluation metrics...")
        metric_analysis = self.metric_analyzer.analyze_metrics(
            validation_report, privacy_report, bias_report
        )
        report.metric_analysis = metric_analysis
        self.logger.info(
            "Overall status: %s, Weakest metrics: %s",
            metric_analysis.overall_status,
            metric_analysis.weakest_metrics[:3],
        )

        # Step 4: Diagnose root causes
        self.logger.info("Step 4: Diagnosing root causes...")
        root_causes = self.root_cause_analyzer.diagnose(
            metric_analysis, data_profile, training_config
        )
        report.root_causes = root_causes
        self.logger.info("Found %d root causes", len(root_causes))
        for rc in root_causes[:3]:
            self.logger.info(
                "  - %s (likelihood: %.0f%%, impact: %s)",
                rc.cause_name, rc.likelihood * 100, rc.impact,
            )

        # Step 5: Feature-level diagnostics
        self.logger.info("Step 5: Running feature-level diagnostics...")
        column_diagnostics = self.feature_diagnostics.analyze_all_columns(
            synthetic_data, real_data
        )
        column_diagnostics = self.feature_diagnostics.prioritize_features(
            column_diagnostics
        )
        report.column_diagnostics = column_diagnostics

        critical_cols = [
            cd for cd in column_diagnostics
            if cd.severity == QualityLevel.CRITICAL
        ]
        self.logger.info(
            "Analyzed %d columns, %d critical",
            len(column_diagnostics), len(critical_cols),
        )

        # Step 6: Correlation analysis
        self.logger.info("Step 6: Analyzing correlation preservation...")
        correlation_analysis = self.feature_diagnostics.analyze_correlations(
            synthetic_data, real_data
        )
        report.correlation_analysis = correlation_analysis
        self.logger.info(
            "Correlation similarity: %.3f, Lost pairs: %d",
            correlation_analysis.overall_similarity,
            len(correlation_analysis.lost_pairs),
        )

        # Step 7: Diversity monitoring
        self.logger.info("Step 7: Monitoring diversity...")
        diversity_report = self.diversity_monitor.analyze_diversity(
            synthetic_data, real_data
        )
        report.diversity_report = diversity_report
        self.logger.info(
            "Unique ratio: %.3f, Duplicates: %d, Mode collapse: %s",
            diversity_report.unique_row_ratio,
            diversity_report.duplicate_count,
            diversity_report.mode_collapse_detected,
        )

        # Step 8: Constraint validation
        self.logger.info("Step 8: Validating constraints...")
        constraint_result = self.constraint_validator.validate_dataset(synthetic_data)
        report.constraint_validation = constraint_result
        self.logger.info(
            "Constraints: %d/%d valid (%.1f%% filtered)",
            constraint_result.valid_records,
            constraint_result.total_records,
            constraint_result.filter_percentage * 100,
        )

        # Step 9: Benchmark tracking
        self.logger.info("Step 9: Tracking benchmarks...")
        flat_metrics = self._extract_flat_metrics(metric_analysis)
        benchmark_status = self.benchmark_tracker.track_benchmarks(
            flat_metrics, self.cycle_count
        )
        report.benchmarks_met = benchmark_status.get('all_met', False)
        self.logger.info(
            "Benchmarks met: %d/%d, All met: %s",
            len(benchmark_status.get('met', [])),
            len(benchmark_status.get('met', [])) + len(benchmark_status.get('not_met', [])),
            report.benchmarks_met,
        )

        # Step 10: Generate recommendations
        self.logger.info("Step 10: Generating improvement recommendations...")
        recommendations = self.pipeline_optimizer.generate_recommendations(
            metric_analysis=metric_analysis,
            root_causes=root_causes,
            data_profile=data_profile,
            column_diagnostics=column_diagnostics,
            diversity_report=diversity_report,
            current_config=current_config,
        )
        # Add size-based recommendations
        if size_analysis.get('recommendations'):
            recommendations.extend(size_analysis['recommendations'])
        recommendations = self.pipeline_optimizer.prioritize_recommendations(
            recommendations
        )
        report.recommendations = recommendations
        self.logger.info(
            "Generated %d recommendations (top priority: %s)",
            len(recommendations),
            recommendations[0].title if recommendations else "none",
        )

        # Step 11: Plan next experiment
        self.logger.info("Step 11: Planning next experiment...")
        experiment_plan = self.experiment_planner.plan_next_experiment(
            recommendations, current_config
        )
        report.experiment_plan = experiment_plan
        self.logger.info("Experiment plan: %s", experiment_plan.plan_id)

        # Step 12: Check for stall
        self.logger.info("Step 12: Checking optimization progress...")
        report.optimization_stalled = self._detect_stall()
        if report.optimization_stalled:
            self.logger.warning(
                "Optimization stalled! No improvement for %d consecutive runs.",
                STALL_DETECTION_RUNS,
            )

        # Step 13: Log experiment
        self.logger.info("Step 13: Logging experiment...")
        experiment = Experiment(
            dataset_version=data_profile.dataset_id,
            model_type=training_config.get('model_type', 'unknown') if training_config else 'unknown',
            model_config=training_config or {},
            training_config=current_config or {},
            metrics=flat_metrics,
            diagnostic_summary={
                'overall_status': metric_analysis.overall_status,
                'root_cause_count': len(root_causes),
                'critical_columns': len(critical_cols),
                'benchmarks_met': report.benchmarks_met,
            },
            status=ExperimentStatus.COMPLETED.value,
        )
        exp_id = self.experiment_tracker.log_experiment(experiment)
        self.logger.info("Experiment logged: %s", exp_id)

        # Store in history
        self.optimization_history.append(report)

        if report.benchmarks_met:
            self.is_complete = True
            self.logger.info("ALL BENCHMARKS MET! Optimization complete.")

        self.logger.info("=" * 60)
        self.logger.info("Diagnostic Cycle #%d Complete", self.cycle_count)
        self.logger.info("=" * 60)

        return report

    def _detect_stall(self) -> bool:
        """Detect if optimization has stalled (no improvement for N runs)."""
        if len(self.optimization_history) < STALL_DETECTION_RUNS:
            return False

        recent = self.optimization_history[-STALL_DETECTION_RUNS:]
        # Check if quality score has improved
        scores = []
        for report in recent:
            if report.metric_analysis:
                quality = report.metric_analysis.quality_metrics.get('quality_score')
                if quality:
                    scores.append(quality[0])  # value from (value, level) tuple

        if len(scores) < STALL_DETECTION_RUNS:
            return False

        # Stall if no improvement: each score <= first score
        first_score = scores[0]
        return all(s <= first_score + 0.001 for s in scores[1:])

    def _extract_flat_metrics(self, analysis: MetricAnalysis) -> Dict[str, float]:
        """Extract flat dict of metric_name -> value from MetricAnalysis."""
        flat = {}
        for metrics_dict in [
            analysis.quality_metrics,
            analysis.utility_metrics,
            analysis.privacy_metrics,
            analysis.bias_metrics,
        ]:
            for name, (value, _level) in metrics_dict.items():
                flat[name] = value
        return flat

    def get_optimization_summary(self) -> dict:
        """Get a summary of the optimization progress."""
        return {
            'total_cycles': self.cycle_count,
            'is_complete': self.is_complete,
            'benchmarks_met': (
                self.optimization_history[-1].benchmarks_met
                if self.optimization_history else False
            ),
            'latest_status': (
                self.optimization_history[-1].metric_analysis.overall_status
                if self.optimization_history and self.optimization_history[-1].metric_analysis
                else 'No cycles run'
            ),
            'optimization_stalled': (
                self.optimization_history[-1].optimization_stalled
                if self.optimization_history else False
            ),
            'experiment_count': self.experiment_tracker.get_experiment_count(),
        }

    def get_metric_trends(self) -> Dict[str, List[float]]:
        """Get metric values across all optimization cycles."""
        trends: Dict[str, List[float]] = {}
        for report in self.optimization_history:
            if report.metric_analysis:
                for name, (value, _) in report.metric_analysis.get_all_metrics().items():
                    if name not in trends:
                        trends[name] = []
                    trends[name].append(value)
        return trends

    def get_benchmark_dashboard(self) -> str:
        """Get the current benchmark status dashboard."""
        if not self.optimization_history:
            return "No optimization cycles run yet."
        latest = self.optimization_history[-1]
        if latest.metric_analysis:
            flat = self._extract_flat_metrics(latest.metric_analysis)
            return self.benchmark_tracker.generate_dashboard(flat)
        return "No metrics available."

    def generate_final_report(self) -> str:
        """Generate a final optimization report summarizing the journey."""
        lines = []
        lines.append("=" * 70)
        lines.append("FINAL OPTIMIZATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Total Optimization Cycles: {self.cycle_count}")
        lines.append(f"Benchmarks Achieved: {self.is_complete}")
        lines.append(f"Total Experiments: {self.experiment_tracker.get_experiment_count()}")
        lines.append("")

        # Metric evolution
        trends = self.get_metric_trends()
        if trends:
            lines.append("--- Metric Evolution ---")
            for metric, values in trends.items():
                if values:
                    lines.append(
                        f"  {metric}: {values[0]:.4f} -> {values[-1]:.4f} "
                        f"(delta: {values[-1] - values[0]:+.4f})"
                    )
            lines.append("")

        # Latest benchmark status
        if self.optimization_history:
            latest = self.optimization_history[-1]
            if latest.metric_analysis:
                flat = self._extract_flat_metrics(latest.metric_analysis)
                status = self.benchmark_tracker.get_benchmark_status(flat)
                lines.append("--- Final Benchmark Status ---")
                for metric, info in status.items():
                    met_str = "MET" if info['met'] else "NOT MET"
                    lines.append(
                        f"  [{met_str}] {metric}: {info['current']:.4f} "
                        f"(target: {info['target']:.4f}, progress: {info['progress']:.1f}%)"
                    )
                lines.append("")

            # Summary of last cycle's root causes
            if latest.root_causes:
                lines.append("--- Remaining Issues ---")
                for rc in latest.root_causes[:5]:
                    lines.append(f"  - {rc.cause_name} ({rc.impact} impact)")
                lines.append("")

            # Remaining recommendations
            if latest.recommendations:
                lines.append("--- Pending Recommendations ---")
                for rec in latest.recommendations[:5]:
                    lines.append(
                        f"  [P{rec.priority}] {rec.title}: {rec.description}"
                    )
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)
