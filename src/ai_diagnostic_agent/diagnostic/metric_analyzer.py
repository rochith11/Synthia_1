"""Metric Analyzer — classifies, benchmarks, and interprets evaluation metrics.

Takes raw report dicts from Synthia's validation, privacy, and bias components
and produces a structured MetricAnalysis with quality classifications,
benchmark deltas, weakest-metric identification, and human-readable
interpretation reports.
"""

from typing import Dict, List

from src.ai_diagnostic_agent.models import MetricAnalysis, QualityLevel
from src.ai_diagnostic_agent.config import (
    ENTERPRISE_BENCHMARKS,
    CLASSIFICATION_THRESHOLDS,
)


class MetricAnalyzer:
    """Analyze, classify, and interpret synthetic-data evaluation metrics."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, benchmarks: Dict[str, Dict] = None):
        """Initialize with enterprise benchmark targets.

        Args:
            benchmarks: Optional override for ENTERPRISE_BENCHMARKS.
                        Each key maps to ``{'target': float, 'direction': str}``
                        where direction is ``'higher_better'`` or
                        ``'lower_better'``.  Defaults to the values defined in
                        ``src.ai_diagnostic_agent.config``.
        """
        self.benchmarks = benchmarks or ENTERPRISE_BENCHMARKS

    # ------------------------------------------------------------------ #
    # Primary entry point
    # ------------------------------------------------------------------ #

    def analyze_metrics(
        self,
        validation_report: dict,
        privacy_report: dict,
        bias_report: dict,
    ) -> MetricAnalysis:
        """Analyze all metrics from validation, privacy, and bias reports.

        Extracts numeric metrics from the report dicts (as returned by the
        existing Synthia ``DataValidator``, ``PrivacyAnalyzer``, and
        ``BiasDetector`` components), classifies each metric against
        enterprise thresholds, computes benchmark deltas, and identifies
        the weakest metrics requiring attention.

        Args:
            validation_report: Dict produced by
                ``ValidationReport.to_dict()``.
            privacy_report: Dict produced by ``PrivacyReport.to_dict()``.
            bias_report: Dict produced by ``BiasDetector.analyze_bias()``
                or ``BiasReport.to_dict()``.

        Returns:
            A fully-populated ``MetricAnalysis`` dataclass.
        """

        # -- 1. Extract raw metric values --------------------------------

        # Quality / statistical metrics from validation_report
        quality_score = validation_report.get("overall_quality_score", 0.0)

        stat_metrics = validation_report.get("statistical_metrics", {})
        summary = stat_metrics.get("summary", {})
        mean_ks = summary.get("mean_ks_statistic", summary.get("mean_ks", 0.0))
        mean_jsd = summary.get("mean_jsd", 0.0)
        max_jsd = summary.get("max_jsd", 0.0)

        # Utility metrics from validation_report
        utility_section = validation_report.get("utility_metrics", {})
        cross_test = utility_section.get("cross_test", {})
        syn_to_real = cross_test.get("synthetic_to_real", {})
        ml_accuracy = syn_to_real.get("accuracy", 0.0)
        f1_score_val = syn_to_real.get("f1_score", 0.0)
        auc_val = syn_to_real.get("auc", 0.0)

        # Privacy metrics
        privacy_score = privacy_report.get("privacy_score", 0.0)
        nnd_section = privacy_report.get("nearest_neighbor_distances", {})
        mean_nnd = nnd_section.get("mean_nnd", nnd_section.get("mean", 0.0))
        reid_section = privacy_report.get("reidentification_risk", {})
        high_risk_pct = reid_section.get("high_risk_percentage", 0.0)

        # Bias metrics
        feat_dist = bias_report.get("feature_distributions", {})
        max_column_jsd = self._extract_max_column_jsd(feat_dist)

        amplification_section = bias_report.get("amplification_results",
                                                bias_report.get("bias_amplification",
                                                bias_report.get("amplification", {})))
        max_amplification_ratio = self._extract_max_amplification_ratio(
            amplification_section
        )

        # -- 2. Classify each metric -------------------------------------

        quality_metrics = {
            "quality_score": (quality_score, self.classify_metric("quality_score", quality_score)),
            "mean_ks": (mean_ks, self.classify_metric("mean_ks", mean_ks)),
            "mean_jsd": (mean_jsd, self.classify_metric("mean_jsd", mean_jsd)),
            "max_jsd": (max_jsd, self.classify_metric("max_jsd", max_jsd)),
        }

        utility_metrics = {
            "ml_accuracy": (ml_accuracy, self.classify_metric("ml_accuracy", ml_accuracy)),
            "f1_score": (f1_score_val, self.classify_metric("f1_score", f1_score_val)),
            "auc": (auc_val, self.classify_metric("auc", auc_val)),
        }

        privacy_metrics = {
            "privacy_score": (privacy_score, self.classify_metric("privacy_score", privacy_score)),
            "mean_nnd": (mean_nnd, self.classify_metric("mean_nnd", mean_nnd)),
            "high_risk_pct": (high_risk_pct, self.classify_metric("high_risk_pct", high_risk_pct)),
        }

        bias_metrics = {
            "max_column_jsd": (max_column_jsd, self.classify_metric("max_column_jsd", max_column_jsd)),
            "max_amplification_ratio": (
                max_amplification_ratio,
                self.classify_metric("max_amplification_ratio", max_amplification_ratio),
            ),
        }

        # -- 3. Benchmark deltas -----------------------------------------

        all_values: Dict[str, float] = {}
        for bucket in (quality_metrics, utility_metrics, privacy_metrics, bias_metrics):
            for name, (value, _level) in bucket.items():
                all_values[name] = value

        benchmark_deltas = self.compare_to_benchmarks(all_values)

        # -- 4. Build analysis and derive weakest metrics ----------------

        analysis = MetricAnalysis(
            quality_metrics=quality_metrics,
            utility_metrics=utility_metrics,
            privacy_metrics=privacy_metrics,
            bias_metrics=bias_metrics,
            benchmark_deltas=benchmark_deltas,
        )

        analysis.weakest_metrics = self.identify_weakest_metrics(analysis)

        # -- 5. Determine overall status ---------------------------------

        analysis.overall_status = self._compute_overall_status(analysis)

        return analysis

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #

    def classify_metric(self, metric_name: str, value: float) -> QualityLevel:
        """Classify a single metric value into a ``QualityLevel``.

        If the metric is defined in ``CLASSIFICATION_THRESHOLDS``, the value
        is checked against the explicit (min, max) range for each level.
        Otherwise, the method falls back to a heuristic comparison against
        the enterprise benchmark target.

        Args:
            metric_name: Canonical metric name (e.g. ``'quality_score'``).
            value: The numeric metric value.

        Returns:
            One of ``QualityLevel.EXCELLENT``, ``ACCEPTABLE``, ``WARNING``,
            or ``CRITICAL``.
        """
        thresholds = CLASSIFICATION_THRESHOLDS.get(metric_name)

        if thresholds is not None:
            return self._classify_by_thresholds(value, thresholds)

        # Fallback: benchmark-relative classification
        benchmark = self.benchmarks.get(metric_name)
        if benchmark is not None:
            return self._classify_by_benchmark(value, benchmark)

        # No information at all — default to Acceptable
        return QualityLevel.ACCEPTABLE

    # ------------------------------------------------------------------ #
    # Weakest-metric identification
    # ------------------------------------------------------------------ #

    def identify_weakest_metrics(self, analysis: MetricAnalysis) -> List[str]:
        """Identify metrics requiring immediate attention.

        Returns metric names ordered by severity: all ``CRITICAL`` metrics
        first (sorted by value ascending for higher-is-better, descending
        for lower-is-better), then all ``WARNING`` metrics in the same
        order.

        Args:
            analysis: A populated ``MetricAnalysis`` instance.

        Returns:
            Ordered list of metric names needing attention.
        """
        all_metrics = analysis.get_all_metrics()

        critical: List[str] = []
        warning: List[str] = []

        for name, (value, level) in all_metrics.items():
            if level == QualityLevel.CRITICAL:
                critical.append(name)
            elif level == QualityLevel.WARNING:
                warning.append(name)

        # Within each severity band, sort by how far below the benchmark
        # the metric is (largest negative delta first).
        def _sort_key(metric_name: str) -> float:
            delta = analysis.benchmark_deltas.get(metric_name, 0.0)
            # More negative delta = worse = should appear first
            return delta

        critical.sort(key=_sort_key)
        warning.sort(key=_sort_key)

        return critical + warning

    # ------------------------------------------------------------------ #
    # Benchmark comparison
    # ------------------------------------------------------------------ #

    def compare_to_benchmarks(
        self, metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare metric values to enterprise benchmark targets.

        For ``higher_better`` metrics the delta is ``value - target``
        (positive means the metric exceeds the target).

        For ``lower_better`` metrics the delta is ``target - value``
        (positive means the metric is better than — i.e. below — the
        target).

        Args:
            metrics: Mapping of metric name to its numeric value.

        Returns:
            Mapping of metric name to its signed delta.  A positive delta
            always means "better than target".
        """
        deltas: Dict[str, float] = {}

        for name, value in metrics.items():
            benchmark = self.benchmarks.get(name)
            if benchmark is None:
                continue

            target = benchmark["target"]
            direction = benchmark.get("direction", "higher_better")

            if direction == "higher_better":
                deltas[name] = round(value - target, 6)
            else:
                # lower_better: being below target is good → positive delta
                deltas[name] = round(target - value, 6)

        return deltas

    # ------------------------------------------------------------------ #
    # Interpretation report
    # ------------------------------------------------------------------ #

    def generate_interpretation_report(self, analysis: MetricAnalysis) -> str:
        """Generate a human-readable metric interpretation report.

        The report is divided into sections for each metric category
        (Quality, Utility, Privacy, Bias), followed by a benchmark
        comparison summary and an action-items list derived from the
        weakest metrics.

        Args:
            analysis: A fully-populated ``MetricAnalysis`` instance.

        Returns:
            Multi-line string suitable for logging or display.
        """
        lines: List[str] = []
        separator = "=" * 68

        lines.append(separator)
        lines.append("  METRIC INTERPRETATION REPORT")
        lines.append(separator)
        lines.append(f"  Overall Status: {analysis.overall_status}")
        lines.append(f"  Timestamp:      {analysis.timestamp}")
        lines.append(separator)

        # -- Quality metrics section ------------------------------------
        lines.append("")
        lines.append("  [1] STATISTICAL QUALITY METRICS")
        lines.append("  " + "-" * 40)
        lines.extend(
            self._format_metric_section(
                analysis.quality_metrics,
                analysis.benchmark_deltas,
                descriptions={
                    "quality_score": "Overall synthetic data quality score",
                    "mean_ks": "Mean Kolmogorov-Smirnov statistic (lower is better)",
                    "mean_jsd": "Mean Jensen-Shannon divergence (lower is better)",
                    "max_jsd": "Worst-case column divergence (lower is better)",
                },
            )
        )

        # -- Utility metrics section ------------------------------------
        lines.append("")
        lines.append("  [2] ML UTILITY METRICS")
        lines.append("  " + "-" * 40)
        lines.extend(
            self._format_metric_section(
                analysis.utility_metrics,
                analysis.benchmark_deltas,
                descriptions={
                    "ml_accuracy": "Train-on-synthetic / test-on-real accuracy",
                    "f1_score": "Weighted F1 score (synthetic-to-real)",
                    "auc": "ROC-AUC (synthetic-to-real, weighted OVR)",
                },
            )
        )

        # -- Privacy metrics section ------------------------------------
        lines.append("")
        lines.append("  [3] PRIVACY METRICS")
        lines.append("  " + "-" * 40)
        lines.extend(
            self._format_metric_section(
                analysis.privacy_metrics,
                analysis.benchmark_deltas,
                descriptions={
                    "privacy_score": "Composite privacy score (higher is better)",
                    "mean_nnd": "Mean nearest-neighbor distance (higher is better)",
                    "high_risk_pct": "Fraction of high re-identification risk records (lower is better)",
                },
            )
        )

        # -- Bias metrics section ---------------------------------------
        lines.append("")
        lines.append("  [4] BIAS METRICS")
        lines.append("  " + "-" * 40)
        lines.extend(
            self._format_metric_section(
                analysis.bias_metrics,
                analysis.benchmark_deltas,
                descriptions={
                    "max_column_jsd": "Worst-case feature distribution divergence (lower is better)",
                    "max_amplification_ratio": "Worst-case bias amplification ratio (lower is better)",
                },
            )
        )

        # -- Benchmark summary ------------------------------------------
        lines.append("")
        lines.append("  [5] BENCHMARK COMPARISON SUMMARY")
        lines.append("  " + "-" * 40)

        met_count = sum(1 for d in analysis.benchmark_deltas.values() if d >= 0)
        total = len(analysis.benchmark_deltas)
        lines.append(
            f"  Benchmarks met: {met_count}/{total}"
        )

        if analysis.benchmark_deltas:
            worst_name = min(analysis.benchmark_deltas, key=analysis.benchmark_deltas.get)
            worst_delta = analysis.benchmark_deltas[worst_name]
            lines.append(
                f"  Largest gap:    {worst_name} (delta: {worst_delta:+.4f})"
            )

            best_name = max(analysis.benchmark_deltas, key=analysis.benchmark_deltas.get)
            best_delta = analysis.benchmark_deltas[best_name]
            lines.append(
                f"  Best surplus:   {best_name} (delta: {best_delta:+.4f})"
            )

        # -- Action items -----------------------------------------------
        lines.append("")
        lines.append("  [6] ACTION ITEMS")
        lines.append("  " + "-" * 40)

        if not analysis.weakest_metrics:
            lines.append("  No metrics require immediate attention.")
        else:
            for idx, metric_name in enumerate(analysis.weakest_metrics, 1):
                all_metrics = analysis.get_all_metrics()
                value, level = all_metrics.get(metric_name, (0.0, QualityLevel.ACCEPTABLE))
                delta = analysis.benchmark_deltas.get(metric_name, 0.0)
                lines.append(
                    f"  {idx}. [{level.value}] {metric_name}: "
                    f"{value:.4f} (benchmark delta: {delta:+.4f})"
                )
                lines.append(f"     -> {self._action_hint(metric_name, level)}")

        lines.append("")
        lines.append(separator)

        return "\n".join(lines)

    # ================================================================== #
    # Internal helpers
    # ================================================================== #

    @staticmethod
    def _classify_by_thresholds(
        value: float, thresholds: Dict[str, tuple]
    ) -> QualityLevel:
        """Match *value* to the first threshold range it falls within."""
        for level_name in ("Excellent", "Acceptable", "Warning", "Critical"):
            lo, hi = thresholds.get(level_name, (None, None))
            if lo is None:
                continue
            if lo <= value < hi:
                return QualityLevel(level_name)
        # Edge case: if value equals hi of Excellent (inf boundary)
        return QualityLevel.ACCEPTABLE

    @staticmethod
    def _classify_by_benchmark(
        value: float, benchmark: Dict
    ) -> QualityLevel:
        """Heuristic classification when explicit thresholds are absent."""
        target = benchmark["target"]
        direction = benchmark.get("direction", "higher_better")

        if direction == "higher_better":
            if value >= target * 1.15:
                return QualityLevel.EXCELLENT
            if value >= target:
                return QualityLevel.ACCEPTABLE
            if value >= target * 0.85:
                return QualityLevel.WARNING
            return QualityLevel.CRITICAL
        else:
            # lower_better
            if value <= target * 0.70:
                return QualityLevel.EXCELLENT
            if value <= target:
                return QualityLevel.ACCEPTABLE
            if value <= target * 1.30:
                return QualityLevel.WARNING
            return QualityLevel.CRITICAL

    @staticmethod
    def _extract_max_column_jsd(feat_dist: dict) -> float:
        """Return the maximum per-column JSD from the feature distributions dict."""
        if not feat_dist:
            return 0.0
        jsd_values = []
        for col_info in feat_dist.values():
            if isinstance(col_info, dict):
                jsd_values.append(col_info.get("jsd", 0.0))
        return max(jsd_values) if jsd_values else 0.0

    @staticmethod
    def _extract_max_amplification_ratio(amplification: dict) -> float:
        """Return the worst (highest) amplification ratio across columns.

        Handles two formats:
        1. Nested: ``{'columns': {'col': {'real_imbalance_ratio': ..., 'synthetic_imbalance_ratio': ...}}}``
        2. Flat: ``{'col': {'amplification_ratio': ...}}``
        """
        if not amplification:
            return 1.0

        # Try nested format first
        columns = amplification.get("columns", {})
        if not columns:
            # Try flat format: each key is a column name -> dict with amplification_ratio
            columns = amplification

        ratios: List[float] = []
        for col_info in columns.values():
            if not isinstance(col_info, dict):
                continue
            # Flat format: amplification_ratio directly
            if 'amplification_ratio' in col_info:
                ratios.append(col_info['amplification_ratio'])
            # Nested format: compute from real/synthetic ratios
            elif 'real_imbalance_ratio' in col_info:
                real_ratio = col_info.get("real_imbalance_ratio", 0.0)
                syn_ratio = col_info.get("synthetic_imbalance_ratio", 0.0)
                if real_ratio > 0:
                    ratios.append(syn_ratio / real_ratio)
                else:
                    ratios.append(syn_ratio if syn_ratio > 0 else 1.0)

        return max(ratios) if ratios else 1.0

    @staticmethod
    def _compute_overall_status(analysis: MetricAnalysis) -> str:
        """Derive a single-word overall status from classified metrics."""
        all_metrics = analysis.get_all_metrics()
        levels = [level for (_value, level) in all_metrics.values()]

        if not levels:
            return "Unknown"

        if any(l == QualityLevel.CRITICAL for l in levels):
            return "Critical — immediate action required"

        if any(l == QualityLevel.WARNING for l in levels):
            return "Warning — improvements recommended"

        if all(l == QualityLevel.EXCELLENT for l in levels):
            return "Excellent — all metrics exceed targets"

        return "Acceptable — meets enterprise benchmarks"

    def _format_metric_section(
        self,
        metrics: dict,
        deltas: dict,
        descriptions: Dict[str, str],
    ) -> List[str]:
        """Format a group of metrics for the interpretation report."""
        lines: List[str] = []
        for name, (value, level) in metrics.items():
            desc = descriptions.get(name, name)
            delta = deltas.get(name)
            delta_str = f" (delta: {delta:+.4f})" if delta is not None else ""
            status_icon = self._status_icon(level)

            lines.append(f"  {status_icon} {name}: {value:.4f}  [{level.value}]{delta_str}")
            lines.append(f"       {desc}")

        return lines

    @staticmethod
    def _status_icon(level: QualityLevel) -> str:
        """Return a plain-text status indicator for a quality level."""
        mapping = {
            QualityLevel.EXCELLENT: "[OK]",
            QualityLevel.ACCEPTABLE: "[OK]",
            QualityLevel.WARNING: "[!!]",
            QualityLevel.CRITICAL: "[XX]",
        }
        return mapping.get(level, "[??]")

    @staticmethod
    def _action_hint(metric_name: str, level: QualityLevel) -> str:
        """Return a short, actionable hint for a struggling metric."""
        hints = {
            "quality_score": (
                "Overall quality is low. Review statistical divergence metrics "
                "and consider retraining with more epochs or a different model."
            ),
            "mean_ks": (
                "Numerical distributions diverge significantly. Check for "
                "preprocessing issues or insufficient training."
            ),
            "mean_jsd": (
                "Average distribution divergence is elevated. Ensure training "
                "data is representative and increase model capacity if needed."
            ),
            "max_jsd": (
                "At least one column has high divergence. Identify the column "
                "via per-feature diagnostics and apply targeted fixes."
            ),
            "ml_accuracy": (
                "Synthetic data produces poor downstream ML accuracy. "
                "Investigate feature correlations and label distribution fidelity."
            ),
            "f1_score": (
                "F1 score is below target. Check minority-class preservation "
                "in the synthetic data."
            ),
            "auc": (
                "AUC is below target. The synthetic data may not preserve "
                "class-separability. Consider model or sampling adjustments."
            ),
            "privacy_score": (
                "Privacy score is low. Synthetic records may be too close to "
                "real records. Add noise or reduce training fidelity."
            ),
            "mean_nnd": (
                "Nearest-neighbor distances are small, indicating potential "
                "memorization. Increase privacy mechanisms or reduce epochs."
            ),
            "high_risk_pct": (
                "Too many synthetic records are near real records. Apply "
                "differential privacy or increase noise injection."
            ),
            "max_column_jsd": (
                "A feature shows high distributional bias. Investigate the "
                "specific column and consider rebalancing strategies."
            ),
            "max_amplification_ratio": (
                "Bias is being amplified in the synthetic data. Review "
                "categorical distributions and apply post-processing corrections."
            ),
        }

        default_hint = (
            f"Metric is at {level.value} level. Review related pipeline "
            f"components and consult per-feature diagnostics."
        )

        return hints.get(metric_name, default_hint)
