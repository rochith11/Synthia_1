"""Benchmark Tracker module for monitoring enterprise benchmark targets,
computing progress toward goals, detecting trends, and generating
human-readable dashboards and reports."""

import logging
from typing import Dict, List, Optional, Tuple

from src.ai_diagnostic_agent.models import QualityLevel
from src.ai_diagnostic_agent.config import ENTERPRISE_BENCHMARKS


class BenchmarkTracker:
    """Track metrics against enterprise benchmark targets.

    Maintains a history of (iteration, metrics_dict) observations and
    provides utilities for progress computation, trend detection,
    critical-violation flagging, and reporting.
    """

    # Critical thresholds -- if a metric drops below these values the
    # violation is flagged as *critical* regardless of the benchmark target.
    _CRITICAL_THRESHOLDS: Dict[str, Tuple[float, str]] = {
        "privacy_score": (0.75, "higher_better"),
        "quality_score": (0.60, "higher_better"),
        "ml_accuracy": (0.55, "higher_better"),
        "f1_score": (0.50, "higher_better"),
        "auc": (0.60, "higher_better"),
        "mean_nnd": (0.40, "higher_better"),
        "mean_jsd": (0.08, "lower_better"),
        "max_jsd": (0.20, "lower_better"),
        "mean_ks": (0.35, "lower_better"),
        "high_risk_pct": (0.05, "lower_better"),
        "max_amplification_ratio": (1.50, "lower_better"),
    }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, benchmarks: dict = None):
        """Initialize with enterprise benchmark targets.

        Parameters
        ----------
        benchmarks : dict, optional
            Custom benchmark definitions.  Each key is a metric name and
            the value is a dict with ``target`` (float) and ``direction``
            (``'higher_better'`` or ``'lower_better'``).  When *None* the
            default ``ENTERPRISE_BENCHMARKS`` from the project config are
            used.
        """
        self.benchmarks: Dict[str, dict] = benchmarks or ENTERPRISE_BENCHMARKS
        self.history: List[Tuple[int, dict]] = []
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Core tracking
    # ------------------------------------------------------------------

    def track_benchmarks(self, metrics: dict, iteration: int = None) -> dict:
        """Compare *metrics* to benchmark targets and record in history.

        Parameters
        ----------
        metrics : dict
            Mapping of metric names to their current float values.
        iteration : int, optional
            Iteration / cycle number.  When *None* it is inferred from
            the length of the existing history.

        Returns
        -------
        dict
            ``met``      -- list of metric names that meet benchmarks.
            ``not_met``  -- list of metric names that fall short.
            ``progress`` -- dict of metric name -> percentage toward target.
            ``all_met``  -- bool indicating whether every benchmark is met.
        """
        if iteration is None:
            iteration = len(self.history) + 1

        # Record observation
        self.history.append((iteration, dict(metrics)))
        self.logger.info(
            "Iteration %d: tracked %d metrics against %d benchmarks.",
            iteration,
            len(metrics),
            len(self.benchmarks),
        )

        met: List[str] = []
        not_met: List[str] = []

        for metric_name, bench in self.benchmarks.items():
            if metric_name not in metrics:
                not_met.append(metric_name)
                continue

            value = metrics[metric_name]
            target = bench["target"]
            direction = bench["direction"]

            if self._is_met(value, target, direction):
                met.append(metric_name)
            else:
                not_met.append(metric_name)

        progress = self.compute_progress(metrics)
        all_met = len(not_met) == 0

        if all_met:
            self.logger.info(
                "Iteration %d: ALL enterprise benchmarks met!", iteration
            )
        else:
            self.logger.info(
                "Iteration %d: %d/%d benchmarks met.  Not met: %s",
                iteration,
                len(met),
                len(self.benchmarks),
                ", ".join(not_met),
            )

        return {
            "met": met,
            "not_met": not_met,
            "progress": progress,
            "all_met": all_met,
        }

    # ------------------------------------------------------------------
    # Progress computation
    # ------------------------------------------------------------------

    def compute_progress(self, metrics: dict) -> dict:
        """Compute percentage progress toward each benchmark.

        For ``higher_better`` metrics::

            progress = min(value / target * 100, 100)

        For ``lower_better`` metrics::

            progress = min(target / value * 100, 100)  if value > 0
                     = 100                              otherwise

        Metrics not present in *metrics* receive a progress of ``0.0``.

        Returns
        -------
        dict
            Mapping of metric name -> progress percentage (0-100).
        """
        progress: Dict[str, float] = {}

        for metric_name, bench in self.benchmarks.items():
            target = bench["target"]
            direction = bench["direction"]

            if metric_name not in metrics:
                progress[metric_name] = 0.0
                continue

            value = metrics[metric_name]

            if direction == "higher_better":
                if target > 0:
                    progress[metric_name] = min(value / target * 100, 100.0)
                else:
                    # Target of zero or negative is trivially met for
                    # higher_better when value >= target.
                    progress[metric_name] = 100.0 if value >= target else 0.0
            else:  # lower_better
                if value > 0:
                    progress[metric_name] = min(target / value * 100, 100.0)
                else:
                    # A value of zero is the best possible for lower_better.
                    progress[metric_name] = 100.0

        return progress

    # ------------------------------------------------------------------
    # Benchmark satisfaction
    # ------------------------------------------------------------------

    def check_all_benchmarks_met(self, metrics: dict) -> bool:
        """Check if ALL enterprise benchmarks are satisfied.

        Returns *True* only when every benchmark metric is present in
        *metrics* and meets or exceeds its target.
        """
        for metric_name, bench in self.benchmarks.items():
            if metric_name not in metrics:
                return False
            if not self._is_met(
                metrics[metric_name], bench["target"], bench["direction"]
            ):
                return False
        return True

    # ------------------------------------------------------------------
    # Detailed per-metric status
    # ------------------------------------------------------------------

    def get_benchmark_status(self, metrics: dict) -> dict:
        """Get per-metric benchmark status.

        Returns
        -------
        dict
            Mapping of metric name -> {
                ``target``:   benchmark target,
                ``current``:  current value (or None),
                ``met``:      bool,
                ``progress``: percentage toward target,
            }
        """
        progress = self.compute_progress(metrics)
        status: Dict[str, dict] = {}

        for metric_name, bench in self.benchmarks.items():
            target = bench["target"]
            direction = bench["direction"]
            current = metrics.get(metric_name)

            if current is not None:
                met = self._is_met(current, target, direction)
            else:
                met = False

            status[metric_name] = {
                "target": target,
                "current": current,
                "met": met,
                "progress": progress.get(metric_name, 0.0),
            }

        return status

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trends(self) -> dict:
        """Detect improvement trends across recorded history.

        For each metric that appears in the history at least twice, a
        simple linear-regression slope is used to classify the trend:

        - ``improving`` -- the metric is moving toward its target.
        - ``stable``    -- the metric is roughly flat (|slope| < 1 % of
          the target per iteration).
        - ``degrading`` -- the metric is moving away from its target.

        Returns
        -------
        dict
            Mapping of metric name -> {
                ``trend``:  ``'improving'`` | ``'stable'`` | ``'degrading'``,
                ``values``: list of observed values in chronological order,
            }
        """
        if len(self.history) < 2:
            self.logger.debug(
                "Not enough history (%d entries) to detect trends.",
                len(self.history),
            )
            return {}

        # Collect per-metric time series
        metric_series: Dict[str, List[Tuple[int, float]]] = {}
        for iteration, metrics_dict in self.history:
            for metric_name, value in metrics_dict.items():
                if metric_name in self.benchmarks:
                    metric_series.setdefault(metric_name, []).append(
                        (iteration, value)
                    )

        trends: Dict[str, dict] = {}
        for metric_name, series in metric_series.items():
            if len(series) < 2:
                continue

            values = [v for _, v in series]
            slope = self._compute_slope(series)

            bench = self.benchmarks[metric_name]
            direction = bench["direction"]
            target = bench["target"]

            # Determine a stability threshold -- 1% of the target value.
            stability_threshold = abs(target) * 0.01 if target != 0 else 0.005

            if abs(slope) < stability_threshold:
                trend_label = "stable"
            elif direction == "higher_better":
                trend_label = "improving" if slope > 0 else "degrading"
            else:  # lower_better
                trend_label = "improving" if slope < 0 else "degrading"

            trends[metric_name] = {
                "trend": trend_label,
                "values": values,
            }

        return trends

    # ------------------------------------------------------------------
    # Critical violation detection
    # ------------------------------------------------------------------

    def detect_critical_violations(self, metrics: dict) -> list:
        """Detect critical benchmark violations.

        Critical thresholds are *stricter* than the enterprise benchmark
        targets and represent hard limits below which the synthetic data
        quality is unacceptable.

        Returns
        -------
        list
            List of human-readable violation description strings.
        """
        violations: List[str] = []

        for metric_name, (threshold, direction) in self._CRITICAL_THRESHOLDS.items():
            if metric_name not in metrics:
                continue

            value = metrics[metric_name]

            if direction == "higher_better" and value < threshold:
                violations.append(
                    f"CRITICAL: {metric_name} = {value:.4f} is below the "
                    f"critical threshold of {threshold:.4f}."
                )
            elif direction == "lower_better" and value > threshold:
                violations.append(
                    f"CRITICAL: {metric_name} = {value:.4f} exceeds the "
                    f"critical threshold of {threshold:.4f}."
                )

        # Also flag any benchmark metric that is missing entirely.
        for metric_name in self.benchmarks:
            if metric_name not in metrics:
                violations.append(
                    f"CRITICAL: {metric_name} is missing from the supplied "
                    f"metrics -- unable to assess benchmark compliance."
                )

        if violations:
            for v in violations:
                self.logger.warning(v)

        return violations

    # ------------------------------------------------------------------
    # Dashboard generation
    # ------------------------------------------------------------------

    def generate_dashboard(self, metrics: dict) -> str:
        """Generate a benchmark status dashboard as formatted text.

        The dashboard shows each benchmark metric with its current value,
        target, a visual progress bar, and a MET / NOT MET label.

        Returns
        -------
        str
            Multi-line formatted string suitable for console output.
        """
        status = self.get_benchmark_status(metrics)
        progress = self.compute_progress(metrics)
        violations = self.detect_critical_violations(metrics)

        bar_width = 20
        lines: List[str] = []

        lines.append("=" * 72)
        lines.append("  ENTERPRISE BENCHMARK DASHBOARD")
        lines.append("=" * 72)
        lines.append("")

        # Header
        lines.append(
            f"  {'Metric':<28} {'Current':>8}  {'Target':>8}  "
            f"{'Progress':>{bar_width + 6}}  {'Status':<8}"
        )
        lines.append("  " + "-" * 68)

        for metric_name in sorted(status.keys()):
            info = status[metric_name]
            target = info["target"]
            current = info["current"]
            met = info["met"]
            pct = progress.get(metric_name, 0.0)

            current_str = f"{current:.4f}" if current is not None else "  N/A "
            target_str = f"{target:.4f}"

            # Build a simple text progress bar
            filled = int(round(pct / 100 * bar_width))
            bar = "#" * filled + "-" * (bar_width - filled)
            pct_str = f"[{bar}] {pct:5.1f}%"

            status_str = "MET" if met else "NOT MET"

            lines.append(
                f"  {metric_name:<28} {current_str:>8}  {target_str:>8}  "
                f"{pct_str}  {status_str:<8}"
            )

        lines.append("")

        # Summary
        met_count = sum(1 for v in status.values() if v["met"])
        total_count = len(status)
        overall_pct = (
            sum(progress.values()) / len(progress) if progress else 0.0
        )

        lines.append(f"  Benchmarks met: {met_count}/{total_count}")
        lines.append(f"  Overall progress: {overall_pct:.1f}%")
        lines.append(
            f"  All benchmarks satisfied: "
            f"{'YES' if met_count == total_count else 'NO'}"
        )

        # Critical violations
        if violations:
            lines.append("")
            lines.append("  !! CRITICAL VIOLATIONS !!")
            for v in violations:
                lines.append(f"    - {v}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, metrics: dict) -> str:
        """Generate a comprehensive benchmark tracking report.

        The report includes the dashboard, trend analysis (when history
        is available), and actionable recommendations.

        Returns
        -------
        str
            Multi-line formatted report string.
        """
        sections: List[str] = []

        # ---- Section 1: Dashboard ----
        sections.append(self.generate_dashboard(metrics))

        # ---- Section 2: Trend Analysis ----
        trends = self.detect_trends()
        if trends:
            trend_lines: List[str] = []
            trend_lines.append("")
            trend_lines.append("  TREND ANALYSIS")
            trend_lines.append("  " + "-" * 40)

            for metric_name in sorted(trends.keys()):
                info = trends[metric_name]
                trend = info["trend"]
                values = info["values"]

                if len(values) >= 2:
                    delta = values[-1] - values[-2]
                    delta_str = f"({'+' if delta >= 0 else ''}{delta:.4f})"
                else:
                    delta_str = ""

                trend_display = trend.upper()
                trend_lines.append(
                    f"    {metric_name:<28} {trend_display:<12} "
                    f"last={values[-1]:.4f} {delta_str}"
                )

            sections.append("\n".join(trend_lines))

        # ---- Section 3: History Summary ----
        if self.history:
            history_lines: List[str] = []
            history_lines.append("")
            history_lines.append("  HISTORY SUMMARY")
            history_lines.append("  " + "-" * 40)
            history_lines.append(
                f"    Total iterations tracked: {len(self.history)}"
            )

            # Show progress of overall quality across iterations
            if len(self.history) >= 2:
                first_iter, first_metrics = self.history[0]
                last_iter, last_metrics = self.history[-1]
                history_lines.append(
                    f"    First iteration: {first_iter}  |  "
                    f"Latest iteration: {last_iter}"
                )

                # Compare overall benchmark satisfaction
                first_met = sum(
                    1
                    for mn, b in self.benchmarks.items()
                    if mn in first_metrics
                    and self._is_met(
                        first_metrics[mn], b["target"], b["direction"]
                    )
                )
                last_met = sum(
                    1
                    for mn, b in self.benchmarks.items()
                    if mn in last_metrics
                    and self._is_met(
                        last_metrics[mn], b["target"], b["direction"]
                    )
                )
                history_lines.append(
                    f"    Benchmarks met (first): {first_met}/{len(self.benchmarks)}"
                )
                history_lines.append(
                    f"    Benchmarks met (latest): {last_met}/{len(self.benchmarks)}"
                )

            sections.append("\n".join(history_lines))

        # ---- Section 4: Recommendations ----
        violations = self.detect_critical_violations(metrics)
        status = self.get_benchmark_status(metrics)
        not_met = [
            name for name, info in status.items() if not info["met"]
        ]

        rec_lines: List[str] = []
        rec_lines.append("")
        rec_lines.append("  RECOMMENDATIONS")
        rec_lines.append("  " + "-" * 40)

        if not not_met and not violations:
            rec_lines.append(
                "    All benchmarks are met.  Continue monitoring for "
                "regressions."
            )
        else:
            if violations:
                rec_lines.append(
                    "    [!] Address critical violations immediately:"
                )
                for v in violations:
                    rec_lines.append(f"        - {v}")
                rec_lines.append("")

            if not_met:
                rec_lines.append(
                    "    The following benchmarks require improvement:"
                )
                for metric_name in sorted(not_met):
                    info = status[metric_name]
                    current = info["current"]
                    target = info["target"]
                    pct = info["progress"]
                    current_str = (
                        f"{current:.4f}" if current is not None else "N/A"
                    )
                    rec_lines.append(
                        f"      - {metric_name}: current={current_str}, "
                        f"target={target:.4f}, progress={pct:.1f}%"
                    )

                rec_lines.append("")

                # Suggest focusing on the metric furthest from its target
                worst_metric = min(
                    not_met, key=lambda m: status[m]["progress"]
                )
                rec_lines.append(
                    f"    >> Priority: Focus on '{worst_metric}' which has "
                    f"the lowest progress ({status[worst_metric]['progress']:.1f}%)."
                )

                # Trend-aware advice
                if trends and worst_metric in trends:
                    trend_info = trends[worst_metric]["trend"]
                    if trend_info == "degrading":
                        rec_lines.append(
                            f"    >> WARNING: '{worst_metric}' is actively "
                            f"degrading.  Consider reverting recent changes."
                        )
                    elif trend_info == "stable":
                        rec_lines.append(
                            f"    >> '{worst_metric}' has stalled.  Consider "
                            f"a different optimisation strategy."
                        )

        sections.append("\n".join(rec_lines))

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_met(value: float, target: float, direction: str) -> bool:
        """Return *True* if *value* satisfies the benchmark *target*."""
        if direction == "higher_better":
            return value >= target
        else:  # lower_better
            return value <= target

    @staticmethod
    def _compute_slope(
        series: List[Tuple[int, float]],
    ) -> float:
        """Compute the ordinary-least-squares slope for *(x, y)* pairs.

        Uses the standard formula::

            slope = (n * sum(x*y) - sum(x)*sum(y))
                  / (n * sum(x^2) - (sum(x))^2)

        Returns 0.0 when the denominator is zero (all x values identical).
        """
        n = len(series)
        if n < 2:
            return 0.0

        sum_x = sum(x for x, _ in series)
        sum_y = sum(y for _, y in series)
        sum_xy = sum(x * y for x, y in series)
        sum_x2 = sum(x * x for x, _ in series)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator
