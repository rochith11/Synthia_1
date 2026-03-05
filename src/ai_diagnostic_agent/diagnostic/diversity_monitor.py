"""Diversity monitoring for synthetic data quality assessment.

Detects mode collapse, measures uniqueness, computes Shannon entropy
ratios, and produces a composite diversity score comparing synthetic
data to the original real dataset.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

from src.ai_diagnostic_agent.models import DiversityReport, QualityLevel
from src.ai_diagnostic_agent.config import DIVERSITY_THRESHOLDS


class DiversityMonitor:
    """Monitors the diversity of synthetic datasets relative to real data.

    Provides uniqueness ratio computation, duplicate detection, Shannon
    entropy comparison per categorical column, mode-collapse detection,
    and a composite diversity score that rolls all sub-metrics into a
    single [0, 1] value.
    """

    def __init__(self):
        """Initialize diversity monitor."""
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_diversity(
        self, synthetic: pd.DataFrame, real: pd.DataFrame
    ) -> DiversityReport:
        """Full diversity analysis comparing synthetic to real data.

        Computes all diversity sub-metrics, assembles them into a
        ``DiversityReport``, derives the composite score and severity
        level, and populates a human-readable issues list.

        Parameters
        ----------
        synthetic : pd.DataFrame
            The synthetic dataset to evaluate.
        real : pd.DataFrame
            The real (reference) dataset.

        Returns
        -------
        DiversityReport
            Comprehensive diversity analysis results.
        """
        self.logger.info(
            "Starting diversity analysis (synthetic=%d rows, real=%d rows)",
            len(synthetic),
            len(real),
        )

        # Uniqueness and duplicates
        unique_ratio = self.compute_uniqueness(synthetic)
        duplicate_count, duplicate_indices = self.detect_duplicates(synthetic)
        duplicate_rate = duplicate_count / max(len(synthetic), 1)

        # Entropy per categorical column
        common_columns = [c for c in real.columns if c in synthetic.columns]
        categorical_columns = [
            c for c in common_columns if self._is_categorical(real[c])
        ]

        syn_entropy: Dict[str, float] = {}
        real_entropy_map: Dict[str, float] = {}
        entropy_ratios: Dict[str, float] = {}

        for col in categorical_columns:
            s_ent = self.compute_entropy(synthetic, col)
            r_ent = self.compute_entropy(real, col)
            syn_entropy[col] = s_ent
            real_entropy_map[col] = r_ent
            # Avoid division by zero
            if r_ent == 0.0:
                entropy_ratios[col] = 1.0
            else:
                entropy_ratios[col] = s_ent / r_ent

        # Mode collapse
        is_collapsed, collapsed_columns = self.detect_mode_collapse(synthetic)

        # Assemble initial report (score and severity filled below)
        report = DiversityReport(
            unique_row_ratio=unique_ratio,
            duplicate_count=duplicate_count,
            duplicate_rate=duplicate_rate,
            categorical_entropy=syn_entropy,
            real_entropy=real_entropy_map,
            entropy_ratios=entropy_ratios,
            mode_collapse_detected=is_collapsed,
            mode_collapse_columns=collapsed_columns,
            diversity_score=0.0,
            issues=[],
            severity=QualityLevel.ACCEPTABLE,
        )

        # Composite score
        report.diversity_score = self._compute_diversity_score(report)

        # Severity classification
        report.severity = self._determine_severity(report)

        # Collect human-readable issues
        report.issues = self._collect_issues(report)

        self.logger.info(
            "Diversity analysis complete: score=%.4f, severity=%s",
            report.diversity_score,
            report.severity.value,
        )

        return report

    def compute_uniqueness(self, data: pd.DataFrame) -> float:
        """Compute ratio of unique rows in the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset to evaluate.

        Returns
        -------
        float
            ``unique_rows / total_rows``.  Returns 1.0 for an empty
            DataFrame.
        """
        if len(data) == 0:
            return 1.0
        unique_rows = len(data.drop_duplicates())
        return unique_rows / len(data)

    def detect_duplicates(self, data: pd.DataFrame) -> Tuple[int, List[int]]:
        """Detect duplicate records in the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset to check.

        Returns
        -------
        Tuple[int, List[int]]
            ``(duplicate_count, list_of_duplicate_row_indices)``
        """
        duplicated_mask = data.duplicated(keep="first")
        duplicate_indices = list(data.index[duplicated_mask])
        duplicate_count = int(duplicated_mask.sum())
        return duplicate_count, duplicate_indices

    def compute_entropy(self, data: pd.DataFrame, column: str) -> float:
        """Compute Shannon entropy for a categorical column.

        Uses ``scipy.stats.entropy`` on the normalised value counts,
        which yields the natural-log-based Shannon entropy.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing the column.
        column : str
            Name of the categorical column.

        Returns
        -------
        float
            Shannon entropy of the column's value distribution.
            Returns 0.0 if the column is empty or missing.
        """
        if column not in data.columns or len(data) == 0:
            return 0.0
        value_counts = data[column].value_counts(normalize=True)
        if len(value_counts) == 0:
            return 0.0
        return float(shannon_entropy(value_counts))

    def compare_diversity(
        self, synthetic: pd.DataFrame, real: pd.DataFrame
    ) -> Dict:
        """Compare diversity metrics between synthetic and real datasets.

        Parameters
        ----------
        synthetic : pd.DataFrame
            Synthetic dataset.
        real : pd.DataFrame
            Real (reference) dataset.

        Returns
        -------
        dict
            ``synthetic_unique_ratio`` -- uniqueness of synthetic data.
            ``real_unique_ratio`` -- uniqueness of real data.
            ``entropy_ratios`` -- per-column ``synthetic_entropy / real_entropy``.
        """
        syn_unique = self.compute_uniqueness(synthetic)
        real_unique = self.compute_uniqueness(real)

        common_columns = [c for c in real.columns if c in synthetic.columns]
        categorical_columns = [
            c for c in common_columns if self._is_categorical(real[c])
        ]

        entropy_ratios: Dict[str, float] = {}
        for col in categorical_columns:
            s_ent = self.compute_entropy(synthetic, col)
            r_ent = self.compute_entropy(real, col)
            if r_ent == 0.0:
                entropy_ratios[col] = 1.0
            else:
                entropy_ratios[col] = s_ent / r_ent

        return {
            "synthetic_unique_ratio": syn_unique,
            "real_unique_ratio": real_unique,
            "entropy_ratios": entropy_ratios,
        }

    def detect_mode_collapse(
        self, data: pd.DataFrame, threshold: float = 0.95
    ) -> Tuple[bool, List[str]]:
        """Detect if the generative model has collapsed to limited modes.

        For each categorical column, checks whether any single value
        accounts for more than *threshold* of all observations.  Also
        flags columns where the number of unique values is suspiciously
        low (fewer than two distinct values).

        Parameters
        ----------
        data : pd.DataFrame
            Synthetic dataset to inspect.
        threshold : float, optional
            Maximum acceptable frequency for a single category value.
            Defaults to ``0.95``.

        Returns
        -------
        Tuple[bool, List[str]]
            ``(is_collapsed, list_of_collapsed_column_names)``
        """
        collapsed_columns: List[str] = []

        for col in data.columns:
            if not self._is_categorical(data[col]):
                continue

            value_counts = data[col].value_counts(normalize=True)
            if len(value_counts) == 0:
                continue

            max_freq = value_counts.iloc[0]
            n_unique = len(value_counts)

            # Flag if a single value dominates beyond the threshold
            if max_freq > threshold:
                collapsed_columns.append(col)
                self.logger.warning(
                    "Mode collapse detected in column '%s': "
                    "top value frequency=%.4f (threshold=%.4f)",
                    col,
                    max_freq,
                    threshold,
                )

            # Flag if diversity has been reduced to a single value
            elif n_unique < 2:
                collapsed_columns.append(col)
                self.logger.warning(
                    "Mode collapse detected in column '%s': "
                    "only %d unique value(s)",
                    col,
                    n_unique,
                )

        is_collapsed = len(collapsed_columns) > 0
        return is_collapsed, collapsed_columns

    def generate_report(self, report: DiversityReport) -> str:
        """Generate a human-readable diversity analysis report.

        Parameters
        ----------
        report : DiversityReport
            The completed diversity report.

        Returns
        -------
        str
            Formatted multi-line report text.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DIVERSITY ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append("Summary:")
        lines.append(f"  Diversity score   : {report.diversity_score:.4f}")
        lines.append(f"  Severity          : {report.severity.value}")
        lines.append(f"  Unique row ratio  : {report.unique_row_ratio:.4f}")
        lines.append(f"  Duplicate count   : {report.duplicate_count}")
        lines.append(f"  Duplicate rate    : {report.duplicate_rate:.4f}")
        lines.append(
            f"  Mode collapse     : {'Yes' if report.mode_collapse_detected else 'No'}"
        )
        lines.append("")

        # Entropy ratios
        if report.entropy_ratios:
            lines.append("-" * 60)
            lines.append("  Entropy Ratios (synthetic / real)")
            lines.append("-" * 60)
            for col, ratio in sorted(report.entropy_ratios.items()):
                syn_ent = report.categorical_entropy.get(col, 0.0)
                real_ent = report.real_entropy.get(col, 0.0)
                status = "OK"
                if ratio < DIVERSITY_THRESHOLDS["min_entropy_ratio"]:
                    status = "LOW"
                elif ratio > DIVERSITY_THRESHOLDS["max_entropy_ratio"]:
                    status = "HIGH"
                lines.append(
                    f"  {col:<30s}  ratio={ratio:.4f}  "
                    f"(syn={syn_ent:.4f}, real={real_ent:.4f})  [{status}]"
                )
            lines.append("")

        # Mode collapse details
        if report.mode_collapse_detected:
            lines.append("-" * 60)
            lines.append("  Mode Collapse Details")
            lines.append("-" * 60)
            for col in report.mode_collapse_columns:
                lines.append(f"    - {col}")
            lines.append("")

        # Issues
        if report.issues:
            lines.append("-" * 60)
            lines.append("  Issues")
            lines.append("-" * 60)
            for issue in report.issues:
                lines.append(f"    - {issue}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("  END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scoring and severity
    # ------------------------------------------------------------------

    def _compute_diversity_score(self, report: DiversityReport) -> float:
        """Compute composite diversity score in the range [0, 1].

        The score is a weighted combination of three components:

        * **Unique ratio** (weight 0.4) -- targets > 0.95.
        * **Duplicate rate** (weight 0.3) -- targets < 0.02.
        * **Entropy ratio** (weight 0.3) -- targets all ratios in [0.90, 1.10].

        Parameters
        ----------
        report : DiversityReport
            Partially populated report with raw metrics.

        Returns
        -------
        float
            Composite score between 0 and 1.
        """
        # Unique ratio component (target > 0.95)
        unique_component = min(report.unique_row_ratio / 0.95, 1.0) * 0.4

        # Duplicate rate component (target < 0.02)
        if report.duplicate_rate < 0.02:
            dup_component = max(0, 1.0 - report.duplicate_rate / 0.02) * 0.3
        else:
            dup_component = 0.0

        # Entropy ratio component (target: all ratios between 0.90 and 1.10)
        if report.entropy_ratios:
            entropy_scores = [
                1.0 - abs(r - 1.0) / 0.10
                for r in report.entropy_ratios.values()
            ]
            avg_entropy_score = sum(s for s in entropy_scores) / len(entropy_scores)
            entropy_component = max(0, avg_entropy_score) * 0.3
        else:
            entropy_component = 0.3

        diversity_score = unique_component + dup_component + entropy_component
        return round(min(max(diversity_score, 0.0), 1.0), 6)

    def _determine_severity(self, report: DiversityReport) -> QualityLevel:
        """Determine severity level based on diversity metrics.

        Classification rules (evaluated in order):

        * **Critical** -- mode collapse detected *or* unique ratio < 0.80.
        * **Warning** -- unique ratio < 0.95 *or* duplicate rate > 0.02.
        * **Acceptable** -- any entropy ratio outside [0.90, 1.10].
        * **Excellent** -- all metrics within ideal targets.

        Parameters
        ----------
        report : DiversityReport
            Populated diversity report.

        Returns
        -------
        QualityLevel
            The determined severity level.
        """
        # Critical conditions
        if report.mode_collapse_detected:
            return QualityLevel.CRITICAL
        if report.unique_row_ratio < 0.80:
            return QualityLevel.CRITICAL

        # Warning conditions
        if report.unique_row_ratio < DIVERSITY_THRESHOLDS["min_unique_ratio"]:
            return QualityLevel.WARNING
        if report.duplicate_rate > DIVERSITY_THRESHOLDS["max_duplicate_rate"]:
            return QualityLevel.WARNING

        # Acceptable -- entropy ratios slightly off
        min_ratio = DIVERSITY_THRESHOLDS["min_entropy_ratio"]
        max_ratio = DIVERSITY_THRESHOLDS["max_entropy_ratio"]
        for ratio in report.entropy_ratios.values():
            if ratio < min_ratio or ratio > max_ratio:
                return QualityLevel.ACCEPTABLE

        # All metrics within targets
        return QualityLevel.EXCELLENT

    # ------------------------------------------------------------------
    # Issue collection
    # ------------------------------------------------------------------

    def _collect_issues(self, report: DiversityReport) -> List[str]:
        """Build a list of human-readable issue descriptions.

        Parameters
        ----------
        report : DiversityReport
            Fully scored diversity report.

        Returns
        -------
        List[str]
            Descriptions of each identified problem.
        """
        issues: List[str] = []

        # Uniqueness issues
        if report.unique_row_ratio < 0.80:
            issues.append(
                f"Critically low uniqueness: only {report.unique_row_ratio:.2%} "
                f"of rows are unique."
            )
        elif report.unique_row_ratio < DIVERSITY_THRESHOLDS["min_unique_ratio"]:
            issues.append(
                f"Unique row ratio ({report.unique_row_ratio:.4f}) is below "
                f"the target threshold of {DIVERSITY_THRESHOLDS['min_unique_ratio']:.2f}."
            )

        # Duplicate issues
        if report.duplicate_rate > DIVERSITY_THRESHOLDS["max_duplicate_rate"]:
            issues.append(
                f"Duplicate rate ({report.duplicate_rate:.4f}) exceeds the "
                f"maximum threshold of {DIVERSITY_THRESHOLDS['max_duplicate_rate']:.2f}. "
                f"{report.duplicate_count} duplicate rows detected."
            )

        # Mode collapse
        if report.mode_collapse_detected:
            cols = ", ".join(report.mode_collapse_columns)
            issues.append(
                f"Mode collapse detected in {len(report.mode_collapse_columns)} "
                f"column(s): {cols}. The generative model may be producing "
                f"insufficient variety for these features."
            )

        # Entropy ratio issues
        min_ratio = DIVERSITY_THRESHOLDS["min_entropy_ratio"]
        max_ratio = DIVERSITY_THRESHOLDS["max_entropy_ratio"]
        low_entropy_cols: List[str] = []
        high_entropy_cols: List[str] = []

        for col, ratio in report.entropy_ratios.items():
            if ratio < min_ratio:
                low_entropy_cols.append(f"{col} ({ratio:.4f})")
            elif ratio > max_ratio:
                high_entropy_cols.append(f"{col} ({ratio:.4f})")

        if low_entropy_cols:
            issues.append(
                f"Entropy is lower than expected for: "
                f"{', '.join(low_entropy_cols)}. "
                f"Synthetic data may lack sufficient category diversity."
            )

        if high_entropy_cols:
            issues.append(
                f"Entropy is higher than expected for: "
                f"{', '.join(high_entropy_cols)}. "
                f"Synthetic data may be introducing spurious categories."
            )

        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_categorical(col: pd.Series) -> bool:
        """Determine whether a column should be treated as categorical.

        Parameters
        ----------
        col : pd.Series
            The column to inspect.

        Returns
        -------
        bool
            ``True`` if the column is categorical, ``False`` otherwise.
        """
        if col.dtype == "object" or col.dtype.name == "category":
            return True
        if col.dtype == "bool":
            return True
        # Treat integer columns with very few unique values as categorical
        if pd.api.types.is_integer_dtype(col) and col.nunique() <= 20:
            return True
        return False
