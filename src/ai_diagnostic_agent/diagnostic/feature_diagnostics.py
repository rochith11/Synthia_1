"""Feature-level diagnostics for comparing synthetic and real data distributions."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional

from src.ai_diagnostic_agent.models import ColumnDiagnosis, CorrelationAnalysis, QualityLevel


class FeatureDiagnostics:
    """Analyzes per-column distribution fidelity between synthetic and real data.

    Provides Jensen-Shannon Divergence, Kolmogorov-Smirnov tests, bias
    amplification detection, and correlation preservation analysis to
    identify which features a synthetic data generator reproduces well
    and which ones need improvement.
    """

    def __init__(self):
        """Initialize feature diagnostics."""
        self.epsilon = 1e-10
        self.n_bins = 30
        self.correlation_loss_threshold = 0.20

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_all_columns(
        self, synthetic: pd.DataFrame, real: pd.DataFrame
    ) -> List[ColumnDiagnosis]:
        """Analyze all columns and return list of diagnoses.

        Parameters
        ----------
        synthetic : pd.DataFrame
            The synthetic dataset.
        real : pd.DataFrame
            The real (reference) dataset.

        Returns
        -------
        List[ColumnDiagnosis]
            A diagnosis object for every column present in both datasets.
        """
        common_columns = [c for c in real.columns if c in synthetic.columns]
        diagnoses: List[ColumnDiagnosis] = []

        bias_ratios = self.detect_bias_amplification(synthetic, real)

        for col in common_columns:
            diag = self.analyze_column(synthetic[col], real[col], col)

            # Attach bias amplification ratio if available
            if col in bias_ratios:
                diag.bias_amplification_ratio = bias_ratios[col]
                if bias_ratios[col] > 1.5:
                    diag.issues.append(
                        f"Bias amplification detected (ratio={bias_ratios[col]:.2f})"
                    )
                    diag.recommendations.append(
                        "Apply rebalancing or fairness-aware generation for this column."
                    )

            diagnoses.append(diag)

        return diagnoses

    def analyze_column(
        self, synthetic_col: pd.Series, real_col: pd.Series, column_name: str
    ) -> ColumnDiagnosis:
        """Analyze single column for distribution mismatch.

        Parameters
        ----------
        synthetic_col : pd.Series
            Column from the synthetic dataset.
        real_col : pd.Series
            Corresponding column from the real dataset.
        column_name : str
            Name of the column being analyzed.

        Returns
        -------
        ColumnDiagnosis
            Diagnosis containing JSD, optional KS statistics, severity, and
            human-readable issues / recommendations.
        """
        # Drop NaN values for analysis
        synthetic_clean = synthetic_col.dropna()
        real_clean = real_col.dropna()

        is_categorical = self._is_categorical(real_col)
        col_type = "categorical" if is_categorical else "continuous"

        # Compute Jensen-Shannon Divergence
        jsd = self._compute_jsd(synthetic_clean, real_clean, is_categorical)

        # Compute KS statistic for continuous columns
        ks_stat: Optional[float] = None
        ks_pval: Optional[float] = None
        if not is_categorical and len(synthetic_clean) > 0 and len(real_clean) > 0:
            ks_stat, ks_pval = self._compute_ks(synthetic_clean, real_clean)

        # Classify severity based on JSD
        severity = self._classify_severity(jsd)

        # Build issues and recommendations
        issues: List[str] = []
        recommendations: List[str] = []

        if severity == QualityLevel.CRITICAL:
            issues.append(
                f"Severe distribution mismatch (JSD={jsd:.4f})"
            )
            recommendations.append(
                "Investigate model capacity or training duration for this feature."
            )
        elif severity == QualityLevel.WARNING:
            issues.append(
                f"Moderate distribution drift (JSD={jsd:.4f})"
            )
            recommendations.append(
                "Consider adjusting preprocessing or adding feature-specific loss weighting."
            )
        elif severity == QualityLevel.ACCEPTABLE:
            issues.append(
                f"Minor distribution difference (JSD={jsd:.4f})"
            )

        if ks_pval is not None and ks_pval < 0.05:
            issues.append(
                f"KS test rejects identical distribution (stat={ks_stat:.4f}, p={ks_pval:.4e})"
            )

        return ColumnDiagnosis(
            column_name=column_name,
            column_type=col_type,
            jsd=jsd,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Statistical measures
    # ------------------------------------------------------------------

    def _compute_jsd(
        self, synthetic_col: pd.Series, real_col: pd.Series, is_categorical: bool
    ) -> float:
        """Compute Jensen-Shannon Divergence for a column.

        For categorical columns the value-count distributions are compared
        directly.  For continuous columns the data is binned into histograms
        first.

        Parameters
        ----------
        synthetic_col : pd.Series
            Cleaned synthetic column (no NaN).
        real_col : pd.Series
            Cleaned real column (no NaN).
        is_categorical : bool
            Whether the column should be treated as categorical.

        Returns
        -------
        float
            The JSD value (squared JS distance).
        """
        if len(synthetic_col) == 0 or len(real_col) == 0:
            return 1.0  # Maximum divergence when data is missing

        if is_categorical:
            return self._jsd_categorical(synthetic_col, real_col)
        else:
            return self._jsd_continuous(synthetic_col, real_col)

    def _jsd_categorical(self, synthetic_col: pd.Series, real_col: pd.Series) -> float:
        """JSD for categorical columns via aligned value-count distributions."""
        syn_counts = synthetic_col.value_counts(normalize=True)
        real_counts = real_col.value_counts(normalize=True)

        # Union of all categories
        all_categories = set(syn_counts.index) | set(real_counts.index)

        syn_dist = np.array(
            [syn_counts.get(cat, 0.0) for cat in all_categories], dtype=np.float64
        )
        real_dist = np.array(
            [real_counts.get(cat, 0.0) for cat in all_categories], dtype=np.float64
        )

        # Add epsilon for numerical stability
        syn_dist = syn_dist + self.epsilon
        real_dist = real_dist + self.epsilon

        # Renormalize after adding epsilon
        syn_dist = syn_dist / syn_dist.sum()
        real_dist = real_dist / real_dist.sum()

        # jensenshannon returns the JS *distance*; JSD = distance^2
        js_distance = jensenshannon(syn_dist, real_dist)
        return float(js_distance ** 2)

    def _jsd_continuous(self, synthetic_col: pd.Series, real_col: pd.Series) -> float:
        """JSD for continuous columns via histogram binning."""
        combined = pd.concat([synthetic_col, real_col])
        bins = np.linspace(combined.min(), combined.max(), self.n_bins + 1)

        syn_hist, _ = np.histogram(synthetic_col, bins=bins)
        real_hist, _ = np.histogram(real_col, bins=bins)

        # Normalize to probability distributions
        syn_dist = syn_hist.astype(np.float64)
        real_dist = real_hist.astype(np.float64)

        syn_total = syn_dist.sum()
        real_total = real_dist.sum()

        if syn_total > 0:
            syn_dist = syn_dist / syn_total
        if real_total > 0:
            real_dist = real_dist / real_total

        # Add epsilon for numerical stability
        syn_dist = syn_dist + self.epsilon
        real_dist = real_dist + self.epsilon

        # Renormalize
        syn_dist = syn_dist / syn_dist.sum()
        real_dist = real_dist / real_dist.sum()

        js_distance = jensenshannon(syn_dist, real_dist)
        return float(js_distance ** 2)

    def _compute_ks(
        self, synthetic_col: pd.Series, real_col: pd.Series
    ) -> Tuple[float, float]:
        """Compute KS statistic and p-value for continuous columns.

        Parameters
        ----------
        synthetic_col : pd.Series
            Cleaned synthetic column.
        real_col : pd.Series
            Cleaned real column.

        Returns
        -------
        Tuple[float, float]
            (ks_statistic, p_value)
        """
        stat, pvalue = ks_2samp(real_col, synthetic_col)
        return float(stat), float(pvalue)

    # ------------------------------------------------------------------
    # Bias amplification
    # ------------------------------------------------------------------

    def detect_bias_amplification(
        self, synthetic: pd.DataFrame, real: pd.DataFrame
    ) -> Dict[str, float]:
        """Detect bias amplification ratios per column.

        For each categorical column the *imbalance ratio*
        (max_freq / min_freq) is computed on both the real and synthetic
        data.  The amplification ratio is ``synthetic_ratio / real_ratio``.
        A value > 1 indicates that the synthetic generator amplified the
        existing class imbalance.

        Parameters
        ----------
        synthetic : pd.DataFrame
            Synthetic dataset.
        real : pd.DataFrame
            Real dataset.

        Returns
        -------
        Dict[str, float]
            Column name -> amplification ratio.
        """
        ratios: Dict[str, float] = {}
        common_columns = [c for c in real.columns if c in synthetic.columns]

        for col in common_columns:
            if not self._is_categorical(real[col]):
                continue

            real_freq = real[col].value_counts()
            syn_freq = synthetic[col].value_counts()

            if len(real_freq) < 2 or len(syn_freq) < 2:
                continue

            real_imbalance = real_freq.max() / max(real_freq.min(), self.epsilon)
            syn_imbalance = syn_freq.max() / max(syn_freq.min(), self.epsilon)

            amplification = syn_imbalance / max(real_imbalance, self.epsilon)
            ratios[col] = float(amplification)

        return ratios

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def analyze_correlations(
        self, synthetic: pd.DataFrame, real: pd.DataFrame
    ) -> CorrelationAnalysis:
        """Analyze correlation preservation between synthetic and real data.

        Categorical columns are encoded as integer codes before computing
        Pearson correlation matrices.  Column pairs whose absolute
        correlation difference exceeds the threshold (0.20) are flagged as
        *lost*.

        Parameters
        ----------
        synthetic : pd.DataFrame
            Synthetic dataset.
        real : pd.DataFrame
            Real dataset.

        Returns
        -------
        CorrelationAnalysis
            Preserved/lost pairs, overall similarity, and max loss.
        """
        common_columns = [c for c in real.columns if c in synthetic.columns]

        if len(common_columns) < 2:
            return CorrelationAnalysis(
                overall_similarity=1.0,
                max_loss=0.0,
            )

        # Encode categoricals and select common columns
        real_encoded = self._encode_for_correlation(real[common_columns])
        syn_encoded = self._encode_for_correlation(synthetic[common_columns])

        real_corr = real_encoded.corr()
        syn_corr = syn_encoded.corr()

        # Align matrices (in case of column ordering differences)
        cols = real_corr.columns
        syn_corr = syn_corr.reindex(index=cols, columns=cols)

        diff_matrix = (syn_corr - real_corr).abs()

        preserved_pairs: List[Tuple[str, str]] = []
        lost_pairs: List[Tuple[str, str, float]] = []
        max_loss = 0.0

        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                col_i = cols[i]
                col_j = cols[j]
                diff = diff_matrix.iloc[i, j]

                if np.isnan(diff):
                    continue

                if diff > self.correlation_loss_threshold:
                    lost_pairs.append((col_i, col_j, float(diff)))
                    if diff > max_loss:
                        max_loss = float(diff)
                else:
                    preserved_pairs.append((col_i, col_j))

        # Overall similarity = 1 - mean(|diff_matrix|) over upper triangle
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                val = diff_matrix.iloc[i, j]
                if not np.isnan(val):
                    upper_triangle.append(val)

        if upper_triangle:
            overall_similarity = 1.0 - float(np.mean(upper_triangle))
        else:
            overall_similarity = 1.0

        return CorrelationAnalysis(
            preserved_pairs=preserved_pairs,
            lost_pairs=lost_pairs,
            overall_similarity=overall_similarity,
            max_loss=max_loss,
        )

    # ------------------------------------------------------------------
    # Prioritization & reporting
    # ------------------------------------------------------------------

    def prioritize_features(
        self, diagnoses: List[ColumnDiagnosis]
    ) -> List[ColumnDiagnosis]:
        """Sort diagnoses by severity (Critical first), then by JSD descending.

        Parameters
        ----------
        diagnoses : List[ColumnDiagnosis]
            Unsorted list of column diagnoses.

        Returns
        -------
        List[ColumnDiagnosis]
            Sorted list with the most problematic columns first.
        """
        severity_order = {
            QualityLevel.CRITICAL: 0,
            QualityLevel.WARNING: 1,
            QualityLevel.ACCEPTABLE: 2,
            QualityLevel.EXCELLENT: 3,
        }

        return sorted(
            diagnoses,
            key=lambda d: (severity_order.get(d.severity, 4), -d.jsd),
        )

    def generate_report(
        self,
        diagnoses: List[ColumnDiagnosis],
        correlation_analysis: Optional[CorrelationAnalysis] = None,
    ) -> str:
        """Generate column-level diagnostics report text.

        Parameters
        ----------
        diagnoses : List[ColumnDiagnosis]
            List of per-column diagnoses (ideally already prioritized).
        correlation_analysis : CorrelationAnalysis, optional
            Correlation preservation results to include in the report.

        Returns
        -------
        str
            Human-readable report text.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  FEATURE DIAGNOSTICS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Summary counts
        severity_counts = {level: 0 for level in QualityLevel}
        for d in diagnoses:
            severity_counts[d.severity] += 1

        lines.append("Summary:")
        lines.append(f"  Total columns analyzed : {len(diagnoses)}")
        for level in [QualityLevel.CRITICAL, QualityLevel.WARNING,
                      QualityLevel.ACCEPTABLE, QualityLevel.EXCELLENT]:
            lines.append(f"  {level.value:<12}: {severity_counts[level]}")
        lines.append("")

        # Per-column details
        lines.append("-" * 60)
        lines.append("  Column Details")
        lines.append("-" * 60)

        for d in diagnoses:
            lines.append("")
            lines.append(f"  Column: {d.column_name}  ({d.column_type})")
            lines.append(f"    Severity : {d.severity.value}")
            lines.append(f"    JSD      : {d.jsd:.6f}")

            if d.ks_statistic is not None:
                lines.append(f"    KS stat  : {d.ks_statistic:.6f}")
                lines.append(f"    KS p-val : {d.ks_pvalue:.4e}")

            if d.bias_amplification_ratio != 1.0:
                lines.append(
                    f"    Bias amp.: {d.bias_amplification_ratio:.4f}"
                )

            if d.issues:
                lines.append("    Issues:")
                for issue in d.issues:
                    lines.append(f"      - {issue}")

            if d.recommendations:
                lines.append("    Recommendations:")
                for rec in d.recommendations:
                    lines.append(f"      * {rec}")

        # Correlation section
        if correlation_analysis is not None:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  Correlation Preservation")
            lines.append("-" * 60)
            lines.append(
                f"  Overall similarity : {correlation_analysis.overall_similarity:.4f}"
            )
            lines.append(
                f"  Max loss           : {correlation_analysis.max_loss:.4f}"
            )
            lines.append(
                f"  Preserved pairs    : {len(correlation_analysis.preserved_pairs)}"
            )
            lines.append(
                f"  Lost pairs         : {len(correlation_analysis.lost_pairs)}"
            )

            if correlation_analysis.lost_pairs:
                lines.append("")
                lines.append("  Top lost correlations:")
                sorted_lost = sorted(
                    correlation_analysis.lost_pairs, key=lambda x: -x[2]
                )
                for col1, col2, diff in sorted_lost[:10]:
                    lines.append(f"    {col1} <-> {col2} : diff={diff:.4f}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("  END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_categorical(col: pd.Series) -> bool:
        """Determine whether a column should be treated as categorical."""
        if col.dtype == "object" or col.dtype.name == "category":
            return True
        if col.dtype == "bool":
            return True
        # Treat integer columns with very few unique values as categorical
        if pd.api.types.is_integer_dtype(col) and col.nunique() <= 20:
            return True
        return False

    @staticmethod
    def _encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns as integer codes for correlation."""
        encoded = df.copy()
        for col in encoded.columns:
            if encoded[col].dtype == "object" or encoded[col].dtype.name == "category":
                encoded[col] = encoded[col].astype("category").cat.codes
            elif encoded[col].dtype == "bool":
                encoded[col] = encoded[col].astype(int)
        return encoded

    @staticmethod
    def _classify_severity(jsd: float) -> QualityLevel:
        """Map JSD value to a QualityLevel severity.

        Thresholds
        ----------
        > 0.10  -> Critical
        > 0.05  -> Warning
        > 0.03  -> Acceptable
        <= 0.03 -> Excellent
        """
        if jsd > 0.10:
            return QualityLevel.CRITICAL
        elif jsd > 0.05:
            return QualityLevel.WARNING
        elif jsd > 0.03:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.EXCELLENT
