"""Statistical analysis for comparing real and synthetic data distributions."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import Dict, Tuple, List


class StatisticalAnalyzer:
    """Computes statistical similarity metrics between real and synthetic data."""

    def compute_ks_statistic(self, synthetic: pd.Series, real: pd.Series) -> Tuple[float, float]:
        """Compute Kolmogorov-Smirnov statistic for a numerical column.

        Args:
            synthetic: Synthetic data series (numerical)
            real: Real data series (numerical)

        Returns:
            Tuple of (ks_statistic, p_value)
        """
        # Drop NaN values
        syn_clean = synthetic.dropna().values.astype(float)
        real_clean = real.dropna().values.astype(float)

        if len(syn_clean) == 0 or len(real_clean) == 0:
            return (1.0, 0.0)  # Maximum divergence if empty

        statistic, p_value = ks_2samp(real_clean, syn_clean)
        return (float(statistic), float(p_value))

    def compute_js_divergence(self, synthetic: pd.Series, real: pd.Series,
                              bins: int = 20) -> float:
        """Compute Jensen-Shannon Divergence between two distributions.

        Works for both categorical and numerical features.
        For numerical features, values are binned first.

        Args:
            synthetic: Synthetic data series
            real: Real data series
            bins: Number of bins for numerical data

        Returns:
            JSD value in [0, 1] (0 = identical distributions)
        """
        syn_clean = synthetic.dropna()
        real_clean = real.dropna()

        if len(syn_clean) == 0 or len(real_clean) == 0:
            return 1.0

        # Check if data is categorical (object/string type)
        if real_clean.dtype == 'object' or str(real_clean.dtype) == 'category':
            return self._js_divergence_categorical(syn_clean, real_clean)
        else:
            return self._js_divergence_numerical(syn_clean, real_clean, bins)

    def _js_divergence_categorical(self, synthetic: pd.Series, real: pd.Series) -> float:
        """JSD for categorical data using value counts as probability distributions."""
        # Get union of all categories
        all_categories = set(real.unique()) | set(synthetic.unique())

        # Build probability distributions with Laplace smoothing
        real_counts = real.value_counts()
        syn_counts = synthetic.value_counts()

        # Small smoothing constant to avoid zero probabilities
        smoothing = 1e-10

        real_probs = np.array([real_counts.get(cat, 0) + smoothing for cat in all_categories])
        syn_probs = np.array([syn_counts.get(cat, 0) + smoothing for cat in all_categories])

        # Normalize to valid probability distributions
        real_probs = real_probs / real_probs.sum()
        syn_probs = syn_probs / syn_probs.sum()

        return float(jensenshannon(real_probs, syn_probs) ** 2)

    def _js_divergence_numerical(self, synthetic: pd.Series, real: pd.Series,
                                 bins: int) -> float:
        """JSD for numerical data by binning into histograms."""
        # Determine common bin edges from the union of both distributions
        combined = np.concatenate([real.values, synthetic.values])
        bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)

        # Build histograms
        real_hist, _ = np.histogram(real.values, bins=bin_edges)
        syn_hist, _ = np.histogram(synthetic.values, bins=bin_edges)

        # Convert to probabilities with smoothing
        smoothing = 1e-10
        real_probs = (real_hist + smoothing).astype(float)
        syn_probs = (syn_hist + smoothing).astype(float)

        real_probs = real_probs / real_probs.sum()
        syn_probs = syn_probs / syn_probs.sum()

        return float(jensenshannon(real_probs, syn_probs) ** 2)

    def compute_correlation_similarity(self, synthetic: pd.DataFrame,
                                       real: pd.DataFrame) -> float:
        """Compute correlation matrix similarity between datasets.

        Uses only numerical columns. Computes Frobenius norm of the difference
        between correlation matrices, normalized to [0, 1].

        Args:
            synthetic: Synthetic DataFrame
            real: Real DataFrame

        Returns:
            Similarity score in [0, 1] (1 = identical correlation structure)
        """
        # Select only numerical columns present in both
        num_cols = []
        for col in real.columns:
            if real[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if col in synthetic.columns:
                    num_cols.append(col)

        if len(num_cols) < 2:
            # Need at least 2 numerical columns for correlation
            return 1.0

        real_corr = real[num_cols].corr().values
        syn_corr = synthetic[num_cols].corr().values

        # Replace NaN with 0 in correlation matrices
        real_corr = np.nan_to_num(real_corr, nan=0.0)
        syn_corr = np.nan_to_num(syn_corr, nan=0.0)

        # Frobenius norm of difference, normalized
        diff_norm = np.linalg.norm(real_corr - syn_corr, 'fro')
        max_norm = np.sqrt(2 * len(num_cols) ** 2)  # Max possible Frobenius norm

        if max_norm == 0:
            return 1.0

        similarity = 1.0 - (diff_norm / max_norm)
        return float(max(0.0, min(1.0, similarity)))

    def compute_all_metrics(self, synthetic: pd.DataFrame,
                            real: pd.DataFrame) -> Dict[str, dict]:
        """Compute all statistical metrics for every column.

        Args:
            synthetic: Synthetic DataFrame
            real: Real DataFrame

        Returns:
            Dictionary with per-column and aggregate metrics
        """
        results = {
            'ks_tests': {},
            'js_divergences': {},
            'correlation_similarity': 0.0,
            'summary': {}
        }

        common_cols = [c for c in real.columns if c in synthetic.columns]

        # Per-column metrics
        ks_stats = []
        jsd_values = []

        for col in common_cols:
            # JSD works for all column types
            jsd = self.compute_js_divergence(synthetic[col], real[col])
            results['js_divergences'][col] = jsd
            jsd_values.append(jsd)

            # KS test only for numerical columns
            if real[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                ks_stat, p_val = self.compute_ks_statistic(synthetic[col], real[col])
                results['ks_tests'][col] = {'statistic': ks_stat, 'p_value': p_val}
                ks_stats.append(ks_stat)

        # Correlation similarity
        results['correlation_similarity'] = self.compute_correlation_similarity(synthetic, real)

        # Summary statistics
        results['summary'] = {
            'mean_ks_statistic': float(np.mean(ks_stats)) if ks_stats else 0.0,
            'mean_jsd': float(np.mean(jsd_values)) if jsd_values else 0.0,
            'correlation_similarity': results['correlation_similarity']
        }

        return results
