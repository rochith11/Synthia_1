"""Statistical analysis for comparing real and synthetic data distributions."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import Dict, Tuple, Optional


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
        syn_clean = synthetic.dropna().values.astype(float)
        real_clean = real.dropna().values.astype(float)

        if len(syn_clean) == 0 or len(real_clean) == 0:
            return (1.0, 0.0)

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

        if real_clean.dtype == 'object' or str(real_clean.dtype) == 'category':
            return self._js_divergence_categorical(syn_clean, real_clean)
        else:
            return self._js_divergence_numerical(syn_clean, real_clean, bins)

    def _js_divergence_categorical(self, synthetic: pd.Series, real: pd.Series) -> float:
        """JSD for categorical data using value counts as probability distributions."""
        all_categories = sorted(set(real.unique()) | set(synthetic.unique()))

        real_counts = real.value_counts()
        syn_counts = synthetic.value_counts()

        n_real = len(real)
        n_syn = len(synthetic)

        # Use count-based probabilities with minimal smoothing to avoid
        # masking true divergence. Smoothing of 1/N keeps zero-count categories
        # from being treated as present while still allowing JSD computation.
        smoothing = 1.0 / (max(n_real, n_syn) * 10)

        real_probs = np.array([real_counts.get(cat, 0) / n_real + smoothing
                               for cat in all_categories])
        syn_probs = np.array([syn_counts.get(cat, 0) / n_syn + smoothing
                              for cat in all_categories])

        real_probs = real_probs / real_probs.sum()
        syn_probs = syn_probs / syn_probs.sum()

        # jensenshannon returns the distance (sqrt of divergence), square it for JSD
        return float(jensenshannon(real_probs, syn_probs) ** 2)

    def _js_divergence_numerical(self, synthetic: pd.Series, real: pd.Series,
                                 bins: int) -> float:
        """JSD for numerical data by binning into histograms."""
        combined = np.concatenate([real.values, synthetic.values])
        bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)

        real_hist, _ = np.histogram(real.values, bins=bin_edges)
        syn_hist, _ = np.histogram(synthetic.values, bins=bin_edges)

        n_real = real_hist.sum()
        n_syn = syn_hist.sum()

        smoothing = 1.0 / (max(n_real, n_syn) * 10)

        real_probs = (real_hist / n_real + smoothing).astype(float)
        syn_probs = (syn_hist / n_syn + smoothing).astype(float)

        real_probs = real_probs / real_probs.sum()
        syn_probs = syn_probs / syn_probs.sum()

        return float(jensenshannon(real_probs, syn_probs) ** 2)

    def compute_correlation_similarity(self, synthetic: pd.DataFrame,
                                       real: pd.DataFrame) -> Optional[float]:
        """Compute correlation matrix similarity between datasets.

        Uses only numerical columns. Returns None when fewer than 2 numerical
        columns exist (cannot compute meaningful correlation).

        Args:
            synthetic: Synthetic DataFrame
            real: Real DataFrame

        Returns:
            Similarity score in [0, 1] or None if not computable
        """
        num_cols = []
        for col in real.columns:
            if real[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if col in synthetic.columns:
                    num_cols.append(col)

        if len(num_cols) < 2:
            return None

        real_corr = real[num_cols].corr().values
        syn_corr = synthetic[num_cols].corr().values

        real_corr = np.nan_to_num(real_corr, nan=0.0)
        syn_corr = np.nan_to_num(syn_corr, nan=0.0)

        diff_norm = np.linalg.norm(real_corr - syn_corr, 'fro')
        max_norm = np.sqrt(2 * len(num_cols) ** 2)

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
            'correlation_similarity': None,
            'summary': {}
        }

        common_cols = [c for c in real.columns if c in synthetic.columns]

        ks_stats = []
        jsd_values = []

        for col in common_cols:
            jsd = self.compute_js_divergence(synthetic[col], real[col])
            results['js_divergences'][col] = jsd
            jsd_values.append(jsd)

            if real[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                ks_stat, p_val = self.compute_ks_statistic(synthetic[col], real[col])
                results['ks_tests'][col] = {'statistic': ks_stat, 'p_value': p_val}
                ks_stats.append(ks_stat)

        corr_sim = self.compute_correlation_similarity(synthetic, real)
        results['correlation_similarity'] = corr_sim

        results['summary'] = {
            'mean_ks_statistic': float(np.mean(ks_stats)) if ks_stats else 0.0,
            'mean_jsd': float(np.mean(jsd_values)) if jsd_values else 0.0,
            'max_jsd': float(np.max(jsd_values)) if jsd_values else 0.0,
            'correlation_similarity': corr_sim
        }

        return results
