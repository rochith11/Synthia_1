"""Bias detection for synthetic data — distribution shifts and amplification."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.spatial.distance import jensenshannon


class BiasDetector:
    """Detect distribution bias, rare-class imbalance, and bias amplification
    between synthetic and real data."""

    # ------------------------------------------------------------------
    # Feature-level distribution divergence
    # ------------------------------------------------------------------

    def analyze_feature_distributions(
        self, synthetic: pd.DataFrame, real: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute JSD-based bias score for every common column.

        Args:
            synthetic: Synthetic DataFrame.
            real: Real DataFrame.

        Returns:
            Dict mapping column name → {jsd, status} where status is
            'low', 'moderate', or 'high'.
        """
        results: Dict[str, Any] = {}
        common_cols = [c for c in real.columns if c in synthetic.columns]

        for col in common_cols:
            jsd = self._column_jsd(synthetic[col], real[col])
            if jsd < 0.05:
                status = "low"
            elif jsd < 0.15:
                status = "moderate"
            else:
                status = "high"
            results[col] = {"jsd": float(jsd), "status": status}

        return results

    # ------------------------------------------------------------------
    # Rare class imbalance
    # ------------------------------------------------------------------

    def detect_rare_class_imbalance(
        self,
        synthetic: pd.DataFrame,
        real: pd.DataFrame,
        column: str,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Identify rare classes (below *threshold*) and compare frequencies.

        Args:
            synthetic: Synthetic DataFrame.
            real: Real DataFrame.
            column: Categorical column to inspect.
            threshold: Frequency below which a class is 'rare'.

        Returns:
            Dict with rare class details and summary stats.
        """
        if column not in real.columns or column not in synthetic.columns:
            return {}

        real_freq = real[column].value_counts(normalize=True)
        syn_freq = synthetic[column].value_counts(normalize=True)

        rare_classes = real_freq[real_freq < threshold].index.tolist()

        details: List[Dict[str, Any]] = []
        for cls in rare_classes:
            r = float(real_freq.get(cls, 0))
            s = float(syn_freq.get(cls, 0))
            details.append({
                "class": cls,
                "real_frequency": r,
                "synthetic_frequency": s,
                "absolute_difference": abs(r - s),
                "missing_in_synthetic": s == 0.0,
            })

        return {
            "column": column,
            "threshold": threshold,
            "total_rare_classes": len(rare_classes),
            "rare_classes": details,
        }

    # ------------------------------------------------------------------
    # Bias amplification
    # ------------------------------------------------------------------

    def compute_bias_amplification(
        self,
        synthetic: pd.DataFrame,
        real: pd.DataFrame,
        amplification_threshold: float = 0.20,
    ) -> Dict[str, Any]:
        """Detect bias amplification across all categorical columns.

        A column is *amplified* if the synthetic imbalance ratio exceeds
        the real imbalance ratio by more than ``amplification_threshold``.

        Imbalance ratio = max_frequency / min_frequency for classes present
        in *both* datasets.

        Args:
            synthetic: Synthetic DataFrame.
            real: Real DataFrame.
            amplification_threshold: Fractional increase that triggers a flag
                                     (default 0.20 = 20%).

        Returns:
            Dict with per-column amplification info and list of flagged columns.
        """
        results: Dict[str, Any] = {"columns": {}, "flagged": []}
        cat_cols = real.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            if col not in synthetic.columns:
                continue

            real_freq = real[col].value_counts(normalize=True)
            syn_freq = synthetic[col].value_counts(normalize=True)

            # Use classes present in both to compute imbalance ratio
            common_classes = sorted(set(real_freq.index) & set(syn_freq.index))
            if len(common_classes) < 2:
                continue

            real_vals = np.array([real_freq[c] for c in common_classes])
            syn_vals = np.array([syn_freq[c] for c in common_classes])

            real_ratio = float(real_vals.max() / real_vals.min()) if real_vals.min() > 0 else 0.0
            syn_ratio = float(syn_vals.max() / syn_vals.min()) if syn_vals.min() > 0 else 0.0

            amplified = syn_ratio > real_ratio * (1 + amplification_threshold)

            results["columns"][col] = {
                "real_imbalance_ratio": real_ratio,
                "synthetic_imbalance_ratio": syn_ratio,
                "amplified": amplified,
            }

            if amplified:
                results["flagged"].append(col)

        return results

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze_bias(
        self,
        synthetic: pd.DataFrame,
        real: pd.DataFrame,
        amplification_threshold: float = 0.20,
        rare_class_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Run the complete bias analysis pipeline.

        Args:
            synthetic: Synthetic DataFrame.
            real: Real DataFrame.
            amplification_threshold: Amplification flag threshold.
            rare_class_threshold: Rare-class frequency threshold.

        Returns:
            Dict with feature distributions, rare-class imbalances,
            amplification results, and recommendations.
        """
        print("[i] Analyzing feature distribution bias...")
        feat_dist = self.analyze_feature_distributions(synthetic, real)

        high_bias_cols = [c for c, v in feat_dist.items() if v["status"] == "high"]
        print(f"    High-bias columns: {len(high_bias_cols)}")
        for c in high_bias_cols:
            print(f"      {c}: JSD={feat_dist[c]['jsd']:.4f}")

        print("[i] Detecting rare class imbalances...")
        cat_cols = real.select_dtypes(include=["object", "category"]).columns
        rare_imbalances: Dict[str, Any] = {}
        for col in cat_cols:
            if col in synthetic.columns:
                result = self.detect_rare_class_imbalance(
                    synthetic, real, col, threshold=rare_class_threshold
                )
                if result.get("total_rare_classes", 0) > 0:
                    rare_imbalances[col] = result

        total_rare = sum(r["total_rare_classes"] for r in rare_imbalances.values())
        print(f"    Total rare classes found: {total_rare}")

        print("[i] Checking bias amplification...")
        amplification = self.compute_bias_amplification(
            synthetic, real, amplification_threshold
        )
        flagged = amplification.get("flagged", [])
        print(f"    Amplified columns: {len(flagged)}")
        for col in flagged:
            info = amplification["columns"][col]
            print(
                f"      {col}: real_ratio={info['real_imbalance_ratio']:.2f}, "
                f"syn_ratio={info['synthetic_imbalance_ratio']:.2f}"
            )

        # Build recommendations
        recommendations: List[str] = []
        if high_bias_cols:
            recommendations.append(
                f"High distribution bias in {len(high_bias_cols)} column(s): "
                f"{', '.join(high_bias_cols)}. Consider retraining with more epochs."
            )
        if flagged:
            recommendations.append(
                f"Bias amplified in {len(flagged)} column(s): "
                f"{', '.join(flagged)}. Consider adjusting model or post-processing."
            )
        missing_rare = [
            col for col, info in rare_imbalances.items()
            if any(c["missing_in_synthetic"] for c in info.get("rare_classes", []))
        ]
        if missing_rare:
            recommendations.append(
                f"Rare classes missing in synthetic data for: "
                f"{', '.join(missing_rare)}. Consider oversampling training data."
            )

        if not recommendations:
            recommendations.append("No significant bias issues detected.")

        print(f"[+] Recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")

        return {
            "feature_distributions": feat_dist,
            "rare_class_imbalances": rare_imbalances,
            "amplification": amplification,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _column_jsd(self, synthetic: pd.Series, real: pd.Series) -> float:
        """JSD between two series (categorical or numerical)."""
        syn = synthetic.dropna()
        real_ = real.dropna()
        if len(syn) == 0 or len(real_) == 0:
            return 1.0

        if real_.dtype == "object" or str(real_.dtype) == "category":
            all_cats = sorted(set(real_.unique()) | set(syn.unique()))
            rc = real_.value_counts()
            sc = syn.value_counts()
            n_r, n_s = len(real_), len(syn)
            sm = 1.0 / (max(n_r, n_s) * 10)
            rp = np.array([rc.get(c, 0) / n_r + sm for c in all_cats])
            sp = np.array([sc.get(c, 0) / n_s + sm for c in all_cats])
            rp /= rp.sum()
            sp /= sp.sum()
            return float(jensenshannon(rp, sp) ** 2)

        combined = np.concatenate([real_.values, syn.values])
        edges = np.linspace(combined.min(), combined.max(), 21)
        rh, _ = np.histogram(real_.values, bins=edges)
        sh, _ = np.histogram(syn.values, bins=edges)
        n_r, n_s = rh.sum(), sh.sum()
        sm = 1.0 / (max(n_r, n_s) * 10)
        rp = (rh / n_r + sm).astype(float)
        sp = (sh / n_s + sm).astype(float)
        rp /= rp.sum()
        sp /= sp.sum()
        return float(jensenshannon(rp, sp) ** 2)
