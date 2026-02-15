"""Privacy analysis for synthetic data — nearest-neighbor distance and re-identification risk."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import cdist

from src.models.reports import PrivacyReport


class PrivacyAnalyzer:
    """Compute privacy metrics between synthetic and real data.

    Uses nearest-neighbor distances (NND) to measure how close each synthetic
    record is to the closest real record.  A short distance implies the
    generative model may have memorised (or nearly copied) a real record.

    Risk scoring is *percentile-based*: the user provides a percentile
    (e.g. 90) and the method reports what fraction of synthetic records fall
    within the top (100 - percentile)% of the distance distribution.
    """

    def __init__(self):
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Encode categoricals and scale all features to [0, 1].

        Args:
            df: Input DataFrame (all columns used).
            fit: If True, fit new encoders/scaler; if False, reuse existing.

        Returns:
            2-D numpy array with all values in [0, 1].
        """
        df_encoded = df.copy()

        categorical_cols = df_encoded.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self._label_encoders[col] = le
                else:
                    known = set(le.classes_)
                    df_encoded[col] = df_encoded[col].astype(str).apply(
                        lambda x, k=known, le_=le: x if x in k else le_.classes_[0]
                    )
                    df_encoded[col] = le.transform(df_encoded[col])

        values = df_encoded.values.astype(float)

        if fit:
            self._scaler = MinMaxScaler()
            values = self._scaler.fit_transform(values)
        else:
            if self._scaler is not None:
                values = self._scaler.transform(values)
            else:
                self._scaler = MinMaxScaler()
                values = self._scaler.fit_transform(values)

        return values

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def compute_nearest_neighbor_distances(
        self, synthetic: pd.DataFrame, real: pd.DataFrame
    ) -> np.ndarray:
        """Compute the nearest-neighbor distance for every synthetic record.

        For each synthetic row, find the Euclidean distance to the closest
        real row (after normalisation).  The computation is fully vectorised
        via ``scipy.spatial.distance.cdist``.

        Args:
            synthetic: Synthetic DataFrame.
            real: Real DataFrame.

        Returns:
            1-D array of length ``len(synthetic)`` with the minimum distance
            from each synthetic record to any real record.
        """
        self._label_encoders = {}
        self._scaler = None

        common_cols = [c for c in real.columns if c in synthetic.columns]
        real_sub = real[common_cols]
        syn_sub = synthetic[common_cols]

        real_norm = self.normalize_data(real_sub, fit=True)
        syn_norm = self.normalize_data(syn_sub, fit=False)

        # cdist returns (n_syn, n_real) matrix of pairwise distances
        dist_matrix = cdist(syn_norm, real_norm, metric='euclidean')

        # Nearest-neighbor distance for each synthetic row
        nnd = dist_matrix.min(axis=1)
        return nnd

    def compute_reidentification_risk(
        self,
        synthetic: pd.DataFrame,
        real: pd.DataFrame,
        percentile: int = 90,
    ) -> Dict[str, Any]:
        """Assess re-identification risk using a percentile-based threshold.

        1. Compute NND for every synthetic record.
        2. Derive a distance threshold at the given *percentile* of the NND
           distribution (e.g. percentile=90 → threshold = 90th-percentile
           distance).
        3. Records whose NND falls *below* the threshold are considered
           "high-risk" (i.e. they are closer to a real record than most).

        Args:
            synthetic: Synthetic DataFrame.
            real: Real DataFrame.
            percentile: Percentile (0-100) that defines the risk threshold.
                        Default 90 means the bottom 10% closest records are
                        flagged as high-risk.

        Returns:
            Dict with threshold, high_risk_count, high_risk_percentage,
            and the NND array.
        """
        nnd = self.compute_nearest_neighbor_distances(synthetic, real)

        threshold = float(np.percentile(nnd, 100 - percentile))

        high_risk_mask = nnd <= threshold
        high_risk_count = int(high_risk_mask.sum())
        high_risk_pct = high_risk_count / len(nnd) if len(nnd) > 0 else 0.0

        return {
            'nnd': nnd,
            'threshold': threshold,
            'high_risk_count': high_risk_count,
            'high_risk_percentage': float(high_risk_pct),
        }

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze_privacy(
        self,
        synthetic: pd.DataFrame,
        real: pd.DataFrame,
        percentile: int = 90,
        dataset_id: str = 'unknown',
    ) -> PrivacyReport:
        """Run the complete privacy analysis pipeline.

        Args:
            synthetic: Synthetic DataFrame.
            real: Real DataFrame.
            percentile: Risk percentile (default 90 → top 10% flagged).
            dataset_id: Identifier for the report.

        Returns:
            PrivacyReport with NND statistics, risk info, and a composite
            privacy score.
        """
        print("[i] Computing nearest-neighbor distances...")

        risk_info = self.compute_reidentification_risk(
            synthetic, real, percentile=percentile
        )
        nnd = risk_info['nnd']

        nnd_stats = {
            'mean': float(np.mean(nnd)),
            'median': float(np.median(nnd)),
            'min': float(np.min(nnd)),
            'max': float(np.max(nnd)),
            'std': float(np.std(nnd)),
        }

        print(f"    Mean NND:   {nnd_stats['mean']:.4f}")
        print(f"    Median NND: {nnd_stats['median']:.4f}")
        print(f"    Min NND:    {nnd_stats['min']:.4f}")
        print(f"    Max NND:    {nnd_stats['max']:.4f}")
        print(f"    Std NND:    {nnd_stats['std']:.4f}")

        print(f"[i] Re-identification risk (percentile={percentile}):")
        print(f"    Threshold distance: {risk_info['threshold']:.4f}")
        print(f"    High-risk records:  {risk_info['high_risk_count']} / {len(nnd)}")
        print(f"    High-risk %:        {risk_info['high_risk_percentage']:.2%}")

        # Privacy score: higher mean NND → better privacy.
        # Clamp to [0, 1].  A mean NND of 0 → score 0; NND ≥ 1 → score 1.
        # We also penalise by the high-risk fraction.
        base_score = min(1.0, nnd_stats['mean'])
        risk_penalty = risk_info['high_risk_percentage']
        privacy_score = max(0.0, base_score - risk_penalty)
        privacy_score = min(1.0, max(0.0, privacy_score))

        print(f"[+] Privacy score: {privacy_score:.4f}")

        return PrivacyReport(
            dataset_id=dataset_id,
            nearest_neighbor_distances=nnd_stats,
            reidentification_risk={
                'percentile': percentile,
                'threshold': risk_info['threshold'],
                'high_risk_count': risk_info['high_risk_count'],
                'high_risk_percentage': risk_info['high_risk_percentage'],
            },
            privacy_score=privacy_score,
        )
