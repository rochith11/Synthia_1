"""Data validation orchestrator for comparing synthetic and real data."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.models.reports import ValidationReport


class DataValidator:
    """Orchestrates all validation metrics between synthetic and real data."""

    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self._label_encoders = {}

    def _encode_for_ml(self, df: pd.DataFrame, target_col: str,
                       fit: bool = True) -> tuple:
        """Encode categorical features and separate target for ML tasks.

        Args:
            df: Input DataFrame
            target_col: Name of the target column
            fit: If True, fit new encoders; if False, use existing ones

        Returns:
            Tuple of (X encoded array, y encoded array)
        """
        df_copy = df.copy()

        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self._label_encoders[col] = le
                else:
                    known = set(le.classes_)
                    df_copy[col] = df_copy[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df_copy[col] = le.transform(df_copy[col])

        X = df_copy.drop(columns=[target_col]).values.astype(float)
        y = df_copy[target_col].values

        return X, y

    def evaluate_ml_utility(self, synthetic_train: pd.DataFrame,
                            real_test: pd.DataFrame,
                            target_column: str) -> Dict[str, float]:
        """Evaluate ML utility: train on synthetic, test on real.

        Args:
            synthetic_train: Synthetic data used for training
            real_test: Real data used for testing
            target_column: Column name to predict

        Returns:
            Dict with accuracy, f1_score, auc metrics
        """
        self._label_encoders = {}

        X_train, y_train = self._encode_for_ml(synthetic_train, target_column, fit=True)
        X_test, y_test = self._encode_for_ml(real_test, target_column, fit=False)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        unique_classes = np.unique(np.concatenate([y_train, y_test]))
        n_classes = len(unique_classes)

        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        auc = 0.0
        if n_classes > 1:
            try:
                y_proba = clf.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except (ValueError, IndexError):
                auc = 0.0

        return {
            'accuracy': float(acc),
            'f1_score': float(f1),
            'auc': float(auc)
        }

    def cross_test(self, synthetic: pd.DataFrame, real: pd.DataFrame,
                   target_column: str) -> Dict[str, Dict[str, float]]:
        """Bidirectional cross-testing.

        Direction 1: Train on synthetic, test on real
        Direction 2: Train on real, test on synthetic

        Args:
            synthetic: Synthetic DataFrame
            real: Real DataFrame
            target_column: Column to predict

        Returns:
            Dict with metrics for both directions
        """
        syn_to_real = self.evaluate_ml_utility(synthetic, real, target_column)

        self._label_encoders = {}
        real_to_syn = self.evaluate_ml_utility(real, synthetic, target_column)

        return {
            'synthetic_to_real': syn_to_real,
            'real_to_synthetic': real_to_syn
        }

    def analyze_rare_events(self, synthetic: pd.DataFrame, real: pd.DataFrame,
                            threshold: float = 0.05) -> Dict[str, Any]:
        """Analyze preservation of rare event frequencies.

        Args:
            synthetic: Synthetic DataFrame
            real: Real DataFrame
            threshold: Frequency threshold below which a class is "rare"

        Returns:
            Dict with rare class analysis per categorical column
        """
        results = {}
        categorical_cols = real.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col not in synthetic.columns:
                continue

            real_freq = real[col].value_counts(normalize=True)
            syn_freq = synthetic[col].value_counts(normalize=True)

            rare_classes = real_freq[real_freq < threshold].index.tolist()

            col_result = {
                'rare_classes': [],
                'total_rare_classes': len(rare_classes)
            }

            for cls in rare_classes:
                real_pct = float(real_freq.get(cls, 0))
                syn_pct = float(syn_freq.get(cls, 0))
                diff = abs(real_pct - syn_pct)

                col_result['rare_classes'].append({
                    'class': cls,
                    'real_frequency': real_pct,
                    'synthetic_frequency': syn_pct,
                    'absolute_difference': diff
                })

            if rare_classes:
                results[col] = col_result

        return results

    def _detect_mode_collapse(self, synthetic: pd.DataFrame,
                              dominance_threshold: float = 0.90) -> Dict[str, Any]:
        """Detect mode collapse in categorical columns.

        A column is collapsed if a single value accounts for more than
        dominance_threshold of all rows.

        Args:
            synthetic: Synthetic DataFrame
            dominance_threshold: Fraction above which a single class is "dominant"

        Returns:
            Dict with collapse info per column. Empty dict means no collapse.
        """
        collapsed = {}
        categorical_cols = synthetic.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            freq = synthetic[col].value_counts(normalize=True)
            if len(freq) == 0:
                continue
            top_freq = freq.iloc[0]
            if top_freq >= dominance_threshold:
                collapsed[col] = {
                    'dominant_class': freq.index[0],
                    'dominant_frequency': float(top_freq),
                    'n_unique': len(freq)
                }

        return collapsed

    def validate(self, synthetic: pd.DataFrame, real: pd.DataFrame,
                 target_column: str = 'disease',
                 dataset_id: str = 'unknown') -> ValidationReport:
        """Run complete validation pipeline.

        Args:
            synthetic: Synthetic DataFrame
            real: Real DataFrame
            target_column: Column for ML utility evaluation
            dataset_id: ID for the report

        Returns:
            ValidationReport with all metrics

        Raises:
            ValueError: If synthetic or real data contains NaN values
        """
        # ---- NaN guard ----
        syn_nan = synthetic.isnull().any().any()
        real_nan = real.isnull().any().any()
        if syn_nan or real_nan:
            parts = []
            if syn_nan:
                cols = synthetic.columns[synthetic.isnull().any()].tolist()
                parts.append(f"Synthetic dataset contains NaN values in columns: {cols}")
            if real_nan:
                cols = real.columns[real.isnull().any()].tolist()
                parts.append(f"Real dataset contains NaN values in columns: {cols}")
            raise ValueError("; ".join(parts))

        print("[i] Running statistical validation...")

        # 1. Statistical metrics
        stat_metrics = self.statistical_analyzer.compute_all_metrics(synthetic, real)
        mean_jsd = stat_metrics['summary']['mean_jsd']
        max_jsd = stat_metrics['summary']['max_jsd']
        mean_ks = stat_metrics['summary']['mean_ks_statistic']
        corr_sim = stat_metrics['summary']['correlation_similarity']

        print(f"    Mean KS statistic: {mean_ks:.4f}")
        print(f"    Mean JSD: {mean_jsd:.4f}")
        print(f"    Max JSD:  {max_jsd:.4f}")
        if corr_sim is not None:
            print(f"    Correlation similarity: {corr_sim:.4f}")
        else:
            print(f"    Correlation similarity: N/A (< 2 numerical columns)")

        # 2. ML utility (bidirectional)
        print("[i] Running ML utility evaluation...")
        cross_results = self.cross_test(synthetic, real, target_column)

        syn_to_real = cross_results['synthetic_to_real']
        real_to_syn = cross_results['real_to_synthetic']

        print(f"    Synthetic->Real: Acc={syn_to_real['accuracy']:.4f}, "
              f"F1={syn_to_real['f1_score']:.4f}, AUC={syn_to_real['auc']:.4f}")
        print(f"    Real->Synthetic: Acc={real_to_syn['accuracy']:.4f}, "
              f"F1={real_to_syn['f1_score']:.4f}, AUC={real_to_syn['auc']:.4f}")

        # 3. Rare event analysis
        print("[i] Analyzing rare events...")
        rare_events = self.analyze_rare_events(synthetic, real)
        total_rare = sum(r['total_rare_classes'] for r in rare_events.values())
        print(f"    Rare classes found: {total_rare}")

        # 4. Mode collapse detection
        collapse_info = self._detect_mode_collapse(synthetic)
        if collapse_info:
            print(f"[!] Mode collapse detected in {len(collapse_info)} column(s):")
            for col, info in collapse_info.items():
                print(f"    {col}: '{info['dominant_class']}' at "
                      f"{info['dominant_frequency']:.1%} ({info['n_unique']} unique)")

        # 5. Compute overall quality score
        #
        # Scoring philosophy: distributional realism dominates.
        #
        # Base formula:
        #   quality = 0.5 * (1 - max_jsd)      divergence (worst column)
        #           + 0.3 * (1 - mean_ks)       numerical distribution
        #           + 0.2 * utility_score        ML utility (secondary)
        #
        # If divergence is high (max_jsd > 0.25 or mean_ks > 0.4),
        # utility weight drops to 0.05 so predictive artifacts cannot
        # inflate the score.
        #
        # Mode collapse penalty is multiplicative:
        #   1 collapsed column  → quality *= 0.5
        #   2+ collapsed columns → quality *= 0.3

        jsd_component = 1.0 - max_jsd
        ks_component = 1.0 - mean_ks
        utility_score = syn_to_real['accuracy']

        # Determine utility weight — cap under high divergence
        # When capped, total weight < 1.0, which naturally reduces the score.
        high_divergence = (max_jsd > 0.25) or (mean_ks > 0.4)
        if high_divergence:
            w_jsd = 0.50
            w_ks = 0.30
            w_util = 0.05
        else:
            w_jsd = 0.50
            w_ks = 0.30
            w_util = 0.20

        overall = float(
            w_jsd * jsd_component
            + w_ks * ks_component
            + w_util * utility_score
        )

        # Under high divergence, scale by severity of worst metric.
        # Worse divergence → stronger reduction.
        if high_divergence:
            severity = max(max_jsd, mean_ks)
            divergence_multiplier = 1.0 - 0.5 * severity
            overall *= divergence_multiplier

        # Multiplicative collapse penalty
        n_collapsed = len(collapse_info)
        if n_collapsed >= 2:
            print(f"[!] Collapse penalty: x0.30 ({n_collapsed} columns collapsed)")
            overall *= 0.3
        elif n_collapsed == 1:
            print(f"[!] Collapse penalty: x0.50 (1 column collapsed)")
            overall *= 0.5

        overall = min(1.0, max(0.0, overall))

        # Transparent score breakdown
        print(f"    Score components:")
        print(f"      1 - max_jsd:    {jsd_component:.4f}  (weight {w_jsd:.2f})")
        print(f"      1 - mean_ks:    {ks_component:.4f}  (weight {w_ks:.2f})")
        print(f"      utility_score:  {utility_score:.4f}  (weight {w_util:.2f})")
        if high_divergence:
            print(f"      [!] High divergence detected — utility capped at {w_util:.2f}")
            print(f"      [!] Severity multiplier: x{divergence_multiplier:.2f} "
                  f"(worst metric: {severity:.4f})")
        if n_collapsed > 0:
            mult = 0.3 if n_collapsed >= 2 else 0.5
            print(f"      [!] Collapse multiplier: x{mult:.1f}")

        print(f"[+] Overall quality score: {overall:.4f}")

        return ValidationReport(
            dataset_id=dataset_id,
            statistical_metrics={
                'ks_tests': stat_metrics['ks_tests'],
                'js_divergences': stat_metrics['js_divergences'],
                'correlation_similarity': corr_sim,
                'collapse_info': collapse_info,
                'summary': stat_metrics['summary']
            },
            utility_metrics={
                'cross_test': cross_results,
                'summary': {
                    'synthetic_to_real_accuracy': syn_to_real['accuracy'],
                    'real_to_synthetic_accuracy': real_to_syn['accuracy']
                }
            },
            rare_event_analysis=rare_events,
            overall_quality_score=overall
        )
