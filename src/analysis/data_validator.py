"""Data validation orchestrator for comparing synthetic and real data."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
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
                    # Handle unseen labels by mapping them to a fallback
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

        # Train RandomForest
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)

        # Use weighted average for multiclass
        unique_classes = np.unique(np.concatenate([y_train, y_test]))
        n_classes = len(unique_classes)

        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # AUC: only if we have probability predictions and >1 class
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
        # Direction 1: synthetic -> real
        syn_to_real = self.evaluate_ml_utility(synthetic, real, target_column)

        # Direction 2: real -> synthetic
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

            # Identify rare classes in real data
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
        """
        print("[i] Running statistical validation...")

        # 1. Statistical metrics
        stat_metrics = self.statistical_analyzer.compute_all_metrics(synthetic, real)
        print(f"    Mean KS statistic: {stat_metrics['summary']['mean_ks_statistic']:.4f}")
        print(f"    Mean JSD: {stat_metrics['summary']['mean_jsd']:.4f}")
        print(f"    Correlation similarity: {stat_metrics['summary']['correlation_similarity']:.4f}")

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

        # 4. Compute overall quality score (simple average)
        ks_score = 1.0 - stat_metrics['summary']['mean_ks_statistic']
        jsd_score = 1.0 - stat_metrics['summary']['mean_jsd']
        corr_score = stat_metrics['summary']['correlation_similarity']
        utility_score = syn_to_real['accuracy']

        overall = float(np.mean([ks_score, jsd_score, corr_score, utility_score]))

        print(f"[+] Overall quality score: {overall:.4f}")

        return ValidationReport(
            dataset_id=dataset_id,
            statistical_metrics={
                'ks_tests': stat_metrics['ks_tests'],
                'js_divergences': stat_metrics['js_divergences'],
                'correlation_similarity': stat_metrics['correlation_similarity'],
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
