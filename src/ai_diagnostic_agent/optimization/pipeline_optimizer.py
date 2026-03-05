"""Pipeline Optimizer - generates prioritized improvement recommendations."""

import logging
import uuid
from typing import List, Dict, Any, Optional

from src.ai_diagnostic_agent.models import (
    Recommendation, RecommendationCategory, RootCause, DataProfile,
    MetricAnalysis, QualityLevel, ColumnDiagnosis, DiversityReport,
)
from src.ai_diagnostic_agent.config import DATASET_SIZE_THRESHOLDS


class PipelineOptimizer:
    """Generates prioritized improvement recommendations based on diagnostic results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_recommendations(
        self,
        metric_analysis: MetricAnalysis,
        root_causes: List[RootCause] = None,
        data_profile: DataProfile = None,
        column_diagnostics: List[ColumnDiagnosis] = None,
        diversity_report: DiversityReport = None,
        current_config: dict = None,
    ) -> List[Recommendation]:
        """Generate prioritized improvement recommendations from all diagnostics."""
        recommendations = []

        # Preprocessing recommendations
        if data_profile or column_diagnostics:
            recommendations.extend(
                self.recommend_preprocessing(data_profile, column_diagnostics)
            )

        # Model strategy recommendations
        dataset_size = data_profile.n_rows if data_profile else None
        current_model = (current_config or {}).get('model_type')
        recommendations.extend(
            self.recommend_model_strategy(dataset_size, current_model, metric_analysis)
        )

        # Training optimization recommendations
        recommendations.extend(
            self.recommend_training_optimization(metric_analysis, current_config)
        )

        # Data strategy recommendations
        recommendations.extend(
            self.recommend_data_strategy(data_profile, metric_analysis)
        )

        # Post-processing recommendations
        recommendations.extend(
            self.recommend_post_processing(diversity_report, metric_analysis)
        )

        # Filter out recommendations that risk privacy degradation
        safe_recommendations = []
        for rec in recommendations:
            if self._check_privacy_impact(rec, metric_analysis):
                safe_recommendations.append(rec)
            else:
                self.logger.info(
                    "Filtered recommendation '%s' due to privacy risk", rec.title
                )

        return self.prioritize_recommendations(safe_recommendations)

    def recommend_preprocessing(
        self,
        data_profile: DataProfile = None,
        column_diagnostics: List[ColumnDiagnosis] = None,
    ) -> List[Recommendation]:
        """Recommend data preprocessing improvements."""
        recs = []

        if data_profile:
            # Log-transform skewed features
            for col_name, stats in data_profile.statistics.items():
                if (
                    stats.skewness is not None
                    and abs(stats.skewness) > 1.0
                    and data_profile.column_types.get(col_name) == 'continuous'
                ):
                    recs.append(Recommendation(
                        category=RecommendationCategory.PREPROCESSING.value,
                        priority=2,
                        title=f"Log-transform {col_name}",
                        description=f"Apply log transformation to '{col_name}' (skewness: {stats.skewness:.2f})",
                        expected_improvements={'mean_jsd': 'Reduce by ~0.01', 'quality_score': 'Improve by ~0.03'},
                        implementation={'preprocessing': {col_name: 'log_transform'}},
                        rationale=f"Column '{col_name}' has high skewness ({stats.skewness:.2f}). Log transformation will normalize the distribution for better model learning.",
                    ))

            # Handle missing values
            for col_name, pct in data_profile.missing_values.items():
                if pct > 0:
                    strategy = 'median' if data_profile.column_types.get(col_name) == 'continuous' else 'mode'
                    recs.append(Recommendation(
                        category=RecommendationCategory.PREPROCESSING.value,
                        priority=2,
                        title=f"Impute missing values in {col_name}",
                        description=f"Apply {strategy} imputation (missing: {pct:.1f}%)",
                        expected_improvements={'quality_score': 'Improve by ~0.02'},
                        implementation={'preprocessing': {col_name: f'impute_{strategy}'}},
                        rationale=f"Missing values in '{col_name}' ({pct:.1f}%) may degrade generation quality.",
                    ))

            # Bucket rare values
            for col_name, rare_vals in data_profile.rare_values.items():
                if len(rare_vals) > 3:
                    recs.append(Recommendation(
                        category=RecommendationCategory.PREPROCESSING.value,
                        priority=3,
                        title=f"Bucket rare values in {col_name}",
                        description=f"Group {len(rare_vals)} rare values into 'Other' category",
                        expected_improvements={'mean_jsd': 'Reduce by ~0.005'},
                        implementation={'preprocessing': {col_name: 'bucket_rare'}},
                        rationale=f"Column '{col_name}' has {len(rare_vals)} rare values that may cause encoding issues.",
                    ))

        # High JSD columns from diagnostics
        if column_diagnostics:
            critical_cols = [cd for cd in column_diagnostics if cd.severity == QualityLevel.CRITICAL]
            if critical_cols:
                col_names = [cd.column_name for cd in critical_cols[:3]]
                recs.append(Recommendation(
                    category=RecommendationCategory.PREPROCESSING.value,
                    priority=1,
                    title="Fix high-divergence columns",
                    description=f"Columns with critical JSD: {', '.join(col_names)}. Review encoding and preprocessing.",
                    expected_improvements={'mean_jsd': 'Reduce by ~0.02', 'quality_score': 'Improve by ~0.05'},
                    implementation={'preprocessing': {cn: 'review_encoding' for cn in col_names}},
                    rationale="These columns have very high distribution divergence. Fixing encoding or adding transformations should significantly improve quality.",
                ))

        return recs

    def recommend_model_strategy(
        self,
        dataset_size: int = None,
        current_model: str = None,
        metric_analysis: MetricAnalysis = None,
    ) -> List[Recommendation]:
        """Recommend model selection strategy based on data characteristics."""
        recs = []

        if dataset_size is not None:
            if dataset_size < DATASET_SIZE_THRESHOLDS.get('medium', 5000):
                if current_model != 'TVAE':
                    recs.append(Recommendation(
                        category=RecommendationCategory.MODEL.value,
                        priority=2,
                        title="Switch to TVAE",
                        description=f"Dataset has {dataset_size} rows. TVAE performs better on small datasets.",
                        expected_improvements={'quality_score': 'Improve by ~0.10'},
                        implementation={'model_type': 'TVAE'},
                        rationale="TVAE uses variational autoencoding which is more stable on small datasets than GAN-based approaches.",
                    ))
            elif dataset_size > DATASET_SIZE_THRESHOLDS.get('large', 10000):
                if current_model != 'CTGAN':
                    recs.append(Recommendation(
                        category=RecommendationCategory.MODEL.value,
                        priority=3,
                        title="Use CTGAN for large dataset",
                        description=f"Dataset has {dataset_size} rows. CTGAN excels on larger datasets.",
                        expected_improvements={'quality_score': 'Improve by ~0.05'},
                        implementation={'model_type': 'CTGAN'},
                        rationale="CTGAN can leverage larger datasets more effectively through adversarial training.",
                    ))

        # If quality is poor, suggest trying a different model
        if metric_analysis:
            quality = metric_analysis.quality_metrics.get('quality_score')
            if quality and quality[1] in (QualityLevel.CRITICAL, QualityLevel.WARNING):
                if current_model == 'CTGAN':
                    recs.append(Recommendation(
                        category=RecommendationCategory.MODEL.value,
                        priority=2,
                        title="Try CopulaGAN",
                        description="Quality is poor with CTGAN. CopulaGAN may better capture structured distributions.",
                        expected_improvements={'quality_score': 'Improve by ~0.05-0.10'},
                        implementation={'model_type': 'CopulaGAN'},
                        rationale="CopulaGAN models feature dependencies using copulas, which can help with structured data.",
                    ))

        return recs

    def recommend_training_optimization(
        self,
        metric_analysis: MetricAnalysis = None,
        current_config: dict = None,
    ) -> List[Recommendation]:
        """Recommend training parameter adjustments."""
        recs = []
        config = current_config or {}

        if metric_analysis:
            quality = metric_analysis.quality_metrics.get('quality_score')
            accuracy = metric_analysis.utility_metrics.get('ml_accuracy')

            # Underfitting: increase epochs
            if quality and quality[1] in (QualityLevel.CRITICAL, QualityLevel.WARNING):
                current_epochs = config.get('epochs', 300)
                new_epochs = min(current_epochs + 200, 800)
                if new_epochs > current_epochs:
                    recs.append(Recommendation(
                        category=RecommendationCategory.TRAINING.value,
                        priority=2,
                        title="Increase training epochs",
                        description=f"Increase epochs from {current_epochs} to {new_epochs} to improve convergence.",
                        expected_improvements={'quality_score': 'Improve by ~0.05', 'mean_ks': 'Reduce by ~0.03'},
                        implementation={'epochs': new_epochs},
                        rationale="Quality metrics are below target. More training epochs allow the model to better learn data distributions.",
                    ))

            # Adjust batch size
            current_batch = config.get('batch_size', 500)
            if quality and quality[1] == QualityLevel.CRITICAL and current_batch > 200:
                new_batch = max(current_batch // 2, 100)
                recs.append(Recommendation(
                    category=RecommendationCategory.TRAINING.value,
                    priority=3,
                    title="Reduce batch size",
                    description=f"Reduce batch size from {current_batch} to {new_batch} for finer gradient updates.",
                    expected_improvements={'quality_score': 'Improve by ~0.03'},
                    implementation={'batch_size': new_batch},
                    rationale="Smaller batch sizes can help the model learn more nuanced patterns in the data.",
                ))

            # Increase embedding dimension if utility is poor
            if accuracy and accuracy[1] in (QualityLevel.CRITICAL, QualityLevel.WARNING):
                current_emb = config.get('embedding_dim', 128)
                new_emb = min(current_emb + 64, 256)
                if new_emb > current_emb:
                    recs.append(Recommendation(
                        category=RecommendationCategory.TRAINING.value,
                        priority=3,
                        title="Increase embedding dimension",
                        description=f"Increase embedding from {current_emb} to {new_emb} for richer representations.",
                        expected_improvements={'ml_accuracy': 'Improve by ~0.05', 'f1_score': 'Improve by ~0.03'},
                        implementation={'embedding_dim': new_emb},
                        rationale="Larger embeddings capture more complex feature relationships, improving ML utility.",
                    ))

        return recs

    def recommend_data_strategy(
        self,
        data_profile: DataProfile = None,
        metric_analysis: MetricAnalysis = None,
    ) -> List[Recommendation]:
        """Recommend data strategy improvements."""
        recs = []

        if data_profile:
            if data_profile.n_rows < DATASET_SIZE_THRESHOLDS.get('medium', 5000):
                recs.append(Recommendation(
                    category=RecommendationCategory.DATA_STRATEGY.value,
                    priority=4,
                    title="Increase training dataset size",
                    description=f"Current dataset has {data_profile.n_rows} rows. Target at least 5,000 rows.",
                    expected_improvements={'quality_score': 'Improve by ~0.10-0.15'},
                    implementation={},
                    rationale="Small datasets limit model learning capacity. More data improves generalization.",
                ))

            # Class imbalance
            if data_profile.class_imbalances:
                for col, info in data_profile.class_imbalances.items():
                    recs.append(Recommendation(
                        category=RecommendationCategory.DATA_STRATEGY.value,
                        priority=4,
                        title=f"Address class imbalance in {col}",
                        description=f"Column '{col}' has class imbalance. Consider oversampling minority classes.",
                        expected_improvements={'mean_jsd': 'Reduce by ~0.01'},
                        implementation={'preprocessing': {col: 'balance_classes'}},
                        rationale="Class imbalance causes the model to underrepresent minority categories.",
                    ))

        return recs

    def recommend_post_processing(
        self,
        diversity_report: DiversityReport = None,
        metric_analysis: MetricAnalysis = None,
    ) -> List[Recommendation]:
        """Recommend post-processing improvements."""
        recs = []

        # Recommend constraint validation
        recs.append(Recommendation(
            category=RecommendationCategory.POST_PROCESSING.value,
            priority=3,
            title="Apply constraint validation",
            description="Filter generated records that violate domain constraints.",
            expected_improvements={'quality_score': 'Maintain or improve'},
            implementation={'post_processing': {'constraint_validation': True}},
            rationale="Post-generation constraint validation removes unrealistic records.",
        ))

        # Privacy risk reduction
        if metric_analysis:
            privacy = metric_analysis.privacy_metrics.get('privacy_score')
            high_risk = metric_analysis.privacy_metrics.get('high_risk_pct')

            if privacy and privacy[1] in (QualityLevel.CRITICAL, QualityLevel.WARNING):
                recs.append(Recommendation(
                    category=RecommendationCategory.POST_PROCESSING.value,
                    priority=1,
                    title="Reduce privacy risk records",
                    description="Filter or perturb records with high re-identification risk.",
                    expected_improvements={'privacy_score': 'Improve by ~0.10', 'high_risk_pct': 'Reduce to <2%'},
                    implementation={'post_processing': {'filter_high_risk': True}},
                    rationale="Privacy score is below target. Filtering high-risk records directly improves privacy metrics.",
                ))

            if high_risk and high_risk[0] > 0.02:
                recs.append(Recommendation(
                    category=RecommendationCategory.POST_PROCESSING.value,
                    priority=2,
                    title="Add differential privacy noise",
                    description="Apply noise injection to reduce memorization of rare records.",
                    expected_improvements={'high_risk_pct': 'Reduce by ~50%', 'mean_nnd': 'Improve by ~0.05'},
                    implementation={'post_processing': {'differential_privacy': True}},
                    rationale="High-risk records indicate memorization. Differential privacy prevents this.",
                ))

        # Diversity issues
        if diversity_report and diversity_report.mode_collapse_detected:
            recs.append(Recommendation(
                category=RecommendationCategory.POST_PROCESSING.value,
                priority=2,
                title="Address mode collapse",
                description="Mode collapse detected. Increase generator diversity or resample.",
                expected_improvements={'quality_score': 'Improve by ~0.05'},
                implementation={'post_processing': {'resample_diverse': True}},
                rationale="Mode collapse means the model is generating limited variety. Resampling or retraining needed.",
            ))

        return recs

    def prioritize_recommendations(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Sort recommendations by priority (1=highest first), then by category weight."""
        category_order = {
            RecommendationCategory.POST_PROCESSING.value: 0,
            RecommendationCategory.PREPROCESSING.value: 1,
            RecommendationCategory.TRAINING.value: 2,
            RecommendationCategory.MODEL.value: 3,
            RecommendationCategory.DATA_STRATEGY.value: 4,
        }
        return sorted(
            recommendations,
            key=lambda r: (r.priority, category_order.get(r.category, 5)),
        )

    def _check_privacy_impact(
        self, recommendation: Recommendation, metric_analysis: MetricAnalysis
    ) -> bool:
        """Check if a recommendation is safe from a privacy perspective.

        Returns True if safe to apply, False if it risks degrading privacy.
        """
        if not metric_analysis:
            return True

        privacy = metric_analysis.privacy_metrics.get('privacy_score')
        if not privacy:
            return True

        # If privacy is already Critical, only allow recommendations that
        # explicitly improve privacy or are unrelated to model training
        if privacy[1] == QualityLevel.CRITICAL:
            if recommendation.category in (
                RecommendationCategory.POST_PROCESSING.value,
                RecommendationCategory.PREPROCESSING.value,
            ):
                return True
            # Block model/training changes that could further degrade privacy
            impl = recommendation.implementation
            if 'epochs' in impl and impl.get('epochs', 0) > 800:
                return False  # Very high epochs risk memorization

        return True
