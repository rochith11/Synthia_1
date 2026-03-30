"""Root Cause Analyzer for synthetic data quality issues.

Diagnoses underlying causes when synthetic data metrics fall below
acceptable thresholds. Uses pattern matching against known failure
modes to identify the most likely root causes and provide actionable
recommendations.
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.ai_diagnostic_agent.models import (
    RootCause,
    MetricAnalysis,
    QualityLevel,
    DataProfile,
)
from src.ai_diagnostic_agent.config import DATASET_SIZE_THRESHOLDS

logger = logging.getLogger(__name__)

# Impact severity ordering used for ranking
_IMPACT_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


class RootCauseAnalyzer:
    """Analyzes metric results and data profiles to identify root causes
    of synthetic data quality problems.

    Each private ``_check_*`` method tests for one failure mode and
    returns a ``RootCause`` when evidence is found, or ``None`` otherwise.
    The public ``diagnose`` method orchestrates all checks and returns a
    ranked list of identified causes.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        """Initialize with root cause pattern definitions."""
        self._define_patterns()

    def _define_patterns(self):
        """Define mapping of metric patterns to root causes.

        Each pattern is a dictionary containing:
        - name: short identifier for the root cause
        - description: human-readable explanation
        - indicators: dict mapping metric names to the ``QualityLevel``
          values that would trigger this pattern
        - affected_metrics: list of metric names typically degraded
        - recommendations: default recommendation strings
        """
        self.patterns: List[Dict] = [
            {
                "name": "Small Dataset",
                "description": (
                    "The training dataset is too small for the generative "
                    "model to learn robust statistical patterns, leading to "
                    "poor distributional fidelity and low utility."
                ),
                "indicators": {
                    "quality_score": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                    "mean_jsd": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                },
                "affected_metrics": [
                    "quality_score",
                    "mean_jsd",
                    "mean_ks",
                    "ml_accuracy",
                ],
                "recommendations": [
                    "Collect more real-world training samples (target > 5 000 rows).",
                    "Apply data augmentation on the real data before training.",
                    "Use a simpler generative model (e.g. GaussianCopula) that "
                    "needs fewer samples.",
                    "Reduce the number of features to lower the data requirement.",
                ],
            },
            {
                "name": "Encoding Issues",
                "description": (
                    "Categorical features are not encoded correctly, causing "
                    "the model to misinterpret discrete distributions and "
                    "produce implausible category frequencies."
                ),
                "indicators": {
                    "mean_jsd": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                    "max_jsd": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                },
                "affected_metrics": [
                    "mean_jsd",
                    "max_jsd",
                    "quality_score",
                ],
                "recommendations": [
                    "Verify that all categorical columns are declared in metadata.",
                    "Switch to one-hot or frequency encoding for high-cardinality "
                    "categoricals.",
                    "Ensure ordinal columns preserve ordering during encoding.",
                    "Review preprocessing for any unintended type coercion.",
                ],
            },
            {
                "name": "Distribution Imbalance",
                "description": (
                    "The training data contains heavily skewed or imbalanced "
                    "distributions that the model amplifies, under-representing "
                    "minority classes and rare values."
                ),
                "indicators": {
                    "max_amplification_ratio": [
                        QualityLevel.WARNING,
                        QualityLevel.CRITICAL,
                    ],
                },
                "affected_metrics": [
                    "max_amplification_ratio",
                    "mean_jsd",
                    "f1_score",
                ],
                "recommendations": [
                    "Apply class-conditional sampling or rebalancing before training.",
                    "Use stratified sampling to preserve minority proportions.",
                    "Post-process synthetic output with rejection sampling for "
                    "rare categories.",
                    "Consider oversampling rare classes in the training set.",
                ],
            },
            {
                "name": "Model Underfitting",
                "description": (
                    "The generative model has not been trained long enough or "
                    "lacks capacity, resulting in poor statistical fidelity and "
                    "low downstream utility."
                ),
                "indicators": {
                    "quality_score": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                    "ml_accuracy": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                },
                "affected_metrics": [
                    "quality_score",
                    "ml_accuracy",
                    "f1_score",
                    "auc",
                    "mean_ks",
                ],
                "recommendations": [
                    "Increase the number of training epochs.",
                    "Increase model capacity (embedding dimension, hidden layers).",
                    "Lower the learning rate for more stable convergence.",
                    "Try a different model architecture (TVAE, CopulaGAN).",
                ],
            },
            {
                "name": "Mode Collapse",
                "description": (
                    "The generative model collapses onto a narrow subset of "
                    "the data distribution, producing repetitive or low-diversity "
                    "synthetic records."
                ),
                "indicators": {
                    "quality_score": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                },
                "affected_metrics": [
                    "quality_score",
                    "mean_jsd",
                    "ml_accuracy",
                    "f1_score",
                ],
                "recommendations": [
                    "Reduce batch size to expose the generator to more variation.",
                    "Add noise injection or use a different GAN training strategy "
                    "(Wasserstein, spectral normalisation).",
                    "Increase discriminator updates per generator update.",
                    "Switch to a VAE-based model that is less prone to mode collapse.",
                ],
            },
            {
                "name": "Privacy Leakage",
                "description": (
                    "The model has memorised individual training records, "
                    "creating synthetic rows that are near-duplicates of real "
                    "data and posing re-identification risk."
                ),
                "indicators": {
                    "privacy_score": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                    "high_risk_pct": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                },
                "affected_metrics": [
                    "privacy_score",
                    "mean_nnd",
                    "high_risk_pct",
                ],
                "recommendations": [
                    "Apply differential privacy (DP-SGD) during training.",
                    "Reduce the number of training epochs to prevent memorisation.",
                    "Increase noise in the latent space.",
                    "Post-filter synthetic rows that are too close to real records.",
                ],
            },
            {
                "name": "Correlation Loss",
                "description": (
                    "Inter-feature correlations present in the real data are "
                    "not preserved in the synthetic output, degrading downstream "
                    "model utility even when marginal distributions look acceptable."
                ),
                "indicators": {
                    "ml_accuracy": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                    "f1_score": [QualityLevel.WARNING, QualityLevel.CRITICAL],
                },
                "affected_metrics": [
                    "ml_accuracy",
                    "f1_score",
                    "auc",
                ],
                "recommendations": [
                    "Use a model that captures correlations (CopulaGAN, TVAE).",
                    "Increase embedding dimension to capture richer relationships.",
                    "Add correlation-preserving loss terms if using a custom model.",
                    "Verify that feature preprocessing does not decorrelate columns.",
                ],
            },
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        metric_analysis: MetricAnalysis,
        data_profile: DataProfile = None,
        training_config: dict = None,
    ) -> List[RootCause]:
        """Diagnose root causes based on metric analysis and optional context.

        Runs every available check method and collects the ``RootCause``
        objects that fire.  The results are ranked by impact and
        likelihood before being returned.

        Parameters
        ----------
        metric_analysis:
            The evaluated metric analysis for this cycle.
        data_profile:
            Optional real-data profile used for size / distribution checks.
        training_config:
            Optional dict of training hyper-parameters (epochs, lr, ...).

        Returns
        -------
        List[RootCause]
            Ranked list of identified root causes (most severe first).
        """
        causes: List[RootCause] = []

        # --- Run each specialised check ---
        small_ds = self._check_small_dataset(data_profile, metric_analysis)
        if small_ds is not None:
            causes.append(small_ds)

        encoding = self._check_encoding_issues(metric_analysis)
        if encoding is not None:
            causes.append(encoding)

        imbalance = self._check_distribution_imbalance(metric_analysis, data_profile)
        if imbalance is not None:
            causes.append(imbalance)

        underfit = self._check_model_underfitting(metric_analysis, training_config)
        if underfit is not None:
            causes.append(underfit)

        collapse = self._check_mode_collapse(metric_analysis)
        if collapse is not None:
            causes.append(collapse)

        privacy = self._check_privacy_leakage(metric_analysis)
        if privacy is not None:
            causes.append(privacy)

        correlation = self._check_correlation_loss(metric_analysis)
        if correlation is not None:
            causes.append(correlation)

        ranked = self.rank_causes(causes)
        logger.info("Root cause analysis identified %d cause(s).", len(ranked))
        return ranked

    # ------------------------------------------------------------------
    # Individual root-cause checks
    # ------------------------------------------------------------------

    def _check_small_dataset(
        self,
        data_profile: Optional[DataProfile],
        metric_analysis: MetricAnalysis,
    ) -> Optional[RootCause]:
        """Check if a small dataset size is the primary driver of quality issues.

        Fires when ``data_profile.n_rows`` is below the *medium*
        threshold **and** at least some quality metrics are degraded.
        """
        if data_profile is None:
            return None

        n_rows = data_profile.n_rows
        medium_threshold = DATASET_SIZE_THRESHOLDS.get("medium", 5000)
        small_threshold = DATASET_SIZE_THRESHOLDS.get("small", 1000)
        critical_threshold = DATASET_SIZE_THRESHOLDS.get("critical_min", 500)

        if n_rows >= medium_threshold:
            return None  # Dataset is large enough

        # Check whether quality / utility metrics are actually degraded
        degraded_quality = self._count_degraded(metric_analysis.quality_metrics)
        degraded_utility = self._count_degraded(metric_analysis.utility_metrics)

        if degraded_quality == 0 and degraded_utility == 0:
            return None  # Small dataset but metrics are fine

        # Build evidence
        evidence: List[str] = [
            f"Dataset has only {n_rows:,} rows (threshold for adequate training "
            f"is {medium_threshold:,}).",
        ]
        if n_rows < critical_threshold:
            evidence.append(
                f"Row count ({n_rows:,}) is below the critical minimum "
                f"({critical_threshold:,}). Model training is severely data-starved."
            )
        if degraded_quality > 0:
            evidence.append(
                f"{degraded_quality} quality metric(s) are at Warning or Critical level."
            )
        if degraded_utility > 0:
            evidence.append(
                f"{degraded_utility} utility metric(s) are at Warning or Critical level."
            )

        # Determine likelihood and impact
        if n_rows < critical_threshold:
            likelihood = 0.95
            impact = "Critical"
        elif n_rows < small_threshold:
            likelihood = 0.85
            impact = "High"
        else:
            likelihood = 0.70
            impact = "High" if (degraded_quality + degraded_utility) >= 3 else "Medium"

        pattern = self._get_pattern("Small Dataset")
        return RootCause(
            cause_name="Small Dataset",
            description=pattern["description"],
            affected_metrics=pattern["affected_metrics"],
            likelihood=likelihood,
            impact=impact,
            evidence=evidence,
            recommendations=pattern["recommendations"],
        )

    def _check_encoding_issues(
        self,
        metric_analysis: MetricAnalysis,
    ) -> Optional[RootCause]:
        """Check for incorrect feature encoding issues.

        Fires when JSD-based metrics (particularly for categorical
        columns) are degraded while other quality signals may still be
        passable, pointing to a mismatch between declared and actual
        column types.
        """
        mean_jsd_info = metric_analysis.quality_metrics.get("mean_jsd")
        max_jsd_info = metric_analysis.quality_metrics.get("max_jsd")

        if mean_jsd_info is None and max_jsd_info is None:
            return None

        mean_jsd_bad = (
            mean_jsd_info is not None
            and mean_jsd_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        max_jsd_bad = (
            max_jsd_info is not None
            and max_jsd_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )

        if not (mean_jsd_bad or max_jsd_bad):
            return None

        # Additional signal: quality_score might still be passable while JSD
        # is bad, which points towards an encoding issue rather than a
        # wholesale model problem.
        quality_score_info = metric_analysis.quality_metrics.get("quality_score")
        quality_ok = (
            quality_score_info is not None
            and quality_score_info[1]
            in (QualityLevel.EXCELLENT, QualityLevel.ACCEPTABLE)
        )

        # Check if correlation-based utility is also degraded -- encoding
        # problems typically hurt correlations too.
        ml_acc_info = metric_analysis.utility_metrics.get("ml_accuracy")
        ml_acc_bad = (
            ml_acc_info is not None
            and ml_acc_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )

        evidence: List[str] = []
        if mean_jsd_bad:
            evidence.append(
                f"mean_jsd is {mean_jsd_info[1].value} "
                f"(value={mean_jsd_info[0]:.4f}), indicating categorical "
                "distributions diverge significantly."
            )
        if max_jsd_bad:
            evidence.append(
                f"max_jsd is {max_jsd_info[1].value} "
                f"(value={max_jsd_info[0]:.4f}), at least one column has "
                "very high distributional divergence."
            )
        if quality_ok:
            evidence.append(
                "Overall quality_score is still acceptable, suggesting the "
                "issue is localised to specific column encodings."
            )
        if ml_acc_bad:
            evidence.append(
                "ML accuracy is also degraded, consistent with encoding "
                "errors propagating to downstream utility."
            )

        # Likelihood
        if mean_jsd_bad and max_jsd_bad:
            likelihood = 0.80 if quality_ok else 0.65
        else:
            likelihood = 0.60

        # Impact
        if max_jsd_info is not None and max_jsd_info[1] == QualityLevel.CRITICAL:
            impact = "High"
        elif mean_jsd_bad:
            impact = "Medium"
        else:
            impact = "Medium"

        pattern = self._get_pattern("Encoding Issues")
        return RootCause(
            cause_name="Encoding Issues",
            description=pattern["description"],
            affected_metrics=pattern["affected_metrics"],
            likelihood=likelihood,
            impact=impact,
            evidence=evidence,
            recommendations=pattern["recommendations"],
        )

    def _check_distribution_imbalance(
        self,
        metric_analysis: MetricAnalysis,
        data_profile: Optional[DataProfile],
    ) -> Optional[RootCause]:
        """Check for unbalanced distribution issues.

        Fires when bias amplification metrics are degraded **or** when
        the data profile reports rare values in multiple columns.
        """
        # Check bias metrics
        amp_info = metric_analysis.bias_metrics.get("max_amplification_ratio")
        amp_bad = (
            amp_info is not None
            and amp_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )

        # Check profile for rare values
        has_rare_values = (
            data_profile is not None
            and len(data_profile.rare_values) > 0
        )
        has_class_imbalance = (
            data_profile is not None
            and len(data_profile.class_imbalances) > 0
        )

        if not amp_bad and not has_rare_values and not has_class_imbalance:
            return None

        evidence: List[str] = []
        if amp_bad:
            evidence.append(
                f"max_amplification_ratio is {amp_info[1].value} "
                f"(value={amp_info[0]:.4f}), bias is being amplified in "
                "synthetic output."
            )
        if has_rare_values:
            rare_cols = list(data_profile.rare_values.keys())
            evidence.append(
                f"Rare values detected in {len(rare_cols)} column(s): "
                f"{', '.join(rare_cols[:5])}"
                + (" ..." if len(rare_cols) > 5 else "")
                + "."
            )
        if has_class_imbalance:
            imb_cols = list(data_profile.class_imbalances.keys())
            evidence.append(
                f"Class imbalance detected in {len(imb_cols)} column(s): "
                f"{', '.join(imb_cols[:5])}"
                + (" ..." if len(imb_cols) > 5 else "")
                + "."
            )

        # Determine severity
        if amp_bad and amp_info[1] == QualityLevel.CRITICAL:
            likelihood = 0.90
            impact = "Critical"
        elif amp_bad:
            likelihood = 0.75
            impact = "High"
        elif has_rare_values and has_class_imbalance:
            likelihood = 0.65
            impact = "Medium"
        else:
            likelihood = 0.55
            impact = "Medium"

        pattern = self._get_pattern("Distribution Imbalance")
        return RootCause(
            cause_name="Distribution Imbalance",
            description=pattern["description"],
            affected_metrics=pattern["affected_metrics"],
            likelihood=likelihood,
            impact=impact,
            evidence=evidence,
            recommendations=pattern["recommendations"],
        )

    def _check_model_underfitting(
        self,
        metric_analysis: MetricAnalysis,
        training_config: Optional[dict],
    ) -> Optional[RootCause]:
        """Check for model underfitting.

        Fires when both quality and utility metrics are degraded,
        optionally amplified by evidence of a low epoch count in the
        training configuration.
        """
        quality_info = metric_analysis.quality_metrics.get("quality_score")
        ml_acc_info = metric_analysis.utility_metrics.get("ml_accuracy")

        quality_bad = (
            quality_info is not None
            and quality_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        ml_acc_bad = (
            ml_acc_info is not None
            and ml_acc_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )

        if not (quality_bad and ml_acc_bad):
            return None

        evidence: List[str] = []
        evidence.append(
            f"quality_score is {quality_info[1].value} "
            f"(value={quality_info[0]:.4f})."
        )
        evidence.append(
            f"ml_accuracy is {ml_acc_info[1].value} "
            f"(value={ml_acc_info[0]:.4f})."
        )

        # Additional utility metrics
        f1_info = metric_analysis.utility_metrics.get("f1_score")
        if f1_info is not None and f1_info[1] in (
            QualityLevel.WARNING,
            QualityLevel.CRITICAL,
        ):
            evidence.append(
                f"f1_score is also {f1_info[1].value} "
                f"(value={f1_info[0]:.4f}), corroborating underfitting."
            )

        auc_info = metric_analysis.utility_metrics.get("auc")
        if auc_info is not None and auc_info[1] in (
            QualityLevel.WARNING,
            QualityLevel.CRITICAL,
        ):
            evidence.append(
                f"auc is {auc_info[1].value} (value={auc_info[0]:.4f})."
            )

        # Epoch / training config evidence
        low_epochs = False
        if training_config is not None:
            epochs = training_config.get("epochs")
            if epochs is not None and epochs < 150:
                low_epochs = True
                evidence.append(
                    f"Training ran for only {epochs} epochs, which may be "
                    "insufficient for convergence."
                )
            lr = training_config.get("learning_rate") or training_config.get(
                "generator_lr"
            )
            if lr is not None and lr > 5e-3:
                evidence.append(
                    f"Learning rate ({lr}) is relatively high, which can "
                    "prevent stable convergence."
                )

        # Determine severity
        both_critical = (
            quality_info[1] == QualityLevel.CRITICAL
            and ml_acc_info[1] == QualityLevel.CRITICAL
        )
        if both_critical:
            likelihood = 0.90
            impact = "Critical"
        elif low_epochs:
            likelihood = 0.85
            impact = "High"
        else:
            likelihood = 0.70
            impact = "High"

        pattern = self._get_pattern("Model Underfitting")
        return RootCause(
            cause_name="Model Underfitting",
            description=pattern["description"],
            affected_metrics=pattern["affected_metrics"],
            likelihood=likelihood,
            impact=impact,
            evidence=evidence,
            recommendations=pattern["recommendations"],
        )

    def _check_mode_collapse(
        self,
        metric_analysis: MetricAnalysis,
    ) -> Optional[RootCause]:
        """Check for mode collapse / overfitting.

        Mode collapse manifests as seemingly decent point-metric values
        but very low diversity -- quality metrics are poor because the
        generator only covers a narrow slice of the real distribution.
        """
        quality_info = metric_analysis.quality_metrics.get("quality_score")
        mean_jsd_info = metric_analysis.quality_metrics.get("mean_jsd")
        mean_ks_info = metric_analysis.quality_metrics.get("mean_ks")

        quality_bad = (
            quality_info is not None
            and quality_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        jsd_bad = (
            mean_jsd_info is not None
            and mean_jsd_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        ks_bad = (
            mean_ks_info is not None
            and mean_ks_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )

        # Mode collapse typically shows degraded quality and distribution
        # metrics simultaneously. We require at least two degraded signals.
        degraded_count = sum([quality_bad, jsd_bad, ks_bad])
        if degraded_count < 2:
            return None

        # Privacy being good (or even excellent) while quality is bad is a
        # strong signal -- the model memorises a few modes well.
        privacy_info = metric_analysis.privacy_metrics.get("privacy_score")
        privacy_good = (
            privacy_info is not None
            and privacy_info[1]
            in (QualityLevel.EXCELLENT, QualityLevel.ACCEPTABLE)
        )

        evidence: List[str] = []
        if quality_bad:
            evidence.append(
                f"quality_score is {quality_info[1].value} "
                f"(value={quality_info[0]:.4f}), suggesting the generator "
                "does not cover the full distribution."
            )
        if jsd_bad:
            evidence.append(
                f"mean_jsd is {mean_jsd_info[1].value} "
                f"(value={mean_jsd_info[0]:.4f}), marginal distributions "
                "are poorly reproduced."
            )
        if ks_bad:
            evidence.append(
                f"mean_ks is {mean_ks_info[1].value} "
                f"(value={mean_ks_info[0]:.4f}), continuous distributions "
                "diverge from real data."
            )
        if privacy_good:
            evidence.append(
                "Privacy score is acceptable/excellent despite poor quality, "
                "consistent with mode collapse (model produces limited but "
                "non-memorised modes)."
            )

        if degraded_count == 3:
            likelihood = 0.85
            impact = "Critical"
        elif privacy_good:
            likelihood = 0.75
            impact = "High"
        else:
            likelihood = 0.65
            impact = "High"

        pattern = self._get_pattern("Mode Collapse")
        return RootCause(
            cause_name="Mode Collapse",
            description=pattern["description"],
            affected_metrics=pattern["affected_metrics"],
            likelihood=likelihood,
            impact=impact,
            evidence=evidence,
            recommendations=pattern["recommendations"],
        )

    def _check_privacy_leakage(
        self,
        metric_analysis: MetricAnalysis,
    ) -> Optional[RootCause]:
        """Check for privacy leakage / memorisation.

        Fires when privacy-specific metrics indicate that synthetic
        records are dangerously close to real records.
        """
        privacy_info = metric_analysis.privacy_metrics.get("privacy_score")
        nnd_info = metric_analysis.privacy_metrics.get("mean_nnd")
        risk_info = metric_analysis.privacy_metrics.get("high_risk_pct")

        privacy_bad = (
            privacy_info is not None
            and privacy_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        risk_bad = (
            risk_info is not None
            and risk_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        nnd_bad = (
            nnd_info is not None
            and nnd_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )

        if not (privacy_bad or risk_bad):
            return None

        evidence: List[str] = []
        if privacy_bad:
            evidence.append(
                f"privacy_score is {privacy_info[1].value} "
                f"(value={privacy_info[0]:.4f}), overall privacy is degraded."
            )
        if risk_bad:
            evidence.append(
                f"high_risk_pct is {risk_info[1].value} "
                f"(value={risk_info[0]:.4f}), an elevated fraction of "
                "synthetic records are dangerously close to real records."
            )
        if nnd_bad:
            evidence.append(
                f"mean_nnd is {nnd_info[1].value} "
                f"(value={nnd_info[0]:.4f}), average nearest-neighbour "
                "distance is too low."
            )

        # Quality being good alongside bad privacy is a strong memorisation
        # signal.
        quality_info = metric_analysis.quality_metrics.get("quality_score")
        quality_good = (
            quality_info is not None
            and quality_info[1]
            in (QualityLevel.EXCELLENT, QualityLevel.ACCEPTABLE)
        )
        if quality_good:
            evidence.append(
                "Quality score is acceptable while privacy is poor -- the "
                "model may be copying real records rather than learning the "
                "distribution."
            )

        # Severity
        if privacy_bad and risk_bad:
            likelihood = 0.90
            impact = "Critical"
        elif risk_bad and risk_info[1] == QualityLevel.CRITICAL:
            likelihood = 0.85
            impact = "Critical"
        elif privacy_bad and privacy_info[1] == QualityLevel.CRITICAL:
            likelihood = 0.80
            impact = "High"
        else:
            likelihood = 0.70
            impact = "High"

        pattern = self._get_pattern("Privacy Leakage")
        return RootCause(
            cause_name="Privacy Leakage",
            description=pattern["description"],
            affected_metrics=pattern["affected_metrics"],
            likelihood=likelihood,
            impact=impact,
            evidence=evidence,
            recommendations=pattern["recommendations"],
        )

    def _check_correlation_loss(
        self,
        metric_analysis: MetricAnalysis,
    ) -> Optional[RootCause]:
        """Check for correlation loss between features.

        Fires when quality metrics are acceptable but utility metrics
        (which depend on multivariate relationships) are degraded.
        """
        quality_info = metric_analysis.quality_metrics.get("quality_score")
        ml_acc_info = metric_analysis.utility_metrics.get("ml_accuracy")
        f1_info = metric_analysis.utility_metrics.get("f1_score")
        auc_info = metric_analysis.utility_metrics.get("auc")

        # Quality should be at least passable
        quality_ok = (
            quality_info is not None
            and quality_info[1]
            in (QualityLevel.EXCELLENT, QualityLevel.ACCEPTABLE, QualityLevel.WARNING)
        )

        # At least one utility metric must be degraded
        ml_bad = (
            ml_acc_info is not None
            and ml_acc_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        f1_bad = (
            f1_info is not None
            and f1_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
        auc_bad = (
            auc_info is not None
            and auc_info[1] in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )

        utility_degraded = sum([ml_bad, f1_bad, auc_bad])
        if not (quality_ok and utility_degraded >= 1):
            return None

        evidence: List[str] = []
        if quality_info is not None:
            evidence.append(
                f"quality_score is {quality_info[1].value} "
                f"(value={quality_info[0]:.4f}), marginal distributions "
                "are reasonably preserved."
            )
        if ml_bad:
            evidence.append(
                f"ml_accuracy is {ml_acc_info[1].value} "
                f"(value={ml_acc_info[0]:.4f}), downstream model performance "
                "is poor despite acceptable quality."
            )
        if f1_bad:
            evidence.append(
                f"f1_score is {f1_info[1].value} "
                f"(value={f1_info[0]:.4f}), class-level prediction quality "
                "is degraded."
            )
        if auc_bad:
            evidence.append(
                f"auc is {auc_info[1].value} (value={auc_info[0]:.4f})."
            )
        evidence.append(
            "Gap between quality (marginals) and utility (multivariate) "
            "suggests inter-feature correlations are not preserved."
        )

        if utility_degraded >= 3:
            likelihood = 0.85
            impact = "High"
        elif utility_degraded == 2:
            likelihood = 0.75
            impact = "High"
        else:
            likelihood = 0.60
            impact = "Medium"

        # Elevate if quality is actually good (stronger signal)
        if quality_info is not None and quality_info[1] in (
            QualityLevel.EXCELLENT,
            QualityLevel.ACCEPTABLE,
        ):
            likelihood = min(likelihood + 0.10, 1.0)

        pattern = self._get_pattern("Correlation Loss")
        return RootCause(
            cause_name="Correlation Loss",
            description=pattern["description"],
            affected_metrics=pattern["affected_metrics"],
            likelihood=likelihood,
            impact=impact,
            evidence=evidence,
            recommendations=pattern["recommendations"],
        )

    # ------------------------------------------------------------------
    # Ranking & reporting
    # ------------------------------------------------------------------

    def rank_causes(self, causes: List[RootCause]) -> List[RootCause]:
        """Rank causes by impact severity (Critical first) then by
        likelihood descending.

        Parameters
        ----------
        causes:
            Unordered list of identified root causes.

        Returns
        -------
        List[RootCause]
            Sorted list (most impactful / most likely first).
        """
        return sorted(
            causes,
            key=lambda c: (
                _IMPACT_ORDER.get(c.impact, 99),
                -c.likelihood,
            ),
        )

    def generate_report(self, causes: List[RootCause]) -> str:
        """Generate a human-readable root cause analysis report.

        Parameters
        ----------
        causes:
            Ranked list of root causes (as returned by ``diagnose``).

        Returns
        -------
        str
            Multi-line report suitable for logging or display.
        """
        if not causes:
            return (
                "=== Root Cause Analysis Report ===\n"
                "No root causes identified. All metrics appear healthy.\n"
            )

        lines: List[str] = [
            "=== Root Cause Analysis Report ===",
            f"Identified {len(causes)} potential root cause(s).\n",
        ]

        for idx, cause in enumerate(causes, start=1):
            lines.append(f"--- Cause #{idx}: {cause.cause_name} ---")
            lines.append(f"  Impact:     {cause.impact}")
            lines.append(f"  Likelihood: {cause.likelihood:.0%}")
            lines.append(f"  Description: {cause.description}")
            lines.append(f"  Affected metrics: {', '.join(cause.affected_metrics)}")

            if cause.evidence:
                lines.append("  Evidence:")
                for ev in cause.evidence:
                    lines.append(f"    - {ev}")

            if cause.recommendations:
                lines.append("  Recommendations:")
                for rec in cause.recommendations:
                    lines.append(f"    * {rec}")

            lines.append("")  # blank separator

        lines.append("=== End of Report ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pattern(self, name: str) -> Dict:
        """Retrieve a pattern definition by name."""
        for pattern in self.patterns:
            if pattern["name"] == name:
                return pattern
        # Fallback -- should never happen if patterns are defined correctly.
        return {
            "name": name,
            "description": "",
            "indicators": {},
            "affected_metrics": [],
            "recommendations": [],
        }

    @staticmethod
    def _count_degraded(
        metrics: Dict[str, Tuple[float, QualityLevel]],
    ) -> int:
        """Count how many metrics in a dict are at Warning or Critical."""
        return sum(
            1
            for _, (_, level) in metrics.items()
            if level in (QualityLevel.WARNING, QualityLevel.CRITICAL)
        )
