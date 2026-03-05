"""Dataset size diagnostics for evaluating whether a dataset is large enough
for reliable synthetic data generation."""

import logging
import math
from typing import List

from src.ai_diagnostic_agent.models import RootCause, Recommendation, QualityLevel
from src.ai_diagnostic_agent.config import DATASET_SIZE_THRESHOLDS


class DatasetSizeDiagnostics:
    """Evaluates dataset dimensions against recommended minimums and provides
    model selection guidance, training parameter adjustments, and actionable
    recommendations based on dataset size.

    Size categories (from ``DATASET_SIZE_THRESHOLDS``):
        * **critical** -- fewer than 500 rows
        * **small**    -- fewer than 1 000 rows
        * **medium**   -- fewer than 5 000 rows
        * **large**    -- 10 000 rows or more

    The diagnostics account for the number of columns and the proportion of
    categorical features when computing the minimum recommended size.
    """

    def __init__(self):
        """Initialize dataset size diagnostics."""
        self.thresholds = DATASET_SIZE_THRESHOLDS
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_dataset_size(
        self,
        n_rows: int,
        n_columns: int,
        n_categorical_cols: int = 0,
    ) -> dict:
        """Analyze dataset size and return diagnostics.

        Parameters
        ----------
        n_rows : int
            Number of rows in the dataset.
        n_columns : int
            Total number of columns in the dataset.
        n_categorical_cols : int, optional
            Number of categorical columns (default 0).

        Returns
        -------
        dict
            Diagnostic results containing:

            - ``n_rows`` -- row count
            - ``n_columns`` -- column count
            - ``size_category`` -- one of ``'critical'``, ``'small'``,
              ``'medium'``, or ``'large'``
            - ``min_recommended_size`` -- computed minimum dataset size
            - ``is_sufficient`` -- whether the dataset meets the minimum
            - ``severity`` -- :class:`QualityLevel` classification
            - ``warnings`` -- list of human-readable warning strings
            - ``recommendations`` -- list of :class:`Recommendation` objects
        """
        min_recommended = self.compute_minimum_size(n_columns, n_categorical_cols)
        size_category = self._classify_size(n_rows)
        severity = self._map_severity(n_rows)
        is_sufficient = n_rows >= min_recommended

        warnings: List[str] = []
        recommendations: List[Recommendation] = []

        # --- Warnings ------------------------------------------------
        if n_rows < self.thresholds["critical_min"]:
            warnings.append(
                f"Dataset has only {n_rows} rows which is below the critical "
                f"minimum of {self.thresholds['critical_min']}. Synthetic data "
                "quality will be severely compromised."
            )
        elif n_rows < self.thresholds["small"]:
            warnings.append(
                f"Dataset has {n_rows} rows which is below the recommended "
                f"minimum of {self.thresholds['small']}. GAN-based models "
                "may produce unstable results."
            )
        elif n_rows < self.thresholds["medium"]:
            warnings.append(
                f"Dataset has {n_rows} rows. TVAE is preferred over "
                "GAN-based models at this size."
            )

        if not is_sufficient:
            warnings.append(
                f"Dataset size ({n_rows}) is below the computed minimum "
                f"recommended size ({min_recommended}) given {n_columns} "
                f"columns and {n_categorical_cols} categorical features."
            )

        # --- Recommendations -----------------------------------------
        model_recs = self.get_model_recommendations(n_rows)
        recommendations.extend(model_recs)

        training_adjustments = self.get_training_adjustments(n_rows)
        if training_adjustments.get("adjustments_needed", False):
            recommendations.append(
                Recommendation(
                    category="training",
                    priority=2,
                    title="Adjust training parameters for dataset size",
                    description=(
                        "Modify training hyperparameters to better suit the "
                        f"current dataset size of {n_rows} rows."
                    ),
                    expected_improvements={
                        "stability": "Improved training stability",
                        "convergence": "Better convergence for small data",
                    },
                    implementation=training_adjustments.get("parameters", {}),
                    rationale=(
                        "Smaller datasets benefit from more training epochs, "
                        "smaller batch sizes, and reduced model capacity to "
                        "avoid overfitting and mode collapse."
                    ),
                )
            )

        if not is_sufficient:
            recommendations.append(
                Recommendation(
                    category="data_strategy",
                    priority=1,
                    title="Increase dataset size",
                    description=(
                        f"The dataset has {n_rows} rows but at least "
                        f"{min_recommended} are recommended given the number "
                        "of features. Consider collecting more data or "
                        "reducing the feature set."
                    ),
                    expected_improvements={
                        "quality_score": "Significant improvement expected",
                        "distribution_fidelity": "Better column distributions",
                    },
                    implementation={
                        "options": [
                            "Collect additional real data",
                            "Remove low-importance features to reduce dimensionality",
                            "Combine related categorical columns to lower cardinality",
                        ]
                    },
                    rationale=(
                        "Synthetic data generators need sufficient examples to "
                        "learn the joint distribution. With too few rows "
                        "relative to the feature space, the model cannot "
                        "capture all relevant patterns."
                    ),
                )
            )

        self.logger.info(
            "Dataset size analysis complete: %d rows, %d cols -> %s (%s)",
            n_rows,
            n_columns,
            size_category,
            severity.value,
        )

        return {
            "n_rows": n_rows,
            "n_columns": n_columns,
            "size_category": size_category,
            "min_recommended_size": min_recommended,
            "is_sufficient": is_sufficient,
            "severity": severity,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def compute_minimum_size(
        self, n_columns: int, n_categorical_cols: int = 0
    ) -> int:
        """Compute the minimum recommended dataset size.

        The heuristic is::

            min_size = max(500, n_columns * 100 + n_categorical_cols * 50)

        For datasets with many categorical columns (more than half of all
        columns) an additional 20 % uplift is applied to account for the
        higher cardinality demands.

        Parameters
        ----------
        n_columns : int
            Total number of columns.
        n_categorical_cols : int, optional
            Number of categorical columns (default 0).

        Returns
        -------
        int
            Minimum recommended number of rows.
        """
        base_size = n_columns * 100 + n_categorical_cols * 50
        min_size = max(self.thresholds["critical_min"], base_size)

        # Uplift for high proportion of categorical columns
        if n_columns > 0 and (n_categorical_cols / n_columns) > 0.5:
            min_size = math.ceil(min_size * 1.20)

        return int(min_size)

    def get_model_recommendations(self, n_rows: int) -> List[Recommendation]:
        """Recommend appropriate models based on dataset size.

        Guidelines:
            * **< 1 000 rows** -- Warn about unstable GAN training; strongly
              prefer TVAE.
            * **< 5 000 rows** -- Prefer TVAE; increase epochs; reduce
              generator complexity.
            * **< 10 000 rows** -- TVAE or CTGAN with careful tuning.
            * **>= 10 000 rows** -- CTGAN or CopulaGAN work well.

        Parameters
        ----------
        n_rows : int
            Number of rows in the dataset.

        Returns
        -------
        List[Recommendation]
            Model selection recommendations.
        """
        recommendations: List[Recommendation] = []

        if n_rows < self.thresholds["small"]:
            recommendations.append(
                Recommendation(
                    category="model",
                    priority=1,
                    title="Use TVAE for small dataset",
                    description=(
                        f"With only {n_rows} rows, GAN-based models (CTGAN, "
                        "CopulaGAN) are likely to experience unstable training "
                        "and mode collapse. TVAE is strongly recommended as it "
                        "is more stable on small datasets."
                    ),
                    expected_improvements={
                        "stability": "Significantly more stable training",
                        "quality_score": "Higher quality for small data",
                    },
                    implementation={
                        "model_type": "TVAE",
                        "avoid": ["CTGAN", "CopulaGAN"],
                    },
                    rationale=(
                        "VAE-based models learn a continuous latent space via "
                        "maximum likelihood, which is more data-efficient than "
                        "the adversarial objective used by GANs."
                    ),
                )
            )
        elif n_rows < self.thresholds["medium"]:
            recommendations.append(
                Recommendation(
                    category="model",
                    priority=2,
                    title="Prefer TVAE; increase epochs and reduce complexity",
                    description=(
                        f"With {n_rows} rows TVAE is the preferred model. If "
                        "a GAN is used, increase training epochs and reduce "
                        "generator complexity to improve stability."
                    ),
                    expected_improvements={
                        "stability": "Improved training convergence",
                        "quality_score": "Better distribution fidelity",
                    },
                    implementation={
                        "preferred_model": "TVAE",
                        "alternative_model": "CTGAN",
                        "ctgan_adjustments": {
                            "epochs": 500,
                            "generator_dim": [128, 128],
                            "discriminator_dim": [128, 128],
                        },
                    },
                    rationale=(
                        "Medium-sized datasets can work with GANs but benefit "
                        "from reduced model capacity to prevent memorization "
                        "and longer training to ensure convergence."
                    ),
                )
            )
        elif n_rows < self.thresholds["large"]:
            recommendations.append(
                Recommendation(
                    category="model",
                    priority=3,
                    title="TVAE or CTGAN both viable",
                    description=(
                        f"With {n_rows} rows, both TVAE and CTGAN can produce "
                        "good results. CTGAN may capture more complex "
                        "distributions but requires careful hyperparameter "
                        "tuning."
                    ),
                    expected_improvements={
                        "flexibility": "Wider model choice",
                    },
                    implementation={
                        "recommended_models": ["TVAE", "CTGAN"],
                    },
                    rationale=(
                        "At this dataset size the data is sufficient for "
                        "stable GAN training, though TVAE remains a solid "
                        "default."
                    ),
                )
            )
        else:
            recommendations.append(
                Recommendation(
                    category="model",
                    priority=4,
                    title="Full model selection available",
                    description=(
                        f"With {n_rows} rows, all model types (CTGAN, "
                        "CopulaGAN, TVAE) are expected to work well. "
                        "CopulaGAN may provide additional benefits for "
                        "datasets with strong inter-column dependencies."
                    ),
                    expected_improvements={
                        "flexibility": "All models viable",
                        "quality_score": "High quality expected across models",
                    },
                    implementation={
                        "recommended_models": ["CTGAN", "CopulaGAN", "TVAE"],
                    },
                    rationale=(
                        "Large datasets provide enough signal for GAN "
                        "discriminators to give useful gradients, enabling "
                        "high-fidelity synthetic data generation."
                    ),
                )
            )

        return recommendations

    def get_training_adjustments(self, n_rows: int) -> dict:
        """Recommend training parameter adjustments for dataset size.

        Returns a dictionary with:
            * ``adjustments_needed`` -- whether any changes are recommended
            * ``parameters`` -- suggested hyperparameter values
            * ``rationale`` -- brief explanation

        Guidance:
            * **Small (< 1 000)**: more epochs (500), smaller batch_size
              (100), smaller embedding_dim (64).
            * **Medium (1 000 -- 5 000)**: standard parameters.
            * **Large (>= 5 000)**: can use larger batch_size; standard
              epochs.

        Parameters
        ----------
        n_rows : int
            Number of rows in the dataset.

        Returns
        -------
        dict
            Training adjustment recommendations.
        """
        if n_rows < self.thresholds["small"]:
            return {
                "adjustments_needed": True,
                "parameters": {
                    "epochs": 500,
                    "batch_size": 100,
                    "embedding_dim": 64,
                    "learning_rate": 1e-4,
                    "pac": 1,
                },
                "rationale": (
                    "Small datasets require more training epochs for "
                    "convergence, smaller batch sizes to see each sample "
                    "more often, and reduced embedding dimensions to "
                    "prevent overfitting."
                ),
            }
        elif n_rows < self.thresholds["medium"]:
            return {
                "adjustments_needed": True,
                "parameters": {
                    "epochs": 300,
                    "batch_size": 200,
                    "embedding_dim": 128,
                    "learning_rate": 2e-4,
                },
                "rationale": (
                    "Medium-small datasets benefit from moderately increased "
                    "epochs and slightly reduced batch sizes compared to the "
                    "defaults."
                ),
            }
        elif n_rows < self.thresholds["large"]:
            return {
                "adjustments_needed": False,
                "parameters": {
                    "epochs": 300,
                    "batch_size": 500,
                    "embedding_dim": 128,
                },
                "rationale": (
                    "Standard training parameters are appropriate for this "
                    "dataset size."
                ),
            }
        else:
            return {
                "adjustments_needed": False,
                "parameters": {
                    "epochs": 300,
                    "batch_size": 1000,
                    "embedding_dim": 256,
                },
                "rationale": (
                    "Large datasets can use bigger batch sizes and higher "
                    "capacity models without risk of overfitting."
                ),
            }

    def generate_report(self, analysis: dict) -> str:
        """Generate a human-readable dataset size diagnostic report.

        Parameters
        ----------
        analysis : dict
            The dictionary returned by :meth:`analyze_dataset_size`.

        Returns
        -------
        str
            Formatted report text.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DATASET SIZE DIAGNOSTICS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # --- Overview ------------------------------------------------
        lines.append("Overview:")
        lines.append(f"  Rows              : {analysis['n_rows']:,}")
        lines.append(f"  Columns           : {analysis['n_columns']}")
        lines.append(f"  Size category     : {analysis['size_category']}")
        lines.append(f"  Severity          : {analysis['severity'].value}")
        lines.append(
            f"  Min recommended   : {analysis['min_recommended_size']:,}"
        )
        lines.append(
            f"  Sufficient        : {'Yes' if analysis['is_sufficient'] else 'No'}"
        )
        lines.append("")

        # --- Warnings ------------------------------------------------
        warnings = analysis.get("warnings", [])
        if warnings:
            lines.append("-" * 60)
            lines.append("  Warnings")
            lines.append("-" * 60)
            for warning in warnings:
                lines.append(f"  ! {warning}")
            lines.append("")

        # --- Model recommendations -----------------------------------
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            lines.append("-" * 60)
            lines.append("  Recommendations")
            lines.append("-" * 60)
            for rec in recommendations:
                lines.append("")
                lines.append(f"  [{rec.category.upper()}] {rec.title}")
                lines.append(f"    Priority    : {rec.priority}")
                lines.append(f"    Description : {rec.description}")
                if rec.rationale:
                    lines.append(f"    Rationale   : {rec.rationale}")
                if rec.implementation:
                    lines.append("    Implementation:")
                    for key, value in rec.implementation.items():
                        lines.append(f"      {key}: {value}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("  END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_size(self, n_rows: int) -> str:
        """Map row count to a size category string.

        Parameters
        ----------
        n_rows : int
            Number of rows.

        Returns
        -------
        str
            One of ``'critical'``, ``'small'``, ``'medium'``, or ``'large'``.
        """
        if n_rows < self.thresholds["critical_min"]:
            return "critical"
        elif n_rows < self.thresholds["small"]:
            return "small"
        elif n_rows < self.thresholds["medium"]:
            return "medium"
        else:
            return "large"

    def _map_severity(self, n_rows: int) -> QualityLevel:
        """Map row count to a :class:`QualityLevel` severity.

        Mapping:
            * < 500   -> Critical
            * < 1 000 -> Warning
            * < 5 000 -> Acceptable
            * >= 5 000 -> Excellent

        Parameters
        ----------
        n_rows : int
            Number of rows.

        Returns
        -------
        QualityLevel
            The corresponding severity level.
        """
        if n_rows < self.thresholds["critical_min"]:
            return QualityLevel.CRITICAL
        elif n_rows < self.thresholds["small"]:
            return QualityLevel.WARNING
        elif n_rows < self.thresholds["medium"]:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.EXCELLENT
