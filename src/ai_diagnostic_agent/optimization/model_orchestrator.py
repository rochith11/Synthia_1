"""Model Orchestrator for training, evaluating, and selecting the best generative model.

This module provides the ModelOrchestrator class which manages the lifecycle of
multiple synthetic data generation models (CTGAN, TVAE, CopulaGAN).  It trains
all models on the same real dataset, generates synthetic data from each, evaluates
them through the validation / privacy / bias pipelines, and selects the
best-performing model based on a weighted composite score.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import concurrent.futures

from src.ai_diagnostic_agent.models import QualityLevel
from src.ai_diagnostic_agent.config import MODEL_SELECTION_WEIGHTS


logger = logging.getLogger(__name__)


class ModelOrchestrator:
    """Orchestrate training, generation, evaluation, and selection across
    multiple generative model architectures.

    Supported model types: CTGAN, TVAE, CopulaGAN.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, models: Optional[List[str]] = None):
        """Initialize orchestrator with list of model types to train.

        Parameters
        ----------
        models:
            Model architecture names to include.  Each must be one of
            ``'CTGAN'``, ``'TVAE'``, or ``'CopulaGAN'``.
            Default: ``['CTGAN', 'TVAE', 'CopulaGAN']``.
        """
        self.models: List[str] = models or ['CTGAN', 'TVAE', 'CopulaGAN']
        self.trained_models: Dict[str, object] = {}
        self.evaluation_results: Dict[str, dict] = {}
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train_all_models(
        self,
        data: pd.DataFrame,
        config: Optional[dict] = None,
    ) -> Dict[str, object]:
        """Train all configured models sequentially.

        Models are trained one at a time (not in parallel) to avoid SDV
        thread-safety issues.  Any model that fails to train is logged and
        skipped; the remaining models continue unaffected.

        Parameters
        ----------
        data:
            Real dataset used to fit every model.
        config:
            Optional per-model configuration overrides.  Expected structure::

                {
                    'CTGAN':     { <CTGANEngine kwargs> },
                    'TVAE':      { <TVAEEngine kwargs> },
                    'CopulaGAN': { <CopulaGANSynthesizer kwargs> },
                }

            If a model key is absent the engine defaults are used.

        Returns
        -------
        dict
            Mapping of ``model_name -> trained_model_object`` for every
            model that trained successfully.
        """
        config = config or {}

        self.logger.info(
            "Starting sequential training for %d model(s): %s",
            len(self.models), self.models,
        )

        for model_type in self.models:
            model_config = config.get(model_type, {})
            self.logger.info("Training %s ...", model_type)

            trained = self._train_single_model(model_type, data, model_config)

            if trained is not None:
                self.trained_models[model_type] = trained
                self.logger.info("%s training completed successfully.", model_type)
            else:
                self.logger.warning(
                    "%s training failed or was skipped. Continuing with remaining models.",
                    model_type,
                )

        self.logger.info(
            "Training phase complete. %d / %d model(s) available.",
            len(self.trained_models), len(self.models),
        )
        return self.trained_models

    def _train_single_model(
        self,
        model_type: str,
        data: pd.DataFrame,
        config: dict,
    ) -> object:
        """Train a single model.

        Parameters
        ----------
        model_type:
            One of ``'CTGAN'``, ``'TVAE'``, ``'CopulaGAN'``.
        data:
            Real training data.
        config:
            Keyword arguments forwarded to the engine constructor (after
            filtering to recognised parameter names).

        Returns
        -------
        object or None
            The trained engine / synthesizer object, or ``None`` on failure.
        """
        discrete_cols = [
            col for col in data.columns if data[col].dtype == 'object'
        ]

        try:
            if model_type == 'CTGAN':
                from src.engines.ctgan_engine import CTGANEngine

                engine_kwargs = self._extract_engine_kwargs(
                    config,
                    accepted_keys={
                        'embedding_dim', 'generator_dim', 'discriminator_dim',
                        'batch_size', 'epochs', 'verbose',
                    },
                )
                engine = CTGANEngine(**engine_kwargs)
                engine.fit(data, discrete_columns=discrete_cols)
                return engine

            elif model_type == 'TVAE':
                from src.engines.tvae_engine import TVAEEngine

                engine_kwargs = self._extract_engine_kwargs(
                    config,
                    accepted_keys={
                        'embedding_dim', 'compress_dims', 'decompress_dims',
                        'latent_dim', 'epochs', 'verbose',
                    },
                )
                engine = TVAEEngine(**engine_kwargs)
                engine.fit(data, discrete_columns=discrete_cols)
                return engine

            elif model_type == 'CopulaGAN':
                try:
                    from sdv.single_table import CopulaGANSynthesizer
                    from sdv.metadata import Metadata
                except ImportError:
                    self.logger.warning(
                        "CopulaGAN not available (sdv package missing or does "
                        "not include CopulaGANSynthesizer). Skipping."
                    )
                    return None

                # Build metadata identical to how CTGANEngine / TVAEEngine do it
                metadata = Metadata()
                metadata.detect_table_from_dataframe(
                    table_name='variants', data=data,
                )
                for col in discrete_cols:
                    if col in data.columns:
                        metadata.update_column(
                            table_name='variants',
                            column_name=col,
                            sdtype='categorical',
                        )
                numerical_cols = [
                    c for c in data.columns if c not in discrete_cols
                ]
                for col in numerical_cols:
                    if col in data.columns:
                        metadata.update_column(
                            table_name='variants',
                            column_name=col,
                            sdtype='numerical',
                        )

                # Extract CopulaGAN-compatible kwargs
                synth_kwargs = self._extract_engine_kwargs(
                    config,
                    accepted_keys={
                        'embedding_dim', 'generator_dim', 'discriminator_dim',
                        'batch_size', 'epochs', 'verbose',
                    },
                )
                synthesizer = CopulaGANSynthesizer(
                    metadata=metadata, **synth_kwargs,
                )
                synthesizer.fit(data)

                # Wrap in a lightweight object that exposes a .sample() method
                # consistent with our other engines so generate_from_all works.
                return _CopulaGANWrapper(synthesizer)

            else:
                self.logger.error("Unknown model type: %s", model_type)
                return None

        except Exception:
            self.logger.exception(
                "Failed to train %s model.", model_type,
            )
            return None

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    def generate_from_all(
        self, n_samples: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data from every trained model.

        Parameters
        ----------
        n_samples:
            Number of rows each model should produce.

        Returns
        -------
        dict
            Mapping of ``model_name -> pd.DataFrame`` for each trained model.
        """
        synthetic_datasets: Dict[str, pd.DataFrame] = {}

        for model_name, model_obj in self.trained_models.items():
            self.logger.info(
                "Generating %d samples from %s ...", n_samples, model_name,
            )
            try:
                if isinstance(model_obj, _CopulaGANWrapper):
                    synthetic = model_obj.sample(n_samples)
                else:
                    # CTGANEngine and TVAEEngine both expose .sample(n)
                    synthetic = model_obj.sample(n_samples)

                synthetic_datasets[model_name] = synthetic
                self.logger.info(
                    "%s generated %d rows.", model_name, len(synthetic),
                )
            except Exception:
                self.logger.exception(
                    "Generation failed for %s. Skipping.", model_name,
                )

        return synthetic_datasets

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #

    def evaluate_all_models(
        self,
        synthetic_datasets: Dict[str, pd.DataFrame],
        real_data: pd.DataFrame,
        target_column: str = 'clinical_significance',
    ) -> Dict[str, dict]:
        """Evaluate every model's synthetic output through the full pipeline.

        Each synthetic dataset is assessed for statistical quality, ML utility,
        privacy, and bias using the project's existing analysis modules.

        Parameters
        ----------
        synthetic_datasets:
            Mapping of ``model_name -> synthetic DataFrame``.
        real_data:
            The original real dataset used as the reference.
        target_column:
            Column used for ML-utility evaluation and bias detection.

        Returns
        -------
        dict
            Mapping of ``model_name`` to an evaluation dict::

                {
                    'validation': <ValidationReport.to_dict()>,
                    'privacy':    <PrivacyReport.to_dict()>,
                    'bias':       <bias analysis dict>,
                    'composite_score': float,
                }
        """
        from src.analysis.data_validator import DataValidator
        from src.analysis.privacy_analyzer import PrivacyAnalyzer
        from src.analysis.bias_detector import BiasDetector

        validator = DataValidator()
        privacy_analyzer = PrivacyAnalyzer()
        bias_detector = BiasDetector()

        results: Dict[str, dict] = {}

        for model_name, synthetic in synthetic_datasets.items():
            self.logger.info("Evaluating %s ...", model_name)

            eval_entry: dict = {
                'validation': {},
                'privacy': {},
                'bias': {},
                'composite_score': 0.0,
            }

            # --- Validation (quality + utility) ---
            try:
                validation_report = validator.validate(
                    synthetic, real_data,
                    target_column=target_column,
                    dataset_id=model_name,
                )
                eval_entry['validation'] = validation_report.to_dict()
            except Exception:
                self.logger.exception(
                    "Validation failed for %s.", model_name,
                )

            # --- Privacy ---
            try:
                privacy_report = privacy_analyzer.analyze_privacy(
                    synthetic, real_data,
                    dataset_id=model_name,
                )
                eval_entry['privacy'] = privacy_report.to_dict()
            except Exception:
                self.logger.exception(
                    "Privacy analysis failed for %s.", model_name,
                )

            # --- Bias ---
            try:
                bias_result = bias_detector.analyze_bias(synthetic, real_data)
                eval_entry['bias'] = bias_result
            except Exception:
                self.logger.exception(
                    "Bias analysis failed for %s.", model_name,
                )

            # --- Composite score ---
            eval_entry['composite_score'] = self._compute_composite_score(
                eval_entry,
            )

            results[model_name] = eval_entry
            self.logger.info(
                "%s composite score: %.4f",
                model_name, eval_entry['composite_score'],
            )

        self.evaluation_results = results
        return results

    # ------------------------------------------------------------------ #
    #  Model selection
    # ------------------------------------------------------------------ #

    def select_best_model(self) -> Tuple[str, object]:
        """Select the best model based on composite evaluation scores.

        Returns
        -------
        tuple
            ``(model_name, model_object)`` for the highest-scoring model.

        Raises
        ------
        ValueError
            If no evaluation results are available (call
            ``evaluate_all_models`` first) or no trained models match.
        """
        if not self.evaluation_results:
            raise ValueError(
                "No evaluation results available. "
                "Run evaluate_all_models() first."
            )

        best_name = max(
            self.evaluation_results,
            key=lambda name: self.evaluation_results[name].get(
                'composite_score', 0.0
            ),
        )

        best_model = self.trained_models.get(best_name)
        if best_model is None:
            raise ValueError(
                f"Best model '{best_name}' is not in trained_models. "
                "This may indicate the model was evaluated externally."
            )

        self.logger.info(
            "Best model: %s (composite score %.4f)",
            best_name,
            self.evaluation_results[best_name]['composite_score'],
        )
        return best_name, best_model

    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Rank all evaluated models by their composite score (descending).

        Returns
        -------
        list
            List of ``(model_name, composite_score)`` tuples sorted from
            best to worst.
        """
        rankings = [
            (name, result.get('composite_score', 0.0))
            for name, result in self.evaluation_results.items()
        ]
        rankings.sort(key=lambda item: item[1], reverse=True)
        return rankings

    # ------------------------------------------------------------------ #
    #  Composite score computation
    # ------------------------------------------------------------------ #

    def _compute_composite_score(self, eval_result: dict) -> float:
        """Compute a weighted composite score from evaluation results.

        The score is a weighted sum of four normalised components:

        * **quality_score** (weight 0.30) -- overall distributional quality
          from the ValidationReport.
        * **utility_score** (weight 0.30) -- ML utility (synthetic-to-real
          accuracy) from the ValidationReport.
        * **privacy_score** (weight 0.25) -- distance-based privacy metric
          from the PrivacyReport.
        * **(1 - bias_level)** (weight 0.15) -- inverse of the fraction of
          high-bias columns detected by BiasDetector.

        All individual components are clamped to [0, 1] before weighting.

        Parameters
        ----------
        eval_result:
            Dictionary with ``'validation'``, ``'privacy'``, and ``'bias'``
            sub-dicts as produced by ``evaluate_all_models``.

        Returns
        -------
        float
            Composite score in [0, 1].
        """
        weights = MODEL_SELECTION_WEIGHTS

        # --- Quality score ---
        quality_score = self._safe_get_nested(
            eval_result, 'validation', 'overall_quality_score', default=0.0,
        )

        # --- Utility score (synthetic -> real accuracy) ---
        utility_score = self._safe_get_nested(
            eval_result,
            'validation', 'utility_metrics', 'summary',
            'synthetic_to_real_accuracy',
            default=0.0,
        )

        # --- Privacy score ---
        privacy_score = self._safe_get_nested(
            eval_result, 'privacy', 'privacy_score', default=0.0,
        )

        # --- Bias level (fraction of columns with high bias) ---
        bias_level = self._compute_bias_level(eval_result.get('bias', {}))

        # Clamp all components to [0, 1]
        quality_score = float(np.clip(quality_score, 0.0, 1.0))
        utility_score = float(np.clip(utility_score, 0.0, 1.0))
        privacy_score = float(np.clip(privacy_score, 0.0, 1.0))
        bias_level = float(np.clip(bias_level, 0.0, 1.0))

        composite = (
            weights.get('quality_score', 0.30) * quality_score
            + weights.get('utility_score', 0.30) * utility_score
            + weights.get('privacy_score', 0.25) * privacy_score
            + weights.get('bias_score', 0.15) * (1.0 - bias_level)
        )

        composite = float(np.clip(composite, 0.0, 1.0))

        self.logger.debug(
            "Composite breakdown: quality=%.3f, utility=%.3f, "
            "privacy=%.3f, bias_level=%.3f -> composite=%.4f",
            quality_score, utility_score, privacy_score, bias_level, composite,
        )

        return composite

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_engine_kwargs(config: dict, accepted_keys: set) -> dict:
        """Filter *config* to only keys recognised by the target engine."""
        return {k: v for k, v in config.items() if k in accepted_keys}

    @staticmethod
    def _safe_get_nested(d: dict, *keys, default=None):
        """Safely traverse nested dicts.

        Example::

            _safe_get_nested(d, 'a', 'b', 'c', default=0.0)
            # equivalent to d.get('a', {}).get('b', {}).get('c', 0.0)
        """
        current = d
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return default
            if current is None:
                return default
        return current

    @staticmethod
    def _compute_bias_level(bias_result: dict) -> float:
        """Derive a scalar bias level in [0, 1] from a bias analysis dict.

        The bias level is the fraction of analysed columns whose JSD-based
        status is ``'high'``.  If no feature distribution data is available
        the level defaults to 0.0 (optimistic -- no evidence of bias).
        """
        feat_dist = bias_result.get('feature_distributions', {})
        if not feat_dist:
            return 0.0

        total = len(feat_dist)
        high_count = sum(
            1 for v in feat_dist.values()
            if isinstance(v, dict) and v.get('status') == 'high'
        )
        return high_count / total if total > 0 else 0.0


# ---------------------------------------------------------------------- #
#  Lightweight CopulaGAN wrapper
# ---------------------------------------------------------------------- #

class _CopulaGANWrapper:
    """Thin wrapper around ``CopulaGANSynthesizer`` to expose a ``.sample()``
    interface consistent with CTGANEngine / TVAEEngine."""

    def __init__(self, synthesizer):
        self._synthesizer = synthesizer

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generate *n_samples* synthetic rows."""
        return self._synthesizer.sample(num_rows=n_samples)

    def get_config(self) -> dict:
        """Return a basic configuration descriptor."""
        return {'model_type': 'CopulaGAN'}
