"""Experiment Planner module for designing and validating optimization experiments.

This module provides the ExperimentPlanner class which takes diagnostic
recommendations and translates them into concrete, validated experiment
configurations.  It tracks past configurations to avoid redundant runs
and predicts which metrics each planned change is likely to improve.
"""

import copy
import json
import logging
from typing import Dict, List, Optional

from src.ai_diagnostic_agent.models import ExperimentPlan, Recommendation
from src.ai_diagnostic_agent.config import HYPERPARAMETER_SEARCH_SPACES

logger = logging.getLogger(__name__)

# ── Default baseline configuration ──────────────────────────────────────────

DEFAULT_CONFIG: Dict = {
    "model_type": "CTGAN",
    "epochs": 300,
    "batch_size": 500,
    "embedding_dim": 128,
    "n_samples": 1000,
    "random_seed": 42,
    "preprocessing": {},
    "post_processing": {},
}

# ── Global validation constraints (broader than per-model search spaces) ────

_GLOBAL_CONSTRAINTS = {
    "epochs": (50, 1000),
    "batch_size": (50, 2000),
    "embedding_dim": (32, 512),
    "n_samples": (100, 100000),
}

_LEARNING_RATE_KEYS = {
    "generator_lr",
    "discriminator_lr",
    "learning_rate",
}

_LEARNING_RATE_RANGE = (1e-6, 1e-2)

_VALID_MODEL_TYPES = {"CTGAN", "TVAE", "CopulaGAN"}

# ── Heuristic improvement deltas keyed by recommendation category ───────────

_IMPROVEMENT_HEURISTICS: Dict[str, Dict[str, float]] = {
    "preprocessing": {"quality_score": 0.05, "mean_jsd": -0.01},
    "model": {"quality_score": 0.10},
    "training": {"quality_score": 0.05, "ml_accuracy": 0.05},
    "post_processing": {"privacy_score": 0.05},
}


class ExperimentPlanner:
    """Design, validate, and document optimisation experiments.

    The planner consumes a ranked list of ``Recommendation`` objects produced
    by earlier diagnostic stages and converts them into a fully specified
    ``ExperimentPlan``.  It also maintains a lightweight history of
    previously-tried configurations so that the same experiment is never
    repeated.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        """Initialize experiment planner."""
        self._experiment_history: List[dict] = []  # Track past configs to avoid repeats

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def plan_next_experiment(
        self,
        recommendations: List[Recommendation],
        current_config: Optional[dict] = None,
    ) -> ExperimentPlan:
        """Design next experiment configuration based on recommendations.

        Apply top-priority recommendations to *current_config* (or the
        default baseline when ``None``).  The resulting configuration is
        validated and, if it duplicates a previous experiment, a warning is
        logged but the plan is still returned so callers can decide how to
        proceed.

        Parameters
        ----------
        recommendations:
            Ranked list of ``Recommendation`` objects.  They will be sorted
            internally by ``priority`` (ascending = highest priority first).
        current_config:
            The configuration used in the most recent experiment.  When
            ``None``, ``DEFAULT_CONFIG`` is used as the starting point.

        Returns
        -------
        ExperimentPlan
            A fully populated plan ready for execution.
        """
        base_config = copy.deepcopy(current_config) if current_config is not None else copy.deepcopy(DEFAULT_CONFIG)

        # Sort recommendations by priority (1 = highest)
        sorted_recs = sorted(recommendations, key=lambda r: r.priority)

        # Apply recommendations and capture the new configuration
        new_config = self.apply_recommendations(copy.deepcopy(base_config), sorted_recs)

        # Validate -- clamp or reject values that fall outside constraints
        is_valid = self.validate_configuration(new_config)
        if not is_valid:
            logger.warning(
                "Planned configuration failed validation.  Falling back to "
                "clamped values where possible."
            )
            new_config = self._clamp_configuration(new_config)

        # Detect repeats
        if self._is_repeated_config(new_config):
            logger.warning(
                "Planned configuration has already been tried in a previous "
                "experiment.  Consider diversifying recommendations."
            )

        # Build configuration_changes delta
        configuration_changes = self._compute_changes(base_config, new_config)

        # Assemble the plan
        plan = ExperimentPlan(
            parent_experiment_id=current_config.get("experiment_id") if current_config else None,
            configuration_changes=configuration_changes,
            applied_recommendations=list(sorted_recs),
            rationale=self._build_rationale(sorted_recs),
        )

        # Predict metric improvements
        plan.predicted_improvements = self.predict_improvements(plan)

        # Record configuration for duplicate detection
        self.record_experiment(new_config)

        logger.info("Experiment plan %s created with %d recommendation(s).",
                     plan.plan_id, len(sorted_recs))
        return plan

    def apply_recommendations(
        self,
        config: dict,
        recommendations: List[Recommendation],
    ) -> dict:
        """Apply recommendations to configuration.

        For each recommendation (sorted by priority), the
        ``recommendation.implementation`` dict is merged into *config*.
        Keys whose values are dicts (e.g. ``preprocessing``,
        ``post_processing``) are merged recursively rather than replaced
        outright.

        Parameters
        ----------
        config:
            The mutable configuration dict to update.
        recommendations:
            Recommendations sorted by priority.

        Returns
        -------
        dict
            The updated *config* (same object, mutated in-place).
        """
        sorted_recs = sorted(recommendations, key=lambda r: r.priority)

        for rec in sorted_recs:
            impl = rec.implementation
            if not impl:
                continue

            for key, value in impl.items():
                # For nested sub-dicts (preprocessing / post_processing) we
                # merge rather than overwrite so that independent
                # recommendations can each contribute settings.
                if key in ("preprocessing", "post_processing") and isinstance(value, dict):
                    if key not in config or not isinstance(config[key], dict):
                        config[key] = {}
                    config[key].update(value)
                else:
                    config[key] = value

        return config

    def validate_configuration(self, config: dict) -> bool:
        """Validate configuration against system constraints.

        Checks
        ------
        * ``epochs`` in [50, 1000]
        * ``batch_size`` in [50, 2000]
        * ``embedding_dim`` in [32, 512]
        * All learning-rate keys in [1e-6, 1e-2]
        * ``n_samples`` in [100, 100000]
        * ``model_type`` in {CTGAN, TVAE, CopulaGAN}

        Returns
        -------
        bool
            ``True`` when every check passes, ``False`` otherwise.
        """
        # -- model_type -------------------------------------------------------
        model_type = config.get("model_type", "CTGAN")
        if model_type not in _VALID_MODEL_TYPES:
            logger.warning("Invalid model_type '%s'. Must be one of %s.",
                           model_type, _VALID_MODEL_TYPES)
            return False

        # -- numeric range checks --------------------------------------------
        for param, (lo, hi) in _GLOBAL_CONSTRAINTS.items():
            value = config.get(param)
            if value is not None:
                if not (lo <= value <= hi):
                    logger.warning(
                        "Parameter '%s' value %s is outside valid range [%s, %s].",
                        param, value, lo, hi,
                    )
                    return False

        # -- learning-rate checks --------------------------------------------
        lr_lo, lr_hi = _LEARNING_RATE_RANGE
        for lr_key in _LEARNING_RATE_KEYS:
            value = config.get(lr_key)
            if value is not None:
                if not (lr_lo <= value <= lr_hi):
                    logger.warning(
                        "Learning rate '%s' value %s is outside valid range [%s, %s].",
                        lr_key, value, lr_lo, lr_hi,
                    )
                    return False

        return True

    def predict_improvements(self, plan: ExperimentPlan) -> Dict[str, float]:
        """Predict which metrics should improve based on planned changes.

        Uses heuristic deltas keyed by ``Recommendation.category``.  When
        multiple recommendations target the same metric the predicted
        improvements are summed.

        Parameters
        ----------
        plan:
            An ``ExperimentPlan`` whose ``applied_recommendations`` list has
            already been populated.

        Returns
        -------
        dict
            Mapping of metric name to predicted delta (positive = improvement
            for higher-is-better metrics, negative = improvement for
            lower-is-better metrics such as ``mean_jsd``).
        """
        predictions: Dict[str, float] = {}

        for rec in plan.applied_recommendations:
            category = rec.category
            heuristics = _IMPROVEMENT_HEURISTICS.get(category, {})
            for metric, delta in heuristics.items():
                predictions[metric] = predictions.get(metric, 0.0) + delta

        # Round to avoid floating-point noise
        predictions = {k: round(v, 4) for k, v in predictions.items()}
        return predictions

    def generate_plan_document(self, plan: ExperimentPlan) -> str:
        """Generate a human-readable experiment plan document.

        Parameters
        ----------
        plan:
            A fully populated ``ExperimentPlan``.

        Returns
        -------
        str
            Structured multi-line text suitable for logging or display.
        """
        lines: List[str] = []
        lines.append("=== Experiment Plan ===")
        lines.append(f"Plan ID: {plan.plan_id}")
        lines.append(
            f"Based on: {plan.parent_experiment_id or 'Initial'}"
        )

        # -- Changes ---------------------------------------------------------
        lines.append("Changes:")
        if plan.configuration_changes:
            for key, change in plan.configuration_changes.items():
                # Each change entry may be a simple value or a dict with
                # 'old' / 'new' keys depending on how _compute_changes
                # stored it.
                if isinstance(change, dict) and "old" in change and "new" in change:
                    rationale = self._rationale_for_key(key, plan.applied_recommendations)
                    lines.append(
                        f"  - {key}: {change['old']} -> {change['new']}"
                        f" ({rationale})"
                    )
                else:
                    rationale = self._rationale_for_key(key, plan.applied_recommendations)
                    lines.append(f"  - {key}: {change} ({rationale})")
        else:
            lines.append("  - (no changes)")

        # -- Expected Improvements -------------------------------------------
        lines.append("Expected Improvements:")
        if plan.predicted_improvements:
            for metric, delta in plan.predicted_improvements.items():
                sign = "+" if delta >= 0 else ""
                lines.append(f"  - {metric}: {sign}{delta}")
        else:
            lines.append("  - (none predicted)")

        # -- Rationale -------------------------------------------------------
        lines.append(f"Rationale: {plan.rationale}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  History management
    # ------------------------------------------------------------------ #

    def _is_repeated_config(self, config: dict) -> bool:
        """Check if this configuration was already tried.

        Comparison is performed on a normalised JSON serialisation of the
        config so that ordering differences do not produce false negatives.
        """
        normalised = self._normalise_config(config)
        for past in self._experiment_history:
            if past == normalised:
                return True
        return False

    def record_experiment(self, config: dict) -> None:
        """Record a config in the history to avoid repeats."""
        self._experiment_history.append(self._normalise_config(config))

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_config(config: dict) -> dict:
        """Return a JSON-round-tripped copy with sorted keys for comparison."""
        return json.loads(json.dumps(config, sort_keys=True, default=str))

    @staticmethod
    def _compute_changes(old: dict, new: dict) -> Dict:
        """Compute a delta dict between *old* and *new* configurations.

        Returns a dict whose keys are only those that differ, with each
        value being ``{"old": <old_val>, "new": <new_val>}``.
        """
        changes: Dict = {}
        all_keys = set(old.keys()) | set(new.keys())
        for key in sorted(all_keys):
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                changes[key] = {"old": old_val, "new": new_val}
        return changes

    @staticmethod
    def _build_rationale(recommendations: List[Recommendation]) -> str:
        """Build a combined rationale string from applied recommendations."""
        if not recommendations:
            return "No recommendations applied."

        parts: List[str] = []
        for rec in recommendations:
            summary = rec.rationale if rec.rationale else rec.description
            if summary:
                parts.append(f"[P{rec.priority}] {rec.title}: {summary}")
            else:
                parts.append(f"[P{rec.priority}] {rec.title}")
        return " | ".join(parts)

    @staticmethod
    def _rationale_for_key(
        key: str, recommendations: List[Recommendation]
    ) -> str:
        """Find the rationale snippet associated with a config key change."""
        for rec in recommendations:
            if key in rec.implementation:
                return rec.title or rec.description or "recommendation applied"
        return "adjusted by planner"

    def _clamp_configuration(self, config: dict) -> dict:
        """Clamp numeric parameters to their valid ranges in-place.

        This is a best-effort fallback used when ``validate_configuration``
        reports failure.  After clamping, the config should pass
        validation.
        """
        # model_type
        if config.get("model_type") not in _VALID_MODEL_TYPES:
            logger.info("Resetting invalid model_type to 'CTGAN'.")
            config["model_type"] = "CTGAN"

        # Global numeric constraints
        for param, (lo, hi) in _GLOBAL_CONSTRAINTS.items():
            value = config.get(param)
            if value is not None:
                clamped = max(lo, min(hi, value))
                if clamped != value:
                    logger.info("Clamped '%s' from %s to %s.", param, value, clamped)
                    config[param] = clamped

        # Learning rates
        lr_lo, lr_hi = _LEARNING_RATE_RANGE
        for lr_key in _LEARNING_RATE_KEYS:
            value = config.get(lr_key)
            if value is not None:
                clamped = max(lr_lo, min(lr_hi, value))
                if clamped != value:
                    logger.info("Clamped '%s' from %s to %s.", lr_key, value, clamped)
                    config[lr_key] = clamped

        return config
