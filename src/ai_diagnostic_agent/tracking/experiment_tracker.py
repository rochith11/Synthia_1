"""Experiment Tracker module for logging, retrieving, comparing, and
analysing experiment runs persisted as individual JSON files."""

import json
import os
import uuid
from typing import Dict, List, Optional, Tuple

from src.ai_diagnostic_agent.models import Experiment, ExperimentStatus


class ExperimentTracker:
    """Track, persist, and analyse experiment runs.

    Each experiment is saved as a separate JSON file named
    ``{experiment_id}.json`` inside the configured *storage_path* directory.
    """

    # ------------------------------------------------------------------
    # Initialisation & persistence helpers
    # ------------------------------------------------------------------

    def __init__(self, storage_path: str = "data/experiments"):
        """Initialize experiment tracker with storage location.

        Creates the directory if it doesn't exist and loads any previously
        persisted experiments from storage.
        """
        self.storage_path = storage_path
        self._experiments: Dict[str, Experiment] = {}
        os.makedirs(self.storage_path, exist_ok=True)
        self._load_experiments()

    def _load_experiments(self):
        """Load all experiments from JSON files in the storage directory."""
        if not os.path.isdir(self.storage_path):
            return
        for filename in os.listdir(self.storage_path):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.storage_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                experiment = Experiment.from_dict(data)
                self._experiments[experiment.experiment_id] = experiment
            except (json.JSONDecodeError, TypeError, KeyError):
                # Skip malformed files gracefully.
                continue

    def _save_experiment(self, experiment: Experiment):
        """Save a single experiment to a JSON file."""
        filepath = os.path.join(
            self.storage_path, f"{experiment.experiment_id}.json"
        )
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(experiment.to_dict(), fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Core CRUD operations
    # ------------------------------------------------------------------

    def log_experiment(self, experiment: Experiment) -> str:
        """Log an experiment and return its unique ID.

        If the experiment's ID already exists in the tracker a new unique ID
        is generated so that every stored experiment is guaranteed to have a
        distinct identifier.
        """
        if experiment.experiment_id in self._experiments:
            experiment.experiment_id = str(uuid.uuid4())

        self._experiments[experiment.experiment_id] = experiment
        self._save_experiment(experiment)
        return experiment.experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve a single experiment by its ID, or *None* if not found."""
        return self._experiments.get(experiment_id)

    # ------------------------------------------------------------------
    # Listing & filtering
    # ------------------------------------------------------------------

    def list_experiments(self, filters: dict = None) -> List[Experiment]:
        """List experiments with optional filtering.

        Supported filter keys:
        - ``model_type``  -- exact match on ``experiment.model_type``
        - ``status``      -- exact match on ``experiment.status``
        - ``min_quality_score`` -- only experiments whose
          ``metrics.get('quality_score', 0)`` is >= the given threshold

        Results are sorted by timestamp descending (most recent first).
        """
        results: List[Experiment] = list(self._experiments.values())

        if filters:
            if "model_type" in filters:
                results = [
                    e for e in results
                    if e.model_type == filters["model_type"]
                ]
            if "status" in filters:
                results = [
                    e for e in results
                    if e.status == filters["status"]
                ]
            if "min_quality_score" in filters:
                threshold = filters["min_quality_score"]
                results = [
                    e for e in results
                    if e.metrics.get("quality_score", 0) >= threshold
                ]

        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Comparison & analysis
    # ------------------------------------------------------------------

    def compare_experiments(self, exp_ids: List[str]) -> Dict:
        """Compare multiple experiments.

        Returns a dictionary containing:
        - ``experiments``     -- list of experiment dicts for the requested IDs
        - ``metric_deltas``   -- pairwise metric differences between
          consecutive experiments (in the order supplied)
        - ``best_by_metric``  -- mapping of each metric name to the
          experiment_id that achieved the highest value
        """
        experiments: List[Experiment] = []
        for eid in exp_ids:
            exp = self.get_experiment(eid)
            if exp is not None:
                experiments.append(exp)

        experiment_dicts = [e.to_dict() for e in experiments]

        # --- Pairwise deltas between consecutive experiments -------------
        metric_deltas: List[Dict[str, float]] = []
        for i in range(1, len(experiments)):
            prev_metrics = experiments[i - 1].metrics
            curr_metrics = experiments[i].metrics
            all_keys = set(prev_metrics.keys()) | set(curr_metrics.keys())
            delta: Dict[str, float] = {}
            for key in all_keys:
                prev_val = prev_metrics.get(key, 0.0)
                curr_val = curr_metrics.get(key, 0.0)
                delta[key] = curr_val - prev_val
            metric_deltas.append(delta)

        # --- Best experiment per metric ----------------------------------
        best_by_metric: Dict[str, str] = {}
        all_metric_names: set = set()
        for exp in experiments:
            all_metric_names.update(exp.metrics.keys())

        for metric_name in all_metric_names:
            best_id: Optional[str] = None
            best_val: Optional[float] = None
            for exp in experiments:
                val = exp.metrics.get(metric_name)
                if val is not None and (best_val is None or val > best_val):
                    best_val = val
                    best_id = exp.experiment_id
            if best_id is not None:
                best_by_metric[metric_name] = best_id

        return {
            "experiments": experiment_dicts,
            "metric_deltas": metric_deltas,
            "best_by_metric": best_by_metric,
        }

    def get_best_experiment(
        self, metric: str = "quality_score"
    ) -> Optional[Experiment]:
        """Return the experiment with the highest value for *metric*.

        Only experiments that actually contain the requested metric in their
        ``metrics`` dictionary are considered.  Returns *None* when no
        experiment has the metric.
        """
        best_exp: Optional[Experiment] = None
        best_val: Optional[float] = None
        for exp in self._experiments.values():
            val = exp.metrics.get(metric)
            if val is not None and (best_val is None or val > best_val):
                best_val = val
                best_exp = exp
        return best_exp

    def get_latest_experiment(self) -> Optional[Experiment]:
        """Return the most recent experiment by timestamp."""
        if not self._experiments:
            return None
        return max(self._experiments.values(), key=lambda e: e.timestamp)

    # ------------------------------------------------------------------
    # Counts & metadata
    # ------------------------------------------------------------------

    def get_experiment_count(self) -> int:
        """Return the total number of tracked experiments."""
        return len(self._experiments)

    # ------------------------------------------------------------------
    # Rollback & export
    # ------------------------------------------------------------------

    def rollback_config(self, experiment_id: str) -> dict:
        """Retrieve the full configuration from a historical experiment.

        Returns a dictionary combining the experiment's preprocessing,
        model, and training configurations so they can be re-applied.
        Raises ``ValueError`` if the experiment ID is not found.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(
                f"Experiment '{experiment_id}' not found. "
                "Cannot rollback to a non-existent experiment."
            )
        return {
            "preprocessing_config": experiment.preprocessing_config,
            "model_type": experiment.model_type,
            "model_config": experiment.model_config,
            "training_config": experiment.training_config,
        }

    def export_history(self, format: str = "json") -> str:
        """Export the full experiment history as a JSON string.

        Experiments are ordered by timestamp (oldest first).
        """
        experiments = sorted(
            self._experiments.values(), key=lambda e: e.timestamp
        )
        history = [e.to_dict() for e in experiments]
        return json.dumps(history, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Metric trends
    # ------------------------------------------------------------------

    def get_metric_trends(self, metric: str) -> List[Tuple[str, float]]:
        """Get metric values across all experiments in chronological order.

        Returns a list of ``(experiment_id, metric_value)`` tuples for every
        experiment that contains the requested *metric*.  The list is sorted
        by experiment timestamp ascending.
        """
        relevant = [
            exp
            for exp in self._experiments.values()
            if metric in exp.metrics
        ]
        relevant.sort(key=lambda e: e.timestamp)
        return [(exp.experiment_id, exp.metrics[metric]) for exp in relevant]
