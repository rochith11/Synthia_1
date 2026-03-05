"""Hyperparameter optimization module for synthetic data generation models.

Supports random search and Bayesian optimization (via Optuna) over
configurable hyperparameter search spaces defined in the project config.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.ai_diagnostic_agent.models import Experiment
from src.ai_diagnostic_agent.config import (
    HYPERPARAMETER_SEARCH_SPACES,
    OPTIMIZATION_OBJECTIVE_WEIGHTS,
)


class HyperparameterOptimizer:
    """Hyperparameter optimizer supporting random search and Bayesian optimization.

    Parameters
    ----------
    optimization_method : str
        Optimization strategy to use. ``'random'`` for random search,
        ``'bayesian'`` for Optuna-based Bayesian optimization. If Optuna is
        not installed the optimizer silently falls back to random search.
    """

    def __init__(self, optimization_method: str = 'random'):
        self.method = optimization_method
        self.history: List[Tuple[dict, float]] = []
        self.best_config: Optional[dict] = None
        self.best_score: float = float('-inf')
        self.logger = logging.getLogger(__name__)
        self._optuna_available = False

        try:
            import optuna  # noqa: F401
            self._optuna_available = True
        except ImportError:
            self.logger.info("Optuna not available, using random search")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def define_search_space(self, model_type: str) -> dict:
        """Return the hyperparameter search space for *model_type*.

        Looks up ``HYPERPARAMETER_SEARCH_SPACES[model_type]``. If the
        requested model type is not found the default CTGAN space is
        returned.

        Parameters
        ----------
        model_type : str
            Name of the generative model (e.g. ``'CTGAN'``, ``'TVAE'``,
            ``'CopulaGAN'``).

        Returns
        -------
        dict
            Mapping of parameter names to ``(low, high)`` tuples.
        """
        if model_type in HYPERPARAMETER_SEARCH_SPACES:
            return HYPERPARAMETER_SEARCH_SPACES[model_type]

        self.logger.warning(
            "Model type '%s' not found in search spaces. "
            "Falling back to CTGAN defaults.",
            model_type,
        )
        return HYPERPARAMETER_SEARCH_SPACES.get('CTGAN', {})

    def optimize(
        self,
        data: Any,
        model_type: str = 'CTGAN',
        n_trials: int = 10,
        train_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
    ) -> dict:
        """Run hyperparameter optimization.

        Parameters
        ----------
        data : Any
            Training data passed through to *train_fn* and *evaluate_fn*.
        model_type : str
            Generative model type whose search space should be used.
        n_trials : int
            Number of hyperparameter configurations to evaluate.
        train_fn : callable(data, config) -> model, optional
            Function that trains a model given *data* and a hyperparameter
            *config* dict.  When ``None`` a placeholder that returns
            ``None`` is used.
        evaluate_fn : callable(model, data) -> float, optional
            Function that evaluates a trained *model* against *data* and
            returns an objective score.  When ``None`` a placeholder that
            returns a random score is used.

        Returns
        -------
        dict
            Dictionary with keys ``best_config``, ``best_score``,
            ``n_trials``, and ``history``.
        """
        search_space = self.define_search_space(model_type)

        # Provide placeholder callables when the caller does not supply them
        if train_fn is None:
            def train_fn(d, cfg):  # type: ignore[misc]
                return None

        if evaluate_fn is None:
            def evaluate_fn(model, d):  # type: ignore[misc]
                return float(np.random.uniform(0.0, 1.0))

        self.logger.info(
            "Starting %s optimization for %s with %d trials",
            self.method, model_type, n_trials,
        )

        # Dispatch to the chosen strategy
        if self.method == 'bayesian' and self._optuna_available:
            result = self._bayesian_search(
                search_space, n_trials, train_fn, evaluate_fn, data, model_type,
            )
        else:
            if self.method == 'bayesian' and not self._optuna_available:
                self.logger.warning(
                    "Bayesian search requested but Optuna is unavailable. "
                    "Falling back to random search."
                )
            result = self._random_search(
                search_space, n_trials, train_fn, evaluate_fn, data, model_type,
            )

        self.logger.info(
            "Optimization complete. Best score: %.4f", result['best_score'],
        )
        return result

    def evaluate_configuration(
        self,
        config: dict,
        train_fn: Callable,
        evaluate_fn: Callable,
        data: Any,
    ) -> float:
        """Train a model with *config* and return the objective score.

        Parameters
        ----------
        config : dict
            Hyperparameter configuration to evaluate.
        train_fn : callable
            ``train_fn(data, config) -> model``.
        evaluate_fn : callable
            ``evaluate_fn(model, data) -> float``.
        data : Any
            Training / evaluation data.

        Returns
        -------
        float
            Objective score, or ``-inf`` if training or evaluation fails.
        """
        try:
            model = train_fn(data, config)
            score = evaluate_fn(model, data)
            return float(score)
        except Exception as exc:
            self.logger.error(
                "Configuration evaluation failed: %s", exc,
            )
            return float('-inf')

    def get_best_config(self) -> dict:
        """Return the best hyperparameter configuration found so far.

        Returns
        -------
        dict
            Best configuration, or an empty dict if no optimisation has
            been run yet.
        """
        if self.best_config is not None:
            return self.best_config
        return {}

    def analyze_importance(self) -> dict:
        """Estimate hyperparameter importance from the recorded history.

        Computes the absolute Pearson correlation between each
        hyperparameter value and the objective score across all evaluated
        configurations.

        Returns
        -------
        dict
            Mapping of parameter name to importance score (absolute
            correlation).  Parameters with zero variance or insufficient
            data receive an importance of ``0.0``.
        """
        if len(self.history) < 2:
            self.logger.warning(
                "Not enough history to analyze importance (need >= 2 entries)."
            )
            return {}

        configs = [cfg for cfg, _ in self.history]
        scores = np.array([score for _, score in self.history])

        # Gather the union of all parameter names
        param_names = sorted({k for cfg in configs for k in cfg})

        importance: Dict[str, float] = {}
        for param in param_names:
            values = np.array([
                cfg.get(param, np.nan) for cfg in configs
            ], dtype=float)

            # Skip if any value is missing or if there is zero variance
            if np.any(np.isnan(values)):
                importance[param] = 0.0
                continue
            if np.std(values) == 0 or np.std(scores) == 0:
                importance[param] = 0.0
                continue

            correlation = np.corrcoef(values, scores)[0, 1]
            importance[param] = float(abs(correlation))

        return importance

    def get_optimization_history(self) -> list:
        """Return the full optimization history.

        Returns
        -------
        list of (dict, float)
            Each entry is a ``(config, score)`` tuple.
        """
        return list(self.history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_search(
        self,
        search_space: dict,
        n_trials: int,
        train_fn: Callable,
        evaluate_fn: Callable,
        data: Any,
        model_type: str,
    ) -> dict:
        """Perform random search over the hyperparameter space.

        For each trial a random configuration is sampled, evaluated, and
        recorded. The best configuration seen across all trials is tracked.

        Returns
        -------
        dict
            Optimization result with ``best_config``, ``best_score``,
            ``n_trials``, and ``history``.
        """
        for trial_idx in range(n_trials):
            config = self._sample_random_config(search_space)
            score = self.evaluate_configuration(
                config, train_fn, evaluate_fn, data,
            )

            self.history.append((config, score))

            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                self.logger.info(
                    "Trial %d/%d - new best score: %.4f",
                    trial_idx + 1, n_trials, score,
                )
            else:
                self.logger.debug(
                    "Trial %d/%d - score: %.4f (best: %.4f)",
                    trial_idx + 1, n_trials, score, self.best_score,
                )

        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'n_trials': n_trials,
            'history': list(self.history),
        }

    def _bayesian_search(
        self,
        search_space: dict,
        n_trials: int,
        train_fn: Callable,
        evaluate_fn: Callable,
        data: Any,
        model_type: str,
    ) -> dict:
        """Perform Bayesian optimization using Optuna.

        An Optuna study is created that maximises the objective score. Each
        trial suggests values according to the search space definition and
        evaluates the resulting configuration.

        Returns
        -------
        dict
            Optimization result with ``best_config``, ``best_score``,
            ``n_trials``, and ``history``.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        optimizer_ref = self  # capture for the closure

        def objective(trial: optuna.Trial) -> float:
            config: Dict[str, Any] = {}
            for param, (low, high) in search_space.items():
                if isinstance(low, float) and low < 1:
                    config[param] = trial.suggest_float(param, low, high, log=True)
                elif isinstance(low, int):
                    config[param] = trial.suggest_int(param, low, high)
                else:
                    config[param] = trial.suggest_float(param, low, high)

            score = optimizer_ref.evaluate_configuration(
                config, train_fn, evaluate_fn, data,
            )

            # Keep the shared history / best tracking in sync
            optimizer_ref.history.append((config, score))
            if score > optimizer_ref.best_score:
                optimizer_ref.best_score = score
                optimizer_ref.best_config = config

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Ensure best state is consistent with Optuna's records
        if study.best_trial is not None:
            self.best_config = study.best_params
            self.best_score = study.best_value

        self.logger.info(
            "Bayesian optimization finished. Best value: %.4f",
            self.best_score,
        )

        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'n_trials': n_trials,
            'history': list(self.history),
        }

    def _sample_random_config(self, search_space: dict) -> dict:
        """Sample a random configuration from *search_space*.

        Sampling rules:
        - **Integer parameters** (``epochs``, ``batch_size``,
          ``embedding_dim``, ``pac``, ``latent_dim``): uniform integer
          between *low* and *high* inclusive.
        - **Float parameters** with values < 1 (learning rates):
          log-uniform sample in ``[low, high]``.
        - **Other float parameters**: uniform sample.

        Parameters
        ----------
        search_space : dict
            Mapping of parameter names to ``(low, high)`` tuples.

        Returns
        -------
        dict
            Sampled hyperparameter configuration.
        """
        config: Dict[str, Any] = {}
        for param, (low, high) in search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                config[param] = int(np.random.randint(low, high + 1))
            elif isinstance(low, float) and low < 1:
                # Log-uniform sampling for small-valued float parameters
                config[param] = float(
                    np.exp(np.random.uniform(np.log(low), np.log(high)))
                )
            else:
                config[param] = float(np.random.uniform(low, high))
        return config
