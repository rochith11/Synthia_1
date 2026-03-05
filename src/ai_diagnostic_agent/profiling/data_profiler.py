"""
Data Profiler Module
====================
Provides comprehensive profiling of tabular datasets, including column type
detection, descriptive statistics, missing-value analysis, rare-value
detection, correlation computation, outlier detection (IQR), and
transformation recommendations.

Usage:
    profiler = DataProfiler()
    profile  = profiler.profile_dataset(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.ai_diagnostic_agent.models import (
    ColumnStatistics,
    DataProfile,
    Transformation,
)

logger = logging.getLogger(__name__)


class DataProfiler:
    """Analyse and profile a pandas DataFrame comprehensively.

    The profiler runs a battery of analyses on the input data and packages the
    results into a :class:`DataProfile` dataclass that downstream components
    (diagnostics, optimisation planner, etc.) can consume directly.
    """

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def profile_dataset(self, data: pd.DataFrame) -> DataProfile:
        """Run the full profiling pipeline on *data* and return a
        :class:`DataProfile` instance.

        Parameters
        ----------
        data:
            The DataFrame to profile.  Must contain at least one column.

        Returns
        -------
        DataProfile
            A populated profile object with statistics, missing-value info,
            correlations, outliers, and recommended transformations.
        """
        if data.empty:
            logger.warning("Received an empty DataFrame for profiling.")
            return DataProfile(n_rows=0, n_columns=len(data.columns))

        logger.info(
            "Profiling dataset with %d rows and %d columns.",
            len(data),
            len(data.columns),
        )

        column_types = self.detect_column_types(data)
        statistics = self.compute_statistics(data, column_types)
        missing_values = self.detect_missing_values(data)
        rare_values = self.detect_rare_values(data)
        correlations = self.compute_correlations(data)
        outliers = self.detect_outliers(data)

        profile = DataProfile(
            n_rows=len(data),
            n_columns=len(data.columns),
            column_types=column_types,
            statistics=statistics,
            missing_values=missing_values,
            rare_values=rare_values,
            correlations=correlations,
            outliers=outliers,
        )

        profile.recommended_transformations = self.recommend_transformations(
            profile
        )

        logger.info("Profiling complete. Profile ID: %s", profile.dataset_id)
        return profile

    # ------------------------------------------------------------------
    # Column type detection
    # ------------------------------------------------------------------

    def detect_column_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Classify every column as *identifier*, *continuous*, *ordinal*, or
        *categorical* using simple heuristics.

        Heuristics
        ----------
        1. If the column name contains ``'id'`` (case-insensitive) **or** every
           value in the column is unique, it is labelled ``identifier``.
        2. If the dtype is numeric and the number of unique (non-null) values
           exceeds 20, it is labelled ``continuous``.
        3. If the dtype is numeric and the number of unique (non-null) values is
           at most 20, it is labelled ``ordinal``.
        4. Everything else is labelled ``categorical``.

        Parameters
        ----------
        data:
            The DataFrame to inspect.

        Returns
        -------
        dict
            Mapping of column name to detected type string.
        """
        column_types: Dict[str, str] = {}

        for col in data.columns:
            series = data[col]
            n_unique = series.nunique(dropna=True)
            n_total = len(series.dropna())

            # Numeric checks come first so that continuous floats with
            # high uniqueness are not misclassified as identifiers.
            if pd.api.types.is_numeric_dtype(series):
                # Rule 2 -- continuous
                if n_unique > 20:
                    column_types[col] = "continuous"
                # Rule 3 -- ordinal
                else:
                    column_types[col] = "ordinal"
                continue

            # Rule 1 -- identifier (non-numeric columns only)
            if "id" in str(col).lower() or (n_total > 0 and n_unique == n_total):
                column_types[col] = "identifier"
                continue

            # Rule 4 -- categorical
            column_types[col] = "categorical"

        return column_types

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        data: pd.DataFrame,
        column_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ColumnStatistics]:
        """Compute descriptive statistics for every column.

        For numeric columns the following are calculated: mean, median,
        standard deviation, min, max, skewness (via
        :func:`scipy.stats.skew`), and kurtosis (via
        :func:`scipy.stats.kurtosis`).

        For all columns: unique count, null count, null percentage, and the
        top-5 most-common values with their frequencies.

        Parameters
        ----------
        data:
            The DataFrame to analyse.
        column_types:
            Optional pre-computed column type mapping.  If *None*, types will
            be detected automatically.

        Returns
        -------
        dict
            Mapping of column name to :class:`ColumnStatistics`.
        """
        if column_types is None:
            column_types = self.detect_column_types(data)

        stats: Dict[str, ColumnStatistics] = {}

        for col in data.columns:
            series = data[col]
            col_type = column_types.get(col, "categorical")
            n_null = int(series.isnull().sum())
            n_total = len(series)
            null_pct = (n_null / n_total * 100.0) if n_total > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))

            # Most common values (top 5)
            most_common: List[Tuple[Any, int]] = []
            try:
                vc = series.value_counts(dropna=True).head(5)
                most_common = list(zip(vc.index.tolist(), vc.values.tolist()))
            except Exception:  # pragma: no cover
                pass

            col_stats = ColumnStatistics(
                column_name=col,
                data_type=col_type,
                unique_count=n_unique,
                null_count=n_null,
                null_percentage=round(null_pct, 2),
                most_common=most_common,
            )

            # Numeric-specific statistics
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    col_stats.mean = round(float(clean.mean()), 6)
                    col_stats.median = round(float(clean.median()), 6)
                    col_stats.std = round(float(clean.std()), 6)
                    col_stats.min_val = round(float(clean.min()), 6)
                    col_stats.max_val = round(float(clean.max()), 6)

                    # scipy skewness and kurtosis (Fisher definition)
                    if len(clean) >= 3:
                        col_stats.skewness = round(
                            float(sp_stats.skew(clean, nan_policy="omit")), 6
                        )
                        col_stats.kurtosis = round(
                            float(sp_stats.kurtosis(clean, nan_policy="omit")), 6
                        )

            stats[col] = col_stats

        return stats

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    def detect_missing_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """Return the percentage of missing (null / NaN) values per column.

        Parameters
        ----------
        data:
            The DataFrame to inspect.

        Returns
        -------
        dict
            Mapping of column name to missing-value percentage (0.0 -- 100.0).
        """
        n_total = len(data)
        if n_total == 0:
            return {col: 0.0 for col in data.columns}

        missing: Dict[str, float] = {}
        for col in data.columns:
            pct = data[col].isnull().sum() / n_total * 100.0
            missing[col] = round(float(pct), 2)
        return missing

    # ------------------------------------------------------------------
    # Rare values
    # ------------------------------------------------------------------

    def detect_rare_values(
        self,
        data: pd.DataFrame,
        threshold: float = 0.05,
    ) -> Dict[str, List[str]]:
        """Identify values that appear in fewer than *threshold* fraction of
        rows for each non-numeric column.

        Parameters
        ----------
        data:
            The DataFrame to inspect.
        threshold:
            Frequency threshold below which a value is considered rare.
            Defaults to ``0.05`` (5 %).

        Returns
        -------
        dict
            Mapping of column name to a list of rare value labels (cast to
            ``str``).
        """
        rare: Dict[str, List[str]] = {}
        n_total = len(data)
        if n_total == 0:
            return rare

        for col in data.columns:
            series = data[col].dropna()
            if series.empty:
                continue

            vc = series.value_counts(normalize=True)
            rare_vals = vc[vc < threshold].index.tolist()

            if rare_vals:
                rare[col] = [str(v) for v in rare_vals]

        return rare

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------

    def compute_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the Pearson correlation matrix for all numeric columns.

        Parameters
        ----------
        data:
            The DataFrame to analyse.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.  Returns an empty DataFrame if there are
            fewer than two numeric columns.
        """
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] < 2:
            logger.debug(
                "Fewer than 2 numeric columns; skipping correlation matrix."
            )
            return pd.DataFrame()

        return numeric_data.corr()

    # ------------------------------------------------------------------
    # Outlier detection (IQR)
    # ------------------------------------------------------------------

    def detect_outliers(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers in numeric columns using the IQR method.

        A value is considered an outlier if it lies below ``Q1 - 1.5 * IQR``
        or above ``Q3 + 1.5 * IQR``.

        Parameters
        ----------
        data:
            The DataFrame to inspect.

        Returns
        -------
        dict
            Mapping of column name to a list of **integer row indices** where
            outliers were found.  Columns with no outliers are omitted.
        """
        outliers: Dict[str, List[int]] = {}

        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if series.empty:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_indices = data.index[mask].tolist()

            if outlier_indices:
                outliers[col] = [int(i) for i in outlier_indices]

        return outliers

    # ------------------------------------------------------------------
    # Transformation recommendations
    # ------------------------------------------------------------------

    def recommend_transformations(
        self, profile: DataProfile
    ) -> List[Transformation]:
        """Generate a list of recommended preprocessing transformations based
        on the supplied *profile*.

        Rules
        -----
        * **Log transform** -- continuous columns with ``|skewness| > 1.0``.
        * **Encoding** -- categorical columns with more than 20 unique values
          (high cardinality).
        * **Imputation** -- any column whose missing-value percentage is
          greater than zero.
        * **Normalisation** -- all continuous columns (z-score / min-max).

        Parameters
        ----------
        profile:
            A previously computed :class:`DataProfile`.

        Returns
        -------
        list[Transformation]
            Ordered list of recommended transformations.
        """
        recommendations: List[Transformation] = []
        seen: set = set()  # avoid exact duplicates

        for col, col_type in profile.column_types.items():
            col_stats = profile.statistics.get(col)

            # --- Log transform for skewed continuous columns ---------------
            if col_type == "continuous" and col_stats is not None:
                if (
                    col_stats.skewness is not None
                    and abs(col_stats.skewness) > 1.0
                ):
                    key = (col, "log_transform")
                    if key not in seen:
                        seen.add(key)
                        recommendations.append(
                            Transformation(
                                column_name=col,
                                transformation_type="log_transform",
                                reason=(
                                    f"Column '{col}' is highly skewed "
                                    f"(skewness={col_stats.skewness:.2f}). "
                                    "A log transform can reduce skew and "
                                    "improve model performance."
                                ),
                                parameters={
                                    "skewness": col_stats.skewness,
                                    "method": "log1p",
                                },
                            )
                        )

            # --- Encoding for high-cardinality categorical columns ---------
            if col_type == "categorical" and col_stats is not None:
                if col_stats.unique_count > 20:
                    key = (col, "encode")
                    if key not in seen:
                        seen.add(key)
                        recommendations.append(
                            Transformation(
                                column_name=col,
                                transformation_type="encode",
                                reason=(
                                    f"Column '{col}' is categorical with high "
                                    f"cardinality ({col_stats.unique_count} "
                                    "unique values). Consider target encoding "
                                    "or embedding-based encoding."
                                ),
                                parameters={
                                    "unique_count": col_stats.unique_count,
                                    "suggested_methods": [
                                        "target_encoding",
                                        "frequency_encoding",
                                        "embedding",
                                    ],
                                },
                            )
                        )

            # --- Imputation for columns with missing values ----------------
            missing_pct = profile.missing_values.get(col, 0.0)
            if missing_pct > 0.0:
                key = (col, "impute")
                if key not in seen:
                    seen.add(key)
                    if col_type in ("continuous", "ordinal"):
                        method = "median"
                    else:
                        method = "mode"
                    recommendations.append(
                        Transformation(
                            column_name=col,
                            transformation_type="impute",
                            reason=(
                                f"Column '{col}' has {missing_pct:.2f}% "
                                "missing values. Imputation is recommended "
                                "to avoid data loss."
                            ),
                            parameters={
                                "missing_percentage": missing_pct,
                                "suggested_method": method,
                            },
                        )
                    )

            # --- Normalisation for continuous columns ----------------------
            if col_type == "continuous":
                key = (col, "normalize")
                if key not in seen:
                    seen.add(key)
                    recommendations.append(
                        Transformation(
                            column_name=col,
                            transformation_type="normalize",
                            reason=(
                                f"Column '{col}' is continuous. "
                                "Normalisation (z-score or min-max) can "
                                "improve convergence and comparability."
                            ),
                            parameters={
                                "suggested_methods": [
                                    "standard_scaler",
                                    "min_max_scaler",
                                ],
                            },
                        )
                    )

        logger.info(
            "Generated %d transformation recommendations.",
            len(recommendations),
        )
        return recommendations
