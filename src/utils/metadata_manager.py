"""Metadata manager — capture and persist generation metadata for reproducibility."""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pandas as pd


class MetadataManager:
    """Record generation parameters, data hashes, and timestamps."""

    def __init__(self, output_dir: str = "data/datasets"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute an MD5 hash of a DataFrame for tracking."""
        raw = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.md5(raw).hexdigest()

    def build_metadata(
        self,
        model_type: str,
        random_seed: int,
        hyperparameters: Dict[str, Any],
        training_data: Optional[pd.DataFrame] = None,
        n_records_generated: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a complete metadata record.

        Args:
            model_type: 'CTGAN' or 'TVAE'.
            random_seed: Seed used for generation.
            hyperparameters: Model hyperparameters.
            training_data: Original training DataFrame (for hash).
            n_records_generated: Number of synthetic records produced.
            extra: Any additional fields to include.

        Returns:
            Dict with all metadata fields.
        """
        meta: Dict[str, Any] = {
            "model_type": model_type,
            "random_seed": random_seed,
            "hyperparameters": hyperparameters,
            "n_records_generated": n_records_generated,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm_version": "1.0.0",
        }

        if training_data is not None:
            meta["training_data_hash"] = self.compute_data_hash(training_data)
            meta["training_records"] = len(training_data)

        if extra:
            meta.update(extra)

        return meta

    def save_metadata(self, metadata: Dict[str, Any], filepath: str) -> str:
        """Write metadata to a JSON file.

        Args:
            metadata: Metadata dict.
            filepath: Destination path.

        Returns:
            The filepath written.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return filepath

    def load_metadata(self, filepath: str) -> Dict[str, Any]:
        """Load metadata from a JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)
