"""Dataset persistence — save, load, list, and export synthetic datasets."""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


class DatasetRepository:
    """Manages storage and retrieval of synthetic datasets with metadata.

    Each dataset is identified by a UUID and consists of:
      - ``{uuid}.csv``              — the synthetic data
      - ``{uuid}_metadata.json``    — generation parameters, timestamps, lineage
    """

    def __init__(self, base_dir: str = "data/datasets"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_dataset(
        self,
        data: pd.DataFrame,
        metadata: Dict[str, Any],
        dataset_name: str = "unnamed",
    ) -> str:
        """Save a synthetic dataset and its metadata.

        Args:
            data: Synthetic DataFrame to persist.
            metadata: Generation metadata (model type, seed, scores, etc.).
            dataset_name: Human-readable name for the dataset.

        Returns:
            The unique dataset_id (UUID string).
        """
        dataset_id = str(uuid.uuid4())

        csv_path = os.path.join(self.base_dir, f"{dataset_id}.csv")
        data.to_csv(csv_path, index=False)

        meta = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_records": len(data),
            "columns": list(data.columns),
            **metadata,
        }

        meta_path = os.path.join(self.base_dir, f"{dataset_id}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"[+] Dataset saved: {csv_path}")
        print(f"[+] Metadata saved: {meta_path}")
        return dataset_id

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a dataset and its metadata by ID.

        Args:
            dataset_id: UUID of the dataset.

        Returns:
            Tuple of (DataFrame, metadata dict).

        Raises:
            FileNotFoundError: If the dataset files do not exist.
        """
        csv_path = os.path.join(self.base_dir, f"{dataset_id}.csv")
        meta_path = os.path.join(self.base_dir, f"{dataset_id}_metadata.json")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        data = pd.read_csv(csv_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        return data, metadata

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_dataset(
        self, dataset_id: str, fmt: str = "csv", output_dir: Optional[str] = None
    ) -> str:
        """Export a dataset in the requested format.

        Args:
            dataset_id: UUID of the dataset.
            fmt: Export format — ``'csv'`` or ``'json'``.
            output_dir: Directory for export file. Defaults to base_dir.

        Returns:
            Path to the exported file.
        """
        data, metadata = self.load_dataset(dataset_id)
        out_dir = output_dir or self.base_dir

        if fmt == "csv":
            out_path = os.path.join(out_dir, f"{dataset_id}_export.csv")
            data.to_csv(out_path, index=False)
        elif fmt == "json":
            out_path = os.path.join(out_dir, f"{dataset_id}_export.json")
            export_payload = {
                "metadata": metadata,
                "data": data.to_dict(orient="records"),
            }
            with open(out_path, "w") as f:
                json.dump(export_payload, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'csv' or 'json'.")

        print(f"[+] Exported to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return summaries for all saved datasets.

        Returns:
            List of metadata dicts (without data).
        """
        datasets = []

        for fname in os.listdir(self.base_dir):
            if fname.endswith("_metadata.json"):
                meta_path = os.path.join(self.base_dir, fname)
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                datasets.append(meta)

        datasets.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return datasets

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def get_dataset_lineage(self, dataset_id: str) -> Dict[str, Any]:
        """Return the complete generation lineage for a dataset.

        Args:
            dataset_id: UUID of the dataset.

        Returns:
            Dict with creation info, model params, and data hash.
        """
        _, metadata = self.load_dataset(dataset_id)

        return {
            "dataset_id": dataset_id,
            "dataset_name": metadata.get("dataset_name"),
            "created_at": metadata.get("created_at"),
            "n_records": metadata.get("n_records"),
            "model_type": metadata.get("model_type"),
            "random_seed": metadata.get("random_seed"),
            "training_data_hash": metadata.get("training_data_hash"),
            "hyperparameters": metadata.get("hyperparameters"),
            "algorithm_version": metadata.get("algorithm_version"),
        }
