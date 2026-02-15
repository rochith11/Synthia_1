"""Report schemas for Synthia outputs."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ValidationReport:
    """Report generated from validation analysis.

    Attributes:
        dataset_id: Unique identifier for the dataset
        statistical_metrics: KS tests, JS divergence, correlation similarity
        utility_metrics: ML utility scores (accuracy, F1, AUC)
        rare_event_analysis: Frequency analysis for rare classes
        overall_quality_score: Composite quality score [0, 1]
    """

    dataset_id: str
    statistical_metrics: Dict[str, Any] = field(default_factory=dict)
    utility_metrics: Dict[str, Any] = field(default_factory=dict)
    rare_event_analysis: Dict[str, Any] = field(default_factory=dict)
    overall_quality_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'dataset_id': self.dataset_id,
            'statistical_metrics': self.statistical_metrics,
            'utility_metrics': self.utility_metrics,
            'rare_event_analysis': self.rare_event_analysis,
            'overall_quality_score': self.overall_quality_score,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'ValidationReport(dataset_id={self.dataset_id}'
            f', quality_score={self.overall_quality_score:.3f})'
        )


@dataclass
class PrivacyReport:
    """Report generated from privacy analysis.

    Attributes:
        dataset_id: Unique identifier for the dataset
        nearest_neighbor_distances: NND statistics (mean, median, min, std)
        reidentification_risk: High-risk percentage and distribution
        privacy_score: Composite privacy score [0, 1] (higher is better)
    """

    dataset_id: str
    nearest_neighbor_distances: Dict[str, float] = field(default_factory=dict)
    reidentification_risk: Dict[str, Any] = field(default_factory=dict)
    privacy_score: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'dataset_id': self.dataset_id,
            'nearest_neighbor_distances': self.nearest_neighbor_distances,
            'reidentification_risk': self.reidentification_risk,
            'privacy_score': self.privacy_score,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'PrivacyReport(dataset_id={self.dataset_id}'
            f', privacy_score={self.privacy_score:.3f})'
        )


@dataclass
class BiasReport:
    """Report generated from bias analysis.

    Attributes:
        dataset_id: Unique identifier for the dataset
        feature_distributions: Distribution bias analysis for each feature
        rare_class_imbalances: Identified rare class imbalances
        bias_amplification: Detected bias amplifications
        recommendations: Re-sampling recommendations
    """

    dataset_id: str
    feature_distributions: Dict[str, Any] = field(default_factory=dict)
    rare_class_imbalances: Dict[str, Any] = field(default_factory=dict)
    bias_amplification: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'dataset_id': self.dataset_id,
            'feature_distributions': self.feature_distributions,
            'rare_class_imbalances': self.rare_class_imbalances,
            'bias_amplification': self.bias_amplification,
            'recommendations': self.recommendations,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'BiasReport(dataset_id={self.dataset_id}'
            f', amplifications={len(self.bias_amplification)}'
            f', recommendations={len(self.recommendations)})'
        )
