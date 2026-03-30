"""Data models and schemas for the AI Diagnostic Agent."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import uuid


class QualityLevel(str, Enum):
    """Quality classification levels for metrics."""
    EXCELLENT = "Excellent"
    ACCEPTABLE = "Acceptable"
    WARNING = "Warning"
    CRITICAL = "Critical"


class RecommendationCategory(str, Enum):
    """Categories for pipeline improvement recommendations."""
    PREPROCESSING = "preprocessing"
    MODEL = "model"
    TRAINING = "training"
    DATA_STRATEGY = "data_strategy"
    POST_PROCESSING = "post_processing"


class ExperimentStatus(str, Enum):
    """Status of an experiment run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Data Profiling Models ---

@dataclass
class ColumnStatistics:
    """Statistical summary for a single column."""
    column_name: str
    data_type: str  # categorical, continuous, ordinal, identifier
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    unique_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    most_common: List[Tuple[Any, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'column_name': self.column_name,
            'data_type': self.data_type,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'unique_count': self.unique_count,
            'null_count': self.null_count,
            'null_percentage': self.null_percentage,
            'most_common': self.most_common,
        }


@dataclass
class Transformation:
    """A recommended preprocessing transformation."""
    column_name: str
    transformation_type: str  # e.g., "log_transform", "normalize", "encode", "bin"
    reason: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'column_name': self.column_name,
            'transformation_type': self.transformation_type,
            'reason': self.reason,
            'parameters': self.parameters,
        }


@dataclass
class DataProfile:
    """Comprehensive profile of a dataset."""
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    n_rows: int = 0
    n_columns: int = 0
    column_types: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, ColumnStatistics] = field(default_factory=dict)
    missing_values: Dict[str, float] = field(default_factory=dict)
    rare_values: Dict[str, List[str]] = field(default_factory=dict)
    class_imbalances: Dict[str, Dict[str, float]] = field(default_factory=dict)
    correlations: Optional[Any] = None  # pd.DataFrame stored as dict for serialization
    outliers: Dict[str, List[int]] = field(default_factory=dict)
    recommended_transformations: List[Transformation] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'dataset_id': self.dataset_id,
            'n_rows': self.n_rows,
            'n_columns': self.n_columns,
            'column_types': self.column_types,
            'statistics': {k: v.to_dict() for k, v in self.statistics.items()},
            'missing_values': self.missing_values,
            'rare_values': self.rare_values,
            'class_imbalances': self.class_imbalances,
            'outliers': self.outliers,
            'recommended_transformations': [t.to_dict() for t in self.recommended_transformations],
            'timestamp': self.timestamp,
        }


# --- Metric Analysis Models ---

@dataclass
class MetricAnalysis:
    """Result of analyzing evaluation metrics."""
    quality_metrics: Dict[str, Tuple[float, QualityLevel]] = field(default_factory=dict)
    utility_metrics: Dict[str, Tuple[float, QualityLevel]] = field(default_factory=dict)
    privacy_metrics: Dict[str, Tuple[float, QualityLevel]] = field(default_factory=dict)
    bias_metrics: Dict[str, Tuple[float, QualityLevel]] = field(default_factory=dict)
    weakest_metrics: List[str] = field(default_factory=list)
    benchmark_deltas: Dict[str, float] = field(default_factory=dict)
    overall_status: str = "Unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        def serialize_metrics(metrics):
            return {k: {'value': v[0], 'level': v[1].value} for k, v in metrics.items()}
        return {
            'quality_metrics': serialize_metrics(self.quality_metrics),
            'utility_metrics': serialize_metrics(self.utility_metrics),
            'privacy_metrics': serialize_metrics(self.privacy_metrics),
            'bias_metrics': serialize_metrics(self.bias_metrics),
            'weakest_metrics': self.weakest_metrics,
            'benchmark_deltas': self.benchmark_deltas,
            'overall_status': self.overall_status,
            'timestamp': self.timestamp,
        }

    def get_all_metrics(self) -> Dict[str, Tuple[float, QualityLevel]]:
        """Return all metrics combined."""
        all_metrics = {}
        all_metrics.update(self.quality_metrics)
        all_metrics.update(self.utility_metrics)
        all_metrics.update(self.privacy_metrics)
        all_metrics.update(self.bias_metrics)
        return all_metrics


# --- Root Cause Analysis Models ---

@dataclass
class RootCause:
    """An identified root cause for quality issues."""
    cause_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cause_name: str = ""
    description: str = ""
    affected_metrics: List[str] = field(default_factory=list)
    likelihood: float = 0.0  # 0.0 to 1.0
    impact: str = "Low"  # Low, Medium, High, Critical
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'cause_id': self.cause_id,
            'cause_name': self.cause_name,
            'description': self.description,
            'affected_metrics': self.affected_metrics,
            'likelihood': self.likelihood,
            'impact': self.impact,
            'evidence': self.evidence,
            'recommendations': self.recommendations,
        }


# --- Feature Diagnostics Models ---

@dataclass
class ColumnDiagnosis:
    """Diagnostic result for a single column."""
    column_name: str = ""
    column_type: str = ""
    jsd: float = 0.0
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    bias_amplification_ratio: float = 1.0
    severity: QualityLevel = QualityLevel.ACCEPTABLE
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'column_name': self.column_name,
            'column_type': self.column_type,
            'jsd': self.jsd,
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'bias_amplification_ratio': self.bias_amplification_ratio,
            'severity': self.severity.value,
            'issues': self.issues,
            'recommendations': self.recommendations,
        }


@dataclass
class CorrelationAnalysis:
    """Analysis of correlation preservation between synthetic and real data."""
    preserved_pairs: List[Tuple[str, str]] = field(default_factory=list)
    lost_pairs: List[Tuple[str, str, float]] = field(default_factory=list)  # (col1, col2, diff)
    overall_similarity: float = 0.0
    max_loss: float = 0.0

    def to_dict(self) -> dict:
        return {
            'preserved_pairs': self.preserved_pairs,
            'lost_pairs': [(p[0], p[1], p[2]) for p in self.lost_pairs],
            'overall_similarity': self.overall_similarity,
            'max_loss': self.max_loss,
        }


# --- Pipeline Optimization Models ---

@dataclass
class Recommendation:
    """A pipeline improvement recommendation."""
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: str = ""  # preprocessing, model, training, data_strategy, post_processing
    priority: int = 3  # 1 (highest) to 5 (lowest)
    title: str = ""
    description: str = ""
    expected_improvements: Dict[str, str] = field(default_factory=dict)
    implementation: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            'recommendation_id': self.recommendation_id,
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'expected_improvements': self.expected_improvements,
            'implementation': self.implementation,
            'rationale': self.rationale,
        }


@dataclass
class ExperimentPlan:
    """Plan for the next optimization experiment."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_experiment_id: Optional[str] = None
    configuration_changes: Dict[str, Any] = field(default_factory=dict)
    applied_recommendations: List[Recommendation] = field(default_factory=list)
    predicted_improvements: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'plan_id': self.plan_id,
            'parent_experiment_id': self.parent_experiment_id,
            'configuration_changes': self.configuration_changes,
            'applied_recommendations': [r.to_dict() for r in self.applied_recommendations],
            'predicted_improvements': self.predicted_improvements,
            'rationale': self.rationale,
            'timestamp': self.timestamp,
        }


# --- Experiment Tracking Models ---

@dataclass
class Experiment:
    """Record of a single experiment run."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_version: str = ""
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    model_type: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnostic_summary: Dict[str, Any] = field(default_factory=dict)
    status: str = ExperimentStatus.PENDING.value

    def to_dict(self) -> dict:
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'dataset_version': self.dataset_version,
            'preprocessing_config': self.preprocessing_config,
            'model_type': self.model_type,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'metrics': self.metrics,
            'diagnostic_summary': self.diagnostic_summary,
            'status': self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Experiment':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --- Diversity Monitoring Models ---

@dataclass
class DiversityReport:
    """Report on synthetic data diversity."""
    unique_row_ratio: float = 0.0
    duplicate_count: int = 0
    duplicate_rate: float = 0.0
    categorical_entropy: Dict[str, float] = field(default_factory=dict)
    real_entropy: Dict[str, float] = field(default_factory=dict)
    entropy_ratios: Dict[str, float] = field(default_factory=dict)
    mode_collapse_detected: bool = False
    mode_collapse_columns: List[str] = field(default_factory=list)
    diversity_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    severity: QualityLevel = QualityLevel.ACCEPTABLE

    def to_dict(self) -> dict:
        return {
            'unique_row_ratio': self.unique_row_ratio,
            'duplicate_count': self.duplicate_count,
            'duplicate_rate': self.duplicate_rate,
            'categorical_entropy': self.categorical_entropy,
            'real_entropy': self.real_entropy,
            'entropy_ratios': self.entropy_ratios,
            'mode_collapse_detected': self.mode_collapse_detected,
            'mode_collapse_columns': self.mode_collapse_columns,
            'diversity_score': self.diversity_score,
            'issues': self.issues,
            'severity': self.severity.value,
        }


# --- Constraint Validation Models ---

@dataclass
class Constraint:
    """A domain constraint rule."""
    name: str = ""
    description: str = ""
    rule_type: str = "categorical_compatibility"  # categorical_compatibility, numerical_range, logical_consistency
    columns: List[str] = field(default_factory=list)
    rule_fn: Optional[Any] = None  # Callable

    def validate(self, row) -> bool:
        """Validate a single row against this constraint."""
        if self.rule_fn is not None:
            try:
                return bool(self.rule_fn(row))
            except Exception:
                return True  # If rule can't be evaluated, pass by default
        return True


@dataclass
class ConstraintValidationResult:
    """Result of constraint validation on a dataset."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    filter_percentage: float = 0.0
    violations_by_constraint: Dict[str, int] = field(default_factory=dict)
    violation_details: List[Dict[str, Any]] = field(default_factory=list)
    is_critical: bool = False
    severity: QualityLevel = QualityLevel.ACCEPTABLE

    def to_dict(self) -> dict:
        return {
            'total_records': self.total_records,
            'valid_records': self.valid_records,
            'invalid_records': self.invalid_records,
            'filter_percentage': self.filter_percentage,
            'violations_by_constraint': self.violations_by_constraint,
            'is_critical': self.is_critical,
            'severity': self.severity.value,
        }


# --- Diagnostic Report Models ---

@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report for an optimization cycle."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cycle_number: int = 0
    metric_analysis: Optional[MetricAnalysis] = None
    root_causes: List[RootCause] = field(default_factory=list)
    column_diagnostics: List[ColumnDiagnosis] = field(default_factory=list)
    correlation_analysis: Optional[CorrelationAnalysis] = None
    diversity_report: Optional[DiversityReport] = None
    constraint_validation: Optional[ConstraintValidationResult] = None
    recommendations: List[Recommendation] = field(default_factory=list)
    experiment_plan: Optional[ExperimentPlan] = None
    benchmarks_met: bool = False
    optimization_stalled: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'report_id': self.report_id,
            'cycle_number': self.cycle_number,
            'metric_analysis': self.metric_analysis.to_dict() if self.metric_analysis else None,
            'root_causes': [rc.to_dict() for rc in self.root_causes],
            'column_diagnostics': [cd.to_dict() for cd in self.column_diagnostics],
            'correlation_analysis': self.correlation_analysis.to_dict() if self.correlation_analysis else None,
            'diversity_report': self.diversity_report.to_dict() if self.diversity_report else None,
            'constraint_validation': self.constraint_validation.to_dict() if self.constraint_validation else None,
            'recommendations': [r.to_dict() for r in self.recommendations],
            'experiment_plan': self.experiment_plan.to_dict() if self.experiment_plan else None,
            'benchmarks_met': self.benchmarks_met,
            'optimization_stalled': self.optimization_stalled,
            'timestamp': self.timestamp,
        }
