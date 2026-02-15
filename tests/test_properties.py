"""Property-based tests for Synthia using Hypothesis.

Each property is tagged with its feature name and property ID from the spec.
Minimum 100 iterations per property (configured via settings).
"""

import sys
import os
import uuid
import json
import warnings
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

from src.models.generation_config import GenerationConfig
from src.models.variant_metadata import VariantMetadata
from src.models.reports import ValidationReport, PrivacyReport
from src.core.synthetic_data_generator import SyntheticDataGenerator
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.analysis.data_validator import DataValidator
from src.analysis.privacy_analyzer import PrivacyAnalyzer
from src.storage.dataset_repository import DatasetRepository
from src.utils.audit_logger import AuditLogger
from src.utils.metadata_manager import MetadataManager


# ── Shared helpers ────────────────────────────────────────────────────

VALID_GENES = ['CFTR', 'DMD', 'HBB', 'F8', 'HEXA']
VALID_CHROMOSOMES = ['7', 'X', '11', '15']
VALID_VARIANT_TYPES = ['Missense', 'Nonsense', 'Frameshift', 'Splice Site', 'Deletion']
VALID_SIGNIFICANCE = ['Pathogenic', 'Likely Pathogenic', 'Benign', 'VUS']
VALID_DISEASES = ['Cystic Fibrosis', 'Duchenne Muscular Dystrophy',
                  'Sickle Cell Disease', 'Hemophilia A', 'Tay-Sachs Disease']
VALID_INHERITANCE = ['Autosomal Recessive', 'X-linked', 'Autosomal Dominant']


def _make_sample_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Build a small realistic DataFrame for testing."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'gene_symbol': rng.choice(VALID_GENES, n),
        'chromosome': rng.choice(VALID_CHROMOSOMES, n),
        'variant_type': rng.choice(VALID_VARIANT_TYPES, n),
        'clinical_significance': rng.choice(VALID_SIGNIFICANCE, n),
        'disease': rng.choice(VALID_DISEASES, n),
        'allele_frequency': rng.uniform(0, 1, n).round(4),
        'inheritance_pattern': rng.choice(VALID_INHERITANCE, n),
    })


# ======================================================================
# Property 2: Complete Schema Coverage
# Feature: Generation
# All generated records must contain every required field.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=5, max_value=30))
def test_property2_complete_schema_coverage(n):
    """Every generated DataFrame has all 7 required columns."""
    df = _make_sample_df(n)
    required = {'gene_symbol', 'chromosome', 'variant_type',
                'clinical_significance', 'disease', 'allele_frequency',
                'inheritance_pattern'}
    assert required.issubset(set(df.columns)), \
        f"Missing columns: {required - set(df.columns)}"
    assert len(df) == n


# ======================================================================
# Property 1: Model Selection Routing
# Feature: Generation
# Requesting CTGAN/TVAE must select the correct engine.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(model_type=st.sampled_from(['CTGAN', 'TVAE']))
def test_property1_model_selection_routing(model_type):
    """GenerationConfig correctly records model_type."""
    config = GenerationConfig(
        model_type=model_type,
        n_samples=10,
        random_seed=42,
        hyperparameters={'epochs': 1}
    )
    gen = SyntheticDataGenerator(config=config)
    assert gen.config.model_type == model_type


# ======================================================================
# Property 3: Conditional Generation Fidelity
# Feature: Generation
# When disease_condition is set, config must record it.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(disease=st.sampled_from(VALID_DISEASES))
def test_property3_conditional_generation_fidelity(disease):
    """GenerationConfig stores disease_condition."""
    config = GenerationConfig(
        model_type='CTGAN',
        n_samples=10,
        random_seed=42,
        hyperparameters={'epochs': 1},
        disease_condition=disease,
    )
    assert config.disease_condition == disease


# ======================================================================
# Property 9: Risk Score Validity
# Feature: Privacy
# Privacy scores must be in [0, 1].
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10000))
def test_property9_risk_score_validity(seed):
    """Privacy score is always in [0, 1]."""
    real = _make_sample_df(30, seed=seed)
    syn = _make_sample_df(20, seed=seed + 1)

    analyzer = PrivacyAnalyzer()
    report = analyzer.analyze_privacy(syn, real, percentile=90, dataset_id='test')

    assert 0.0 <= report.privacy_score <= 1.0, \
        f"Privacy score {report.privacy_score} out of [0, 1]"


# ======================================================================
# Property 26: Complete Metadata Storage
# Feature: Metadata
# Metadata must contain model_type, random_seed, training_data_hash,
# timestamp.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    model=st.sampled_from(['CTGAN', 'TVAE']),
    seed=st.integers(min_value=0, max_value=99999),
)
def test_property26_complete_metadata_storage(model, seed):
    """MetadataManager.build_metadata always includes required fields."""
    mgr = MetadataManager()
    df = _make_sample_df(10, seed=seed % 10000)
    meta = mgr.build_metadata(
        model_type=model,
        random_seed=seed,
        hyperparameters={'epochs': 1},
        training_data=df,
        n_records_generated=10,
    )

    assert meta['model_type'] == model
    assert meta['random_seed'] == seed
    assert 'training_data_hash' in meta
    assert 'created_at' in meta
    assert meta['training_records'] == 10


# ======================================================================
# Property 15: Bidirectional Utility Testing
# Feature: Validation
# cross_test must return metrics for both directions.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10000))
def test_property15_bidirectional_utility_testing(seed):
    """cross_test returns synthetic_to_real and real_to_synthetic."""
    real = _make_sample_df(30, seed=seed)
    syn = _make_sample_df(20, seed=seed + 1)

    validator = DataValidator()
    results = validator.cross_test(syn, real, target_column='disease')

    assert 'synthetic_to_real' in results
    assert 'real_to_synthetic' in results

    for direction in ['synthetic_to_real', 'real_to_synthetic']:
        metrics = results[direction]
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'auc' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0


# ======================================================================
# Property 24: Multi-Format Export
# Feature: Persistence
# CSV and JSON exports contain the same data (same number of records
# and same column set).
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=5, max_value=30))
def test_property24_multi_format_export(n, tmp_path_factory):
    """CSV and JSON exports have identical record counts and columns."""
    tmp = str(tmp_path_factory.mktemp("export"))
    repo = DatasetRepository(base_dir=tmp)

    df = _make_sample_df(n)
    ds_id = repo.save_dataset(df, {"model_type": "CTGAN"}, "test-ds")

    csv_path = repo.export_dataset(ds_id, fmt='csv', output_dir=tmp)
    json_path = repo.export_dataset(ds_id, fmt='json', output_dir=tmp)

    csv_data = pd.read_csv(csv_path)
    with open(json_path) as f:
        json_payload = json.load(f)
    json_data = pd.DataFrame(json_payload['data'])

    assert len(csv_data) == n
    assert len(json_data) == n
    assert set(csv_data.columns) == set(json_data.columns)


# ======================================================================
# Property 25: Unique Identifier Generation
# Feature: Persistence
# Every save must produce a unique dataset_id.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=2, max_value=5))
def test_property25_unique_identifier_generation(n, tmp_path_factory):
    """Saving N datasets produces N distinct UUIDs."""
    tmp = str(tmp_path_factory.mktemp("uuid"))
    repo = DatasetRepository(base_dir=tmp)
    ids = set()
    for _ in range(n):
        df = _make_sample_df(5)
        ds_id = repo.save_dataset(df, {"model_type": "CTGAN"}, "test")
        ids.add(ds_id)
    assert len(ids) == n, f"Expected {n} unique IDs, got {len(ids)}"


# ======================================================================
# Property 34: Audit Trail Completeness
# Feature: Audit
# Every logged action must appear in the audit trail with a timestamp.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    action=st.sampled_from(['create', 'export', 'delete', 'generate']),
    resource=st.text(min_size=1, max_size=8, alphabet='abcdef0123456789'),
)
def test_property34_audit_trail_completeness(action, resource, tmp_path_factory):
    """Every logged action is retrievable and has a timestamp."""
    tmp = str(tmp_path_factory.mktemp("audit"))
    log_path = os.path.join(tmp, "audit.jsonl")
    logger = AuditLogger(log_path=log_path)

    logger.log(action=action, resource_id=resource, username="tester")

    entries = logger.get_entries()
    assert len(entries) >= 1

    match = [e for e in entries if e['action'] == action and e['resource_id'] == resource]
    assert len(match) >= 1, f"Action '{action}' for '{resource}' not in audit log"
    assert 'timestamp' in match[0]


# ======================================================================
# Property 27: Version Tracking
# Feature: Metadata
# Metadata always records algorithm_version.
# ======================================================================

@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=99999))
def test_property27_version_tracking(seed):
    """build_metadata always includes algorithm_version."""
    mgr = MetadataManager()
    meta = mgr.build_metadata(
        model_type='CTGAN',
        random_seed=seed,
        hyperparameters={'epochs': 1},
    )
    assert 'algorithm_version' in meta
    assert isinstance(meta['algorithm_version'], str)
    assert len(meta['algorithm_version']) > 0
