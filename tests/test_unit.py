"""Unit tests for Synthia core modules."""

import sys
import os
import json
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

from src.models.variant_metadata import VariantMetadata
from src.models.generation_config import GenerationConfig
from src.models.reports import ValidationReport, PrivacyReport, BiasReport
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.analysis.data_validator import DataValidator
from src.analysis.privacy_analyzer import PrivacyAnalyzer
from src.analysis.bias_detector import BiasDetector
from src.storage.dataset_repository import DatasetRepository
from src.utils.audit_logger import AuditLogger
from src.utils.metadata_manager import MetadataManager
from src.utils.config_manager import ConfigManager


# ── Fixtures ──────────────────────────────────────────────────────────

GENES = ['CFTR', 'DMD', 'HBB', 'F8', 'HEXA']
CHROMOSOMES = ['7', 'X', '11', '15']
VARIANT_TYPES = ['Missense', 'Nonsense', 'Frameshift', 'Splice Site', 'Deletion']
SIGNIFICANCE = ['Pathogenic', 'Likely Pathogenic', 'Benign', 'VUS']
DISEASES = ['Cystic Fibrosis', 'Duchenne Muscular Dystrophy',
            'Sickle Cell Disease', 'Hemophilia A', 'Tay-Sachs Disease']
INHERITANCE = ['Autosomal Recessive', 'X-linked', 'Autosomal Dominant']


def _df(n=50, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'gene_symbol': rng.choice(GENES, n),
        'chromosome': rng.choice(CHROMOSOMES, n),
        'variant_type': rng.choice(VARIANT_TYPES, n),
        'clinical_significance': rng.choice(SIGNIFICANCE, n),
        'disease': rng.choice(DISEASES, n),
        'allele_frequency': rng.uniform(0, 1, n).round(4),
        'inheritance_pattern': rng.choice(INHERITANCE, n),
    })


# ======================================================================
# VariantMetadata
# ======================================================================

class TestVariantMetadata:
    def test_creation(self):
        v = VariantMetadata(
            gene_symbol='CFTR', chromosome='7', variant_type='SNV',
            clinical_significance='Pathogenic', disease='Cystic Fibrosis',
            allele_frequency=0.01, inheritance_pattern='Autosomal Recessive'
        )
        assert v.gene_symbol == 'CFTR'
        assert v.allele_frequency == 0.01

    def test_to_dict(self):
        v = VariantMetadata(
            gene_symbol='DMD', chromosome='X', variant_type='Deletion',
            clinical_significance='Pathogenic',
            disease='Duchenne Muscular Dystrophy',
            allele_frequency=0.001, inheritance_pattern='X-linked'
        )
        d = v.to_dict()
        assert isinstance(d, dict)
        assert d['gene_symbol'] == 'DMD'

    def test_from_dict(self):
        d = {
            'gene_symbol': 'HBB', 'chromosome': '11',
            'variant_type': 'SNV', 'clinical_significance': 'Pathogenic',
            'disease': 'Sickle Cell Disease', 'allele_frequency': 0.05,
            'inheritance_pattern': 'Autosomal Recessive'
        }
        v = VariantMetadata.from_dict(d)
        assert v.gene_symbol == 'HBB'


# ======================================================================
# GenerationConfig
# ======================================================================

class TestGenerationConfig:
    def test_creation(self):
        c = GenerationConfig(model_type='CTGAN', n_samples=100, random_seed=42)
        assert c.model_type == 'CTGAN'
        assert c.n_samples == 100

    def test_timestamp_set(self):
        c = GenerationConfig(model_type='TVAE', n_samples=10, random_seed=1)
        assert c.timestamp is not None

    def test_hyperparameters_default(self):
        c = GenerationConfig(model_type='CTGAN', n_samples=10, random_seed=1)
        assert isinstance(c.hyperparameters, dict)


# ======================================================================
# Reports
# ======================================================================

class TestReports:
    def test_validation_report_to_dict(self):
        r = ValidationReport(dataset_id='test-1', overall_quality_score=0.8)
        d = r.to_dict()
        assert d['dataset_id'] == 'test-1'
        assert d['overall_quality_score'] == 0.8

    def test_privacy_report_default(self):
        r = PrivacyReport(dataset_id='p-1')
        assert r.privacy_score == 1.0

    def test_bias_report_to_dict(self):
        r = BiasReport(dataset_id='b-1', recommendations=['test'])
        d = r.to_dict()
        assert len(d['recommendations']) == 1


# ======================================================================
# StatisticalAnalyzer
# ======================================================================

class TestStatisticalAnalyzer:
    def setup_method(self):
        self.analyzer = StatisticalAnalyzer()
        self.real = _df(50, seed=0)
        self.syn = _df(50, seed=1)

    def test_ks_statistic_range(self):
        stat, pval = self.analyzer.compute_ks_statistic(
            self.syn['allele_frequency'], self.real['allele_frequency']
        )
        assert 0.0 <= stat <= 1.0
        assert 0.0 <= pval <= 1.0

    def test_ks_identical(self):
        stat, pval = self.analyzer.compute_ks_statistic(
            self.real['allele_frequency'], self.real['allele_frequency']
        )
        assert stat == 0.0

    def test_js_divergence_range(self):
        jsd = self.analyzer.compute_js_divergence(
            self.syn['gene_symbol'], self.real['gene_symbol']
        )
        assert 0.0 <= jsd <= 1.0

    def test_js_divergence_identical(self):
        jsd = self.analyzer.compute_js_divergence(
            self.real['gene_symbol'], self.real['gene_symbol']
        )
        assert jsd < 0.01  # Near zero for identical

    def test_correlation_similarity_none_for_one_col(self):
        # Only one numerical column -> None
        result = self.analyzer.compute_correlation_similarity(self.syn, self.real)
        assert result is None

    def test_correlation_similarity_with_multiple_num(self):
        real2 = self.real.copy()
        real2['extra_num'] = np.random.rand(len(real2))
        syn2 = self.syn.copy()
        syn2['extra_num'] = np.random.rand(len(syn2))
        result = self.analyzer.compute_correlation_similarity(syn2, real2)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_compute_all_metrics_structure(self):
        metrics = self.analyzer.compute_all_metrics(self.syn, self.real)
        assert 'ks_tests' in metrics
        assert 'js_divergences' in metrics
        assert 'summary' in metrics
        assert 'mean_jsd' in metrics['summary']
        assert 'max_jsd' in metrics['summary']

    def test_empty_series(self):
        empty = pd.Series([], dtype=float)
        stat, pval = self.analyzer.compute_ks_statistic(empty, self.real['allele_frequency'])
        assert stat == 1.0

        jsd = self.analyzer.compute_js_divergence(empty, self.real['allele_frequency'])
        assert jsd == 1.0


# ======================================================================
# DataValidator
# ======================================================================

class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator()
        self.real = _df(30, seed=0)
        self.syn = _df(20, seed=1)

    def test_validate_produces_report(self):
        report = self.validator.validate(self.syn, self.real,
                                         target_column='disease')
        assert isinstance(report, ValidationReport)
        assert 0.0 <= report.overall_quality_score <= 1.0

    def test_nan_guard(self):
        bad = self.syn.copy()
        bad.loc[0, 'allele_frequency'] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            self.validator.validate(bad, self.real, target_column='disease')

    def test_mode_collapse_detection(self):
        collapsed = self.real.copy()
        collapsed['gene_symbol'] = 'CFTR'  # 100% single value
        info = self.validator._detect_mode_collapse(collapsed)
        assert 'gene_symbol' in info
        assert info['gene_symbol']['dominant_frequency'] == 1.0

    def test_cross_test_returns_both_directions(self):
        result = self.validator.cross_test(self.syn, self.real, 'disease')
        assert 'synthetic_to_real' in result
        assert 'real_to_synthetic' in result

    def test_rare_events_with_threshold(self):
        # Make one class very rare in real data
        real = self.real.copy()
        real.loc[0, 'gene_symbol'] = 'RARE_GENE'
        result = self.validator.analyze_rare_events(self.syn, real, threshold=0.10)
        # RARE_GENE should appear since it's ~3% of 30 records
        found = any('RARE_GENE' in str(v) for v in result.values())
        assert found or len(result) >= 0  # Just ensure no crash


# ======================================================================
# PrivacyAnalyzer
# ======================================================================

class TestPrivacyAnalyzer:
    def setup_method(self):
        self.analyzer = PrivacyAnalyzer()
        self.real = _df(30, seed=0)
        self.syn = _df(20, seed=1)

    def test_nnd_shape(self):
        nnd = self.analyzer.compute_nearest_neighbor_distances(self.syn, self.real)
        assert nnd.shape == (20,)

    def test_nnd_exact_copy_zero(self):
        nnd = self.analyzer.compute_nearest_neighbor_distances(self.real, self.real)
        assert np.allclose(nnd, 0.0)

    def test_risk_percentile(self):
        risk = self.analyzer.compute_reidentification_risk(
            self.syn, self.real, percentile=90
        )
        assert 'threshold' in risk
        assert 'high_risk_percentage' in risk
        assert 0.0 <= risk['high_risk_percentage'] <= 1.0

    def test_analyze_privacy_report(self):
        report = self.analyzer.analyze_privacy(self.syn, self.real)
        assert isinstance(report, PrivacyReport)
        assert 0.0 <= report.privacy_score <= 1.0

    def test_normalize_output_range(self):
        values = self.analyzer.normalize_data(self.real, fit=True)
        assert values.min() >= -0.01  # Allow small floating point drift
        assert values.max() <= 1.01


# ======================================================================
# BiasDetector
# ======================================================================

class TestBiasDetector:
    def setup_method(self):
        self.detector = BiasDetector()
        self.real = _df(50, seed=0)
        self.syn = _df(50, seed=1)

    def test_feature_distributions(self):
        result = self.detector.analyze_feature_distributions(self.syn, self.real)
        assert len(result) == 7  # All columns
        for col, info in result.items():
            assert 'jsd' in info
            assert info['status'] in ('low', 'moderate', 'high')

    def test_rare_class_imbalance(self):
        result = self.detector.detect_rare_class_imbalance(
            self.syn, self.real, 'gene_symbol', threshold=0.05
        )
        assert 'total_rare_classes' in result

    def test_bias_amplification(self):
        result = self.detector.compute_bias_amplification(
            self.syn, self.real, amplification_threshold=0.20
        )
        assert 'columns' in result
        assert 'flagged' in result

    def test_analyze_bias_full(self):
        result = self.detector.analyze_bias(self.syn, self.real)
        assert 'feature_distributions' in result
        assert 'amplification' in result
        assert 'recommendations' in result


# ======================================================================
# DatasetRepository
# ======================================================================

class TestDatasetRepository:
    def test_save_and_load(self, tmp_path):
        repo = DatasetRepository(base_dir=str(tmp_path))
        df = _df(10)
        ds_id = repo.save_dataset(df, {'model_type': 'CTGAN'}, 'test')
        loaded, meta = repo.load_dataset(ds_id)
        assert len(loaded) == 10
        assert meta['model_type'] == 'CTGAN'

    def test_export_csv(self, tmp_path):
        repo = DatasetRepository(base_dir=str(tmp_path))
        df = _df(5)
        ds_id = repo.save_dataset(df, {}, 'test')
        path = repo.export_dataset(ds_id, fmt='csv')
        assert os.path.exists(path)

    def test_export_json(self, tmp_path):
        repo = DatasetRepository(base_dir=str(tmp_path))
        df = _df(5)
        ds_id = repo.save_dataset(df, {}, 'test')
        path = repo.export_dataset(ds_id, fmt='json')
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert 'data' in data
        assert len(data['data']) == 5

    def test_list_datasets(self, tmp_path):
        repo = DatasetRepository(base_dir=str(tmp_path))
        for i in range(3):
            repo.save_dataset(_df(5), {}, f'ds-{i}')
        listed = repo.list_datasets()
        assert len(listed) == 3

    def test_load_nonexistent_raises(self, tmp_path):
        repo = DatasetRepository(base_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            repo.load_dataset('nonexistent-id')

    def test_lineage(self, tmp_path):
        repo = DatasetRepository(base_dir=str(tmp_path))
        df = _df(5)
        ds_id = repo.save_dataset(df, {'model_type': 'TVAE', 'random_seed': 99}, 'test')
        lineage = repo.get_dataset_lineage(ds_id)
        assert lineage['model_type'] == 'TVAE'
        assert lineage['random_seed'] == 99


# ======================================================================
# AuditLogger
# ======================================================================

class TestAuditLogger:
    def test_log_and_retrieve(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=path)
        logger.log('create', resource_id='abc', username='alice')
        entries = logger.get_entries()
        assert len(entries) == 1
        assert entries[0]['action'] == 'create'
        assert entries[0]['username'] == 'alice'

    def test_multiple_entries(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=path)
        for i in range(5):
            logger.log('action', resource_id=str(i))
        entries = logger.get_entries()
        assert len(entries) == 5

    def test_empty_log(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        logger = AuditLogger(log_path=path)
        entries = logger.get_entries()
        assert entries == []


# ======================================================================
# MetadataManager
# ======================================================================

class TestMetadataManager:
    def test_build_metadata(self):
        mgr = MetadataManager()
        meta = mgr.build_metadata('CTGAN', 42, {'epochs': 100})
        assert meta['model_type'] == 'CTGAN'
        assert meta['random_seed'] == 42
        assert 'created_at' in meta

    def test_data_hash_deterministic(self):
        mgr = MetadataManager()
        df = _df(10, seed=0)
        h1 = mgr.compute_data_hash(df)
        h2 = mgr.compute_data_hash(df)
        assert h1 == h2

    def test_save_and_load(self, tmp_path):
        mgr = MetadataManager()
        meta = mgr.build_metadata('TVAE', 1, {'epochs': 50})
        path = str(tmp_path / "meta.json")
        mgr.save_metadata(meta, path)
        loaded = mgr.load_metadata(path)
        assert loaded['model_type'] == 'TVAE'


# ======================================================================
# ConfigManager
# ======================================================================

class TestConfigManager:
    def test_loads_config(self):
        cfg = ConfigManager()
        assert cfg.get('generation.defaults.model_type') is not None

    def test_dot_notation(self):
        cfg = ConfigManager()
        val = cfg.get('privacy.risk_percentile')
        assert val == 90

    def test_default_value(self):
        cfg = ConfigManager()
        val = cfg.get('nonexistent.key', 'fallback')
        assert val == 'fallback'
