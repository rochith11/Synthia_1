"""Enterprise benchmark configuration and thresholds for the AI Diagnostic Agent."""

# Enterprise Benchmark Targets
ENTERPRISE_BENCHMARKS = {
    'quality_score': {'target': 0.70, 'direction': 'higher_better'},
    'mean_ks': {'target': 0.20, 'direction': 'lower_better'},
    'mean_jsd': {'target': 0.03, 'direction': 'lower_better'},
    'max_jsd': {'target': 0.10, 'direction': 'lower_better'},
    'ml_accuracy': {'target': 0.70, 'direction': 'higher_better'},
    'f1_score': {'target': 0.65, 'direction': 'higher_better'},
    'auc': {'target': 0.75, 'direction': 'higher_better'},
    'privacy_score': {'target': 0.75, 'direction': 'higher_better'},
    'mean_nnd': {'target': 0.60, 'direction': 'higher_better'},
    'high_risk_pct': {'target': 0.02, 'direction': 'lower_better'},
    'max_column_jsd': {'target': 0.10, 'direction': 'lower_better'},
    'max_amplification_ratio': {'target': 1.20, 'direction': 'lower_better'},
}

# Classification Thresholds: metric -> {level: (min, max)} or (threshold, direction)
CLASSIFICATION_THRESHOLDS = {
    'quality_score': {
        'Excellent': (0.80, float('inf')),
        'Acceptable': (0.70, 0.80),
        'Warning': (0.60, 0.70),
        'Critical': (float('-inf'), 0.60),
    },
    'privacy_score': {
        'Excellent': (0.85, float('inf')),
        'Acceptable': (0.75, 0.85),
        'Warning': (0.65, 0.75),
        'Critical': (float('-inf'), 0.65),
    },
    'mean_ks': {
        'Excellent': (float('-inf'), 0.10),
        'Acceptable': (0.10, 0.20),
        'Warning': (0.20, 0.30),
        'Critical': (0.30, float('inf')),
    },
    'mean_jsd': {
        'Excellent': (float('-inf'), 0.02),
        'Acceptable': (0.02, 0.03),
        'Warning': (0.03, 0.05),
        'Critical': (0.05, float('inf')),
    },
    'max_jsd': {
        'Excellent': (float('-inf'), 0.05),
        'Acceptable': (0.05, 0.10),
        'Warning': (0.10, 0.15),
        'Critical': (0.15, float('inf')),
    },
    'ml_accuracy': {
        'Excellent': (0.80, float('inf')),
        'Acceptable': (0.70, 0.80),
        'Warning': (0.60, 0.70),
        'Critical': (float('-inf'), 0.60),
    },
    'f1_score': {
        'Excellent': (0.75, float('inf')),
        'Acceptable': (0.65, 0.75),
        'Warning': (0.55, 0.65),
        'Critical': (float('-inf'), 0.55),
    },
    'auc': {
        'Excellent': (0.85, float('inf')),
        'Acceptable': (0.75, 0.85),
        'Warning': (0.65, 0.75),
        'Critical': (float('-inf'), 0.65),
    },
    'mean_nnd': {
        'Excellent': (0.70, float('inf')),
        'Acceptable': (0.60, 0.70),
        'Warning': (0.50, 0.60),
        'Critical': (float('-inf'), 0.50),
    },
    'high_risk_pct': {
        'Excellent': (float('-inf'), 0.01),
        'Acceptable': (0.01, 0.02),
        'Warning': (0.02, 0.05),
        'Critical': (0.05, float('inf')),
    },
}

# Dataset size thresholds
DATASET_SIZE_THRESHOLDS = {
    'critical_min': 500,
    'small': 1000,
    'medium': 5000,
    'large': 10000,
}

# Diversity thresholds
DIVERSITY_THRESHOLDS = {
    'min_unique_ratio': 0.95,
    'max_duplicate_rate': 0.02,
    'min_entropy_ratio': 0.90,
    'max_entropy_ratio': 1.10,
}

# Constraint filtering threshold
CONSTRAINT_FILTER_CRITICAL_THRESHOLD = 0.10  # Flag if > 10% filtered

# Optimization stall detection
STALL_DETECTION_RUNS = 3  # No improvement for N consecutive runs

# Model selection weights
MODEL_SELECTION_WEIGHTS = {
    'quality_score': 0.30,
    'utility_score': 0.30,
    'privacy_score': 0.25,
    'bias_score': 0.15,
}

# Hyperparameter optimization objective weights
OPTIMIZATION_OBJECTIVE_WEIGHTS = {
    'quality_score': 0.35,
    'utility_score': 0.35,
    'privacy_score': 0.30,
}

# Hyperparameter search spaces
HYPERPARAMETER_SEARCH_SPACES = {
    'CTGAN': {
        'epochs': (100, 500),
        'batch_size': (100, 1000),
        'embedding_dim': (64, 256),
        'generator_lr': (1e-5, 1e-3),
        'discriminator_lr': (1e-5, 1e-3),
        'pac': (1, 10),
    },
    'TVAE': {
        'epochs': (100, 500),
        'batch_size': (100, 1000),
        'embedding_dim': (64, 256),
        'latent_dim': (64, 256),
        'learning_rate': (1e-5, 1e-3),
    },
    'CopulaGAN': {
        'epochs': (100, 500),
        'batch_size': (100, 1000),
        'embedding_dim': (64, 256),
        'generator_lr': (1e-5, 1e-3),
        'discriminator_lr': (1e-5, 1e-3),
    },
}
