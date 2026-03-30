# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthia is a synthetic rare disease data generation system built with Python. It uses statistical generative models (CTGAN and TVAE from SDV library) to create realistic synthetic genetic variant data for research purposes while preserving privacy.

## Architecture

### Core Components

1. **Data Models** (`src/models/`)
   - `VariantMetadata`: 7-field dataclass with validation for genetic variants
   - `GenerationConfig`: Configuration dataclass for model parameters
   - Report schemas: ValidationReport, PrivacyReport, BiasReport

2. **Generation Engines** (`src/engines/`)
   - `CTGANEngine`: Wraps SDV's CTGAN with disease-conditioned generation
   - `TVAEEngine`: Wraps SDV's TVAE with lower computational cost

3. **Orchestrator** (`src/core/`)
   - `SyntheticDataGenerator`: Main class that selects and trains appropriate model

4. **Analysis Modules** (`src/analysis/`)
   - `DataValidator`: Statistical validation using KS tests, JSD, correlation analysis
   - `PrivacyAnalyzer`: Nearest neighbor distance privacy metrics
   - `BiasDetector`: Bias amplification detection

5. **AI Diagnostic Agent** (`src/ai_diagnostic_agent/`)
   - Advanced optimization system with root cause analysis
   - Multi-model orchestration and hyperparameter optimization
   - Comprehensive diagnostic reporting

## Common Development Commands

### Running the Application

```bash
# Start the Flask web interface
python app.py

# Run the CLI pipeline with default settings
python run_pipeline.py

# Run diagnostic agent commands
python run_diagnostic.py run          # Full diagnostic cycle
python run_diagnostic.py profile      # Profile training data
python run_diagnostic.py optimize     # Multi-model optimization
python run_diagnostic.py experiments  # List tracked experiments
python run_diagnostic.py benchmarks   # Show benchmark dashboard
```

### Testing

```bash
# Run verification script for Phase 1
python scripts/phase1_verify.py

# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/
```

### Development Workflow

1. **Configuration**: Modify `config.yaml` to change default settings
2. **Data**: Sample data in `data/` directory (100 records with 70/30 train/test split)
3. **Models**: Trained models saved in `models/` directory
4. **Datasets**: Generated datasets saved in `data/datasets/`
5. **Reports**: Diagnostic reports in `data/reports/`

## Key Design Patterns

1. **Modular Architecture**: Separation of concerns with clear component boundaries
2. **Configuration Driven**: All settings in `config.yaml` for easy customization
3. **Extensible Engines**: Easy to add new generative models by implementing engine interface
4. **Comprehensive Validation**: Multi-layer validation of synthetic data quality
5. **Audit Trail**: Full logging of all operations for reproducibility

## File Structure

```
synthia/
├── src/                    # Source code
│   ├── core/              # Main orchestrator
│   ├── engines/           # CTGAN and TVAE wrappers
│   ├── models/            # Data models and schemas
│   ├── analysis/          # Validation, privacy, bias analysis
│   ├── ai_diagnostic_agent/ # Advanced optimization system
│   ├── utils/             # Utility functions
│   └── storage/           # Dataset repository
├── data/                  # Sample data and generated datasets
├── models/                # Saved trained models
├── scripts/               # Verification and utility scripts
├── tests/                 # Unit and integration tests
├── templates/             # Flask web templates
├── static/                # Static web assets
├── app.py                 # Flask web interface
├── run_pipeline.py        # CLI pipeline runner
├── run_diagnostic.py      # Diagnostic agent CLI
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
└── PHASE1_SUMMARY.md      # Phase 1 implementation summary
```

## Enterprise Benchmarks

The diagnostic agent targets these quality thresholds:
- Quality Score: ≥ 0.85
- Privacy Score: ≥ 0.90
- Utility Accuracy: ≥ 0.80
- Max JSD: ≤ 0.10
- Max KS: ≤ 0.15

## Optimization Features

1. **Root Cause Analysis**: Automatic diagnosis of quality issues
2. **Feature-Level Diagnostics**: Per-column analysis and recommendations
3. **Hyperparameter Optimization**: Bayesian/random search for best parameters
4. **Multi-Model Orchestration**: Train and compare CTGAN, TVAE, CopulaGAN
5. **Experiment Tracking**: Full history of optimization cycles
6. **Benchmark Dashboard**: Real-time enterprise compliance monitoring