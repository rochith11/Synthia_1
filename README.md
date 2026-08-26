# Synthia

Synthia is a research oriented engineering project for synthetic rare disease variant data generation, evaluation, and iterative quality optimization. It combines tabular generative modeling with a validation stack for statistical fidelity, machine learning utility, privacy risk, bias analysis, lineage tracking, and diagnostic feedback.

The project is built around an applied question: how can we generate synthetic biomedical style tabular data that is useful for downstream experimentation while remaining auditable, privacy aware, and reproducible? To answer that, Synthia implements an end-to-end pipeline using SDV based generative models, metric driven validation, and an AI diagnostic agent that interprets failures and proposes optimization actions.

## Research Motivation

Rare disease datasets are often small, imbalanced, operationally sensitive, and difficult to share. Synthia approaches this as a systems problem rather than only a modeling problem:

- Generation quality must be measured beyond sample realism.
- Utility must be checked against held out real data.
- Privacy risk must be estimated explicitly.
- Bias amplification and rare-class loss must be surfaced early.
- Every generated artifact should remain traceable to its training context.

This makes the repository suitable as an applied prototype for synthetic health data workflows, academic demonstrations, capstone engineering work, and experimentation on trustworthy data generation pipelines.

## Core Capabilities

- Synthetic tabular data generation using `CTGAN` and `TVAE`
- Optional disease-conditioned generation flow in the main pipeline
- Train/test split workflow for holdout-based evaluation
- Statistical validation with Kolmogorov-Smirnov statistics, Jensen-Shannon divergence, and correlation preservation
- ML utility validation through bidirectional cross-testing
- Privacy analysis using nearest-neighbor distance and re-identification risk estimates
- Bias analysis for feature drift, rare class imbalance, and amplification
- Persistent dataset storage with metadata, lineage, and audit logging
- AI diagnostic agent for profiling, root-cause analysis, benchmark tracking, and recommendation generation
- Flask web interface for interactive experimentation and result inspection
- Unit and property-based testing for core pipeline behavior

## System Overview

Synthia follows a multi stage evaluation loop:

1. Load training and holdout test data.
2. Train a generative model on real training data.
3. Generate synthetic records.
4. Validate synthetic data against held out real data.
5. Measure privacy and bias risk.
6. Save artifacts with metadata and audit trace.
7. Run an AI diagnostic cycle to explain weak metrics and suggest next experiments.

## Methodology

### 1. Generative Models

The generation layer is orchestrated by [`src/core/synthetic_data_generator.py`](src/core/synthetic_data_generator.py) and routes to:

- `CTGAN` for mixed-type tabular generation and conditional sampling
- `TVAE` as an alternative latent-variable baseline

The current sample domain is variant-style rare-disease metadata with seven key fields:

- `gene_symbol`
- `chromosome`
- `variant_type`
- `clinical_significance`
- `disease`
- `allele_frequency`
- `inheritance_pattern`

### 2. Statistical Fidelity

The validation pipeline in [`src/analysis/data_validator.py`](src/analysis/data_validator.py) and [`src/analysis/statistical_analyzer.py`](src/analysis/statistical_analyzer.py) compares synthetic and real distributions through:

- Kolmogorov-Smirnov statistics for numerical features
- Jensen-Shannon divergence for categorical and numerical distributions
- Correlation similarity across numerical structure
- Mode-collapse checks for dominant-category failure modes

### 3. Utility Assessment

Synthia evaluates downstream usefulness with cross-domain predictive testing:

- Train on synthetic, test on real
- Train on real, test on synthetic

This yields accuracy, weighted F1, and AUC estimates, helping separate visually plausible generation from genuinely useful generation.

### 4. Privacy Analysis

The privacy module in [`src/analysis/privacy_analyzer.py`](src/analysis/privacy_analyzer.py) estimates memorization risk using:

- nearest-neighbor distance (NND) between synthetic and real samples
- percentile-based re-identification risk thresholds
- a composite privacy score derived from average distance and high-risk fraction

### 5. Bias and Rare-Class Analysis

The bias module in [`src/analysis/bias_detector.py`](src/analysis/bias_detector.py) checks:

- feature-level distribution divergence
- rare-class underrepresentation or disappearance
- amplification of imbalance ratios in categorical variables

### 6. Diagnostic Optimization

The AI diagnostic subsystem in [`src/ai_diagnostic_agent/`](src/ai_diagnostic_agent/) expands the project from a generator into an optimization framework. It profiles data, classifies metric severity, infers likely root causes, checks benchmark attainment, and proposes next-step interventions such as parameter changes or dataset-focused recommendations.

## Repository Structure

```text
synthia/
|-- app.py
|-- run_pipeline.py
|-- run_diagnostic.py
|-- config.yaml
|-- requirements.txt
|-- data/
|-- scripts/
|-- tests/
`-- src/
    |-- analysis/
    |-- ai_diagnostic_agent/
    |-- core/
    |-- engines/
    |-- models/
    |-- storage/
    `-- utils/
```

### Important Modules

- [`app.py`](app.py): Flask application for the interactive web workflow
- [`run_pipeline.py`](run_pipeline.py): CLI entrypoint for generation, validation, privacy, bias, and persistence
- [`run_diagnostic.py`](run_diagnostic.py): CLI entrypoint for diagnostic cycles, profiling, benchmarks, and model comparison
- [`src/core/`](src/core/): generation orchestration
- [`src/engines/`](src/engines/): SDV model wrappers
- [`src/analysis/`](src/analysis/): evaluation, privacy, and bias analytics
- [`src/storage/`](src/storage/): dataset persistence and lineage
- [`src/utils/`](src/utils/): configuration, loading, logging, metadata, audit
- [`src/ai_diagnostic_agent/`](src/ai_diagnostic_agent/): diagnostic intelligence and optimization planning
- [`tests/`](tests/): unit and property-based tests

## Data and Experimental Setup

The repository includes sample tabular data under [`data/`](data/) with predefined train/test splits:

- `sample_real_variants.csv`
- `sample_real_variants_train.csv`
- `sample_real_variants_test.csv`

These files support a reproducible demonstration workflow for rare-disease variant-style generation. If the expected sample files are missing, the loader utilities can regenerate a small synthetic seed dataset for local experimentation.

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run the end-to-end generation pipeline

```bash
python run_pipeline.py
```

Example with explicit settings:

```bash
python run_pipeline.py --model CTGAN --samples 500 --epochs 300 --seed 42
```

Example with disease filtering:

```bash
python run_pipeline.py --model CTGAN --samples 300 --disease "Cystic Fibrosis"
```

### Launch the web application

```bash
python app.py
```

Then open `http://localhost:5000`.

### Run the diagnostic agent

```bash
python run_diagnostic.py run --model CTGAN --samples 500 --epochs 300 --format text
```

### Profile the training dataset

```bash
python run_diagnostic.py profile
```

### Compare multiple models

```bash
python run_diagnostic.py optimize --models CTGAN TVAE --samples 500
```

## Configuration

Project behavior is controlled through [`config.yaml`](config.yaml). Key configuration areas include:

- default generation parameters
- disease templates
- privacy thresholds
- bias thresholds
- diagnostic benchmark targets
- optimization and stall-detection settings
- dataset-size heuristics

This design makes experiments easier to reproduce and modify without changing code.

## Evaluation Outputs

The pipeline produces several classes of outputs:

- synthetic datasets stored under `data/datasets/`
- dataset metadata and lineage records
- audit logs in `data/audit_log.jsonl`
- experiment tracking files in `data/experiments/` or related diagnostic storage
- diagnostic reports in text, JSON, or HTML form

## Testing

The repository includes both conventional and property-based tests.

Run the full suite with:

```bash
pytest
```

The tests cover:

- schema and configuration models
- statistical and privacy metric validity
- dataset persistence and export behavior
- audit logging behavior
- routing and metadata invariants

## Engineering Characteristics

This project is intentionally structured as applied research software:

- modular pipeline design for ablation and extension
- reproducibility via seeds, config management, and metadata hashing
- explicit trustworthiness checks for privacy and bias
- benchmark-driven diagnostics instead of single-metric reporting
- auditable storage and experiment history for iterative evaluation

## Current Scope and Limitations

- The included dataset is a compact sample dataset intended for experimentation, not clinical deployment.
- Privacy analysis is heuristic and distance-based; it should not be treated as a formal privacy guarantee.
- Model performance and benchmark attainment depend heavily on dataset size, class balance, and feature quality.
- The web application is a research interface for demonstration and inspection, not a hardened production service.

## Future Extension Directions

- richer biomedical schemas and constraint-aware generation
- stronger privacy auditing and membership-inference style attacks
- broader model orchestration beyond CTGAN and TVAE
- automated experiment loops with hyperparameter search
- domain-specific post-generation filtering and calibration
- improved dataset cards and reporting for governance workflows
