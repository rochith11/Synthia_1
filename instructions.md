# Synthia Enhancement: Multi-Model Training and Selection

This document outlines the implementation plan for enhancing Synthia to train multiple generative models on the new `@data/training.csv` dataset and automatically select the best performing model based on data characteristics and performance metrics.

## Overview

Currently, Synthia trains a single model (CTGAN or TVAE) based on configuration. This enhancement will:

1. Enable training of multiple models simultaneously on the new dataset
2. Evaluate models using comprehensive metrics
3. Select the best model based on performance characteristics
4. Integrate seamlessly with the existing diagnostic agent

## Implementation Phases

### Phase 1: Data Integration and Preprocessing

#### Tasks:
- [ ] Update data loading to support the new `@data/training.csv` dataset
- [ ] Implement data profiling to understand characteristics of the new dataset
- [ ] Create preprocessing pipeline for the new dataset format
- [ ] Ensure compatibility with existing discrete column definitions

#### Implementation Details:
- Extend `src/utils/data_loader.py` to load the new training.csv
- Enhance `src/ai_diagnostic_agent/profiling/data_profiler.py` to analyze the new dataset
- Update `src/preprocessing/` with any necessary transformations

### Phase 2: Multi-Model Training Framework

#### Tasks:
- [ ] Extend `src/ai_diagnostic_agent/optimization/model_orchestrator.py` to support the new dataset
- [ ] Implement parallel or sequential training of CTGAN and TVAE models
- [ ] Add support for training configuration based on data characteristics
- [ ] Create model comparison functionality

#### Implementation Details:
- Modify `ModelOrchestrator` to accept the new dataset
- Implement training configuration that adapts to data size/complexity
- Ensure models are trained with appropriate hyperparameters

### Phase 3: Model Evaluation and Selection

#### Tasks:
- [ ] Enhance evaluation metrics collection from all trained models
- [ ] Implement model selection algorithm based on composite scores
- [ ] Add detailed comparison reporting
- [ ] Integrate with existing diagnostic agent metrics

#### Implementation Details:
- Leverage existing `evaluate_all_models` functionality
- Use the composite scoring mechanism in `_compute_composite_score`
- Create selection criteria based on quality, privacy, utility, and bias metrics

### Phase 4: Integration with Existing Pipeline

#### Tasks:
- [ ] Update `run_pipeline.py` to support multi-model training
- [ ] Modify `run_diagnostic.py` to use the new multi-model approach
- [ ] Ensure backward compatibility with single-model configurations
- [ ] Add command-line options for multi-model training

#### Implementation Details:
- Add new command-line arguments for model selection strategy
- Maintain existing functionality while adding new capabilities
- Update configuration handling to support both approaches

### Phase 5: Reporting and Visualization

#### Tasks:
- [ ] Create comparative reports for all trained models
- [ ] Add visualization of model performance metrics
- [ ] Implement selection rationale documentation
- [ ] Enhance diagnostic agent reporting

#### Implementation Details:
- Extend `src/ai_diagnostic_agent/report_generator.py`
- Add model comparison charts and tables
- Document why a particular model was selected

## Technical Considerations

### Performance Optimization
- Implement parallel training if computational resources allow
- Add progress tracking for long-running training processes
- Optimize memory usage during multi-model training

### Error Handling
- Handle training failures gracefully
- Implement fallback mechanisms if one model fails
- Add comprehensive logging for debugging

### Scalability
- Ensure the solution works with datasets of varying sizes
- Implement adaptive hyperparameters based on data characteristics
- Add support for additional model types in the future

## Expected Outcomes

1. Improved synthetic data quality through automatic model selection
2. Better performance metrics across quality, privacy, and utility dimensions
3. Enhanced flexibility in handling diverse datasets
4. Seamless integration with existing Synthia workflows

## Testing Strategy

1. Unit tests for new multi-model training components
2. Integration tests with the new dataset
3. Performance benchmarks comparing single vs multi-model approaches
4. Validation of model selection accuracy

## Rollout Plan

1. Develop in feature branch with incremental commits
2. Test with sample datasets before full deployment
3. Document usage in README and CLAUDE.md
4. Provide migration guide for existing configurations