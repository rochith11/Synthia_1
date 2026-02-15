"""Synthia Flask Web Interface.

Run: python app.py
Visit: http://localhost:5000
"""

import sys
import os
import io
import warnings
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify

from src.utils.config_manager import ConfigManager
from src.utils.data_loader import load_training_data, load_test_data, create_sample_data
from src.models.generation_config import GenerationConfig
from src.core.synthetic_data_generator import SyntheticDataGenerator
from src.analysis.data_validator import DataValidator
from src.analysis.privacy_analyzer import PrivacyAnalyzer
from src.analysis.bias_detector import BiasDetector
from src.storage.dataset_repository import DatasetRepository
from src.utils.audit_logger import AuditLogger

app = Flask(__name__)
app.secret_key = 'synthia-dev-key'

# Shared state
repo = DatasetRepository()
audit = AuditLogger()

# Store latest run results in memory for display
_latest_run = {}
_run_lock = threading.Lock()


def _get_data():
    """Load training and test data, creating sample if needed."""
    try:
        train = load_training_data()
        test = load_test_data()
    except FileNotFoundError:
        _, train, test = create_sample_data()
    return train, test


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route('/')
def index():
    """Landing page with configuration form."""
    try:
        cfg = ConfigManager()
        defaults = cfg.get_model_defaults()
    except FileNotFoundError:
        defaults = {
            'model_type': 'CTGAN', 'n_samples': 1000,
            'random_seed': 42, 'epochs': 300, 'batch_size': 500
        }

    return render_template('index.html', defaults=defaults)


@app.route('/generate', methods=['POST'])
def generate():
    """Run the full pipeline and redirect to results."""
    model_type = request.form.get('model_type', 'CTGAN')
    n_samples = int(request.form.get('n_samples', 100))
    seed = int(request.form.get('random_seed', 42))
    epochs = int(request.form.get('epochs', 100))
    batch_size = int(request.form.get('batch_size', 500))
    disease = request.form.get('disease', '').strip() or None

    train_data, test_data = _get_data()

    if disease:
        train_data = train_data[train_data['disease'] == disease].reset_index(drop=True)
        test_data = test_data[test_data['disease'] == disease].reset_index(drop=True)
        if len(train_data) == 0:
            flash(f"No training records for disease: {disease}", "error")
            return redirect(url_for('index'))

    # Generation
    config = GenerationConfig(
        model_type=model_type,
        n_samples=n_samples,
        random_seed=seed,
        hyperparameters={'epochs': epochs, 'batch_size': batch_size},
        disease_condition=disease,
    )
    generator = SyntheticDataGenerator(config=config)
    generator.train(train_data, config=config)
    synthetic = generator.generate(n_samples=n_samples)

    # Validation
    validator = DataValidator()
    val_report = validator.validate(
        synthetic=synthetic, real=test_data,
        target_column='disease', dataset_id='web-run'
    )

    # Privacy
    privacy = PrivacyAnalyzer()
    priv_report = privacy.analyze_privacy(
        synthetic=synthetic, real=test_data,
        percentile=90, dataset_id='web-run'
    )

    # Bias
    bias = BiasDetector()
    bias_results = bias.analyze_bias(
        synthetic=synthetic, real=test_data,
        amplification_threshold=0.20, rare_class_threshold=0.05
    )

    # Save
    gen_meta = generator.get_metadata()
    gen_meta['quality_score'] = val_report.overall_quality_score
    gen_meta['privacy_score'] = priv_report.privacy_score
    dataset_id = repo.save_dataset(
        data=synthetic, metadata=gen_meta,
        dataset_name=f"{model_type}-{disease or 'all'}-{n_samples}"
    )

    audit.log("generate", resource_id=dataset_id, details={
        "model": model_type, "n_samples": n_samples
    })

    # Store results
    with _run_lock:
        _latest_run['config'] = {
            'model_type': model_type, 'n_samples': n_samples,
            'seed': seed, 'epochs': epochs, 'disease': disease,
        }
        _latest_run['dataset_id'] = dataset_id
        _latest_run['synthetic_preview'] = synthetic.head(20).to_dict(orient='records')
        _latest_run['synthetic_columns'] = list(synthetic.columns)
        _latest_run['n_generated'] = len(synthetic)

        # Validation
        summary = val_report.statistical_metrics.get('summary', {})
        cross = val_report.utility_metrics.get('cross_test', {})
        s2r = cross.get('synthetic_to_real', {})
        _latest_run['validation'] = {
            'mean_ks': summary.get('mean_ks_statistic', 0),
            'mean_jsd': summary.get('mean_jsd', 0),
            'max_jsd': summary.get('max_jsd', 0),
            'corr_sim': summary.get('correlation_similarity'),
            'accuracy': s2r.get('accuracy', 0),
            'f1': s2r.get('f1_score', 0),
            'auc': s2r.get('auc', 0),
            'quality_score': val_report.overall_quality_score,
            'ks_tests': val_report.statistical_metrics.get('ks_tests', {}),
            'js_divergences': val_report.statistical_metrics.get('js_divergences', {}),
            'collapse_info': val_report.statistical_metrics.get('collapse_info', {}),
        }

        # Privacy
        _latest_run['privacy'] = {
            'nnd': priv_report.nearest_neighbor_distances,
            'risk': priv_report.reidentification_risk,
            'score': priv_report.privacy_score,
        }

        # Bias
        _latest_run['bias'] = bias_results

    flash("Generation complete!", "success")
    return redirect(url_for('results'))


@app.route('/results')
def results():
    """Display latest generation results."""
    with _run_lock:
        run = dict(_latest_run)
    if not run:
        flash("No results yet. Run a generation first.", "info")
        return redirect(url_for('index'))
    return render_template('results.html', run=run)


@app.route('/datasets')
def datasets():
    """List all saved datasets."""
    all_ds = repo.list_datasets()
    return render_template('datasets.html', datasets=all_ds)


@app.route('/datasets/<dataset_id>/download')
def download_dataset(dataset_id):
    """Download a dataset as CSV."""
    try:
        data, _ = repo.load_dataset(dataset_id)
        buf = io.StringIO()
        data.to_csv(buf, index=False)
        buf.seek(0)
        audit.log("export", resource_id=dataset_id, details={"format": "csv"})
        return send_file(
            io.BytesIO(buf.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"{dataset_id}.csv"
        )
    except FileNotFoundError:
        flash("Dataset not found.", "error")
        return redirect(url_for('datasets'))


@app.route('/datasets/<dataset_id>')
def dataset_detail(dataset_id):
    """Show dataset details and preview."""
    try:
        data, meta = repo.load_dataset(dataset_id)
        lineage = repo.get_dataset_lineage(dataset_id)
        preview = data.head(20).to_dict(orient='records')
        columns = list(data.columns)
        return render_template('dataset_detail.html',
                               meta=meta, lineage=lineage,
                               preview=preview, columns=columns,
                               dataset_id=dataset_id)
    except FileNotFoundError:
        flash("Dataset not found.", "error")
        return redirect(url_for('datasets'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
