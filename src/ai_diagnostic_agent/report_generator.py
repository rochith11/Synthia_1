"""Automated Report Generator for diagnostic cycle results."""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.ai_diagnostic_agent.models import (
    DiagnosticReport, MetricAnalysis, QualityLevel,
    RootCause, ColumnDiagnosis, Recommendation,
)


class ReportGenerator:
    """Generates comprehensive diagnostic reports in multiple formats."""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        self.report_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

    def generate_diagnostic_report(
        self, report: DiagnosticReport, format: str = 'text'
    ) -> str:
        """Generate a structured diagnostic report.

        Parameters
        ----------
        report : DiagnosticReport
            The diagnostic report data.
        format : str
            Output format: 'text', 'html', or 'json'.

        Returns
        -------
        str
            The formatted report content.
        """
        if format == 'json':
            content = self._generate_json(report)
        elif format == 'html':
            content = self._generate_html(report)
        else:
            content = self._generate_text(report)

        # Save to file
        ext = {'text': 'txt', 'html': 'html', 'json': 'json'}.get(format, 'txt')
        filename = f"diagnostic_report_cycle_{report.cycle_number}_{report.report_id}.{ext}"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info("Report saved to %s", filepath)
        except Exception as e:
            self.logger.error("Failed to save report: %s", e)

        # Record in history
        self.report_history.append({
            'cycle': report.cycle_number,
            'report_id': report.report_id,
            'filepath': filepath,
            'format': format,
            'timestamp': report.timestamp,
        })

        return content

    def _generate_text(self, report: DiagnosticReport) -> str:
        """Generate plain-text diagnostic report."""
        lines = []
        sep = "=" * 70

        # Header
        lines.append(sep)
        lines.append(f"  DIAGNOSTIC REPORT - Cycle #{report.cycle_number}")
        lines.append(f"  Report ID: {report.report_id}")
        lines.append(f"  Generated: {report.timestamp}")
        lines.append(sep)
        lines.append("")

        # Section 1: Metric Interpretation
        lines.append(self._section_header("1. METRIC INTERPRETATION"))
        if report.metric_analysis:
            ma = report.metric_analysis
            lines.append(f"  Overall Status: {ma.overall_status}")
            lines.append("")

            lines.append("  Quality Metrics:")
            for name, (val, level) in ma.quality_metrics.items():
                lines.append(f"    {name:25s} = {val:.4f}  [{level.value}]")

            lines.append("  Utility Metrics:")
            for name, (val, level) in ma.utility_metrics.items():
                lines.append(f"    {name:25s} = {val:.4f}  [{level.value}]")

            lines.append("  Privacy Metrics:")
            for name, (val, level) in ma.privacy_metrics.items():
                lines.append(f"    {name:25s} = {val:.4f}  [{level.value}]")

            lines.append("  Bias Metrics:")
            for name, (val, level) in ma.bias_metrics.items():
                lines.append(f"    {name:25s} = {val:.4f}  [{level.value}]")

            lines.append("")
            if ma.weakest_metrics:
                lines.append(f"  Weakest Metrics: {', '.join(ma.weakest_metrics)}")

            lines.append("")
            lines.append("  Benchmark Deltas:")
            for name, delta in ma.benchmark_deltas.items():
                sign = "+" if delta >= 0 else ""
                status = "ABOVE" if delta >= 0 else "BELOW"
                lines.append(f"    {name:25s}: {sign}{delta:.4f} ({status} target)")
        else:
            lines.append("  No metric analysis available.")
        lines.append("")

        # Section 2: Root Cause Analysis
        lines.append(self._section_header("2. ROOT CAUSE ANALYSIS"))
        if report.root_causes:
            for i, rc in enumerate(report.root_causes, 1):
                lines.append(f"  [{i}] {rc.cause_name}")
                lines.append(f"      Impact: {rc.impact} | Likelihood: {rc.likelihood*100:.0f}%")
                lines.append(f"      {rc.description}")
                if rc.affected_metrics:
                    lines.append(f"      Affected: {', '.join(rc.affected_metrics)}")
                if rc.evidence:
                    lines.append("      Evidence:")
                    for ev in rc.evidence:
                        lines.append(f"        - {ev}")
                if rc.recommendations:
                    lines.append("      Recommendations:")
                    for rec in rc.recommendations:
                        lines.append(f"        - {rec}")
                lines.append("")
        else:
            lines.append("  No root causes identified.")
        lines.append("")

        # Section 3: Column-Level Diagnostics
        lines.append(self._section_header("3. COLUMN-LEVEL DIAGNOSTICS"))
        if report.column_diagnostics:
            critical = [cd for cd in report.column_diagnostics if cd.severity == QualityLevel.CRITICAL]
            warning = [cd for cd in report.column_diagnostics if cd.severity == QualityLevel.WARNING]
            lines.append(f"  Total columns: {len(report.column_diagnostics)}")
            lines.append(f"  Critical: {len(critical)}, Warning: {len(warning)}")
            lines.append("")

            for cd in report.column_diagnostics:
                if cd.severity in (QualityLevel.CRITICAL, QualityLevel.WARNING):
                    lines.append(f"  {cd.column_name} [{cd.severity.value}]")
                    lines.append(f"    JSD: {cd.jsd:.4f}")
                    if cd.ks_statistic is not None:
                        lines.append(f"    KS:  {cd.ks_statistic:.4f} (p={cd.ks_pvalue:.4f})")
                    if cd.bias_amplification_ratio > 1.2:
                        lines.append(f"    Bias Amplification: {cd.bias_amplification_ratio:.2f}x")
                    for issue in cd.issues:
                        lines.append(f"    - {issue}")
                    lines.append("")
        else:
            lines.append("  No column diagnostics available.")
        lines.append("")

        # Section 4: Correlation Analysis
        if report.correlation_analysis:
            lines.append(self._section_header("3b. CORRELATION PRESERVATION"))
            ca = report.correlation_analysis
            lines.append(f"  Overall Similarity: {ca.overall_similarity:.4f}")
            lines.append(f"  Max Loss: {ca.max_loss:.4f}")
            lines.append(f"  Lost Pairs: {len(ca.lost_pairs)}")
            if ca.lost_pairs:
                lines.append("  Top lost correlations:")
                for col1, col2, diff in ca.lost_pairs[:5]:
                    lines.append(f"    {col1} <-> {col2}: diff = {diff:.4f}")
            lines.append("")

        # Section 5: Diversity Report
        if report.diversity_report:
            lines.append(self._section_header("4. DIVERSITY ANALYSIS"))
            dr = report.diversity_report
            lines.append(f"  Unique Row Ratio: {dr.unique_row_ratio:.4f}")
            lines.append(f"  Duplicate Count:  {dr.duplicate_count}")
            lines.append(f"  Duplicate Rate:   {dr.duplicate_rate:.4f}")
            lines.append(f"  Mode Collapse:    {'DETECTED' if dr.mode_collapse_detected else 'No'}")
            lines.append(f"  Diversity Score:  {dr.diversity_score:.4f}")
            lines.append(f"  Severity:         {dr.severity.value}")
            if dr.issues:
                lines.append("  Issues:")
                for issue in dr.issues:
                    lines.append(f"    - {issue}")
            lines.append("")

        # Section 6: Constraint Validation
        if report.constraint_validation:
            lines.append(self._section_header("5. CONSTRAINT VALIDATION"))
            cv = report.constraint_validation
            lines.append(f"  Total Records:   {cv.total_records}")
            lines.append(f"  Valid Records:   {cv.valid_records}")
            lines.append(f"  Invalid Records: {cv.invalid_records}")
            lines.append(f"  Filter Rate:     {cv.filter_percentage*100:.2f}%")
            lines.append(f"  Critical:        {'YES' if cv.is_critical else 'No'}")
            lines.append("")

        # Section 7: Pipeline Improvements
        lines.append(self._section_header("6. PIPELINE IMPROVEMENT RECOMMENDATIONS"))
        if report.recommendations:
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  [{i}] [P{rec.priority}] {rec.title}")
                lines.append(f"      Category: {rec.category}")
                lines.append(f"      {rec.description}")
                lines.append(f"      Rationale: {rec.rationale}")
                if rec.expected_improvements:
                    improvements = ", ".join(
                        f"{k}: {v}" for k, v in rec.expected_improvements.items()
                    )
                    lines.append(f"      Expected: {improvements}")
                lines.append("")
        else:
            lines.append("  No recommendations generated.")
        lines.append("")

        # Section 8: Next Experiment Plan
        lines.append(self._section_header("7. NEXT EXPERIMENT PLAN"))
        if report.experiment_plan:
            ep = report.experiment_plan
            lines.append(f"  Plan ID: {ep.plan_id}")
            lines.append(f"  Based on: {ep.parent_experiment_id or 'Initial'}")
            if ep.configuration_changes:
                lines.append("  Configuration Changes:")
                for key, change in ep.configuration_changes.items():
                    lines.append(f"    {key}: {change}")
            if ep.predicted_improvements:
                lines.append("  Predicted Improvements:")
                for metric, delta in ep.predicted_improvements.items():
                    lines.append(f"    {metric}: {delta:+.4f}")
            lines.append(f"  Rationale: {ep.rationale[:200]}")
        else:
            lines.append("  No experiment plan generated.")
        lines.append("")

        # Footer
        lines.append(sep)
        lines.append(f"  Benchmarks Met: {'YES' if report.benchmarks_met else 'NO'}")
        lines.append(f"  Optimization Stalled: {'YES' if report.optimization_stalled else 'No'}")
        lines.append(sep)

        return "\n".join(lines)

    def _generate_json(self, report: DiagnosticReport) -> str:
        """Generate JSON diagnostic report."""
        return json.dumps(report.to_dict(), indent=2, default=str)

    def _generate_html(self, report: DiagnosticReport) -> str:
        """Generate HTML diagnostic report."""
        lines = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html><head>")
        lines.append("<meta charset='utf-8'>")
        lines.append(f"<title>Diagnostic Report - Cycle #{report.cycle_number}</title>")
        lines.append("<style>")
        lines.append("body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 900px; margin: 2em auto; padding: 1em; }")
        lines.append("h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5em; }")
        lines.append("h2 { color: #34495e; margin-top: 1.5em; }")
        lines.append("table { border-collapse: collapse; width: 100%; margin: 1em 0; }")
        lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        lines.append("th { background: #3498db; color: white; }")
        lines.append("tr:nth-child(even) { background: #f2f2f2; }")
        lines.append(".critical { color: #e74c3c; font-weight: bold; }")
        lines.append(".warning { color: #f39c12; font-weight: bold; }")
        lines.append(".acceptable { color: #27ae60; }")
        lines.append(".excellent { color: #2ecc71; font-weight: bold; }")
        lines.append(".badge { padding: 2px 8px; border-radius: 4px; color: white; font-size: 0.85em; }")
        lines.append(".badge-critical { background: #e74c3c; }")
        lines.append(".badge-warning { background: #f39c12; }")
        lines.append(".badge-acceptable { background: #27ae60; }")
        lines.append(".badge-excellent { background: #2ecc71; }")
        lines.append(".badge-met { background: #2ecc71; }")
        lines.append(".badge-notmet { background: #e74c3c; }")
        lines.append("</style></head><body>")

        lines.append(f"<h1>Diagnostic Report - Cycle #{report.cycle_number}</h1>")
        lines.append(f"<p><strong>Report ID:</strong> {report.report_id} | <strong>Generated:</strong> {report.timestamp}</p>")

        # Metric Analysis
        lines.append("<h2>1. Metric Interpretation</h2>")
        if report.metric_analysis:
            ma = report.metric_analysis
            badge_class = f"badge-{ma.overall_status.lower()}" if ma.overall_status.lower() in ('critical', 'warning', 'acceptable', 'excellent') else ''
            lines.append(f"<p>Overall Status: <span class='badge {badge_class}'>{ma.overall_status}</span></p>")

            lines.append("<table><tr><th>Metric</th><th>Value</th><th>Level</th><th>Benchmark Delta</th></tr>")
            for metrics_dict in [ma.quality_metrics, ma.utility_metrics, ma.privacy_metrics, ma.bias_metrics]:
                for name, (val, level) in metrics_dict.items():
                    level_class = level.value.lower()
                    delta = ma.benchmark_deltas.get(name, 0)
                    delta_str = f"{delta:+.4f}"
                    lines.append(
                        f"<tr><td>{name}</td><td>{val:.4f}</td>"
                        f"<td><span class='badge badge-{level_class}'>{level.value}</span></td>"
                        f"<td>{delta_str}</td></tr>"
                    )
            lines.append("</table>")

        # Root Causes
        lines.append("<h2>2. Root Cause Analysis</h2>")
        if report.root_causes:
            for rc in report.root_causes:
                lines.append(f"<h3>{rc.cause_name} <span class='badge badge-{rc.impact.lower()}'>{rc.impact}</span></h3>")
                lines.append(f"<p>Likelihood: {rc.likelihood*100:.0f}% | {rc.description}</p>")
                if rc.evidence:
                    lines.append("<ul>")
                    for ev in rc.evidence:
                        lines.append(f"<li>{ev}</li>")
                    lines.append("</ul>")
        else:
            lines.append("<p>No root causes identified.</p>")

        # Recommendations
        lines.append("<h2>3. Pipeline Improvement Recommendations</h2>")
        if report.recommendations:
            lines.append("<table><tr><th>Priority</th><th>Title</th><th>Category</th><th>Description</th></tr>")
            for rec in report.recommendations:
                lines.append(
                    f"<tr><td>P{rec.priority}</td><td>{rec.title}</td>"
                    f"<td>{rec.category}</td><td>{rec.description}</td></tr>"
                )
            lines.append("</table>")

        # Benchmark Status
        met_str = "MET" if report.benchmarks_met else "NOT MET"
        met_class = "badge-met" if report.benchmarks_met else "badge-notmet"
        lines.append(f"<h2>Benchmark Status: <span class='badge {met_class}'>{met_str}</span></h2>")

        if report.optimization_stalled:
            lines.append("<p class='critical'>WARNING: Optimization has stalled.</p>")

        lines.append("</body></html>")
        return "\n".join(lines)

    def _section_header(self, title: str) -> str:
        """Format a section header."""
        return f"--- {title} ---"

    def get_report_history(self) -> List[Dict[str, Any]]:
        """Return the report generation history."""
        return list(self.report_history)

    def export_all_reports(self, report: DiagnosticReport) -> Dict[str, str]:
        """Export a report in all supported formats."""
        results = {}
        for fmt in ['text', 'html', 'json']:
            content = self.generate_diagnostic_report(report, format=fmt)
            results[fmt] = content
        return results
