"""Constraint Validator for enforcing domain-specific rules on synthetic data."""

import logging
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from src.ai_diagnostic_agent.models import Constraint, ConstraintValidationResult, QualityLevel
from src.ai_diagnostic_agent.config import CONSTRAINT_FILTER_CRITICAL_THRESHOLD


class ConstraintValidator:
    """Validates synthetic data against domain-specific constraints."""

    def __init__(self, constraints: List[Constraint] = None):
        self.constraints = constraints if constraints is not None else self._default_constraints()
        self.logger = logging.getLogger(__name__)

    def _default_constraints(self) -> List[Constraint]:
        constraints = []
        constraints.append(Constraint(
            name="allele_frequency_range",
            description="Allele frequency must be between 0 and 1",
            rule_type="numerical_range",
            columns=["allele_frequency"],
            rule_fn=lambda row: 0 <= float(row.get('allele_frequency', 0.5)) <= 1
        ))
        constraints.append(Constraint(
            name="inheritance_chromosome_consistency",
            description="X-linked inheritance should have chromosome X",
            rule_type="logical_consistency",
            columns=["inheritance_pattern", "chromosome"],
            rule_fn=lambda row: not (
                str(row.get('inheritance_pattern', '')) == 'X-linked' and
                str(row.get('chromosome', '')) != 'X'
            )
        ))
        constraints.append(Constraint(
            name="variant_type_validity",
            description="Variant type must be a recognized type",
            rule_type="categorical_compatibility",
            columns=["variant_type"],
            rule_fn=lambda row: str(row.get('variant_type', '')) in [
                'Missense', 'Nonsense', 'Frameshift', 'Splice Site', 'Deletion',
                'Insertion', 'Duplication', 'SNV', 'CNV', 'Indel', ''
            ]
        ))
        constraints.append(Constraint(
            name="clinical_significance_validity",
            description="Clinical significance must be a recognized value",
            rule_type="categorical_compatibility",
            columns=["clinical_significance"],
            rule_fn=lambda row: str(row.get('clinical_significance', '')) in [
                'Pathogenic', 'Likely Pathogenic', 'Benign', 'Likely Benign',
                'Uncertain Significance', 'VUS', ''
            ]
        ))
        return constraints

    def validate_dataset(self, data: pd.DataFrame) -> ConstraintValidationResult:
        if data.empty:
            return ConstraintValidationResult(
                total_records=0, valid_records=0, invalid_records=0,
                filter_percentage=0.0, severity=QualityLevel.ACCEPTABLE
            )
        total = len(data)
        violations_by_constraint = {c.name: 0 for c in self.constraints}
        invalid_indices = set()
        violation_details = []

        for idx, row in data.iterrows():
            is_valid, violated = self.validate_record(row)
            if not is_valid:
                invalid_indices.add(idx)
                for v in violated:
                    violations_by_constraint[v] = violations_by_constraint.get(v, 0) + 1
                if len(violation_details) < 50:
                    violation_details.append({
                        'index': idx,
                        'violated_constraints': violated
                    })

        invalid_count = len(invalid_indices)
        valid_count = total - invalid_count
        filter_pct = invalid_count / total if total > 0 else 0.0
        is_critical = filter_pct > CONSTRAINT_FILTER_CRITICAL_THRESHOLD

        if is_critical:
            severity = QualityLevel.CRITICAL
        elif filter_pct > 0.05:
            severity = QualityLevel.WARNING
        elif filter_pct > 0.01:
            severity = QualityLevel.ACCEPTABLE
        else:
            severity = QualityLevel.EXCELLENT

        return ConstraintValidationResult(
            total_records=total,
            valid_records=valid_count,
            invalid_records=invalid_count,
            filter_percentage=filter_pct,
            violations_by_constraint=violations_by_constraint,
            violation_details=violation_details,
            is_critical=is_critical,
            severity=severity,
        )

    def validate_record(self, record) -> Tuple[bool, List[str]]:
        violated = []
        for constraint in self.constraints:
            if not constraint.validate(record):
                violated.append(constraint.name)
        return (len(violated) == 0, violated)

    def filter_invalid_records(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data.copy()
        valid_mask = []
        for _, row in data.iterrows():
            is_valid, _ = self.validate_record(row)
            valid_mask.append(is_valid)
        filtered = data[valid_mask].copy()
        removed = len(data) - len(filtered)
        if removed > 0:
            self.logger.info(f"Filtered {removed} invalid records ({removed/len(data)*100:.1f}%)")
        return filtered.reset_index(drop=True)

    def get_violation_report(self, data: pd.DataFrame) -> dict:
        result = self.validate_dataset(data)
        recommendations = []
        if result.is_critical:
            recommendations.append("Critical: >10% records violate constraints. Review generation model.")
        for name, count in result.violations_by_constraint.items():
            if count > 0:
                pct = count / result.total_records * 100 if result.total_records > 0 else 0
                recommendations.append(f"Constraint '{name}': {count} violations ({pct:.1f}%)")
        return {
            'total_records': result.total_records,
            'valid_records': result.valid_records,
            'invalid_records': result.invalid_records,
            'filter_percentage': result.filter_percentage,
            'violations_by_constraint': result.violations_by_constraint,
            'sample_violations': result.violation_details[:10],
            'is_critical': result.is_critical,
            'recommendations': recommendations,
        }

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def generate_report(self, result: ConstraintValidationResult) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("CONSTRAINT VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Records:   {result.total_records}")
        lines.append(f"Valid Records:   {result.valid_records}")
        lines.append(f"Invalid Records: {result.invalid_records}")
        lines.append(f"Filter Rate:     {result.filter_percentage*100:.2f}%")
        lines.append(f"Severity:        {result.severity.value}")
        lines.append(f"Critical:        {'YES' if result.is_critical else 'No'}")
        lines.append("")
        lines.append("Violations by Constraint:")
        for name, count in result.violations_by_constraint.items():
            status = "FAIL" if count > 0 else "PASS"
            lines.append(f"  [{status}] {name}: {count} violations")
        if result.is_critical:
            lines.append("")
            lines.append("WARNING: Filter rate exceeds 10% threshold.")
            lines.append("Review the generation model and constraint definitions.")
        lines.append("=" * 60)
        return "\n".join(lines)
