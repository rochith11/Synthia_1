"""Variant Metadata model for Synthia."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class VariantType(str, Enum):
    """Valid variant types."""
    SNV = 'SNV'
    INSERTION = 'Insertion'
    DELETION = 'Deletion'
    DUPLICATION = 'Duplication'


class ClinicalSignificance(str, Enum):
    """Valid clinical significance levels."""
    PATHOGENIC = 'Pathogenic'
    LIKELY_PATHOGENIC = 'Likely Pathogenic'
    VUS = 'VUS'
    BENIGN = 'Benign'


class InheritancePattern(str, Enum):
    """Valid inheritance patterns."""
    AUTOSOMAL_DOMINANT = 'Autosomal Dominant'
    AUTOSOMAL_RECESSIVE = 'Autosomal Recessive'
    X_LINKED = 'X-linked'


@dataclass
class VariantMetadata:
    """Represents a genetic variant with metadata.

    Attributes:
        gene_symbol: Gene symbol (e.g., "CFTR", "DMD", "HBB")
        chromosome: Chromosome location (e.g., "chr7", "chrX")
        variant_type: Type of variant (SNV, Insertion, Deletion, Duplication)
        clinical_significance: Clinical significance classification
        disease: Associated rare disease (e.g., "Cystic Fibrosis")
        allele_frequency: Allele frequency [0.0, 1.0]
        inheritance_pattern: Inheritance pattern classification
    """

    gene_symbol: str
    chromosome: str
    variant_type: str
    clinical_significance: str
    disease: str
    allele_frequency: float
    inheritance_pattern: str

    REQUIRED_FIELDS = [
        'gene_symbol', 'chromosome', 'variant_type',
        'clinical_significance', 'disease', 'allele_frequency',
        'inheritance_pattern',
    ]

    def __post_init__(self):
        """Validate on creation."""
        self.validate()

    def validate(self):
        """Validate variant metadata fields."""
        errors = []

        # Validate gene_symbol
        if not isinstance(self.gene_symbol, str) or len(self.gene_symbol) == 0:
            errors.append('gene_symbol must be a non-empty string')

        # Validate chromosome
        if not isinstance(self.chromosome, str) or len(self.chromosome) == 0:
            errors.append('chromosome must be a non-empty string')

        # Validate variant_type
        if self.variant_type not in [v.value for v in VariantType]:
            errors.append(f'variant_type must be one of {[v.value for v in VariantType]}')

        # Validate clinical_significance
        if self.clinical_significance not in [v.value for v in ClinicalSignificance]:
            errors.append(f'clinical_significance must be one of {[v.value for v in ClinicalSignificance]}')

        # Validate disease
        if not isinstance(self.disease, str) or len(self.disease) == 0:
            errors.append('disease must be a non-empty string')

        # Validate inheritance_pattern
        if self.inheritance_pattern not in [v.value for v in InheritancePattern]:
            errors.append(f'inheritance_pattern must be one of {[v.value for v in InheritancePattern]}')

        # Validate allele_frequency
        try:
            allele_freq = float(self.allele_frequency)
            if not (0.0 <= allele_freq <= 1.0):
                errors.append('allele_frequency must be in range [0.0, 1.0]')
        except (ValueError, TypeError):
            errors.append('allele_frequency must be a float')

        if errors:
            raise ValueError(f'VariantMetadata validation failed: {"; ".join(errors)}')

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'gene_symbol': self.gene_symbol,
            'chromosome': self.chromosome,
            'variant_type': self.variant_type,
            'clinical_significance': self.clinical_significance,
            'disease': self.disease,
            'allele_frequency': float(self.allele_frequency),
            'inheritance_pattern': self.inheritance_pattern,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VariantMetadata':
        """Create from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'VariantMetadata(gene={self.gene_symbol}'
            f', chr={self.chromosome}'
            f', type={self.variant_type}'
            f', significance={self.clinical_significance}'
            f', disease={self.disease}'
            f', allele_freq={self.allele_frequency}'
            f', pattern={self.inheritance_pattern})'
        )
