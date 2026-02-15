"""Setup script for Synthia."""

from setuptools import setup, find_packages

setup(
    name='synthia',
    version='0.1.0',
    description='Synthetic Rare Disease Data Generator - Research Prototype',
    author='Final Year Project Team',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'sdv>=0.18.0',
        'Flask>=2.0.0',
        'Flask-WTF>=1.0.0',
        'PyYAML>=5.4.0',
        'python-dateutil>=2.8.0',
        'tqdm>=4.60.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'hypothesis>=6.0.0',
        ],
    },
)
