"""Logging utilities for Synthia."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(name: str, log_dir: str = None, level: str = 'INFO') -> logging.Logger:
    """Set up logging for a module.

    Args:
        name: Logger name
        log_dir: Directory for log files. If None, console-only
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add console handler
    if not logger.handlers:
        logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Common console output formatting
def print_section(title: str, width: int = 60):
    """Print section divider."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def print_success(message: str):
    """Print success message."""
    print(f"[+] {message}")


def print_info(message: str):
    """Print info message."""
    print(f"[i] {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"[!] {message}")


def print_error(message: str):
    """Print error message."""
    print(f"[-] {message}")
