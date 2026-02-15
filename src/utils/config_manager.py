"""Configuration management for Synthia."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages configuration loading and validation."""

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'

    def __init__(self, config_path: str = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration YAML file. Uses default if not provided.
        """
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"[OK] Configuration loaded from: {self.config_path}")
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.

        Args:
            key: Configuration key (e.g., 'generation.defaults.model_type')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_model_defaults(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return self.config.get('generation', {}).get('defaults', {})

    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy configuration."""
        return self.config.get('privacy', {})

    def get_bias_config(self) -> Dict[str, Any]:
        """Get bias detection configuration."""
        return self.config.get('bias', {})


# Global config instance
_config_instance = None


def get_config(config_path: str = None) -> ConfigManager:
    """Get or create global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance
