# src/utils/config_loader.py
from pathlib import Path

import yaml


def load_config():
    """
    Loads and parses the main YAML configuration file for the application.

    This function robustly locates the `config.yaml` file at the project root
    relative to its own location, making the script runnable from any directory.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is not valid YAML.

    Returns:
        A dictionary containing the application configuration.
    """
    # Build a path to the project root (2 levels up from src/utils) and then to config.yaml
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
