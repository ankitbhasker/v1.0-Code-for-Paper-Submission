"""Utility functions for the psychiatric clustering pipeline."""

import os
import load_pickle
import logging
import yaml
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger
    """
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'pipeline.log')
    
    # Create logs directory if needed
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logging.info(f"Loaded pickle from {filepath}")
    return obj


def ensure_dir(directory: str) -> None:
    """Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_icd9_code(code: str) -> str:
    """Format ICD-9 code to standard format.
    
    Args:
        code: Raw ICD-9 code
        
    Returns:
        Formatted ICD-9 code
    """
    if pd.isna(code):
        return None
    code_str = str(code).strip()
    return code_str


def is_psychiatric_icd9(code: str, min_code: int = 290, max_code: int = 319) -> bool:
    """Check if ICD-9 code is psychiatric.
    
    Args:
        code: ICD-9 code
        min_code: Minimum psychiatric code
        max_code: Maximum psychiatric code
        
    Returns:
        True if psychiatric code
    """
    if pd.isna(code):
        return False
    
    try:
        # Extract numeric prefix
        code_str = str(code).strip()
        numeric_part = int(code_str.split('.')[0])
        return min_code <= numeric_part <= max_code
    except (ValueError, IndexError):
        return False
