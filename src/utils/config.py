"""Configuration Module"""

import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # Training
    'num_episodes': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coeff': 0.01,
    'value_loss_coeff': 0.5,
    'max_grad_norm': 0.5,
    
    # GCS
    'gcs_dimension': 3,
    'num_convex_sets': 5,
    'solver_type': 'ECOS',
    
    # Visualization
    'use_meshcat': True,
    'meshcat_url': 'tcp://127.0.0.1:6000',
    'use_plotly': True,
    'use_pyvista': False,
    
    # Logging
    'log_level': 'INFO',
    'log_interval': 10,
    'save_interval': 50,
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
}

class Config:
    """Configuration manager"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize config from dictionary"""
        self.config = DEFAULT_CONFIG.copy()
        if config_dict:
            self.config.update(config_dict)
        
        # Create directories
        os.makedirs(self.config.get('checkpoint_dir', './checkpoints'), exist_ok=True)
        os.makedirs(self.config.get('log_dir', './logs'), exist_ok=True)
        
        logger.info("Configuration initialized")

    def __getitem__(self, key: str) -> Any:
        """Get config value"""
        return self.config.get(key)

    def __setitem__(self, key: str, value: Any):
        """Set config value"""
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default"""
        return self.config.get(key, default)

    def update(self, config_dict: Dict[str, Any]):
        """Update config from dictionary"""
        self.config.update(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Get config as dictionary"""
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation"""
        return f"Config({len(self.config)} items)"
