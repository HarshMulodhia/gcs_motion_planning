"""W&B Integration for training monitoring"""
import wandb
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class WandBLogger:
    """Weights & Biases integration for experiment tracking"""

    def __init__(self, project_name: str = "gcs-motion-planning",
                 config: Dict[str, Any] = None):
        """Initialize W&B logger
        
        Args:
            project_name: W&B project name
            config: Configuration dictionary to log
        """
        self.run = None
        self._setup(project_name, config)

    def _setup(self, project_name, config):
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project_name, config=config, reinit=True)
            logger.info(f"W&B initialized for project: {project_name}")
        except ImportError:
            logger.error("wandb package not installed. Run `pip install wandb`")
        except Exception as e:
            logger.warning(f"Could not initialize W&B: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log training metrics
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step/epoch number
        """
        if self.run is None:
            return
        
        try:
            wandb.log(metrics, step=step)
            logger.debug(f"Logged metrics at step {step}")
        except Exception as e:
            logger.warning(f"Error logging to W&B: {e}")

    def log_trajectory(self, trajectory: np.ndarray, name: str = "trajectory"):
        """Log trajectory visualization
        
        Args:
            trajectory: (N, 3) trajectory array
            name: Name for the trajectory
        """
        if self.run is None:
            return
        
        try:
            # Log as table
            table = wandb.Table(columns=["x", "y", "z"])
            for point in trajectory:
                table.add_data(*point[:3])
            wandb.log({name: table})
        except Exception as e:
            logger.warning(f"Error logging trajectory: {e}")

    def finish(self):
        """Finish W&B logging session"""
        if self.run is not None:
            wandb.finish()
            logger.info("W&B logging finished")
