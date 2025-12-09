"""Training Agent - Main training loop with monitoring, visualization, and config handling"""

import argparse
import numpy as np
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSTrainingAgent:
    """Main training agent for GCS motion planning"""

    def __init__(self, config_path: str = "configs/training_config.yaml", visualize: bool = False):
        """
        Initialize training agent
        Args:
            config_path: Path to training configuration YAML
            visualize: Boolean to enable/disable 3D visualization
        """
        self.visualize = visualize
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_trainer()
        self._init_dashboard()
        self._init_wandb()
        self._init_visualizer()
        
        logger.info("✓ Training agent initialized")

    def _load_config(self, path: str) -> Dict:
        try:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Config load failed ({e}), using defaults")
            return {}

    def _get_nested_config(self, path: str, default: Any = None) -> Any:
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def _init_trainer(self):
        try:
            from src.training.training_utils import GCSTrainer
            self.trainer = GCSTrainer(
                learning_rate=self._get_nested_config('model.learning_rate', 0.001),
                patience=self._get_nested_config('stability.patience', 10)
            )
        except ImportError:
            self.trainer = None

    def _init_dashboard(self):
        try:
            from src.visualization.plotly_dashboard import TrainingDashboard
            self.dashboard = TrainingDashboard()
        except ImportError:
            self.dashboard = None

    def _init_wandb(self):
        if self._get_nested_config('wandb.use_wandb', False):
            try:
                from src.training.wandb_integration import WandBLogger
                self.wandb_logger = WandBLogger(
                    project_name=self._get_nested_config('wandb.project_name'),
                    config=self.config
                )
            except ImportError:
                self.wandb_logger = None
        else:
            self.wandb_logger = None

    def _init_visualizer(self):
        """Initialize MeshCat visualizer if enabled"""
        self.vis = None
        if self.visualize:
            try:
                from src.visualization.meshcat_visualizer import MeshCatVisualizer
                self.vis = MeshCatVisualizer(
                    zmq_url=self._get_nested_config('visualization.meshcat_url', "tcp://127.0.0.1:6000")
                )
                
                # Setup basic scene
                if self.vis.is_connected():
                    self.vis.add_reference_frame(size=0.5)
                    self.vis.add_obstacle_box(
                        center=np.array([0.5, 0.0, 0.2]), 
                        size=np.array([0.2, 0.2, 0.4]),
                        name="static_obstacle"
                    )
            except ImportError as e:
                logger.warning(f"Visualization disabled: {e}")

    def train(self, num_episodes: Optional[int] = None):
        """Run main training loop"""
        num_episodes = num_episodes or self._get_nested_config('training.num_epochs', 100)
        
        logger.info(f"Starting training for {num_episodes} episodes...")
        
        try:
            for episode in range(num_episodes):
                # 1. Simulate Training Step (Replace with real training logic)
                loss, reward, metrics = self._simulate_step(episode)
                
                # 2. Update Visualization (Real-time)
                if self.vis and self.vis.is_connected() and episode % 5 == 0:
                    self._update_visualization(episode, metrics)

                # 3. Log Progress
                if self.trainer:
                    self.trainer.log_progress(episode, loss, reward=reward)
                    if self.trainer.check_early_stopping(loss):
                        print(f"\n✓ Early stopping at episode {episode}")
                        break

                # 4. Update Dashboard/W&B
                if self.dashboard:
                    self.dashboard.update_metrics(episode=episode, loss=loss, reward=reward, **metrics)
                
                if self.wandb_logger:
                    self.wandb_logger.log_metrics({'loss': loss, 'reward': reward, **metrics}, step=episode)
                
                # Small delay for visual effect
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        finally:
            self._save_artifacts()

    def _simulate_step(self, episode):
        """Mock training step - replace with real GCS solve"""
        loss = 100 * np.exp(-episode / 30) + np.random.randn() * 0.5
        reward = 50 * (1 - np.exp(-episode / 20)) + np.random.randn() * 2
        
        # Mock trajectory data for visualization
        metrics = {
            'success_rate': min(0.95, episode / 50),
            'path_length': 10 - 3 * (1 - np.exp(-episode / 25))
        }
        return loss, reward, metrics

    def _update_visualization(self, episode, metrics):
        """Generate and visualize a mock trajectory"""
        # Generate a spiral path that changes over time
        t = np.linspace(0, 10, 50)
        scale = 0.5 + (episode / 100.0) * 0.5
        x = scale * np.cos(t)
        y = scale * np.sin(t)
        z = t * 0.1
        path = np.stack([x, y, z], axis=1)
        
        self.vis.add_trajectory(path, name=f"traj_ep_{episode}", color=(0, 255, 255))
        self.vis.add_sphere(path[-1], radius=0.05, name="agent", color=(255, 0, 0))

    def _save_artifacts(self):
        if self.dashboard:
            path = self._get_nested_config('dashboard.dashboard_file', 'training_dashboard.html')
            self.dashboard.create_dashboard(filename=path)
            logger.info(f"Dashboard saved to {path}")
        
        if self.wandb_logger:
            self.wandb_logger.finish()

def main():
    parser = argparse.ArgumentParser(description='GCS Motion Planning Training')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Config path')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--log-file', type=str, default='training.log', help='Log file path')
    parser.add_argument('--visualize', action='store_true', help='Enable MeshCat visualization')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()

    # Update global config for W&B override
    # (In a real app, you'd merge this cleanly into the config dict)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        agent = GCSTrainingAgent(config_path=args.config, visualize=args.visualize)
        agent.train(num_episodes=args.epochs)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
