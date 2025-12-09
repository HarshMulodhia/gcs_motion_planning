"""Training utilities for GCS"""
import numpy as np
import logging
import time
import os
import json
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name="Process"):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logger.debug(f"{self.name} took {self.interval:.4f} seconds")

class CheckpointManager:
    """Manages saving and loading of training checkpoints"""
    def __init__(self, directory="checkpoints"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        
    def save(self, episode, model_state, metrics):
        filename = os.path.join(self.directory, f"checkpoint_ep{episode}.json")
        data = {
            "episode": episode,
            "metrics": metrics,
            # In real scenario, serialize your model weights here
            "model_state": str(model_state) 
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Checkpoint saved: {filename}")

class GCSTrainer:
    """Training utilities with stability mechanisms"""

    def __init__(self, learning_rate: float = 0.001,
                 max_gradient_norm: float = 1.0, patience: int = 10):
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.patience = patience
        self.loss_history = []
        self.best_loss = float("inf")
        self.no_improve_count = 0
        logger.info("GCSTrainer initialized")

    def gradient_clipping(self, gradients: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(gradients)
        if norm > self.max_gradient_norm:
            gradients = gradients * (self.max_gradient_norm / norm)
            logger.debug(f"Gradient clipped: {norm:.4f} -> {self.max_gradient_norm}")
        return gradients

    def check_early_stopping(self, loss: float) -> bool:
        self.loss_history.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        if self.no_improve_count >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} epochs")
            return True
        return False

    def adaptive_learning_rate(self, epoch: int) -> float:
        return self.learning_rate * (0.95 ** (epoch / 20))

    def log_progress(self, epoch: int, loss: float, **kwargs):
        if epoch % 10 == 0:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in kwargs.items()])
            logger.info(f"Epoch {epoch:4d} | Loss: {loss:.6f} | {metrics_str}")

    def get_loss_trend(self, window: int = 5) -> str:
        if len(self.loss_history) < window:
            return "insufficient_data"
        recent_losses = self.loss_history[-window:]
        trend = recent_losses[0] - recent_losses[-1]
        if trend > 1e-4:
            return "improving"
        elif trend > -1e-4:
            return "stable"
        else:
            return "diverging"

    def reset(self):
        self.loss_history = []
        self.best_loss = float("inf")
        self.no_improve_count = 0
        logger.info("Trainer reset")

def warm_start_optimization(objective_fn, num_restarts: int = 3,
                           initial_point_generator=None) -> Dict:
    best_solution = None
    best_cost = float("inf")
    if initial_point_generator is None:
        initial_point_generator = lambda: np.random.randn(10)
    for restart in range(num_restarts):
        initial_point = initial_point_generator()
        try:
            solution = objective_fn(initial_point)
            cost = solution.get("cost", float("inf"))
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
                logger.info(f"Restart {restart}: Cost={best_cost:.6f}")
        except Exception as e:
            logger.warning(f"Restart {restart} failed: {e}")
    return best_solution if best_solution else {"cost": best_cost}

def early_stopping(losses: List[float], patience: int = 10) -> bool:
    if len(losses) < patience:
        return False
    recent_losses = losses[-patience:]
    improvement = recent_losses[0] - recent_losses[-1]
    return improvement < 1e-4

class GCSOptimizationCache:
    """Cache for optimization results"""
    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.max_size = max_cache_size
        self.access_count = {}
        logger.info(f"Cache initialized with max size {max_cache_size}")

    def cache_solution(self, problem_key: str, solution):
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
            logger.debug(f"Removed LRU entry: {lru_key}")
        self.cache[problem_key] = solution
        self.access_count[problem_key] = 0
        logger.debug(f"Cached solution for {problem_key}")

    def get_cached_solution(self, problem_key: str):
        if problem_key in self.cache:
            self.access_count[problem_key] += 1
            return self.cache[problem_key]
        return None

    def clear(self):
        self.cache.clear()
        self.access_count.clear()
        logger.info("Cache cleared")
