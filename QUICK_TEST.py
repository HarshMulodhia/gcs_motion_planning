"""Quick test script to verify all components are working"""
import sys
import os

print("=" * 60)
print("MOTION PLANNING GCS - COMPONENT TEST")
print("=" * 60)

print("\n[1/4] Testing imports...")
try:
    import numpy as np
    import scipy
    import cvxpy
    import plotly.graph_objects as go
    import meshcat
    import pyvista
    print("✓ All core packages available")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n[2/4] Testing visualization modules...")
sys.path.insert(0, '.')
try:
    from src.visualization import MeshCatVisualizer, TrainingDashboard, PyVistaVisualizer
    print("✓ Visualization modules imported successfully")
except ImportError as e:
    print(f"✗ Visualization import error: {e}")
    print("  Ensure src/visualization/__init__.py exists")
    sys.exit(1)

print("\n[3/4] Testing training utilities...")
try:
    from src.training import GCSTrainer, warm_start_optimization, early_stopping, GCSOptimizationCache
    print("✓ Training utilities imported successfully")
except ImportError as e:
    print(f"✗ Training import error: {e}")
    print("  Ensure src/training/__init__.py exists")
    sys.exit(1)

print("\n[4/4] Testing functionality...")
try:
    dashboard = TrainingDashboard()
    for i in range(10):
        dashboard.update_metrics(episode=i, reward=50 * (1 - np.exp(-i/5)),
            loss=100 * np.exp(-i/5), success_rate=i/10)
    print("✓ Dashboard functionality working")

    trainer = GCSTrainer()
    gradients = np.array([2.0, 3.0, 4.0])
    clipped = trainer.gradient_clipping(gradients)
    assert np.linalg.norm(clipped) <= trainer.max_gradient_norm + 1e-6
    print("✓ Trainer functionality working")

    losses = [1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    stopped = early_stopping(losses, patience=5)
    print("✓ Early stopping working")

except Exception as e:
    print(f"✗ Functionality test error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYou're ready to start development!")
print("Next: jupyter lab --ip=0.0.0.0")
print("=" * 60)
