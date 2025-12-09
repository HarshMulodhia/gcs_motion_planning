# Motion Planning with Graph of Convex Sets (GCS)

Complete motion planning framework using Graph of Convex Sets with visualization, training utilities, and monitoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- VS Code with DevContainers extension

### Installation

```bash
# Clone repository
git clone <repository-url>
cd motion-planning-gcs

# DevContainer setup (automatic)
# Open in VS Code → Dev Containers: Rebuild Container

# Manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Training

```bash
# Make script executable (Linux/Mac)
chmod +x scripts/run_training.sh

# Run training
./scripts/run_training.sh
# or
python src/training/agent.py --config configs/training_config.yaml
```

### Start Jupyter

```bash
source venv/bin/activate
jupyter lab --ip=0.0.0.0
```

## 📁 Project Structure

```
motion-planning-gcs/
├── .devcontainer/
│   └── devcontainer.json              # VS Code container config
├── scripts/
│   ├── install-dependencies.sh        # Auto-installer
│   └── run_training.sh                # Training script
├── src/
│   ├── __init__.py
│   ├── gcs_planner/                   # Core GCS solver
│   │   ├── gcs_builder.py
│   │   └── solver.py
│   ├── visualization/                 # 3D visualization
│   │   ├── meshcat_visualizer.py      # Interactive 3D
│   │   ├── plotly_dashboard.py        # Metrics dashboard
│   │   └── pyvista_visualizer.py      # Advanced 3D
│   ├── training/                      # Training loop
│   │   ├── training_utils.py          # Stability utilities
│   │   ├── wandb_integration.py       # W&B logging
│   │   └── agent.py                   # Main training loop
│   └── utils/
│       └── config.py                  # Configuration loader
├── notebooks/                         # Jupyter notebooks
│   ├── 01_gcs_introduction.ipynb
│   ├── 02_visualization_demo.ipynb
│   ├── 03_training_analysis.ipynb
│   └── 04_results_presentation.ipynb
├── tests/                             # Unit tests
│   ├── test_gcs_planner.py
│   └── test_visualizer.py
├── configs/
│   └── training_config.yaml           # Training configuration
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
└── README.md                          # This file
```

## 🎯 Core Features

### Motion Planning
- **GCS Builder**: Construct graph of convex sets
- **Convex Sets**: Box, sphere, and polytope primitives
- **GCS Solver**: Solve using CVXPY with multiple backends
- **Path Planning**: Shortest path through convex regions

### Visualization
- **MeshCat**: Interactive 3D visualization in browser
- **Plotly**: Publication-quality metrics dashboards
- **PyVista**: Advanced scientific 3D rendering

### Training
- **Gradient Clipping**: Prevent exploding gradients
- **Early Stopping**: Stop when plateaued
- **W&B Integration**: Cloud-based experiment tracking
- **Training Dashboard**: Real-time metrics with HTML export

## 📊 Usage Examples

### Basic GCS Planning

```python
from src.gcs_planner import GCSBuilder, GCSSolver
from src.visualization import MeshCatVisualizer
import numpy as np

# Build GCS graph
builder = GCSBuilder(dimension=3)

# Add convex regions
vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
builder.add_convex_set('region1', vertices)
builder.add_convex_set('region2', vertices + [1,0,0])

# Connect regions
builder.add_edge('region1', 'region2', weight=1.0)

# Solve
solver = GCSSolver()
start = np.array([0, 0, 0])
goal = np.array([2, 1, 0])
solution = solver.solve(builder, start, goal)

# Visualize
if solution:
    vis = MeshCatVisualizer()
    vis.add_trajectory(solution['trajectory'])
    vis.print_url()  # http://localhost:6000
```

### Training with Monitoring

```python
from src.training import GCSTrainer
from src.visualization import TrainingDashboard

trainer = GCSTrainer(learning_rate=0.001, patience=10)
dashboard = TrainingDashboard()

for epoch in range(100):
    # Your training code here
    loss = train_step()
    
    # Automatic stability
    grads = trainer.gradient_clipping(grads)
    if trainer.check_early_stopping(loss):
        break
    
    # Monitor progress
    dashboard.update_metrics(epoch=epoch, loss=loss)
    trainer.log_progress(epoch, loss)

# Save results
dashboard.create_dashboard()
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_gcs_planner.py::TestGCSBuilder::test_initialization -v
```

## 📦 Dependencies

### Core
- **numpy, scipy**: Scientific computing
- **cvxpy**: Convex optimization
- **networkx**: Graph algorithms

### Visualization
- **plotly**: Interactive dashboards
- **meshcat**: 3D visualization
- **pyvista**: Scientific 3D rendering

### Training
- **wandb**: Experiment tracking
- **tensorboard**: TensorFlow monitoring

### Development
- **pytest**: Unit testing
- **black**: Code formatting
- **pylint**: Code linting

## 🔧 Configuration

Edit `configs/training_config.yaml`:

```yaml
learning_rate: 0.001
max_gradient_norm: 1.0
num_epochs: 100
patience: 10

# Visualization
visualize: true
meshcat_port: 6000
jupyter_port: 8888

# Logging
use_wandb: false
```

## 🚀 Advanced Usage

### Custom Convex Sets

```python
from src.gcs_planner import ConvexSetBuilder

# Box
vertices = ConvexSetBuilder.box(
    center=np.array([0, 0, 0]),
    half_lengths=np.array([1, 1, 1])
)

# Sphere
vertices = ConvexSetBuilder.sphere(
    center=np.array([0, 0, 0]),
    radius=1.0,
    num_points=20
)
```

### W&B Integration

```python
from src.training import WandBLogger

wandb_logger = WandBLogger(project_name="my-project")
wandb_logger.log_metrics({
    'loss': 0.5,
    'accuracy': 0.95
}, step=epoch)
wandb_logger.log_trajectory(trajectory)
wandb_logger.finish()
```

## 📚 Notebooks

1. **01_gcs_introduction.ipynb** - GCS concepts and examples
2. **02_visualization_demo.ipynb** - Visualization tools demo
3. **03_training_analysis.ipynb** - Training and monitoring
4. **04_results_presentation.ipynb** - Results and analysis

## 🐛 Troubleshooting

### MeshCat connection failed
```bash
# Start meshcat server
meshcat-server &
# Then access http://localhost:6000
```

### Jupyter not accessible
```bash
# Check port forwarding in DevContainer
# Rebuild container: Ctrl+Shift+P → Dev Containers: Rebuild Container
```

### Import errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## 📖 Documentation

- [Configuration Guide](docs/config.md)
- [GCS Theory](docs/gcs_theory.md)
- [Visualization Guide](docs/visualization.md)
- [Training Guide](docs/training.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Graph of Convex Sets theory: [Drake Robotics Toolbox](http://underactuated.mit.edu/)
- Visualization: MeshCat, Plotly, PyVista communities
- Optimization: CVXPY and solver developers

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review notebook examples

---

**Happy planning!** 🎉
