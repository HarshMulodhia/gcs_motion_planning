#!/bin/bash
set -e

echo "📦 Updating system packages..."
apt-get update
apt-get install -y \
  build-essential \
  cmake \
  git \
  libopenblas-dev \
  gfortran \
  wget

echo "🐍 Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "📥 Installing dependencies..."
pip install --quiet -r requirements.txt

echo "✓ Installation complete!"
python -c "import numpy, scipy, cvxpy, plotly, meshcat; print('✓ All core packages available')"
