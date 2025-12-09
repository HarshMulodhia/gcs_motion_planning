"""Setup configuration for Motion Planning GCS"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gcs_motion_planning",
    version="0.1.0",
    author="Harsh Mulodhia",
    author_email="hajiharsh598@gmail.com",
    description="Motion Planning with Graph of Convex Sets (GCS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HarshMulodhia/gcs_motion_planning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.8.0",
        "cvxpy>=1.3.0",
        "plotly>=5.17.0",
        "meshcat>=0.3.0",
        "pyvista>=0.43.0",
        "networkx>=3.0",
        "wandb>=0.15.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)
