"""
Test Suite for GCS Planner Components
Tests for GCS Builder, Solver, and Convex Set Construction
"""

import pytest
import numpy as np
from src.gcs_planner import GCSBuilder, ConvexSetBuilder, GCSSolver

class TestGCSBuilder:
    """Test cases for GCSBuilder class"""
    
    def test_initialization(self):
        """Test GCSBuilder initialization"""
        builder = GCSBuilder(dimension=3)
        assert builder.dimension == 3
        assert len(builder.convex_sets) == 0
        assert builder.source is None
        assert builder.target is None
    
    def test_add_convex_set(self):
        """Test adding convex sets"""
        builder = GCSBuilder(dimension=3)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        
        builder.add_convex_set('region_1', vertices)
        
        assert 'region_1' in builder.convex_sets
        assert builder.convex_sets['region_1']['dimension'] == 3
        assert np.array_equal(builder.convex_sets['region_1']['vertices'], vertices)
    
    def test_add_edge(self):
        """Test adding edges between convex sets"""
        builder = GCSBuilder(dimension=3)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        
        builder.add_convex_set('region_1', vertices)
        builder.add_convex_set('region_2', vertices)
        builder.add_edge('region_1', 'region_2', weight=1.5)
        
        assert builder.graph.has_edge('region_1', 'region_2')
        assert builder.graph['region_1']['region_2']['weight'] == 1.5
    
    def test_shortest_path(self):
        """Test shortest path computation"""
        builder = GCSBuilder(dimension=3)
        vertices = np.array([[0, 0, 0], [1, 0, 0]])
        
        # Create a simple path graph
        builder.add_convex_set('start', vertices)
        builder.add_convex_set('middle', vertices)
        builder.add_convex_set('goal', vertices)
        
        builder.add_edge('start', 'middle', weight=1.0)
        builder.add_edge('middle', 'goal', weight=1.0)
        
        builder.set_source_target('start', 'goal')
        path = builder.get_shortest_path()
        
        assert path is not None
        assert 'start' in path
        assert 'goal' in path
        assert len(path) == 3


class TestConvexSetBuilder:
    """Test cases for ConvexSetBuilder utility class"""
    
    def test_box_2d(self):
        """Test 2D box construction"""
        center = np.array([0.5, 0.5])
        half_lengths = np.array([0.5, 0.5])
        
        vertices = ConvexSetBuilder.box(center, half_lengths)
        
        assert vertices.shape == (4, 2)  # 2^2 vertices
        assert np.allclose(np.min(vertices, axis=0), center - half_lengths)
        assert np.allclose(np.max(vertices, axis=0), center + half_lengths)
    
    def test_box_3d(self):
        """Test 3D box construction"""
        center = np.array([0.5, 0.5, 0.5])
        half_lengths = np.array([0.5, 0.5, 0.5])
        
        vertices = ConvexSetBuilder.box(center, half_lengths)
        
        assert vertices.shape == (8, 3)  # 2^3 vertices
        assert np.allclose(np.min(vertices, axis=0), center - half_lengths)
        assert np.allclose(np.max(vertices, axis=0), center + half_lengths)
    
    def test_sphere_2d(self):
        """Test 2D sphere approximation"""
        center = np.array([0.0, 0.0])
        radius = 1.0
        num_points = 8
        
        vertices = ConvexSetBuilder.sphere(center, radius, num_points)
        
        assert vertices.shape == (8, 2)
        # Check that vertices are approximately on the circle
        distances = np.linalg.norm(vertices - center, axis=1)
        assert np.allclose(distances, radius, atol=1e-6)
    
    def test_sphere_3d(self):
        """Test 3D sphere approximation"""
        center = np.array([0.0, 0.0, 0.0])
        radius = 2.0
        
        vertices = ConvexSetBuilder.sphere(center, radius)
        
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        # Check that vertices are approximately on the sphere
        distances = np.linalg.norm(vertices - center, axis=1)
        assert np.allclose(distances, radius, atol=1e-6)


class TestGCSSolver:
    """Test cases for GCSSolver class"""
    
    def test_initialization(self):
        """Test GCSSolver initialization"""
        solver = GCSSolver(verbose=False, solver='ECOS')
        
        assert solver.verbose is False
        assert solver.solver == 'ECOS'
        assert solver.solution is None
    
    def test_trajectory_cost(self):
        """Test trajectory cost calculation"""
        solver = GCSSolver()
        
        # Create a simple trajectory
        trajectory = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0]
        ])
        
        # Manually set solution
        solver.solution = {
            'trajectory': trajectory,
            'path': ['start', 'goal'],
            'feasible': True,
            'length': len(trajectory)
        }
        
        cost = solver.get_trajectory_cost()
        
        assert cost is not None
        # Expected cost: distance from [0,0,0] to [1,0,0] + [1,0,0] to [1,1,0]
        expected_cost = np.sqrt(1) + np.sqrt(1)
        assert np.isclose(cost, expected_cost, atol=1e-6)


class TestIntegration:
    """Integration tests for GCS components working together"""
    
    def test_builder_and_solver_integration(self):
        """Test GCSBuilder and GCSSolver working together"""
        # Create builder
        builder = GCSBuilder(dimension=3)
        vertices = np.array([[0, 0, 0], [1, 0, 0]])
        
        builder.add_convex_set('start', vertices)
        builder.add_convex_set('goal', vertices)
        builder.add_edge('start', 'goal', weight=1.0)
        builder.set_source_target('start', 'goal')
        
        # Create solver
        solver = GCSSolver()
        path = builder.get_shortest_path()
        
        assert path is not None
        assert len(path) == 2
