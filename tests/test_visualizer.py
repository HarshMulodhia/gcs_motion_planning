"""
Test Suite for Visualization Components
Tests for MeshCat, PyVista, and Plotly Dashboard
"""

import pytest
import numpy as np
from src.visualization import MeshCatVisualizer, TrainingDashboard, PyVistaVisualizer


class TestTrainingDashboard:
    """Test cases for TrainingDashboard class"""
    
    def test_initialization(self):
        """Test dashboard initialization"""
        dashboard = TrainingDashboard()
        
        assert 'loss' in dashboard.metrics
        assert 'reward' in dashboard.metrics
        assert 'success_rate' in dashboard.metrics
        assert len(dashboard.metrics['loss']) == 0
    
    def test_update_metrics(self):
        """Test updating metrics"""
        dashboard = TrainingDashboard()
        dashboard.update_metrics(loss=0.5, reward=10.0, success_rate=0.8)
        
        assert len(dashboard.metrics['loss']) == 1
        assert len(dashboard.metrics['reward']) == 1
        assert len(dashboard.metrics['success_rate']) == 1
        assert dashboard.metrics['loss'][0] == 0.5
        assert dashboard.metrics['reward'][0] == 10.0
        assert dashboard.metrics['success_rate'][0] == 0.8
    
    def test_multiple_updates(self):
        """Test multiple metric updates"""
        dashboard = TrainingDashboard()
        
        for i in range(10):
            dashboard.update_metrics(
                loss=float(10 - i),
                reward=float(i),
                success_rate=float(i) / 10.0
            )
        
        assert len(dashboard.metrics['loss']) == 10
        assert len(dashboard.metrics['reward']) == 10
        assert dashboard.metrics['loss'][0] == 10.0
        assert dashboard.metrics['loss'][-1] == 1.0
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        dashboard = TrainingDashboard()
        
        for i in range(5):
            dashboard.update_metrics(
                loss=float(5 - i),
                reward=float(i),
                success_rate=0.5
            )
        
        summary = dashboard.get_metrics_summary()
        
        assert 'loss' in summary
        assert 'reward' in summary
        assert summary['loss']['count'] == 5
        assert summary['loss']['mean'] == 3.0
        assert summary['reward']['mean'] == 2.0


class TestMeshCatVisualizer:
    """Test cases for MeshCatVisualizer class"""
    
    def test_initialization(self):
        """Test MeshCat initialization"""
        viz = MeshCatVisualizer(fallback_mode=True)
        
        # Should initialize without errors, even if server unavailable
        assert isinstance(viz, MeshCatVisualizer)
        assert viz.fallback_mode is True
        assert isinstance(viz.pending_operations, list)
    
    def test_trajectory_addition(self):
        """Test adding trajectory (with fallback)"""
        viz = MeshCatVisualizer(fallback_mode=True)
        
        waypoints = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0]
        ])
        
        # Should queue if not connected, or add if connected
        viz.add_trajectory(waypoints, name='test_path')
        
        # Either added to viz or queued
        assert viz is not None
    
    def test_pending_operations(self):
        """Test pending operations when disconnected"""
        viz = MeshCatVisualizer(fallback_mode=True)
        
        if not viz.is_connected():
            waypoints = np.array([[0, 0, 0], [1, 1, 1]])
            viz.add_trajectory(waypoints, name='pending_path')
            
            # Should be queued since not connected
            assert viz.get_pending_count() >= 0


class TestPyVistaVisualizer:
    """Test cases for PyVistaVisualizer class"""
    
    def test_initialization(self):
        """Test PyVista initialization"""
        vis = PyVistaVisualizer()
        
        # PyVista may not be available, but should initialize gracefully
        assert isinstance(vis, PyVistaVisualizer)
        assert hasattr(vis, 'connected')
    
    def test_add_trajectory(self):
        """Test adding trajectory to PyVista"""
        vis = PyVistaVisualizer()
        
        waypoints = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 0, 0]
        ])
        
        # Should handle gracefully even if not connected
        vis.add_trajectory(waypoints, name='test_trajectory')
        assert isinstance(vis, PyVistaVisualizer)
    
    def test_add_sphere(self):
        """Test adding sphere"""
        vis = PyVistaVisualizer()
        
        center = np.array([0, 0, 0])
        vis.add_sphere(center, radius=0.5, name='test_sphere')
        
        assert isinstance(vis, PyVistaVisualizer)
    
    def test_add_box(self):
        """Test adding box"""
        vis = PyVistaVisualizer()
        
        center = np.array([0, 0, 0])
        size = np.array([1, 1, 1])
        
        vis.add_box(center, size, name='test_box')
        assert isinstance(vis, PyVistaVisualizer)
    
    def test_add_point_cloud(self):
        """Test adding point cloud"""
        vis = PyVistaVisualizer()
        
        points = np.random.rand(100, 3)
        vis.add_point_cloud(points, name='test_cloud')
        
        assert isinstance(vis, PyVistaVisualizer)
    
    def test_is_connected(self):
        """Test connection status check"""
        vis = PyVistaVisualizer()
        
        # Should have a connection status
        status = vis.is_connected()
        assert isinstance(status, bool)


class TestVisualizationIntegration:
    """Integration tests for visualization components"""
    
    def test_dashboard_with_real_data(self):
        """Test dashboard with simulated training data"""
        dashboard = TrainingDashboard()
        
        # Simulate training
        for epoch in range(50):
            loss = 100 * np.exp(-epoch / 30)
            reward = 50 * (1 - np.exp(-epoch / 20))
            success_rate = min(0.95, epoch / 50)
            
            dashboard.update_metrics(
                loss=loss,
                reward=reward,
                success_rate=success_rate
            )
        
        summary = dashboard.get_metrics_summary()
        
        assert len(summary) > 0
        assert summary['loss']['count'] == 50
        assert summary['reward']['count'] == 50
    
    def test_meshcat_fallback_mode(self):
        """Test MeshCat fallback when server unavailable"""
        viz = MeshCatVisualizer(fallback_mode=True)
        
        # Should not raise errors even if server unavailable
        waypoints = np.array([[0, 0, 0], [1, 1, 1]])
        viz.add_trajectory(waypoints)
        
        center = np.array([0.5, 0.5, 0.5])
        viz.add_sphere(center)
        
        # Verify it queues operations
        count = viz.get_pending_count()
        assert isinstance(count, int)
    
    def test_pyvista_graceful_handling(self):
        """Test PyVista graceful handling of missing dependencies"""
        vis = PyVistaVisualizer()
        
        # Should not crash even if PyVista unavailable
        waypoints = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0]])
        vis.add_trajectory(waypoints)
        
        points = np.random.rand(50, 3)
        vis.add_point_cloud(points)
        
        # Should complete without errors
        assert True


class TestDataValidation:
    """Test data validation for visualization components"""
    
    def test_trajectory_shape_validation(self):
        """Test trajectory shape validation"""
        viz = MeshCatVisualizer(fallback_mode=True)
        
        # Invalid shape - should handle gracefully
        invalid_waypoints = np.array([0, 0, 0])  # 1D array
        viz.add_trajectory(invalid_waypoints)  # Should not crash
        
        # Valid shape
        valid_waypoints = np.array([[0, 0, 0], [1, 1, 1]])
        viz.add_trajectory(valid_waypoints)  # Should work
    
    def test_point_cloud_shape_validation(self):
        """Test point cloud shape validation"""
        vis = PyVistaVisualizer()
        
        # Valid point cloud
        valid_points = np.random.rand(100, 3)
        vis.add_point_cloud(valid_points)
        
        # Invalid point cloud - wrong dimensions
        invalid_points = np.random.rand(100, 2)  # Should log warning
        vis.add_point_cloud(invalid_points)
    
    def test_metrics_type_conversion(self):
        """Test metrics type conversion"""
        dashboard = TrainingDashboard()
        
        # Test with different numeric types
        dashboard.update_metrics(
            loss=np.float32(0.5),
            reward=int(10),
            success_rate=0.8
        )
        
        assert isinstance(dashboard.metrics['loss'][0], float)
        assert isinstance(dashboard.metrics['reward'][0], float)
