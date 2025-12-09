"""GCS Builder - Construct Graph of Convex Sets for motion planning"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GCSBuilder:
    """Build Graph of Convex Sets for motion planning problems"""

    def __init__(self, dimension: int = 3):
        """Initialize GCS builder
        
        Args:
            dimension: Problem dimension (2D or 3D)
        """
        self.dimension = dimension
        self.graph = nx.DiGraph()
        self.convex_sets = {}
        self.source = None
        self.target = None
        logger.info(f"GCSBuilder initialized for {dimension}D problems")

    def add_convex_set(self, set_id: str, vertices: np.ndarray, 
                       costs: Optional[Dict] = None):
        """Add a convex set (region) to the graph
        
        Args:
            set_id: Unique identifier for the convex set
            vertices: (N, dimension) array of vertices defining the polytope
            costs: Cost parameters for traversing this set
        """
        self.convex_sets[set_id] = {
            'vertices': vertices,
            'costs': costs or {},
            'dimension': vertices.shape[1]
        }
        self.graph.add_node(set_id)
        logger.debug(f"Added convex set: {set_id}")

    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0):
        """Add edge between two convex sets
        
        Args:
            source_id: Source convex set
            target_id: Target convex set
            weight: Edge weight (cost)
        """
        self.graph.add_edge(source_id, target_id, weight=weight)
        logger.debug(f"Added edge: {source_id} -> {target_id}")

    def set_source_target(self, source_id: str, target_id: str):
        """Set source and target nodes for planning
        
        Args:
            source_id: Starting convex set
            target_id: Goal convex set
        """
        self.source = source_id
        self.target = target_id
        logger.info(f"Set source: {source_id}, target: {target_id}")

    def get_shortest_path(self) -> Optional[List[str]]:
        """Get shortest path through the graph
        
        Returns:
            List of convex set IDs forming the path
        """
        if self.source is None or self.target is None:
            logger.warning("Source or target not set")
            return None
        
        try:
            path = nx.shortest_path(self.graph, self.source, self.target, weight='weight')
            logger.info(f"Found path with {len(path)} regions")
            return path
        except nx.NetworkXNoPath:
            logger.warning("No path found between source and target")
            return None

    def get_graph_info(self) -> Dict:
        """Get information about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'dimension': self.dimension,
            'num_convex_sets': len(self.convex_sets)
        }

class ConvexSetBuilder:
    """Build individual convex sets (polytopes)"""

    @staticmethod
    def box(center: np.ndarray, half_lengths: np.ndarray) -> np.ndarray:
        """Create a box (hyperrectangle) polytope
        
        Args:
            center: Center point
            half_lengths: Half-lengths in each dimension
            
        Returns:
            Vertices of the box
        """
        dimension = len(center)
        vertices = []
        for i in range(2 ** dimension):
            vertex = center.copy()
            for d in range(dimension):
                if (i >> d) & 1:
                    vertex[d] += half_lengths[d]
                else:
                    vertex[d] -= half_lengths[d]
            vertices.append(vertex)
        return np.array(vertices)

    @staticmethod
    def sphere(center: np.ndarray, radius: float, num_points: int = 8) -> np.ndarray:
        """Create a polytope approximating a sphere
        
        Args:
            center: Center point
            radius: Sphere radius
            num_points: Number of vertices
            
        Returns:
            Vertices approximating the sphere
        """
        dimension = len(center)
        if dimension == 2:
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            vertices = center[:, np.newaxis] + radius * np.array([np.cos(angles), np.sin(angles)])
            return vertices.T
        elif dimension == 3:
            # Icosphere approximation
            phi = (1 + np.sqrt(5)) / 2
            vertices = [
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ]
            vertices = np.array(vertices)
            vertices = vertices / np.linalg.norm(vertices[0]) * radius
            return vertices + center
        else:
            raise NotImplementedError(f"Sphere approximation not implemented for {dimension}D")

    @staticmethod
    def polytope_from_halfspaces(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Create polytope from halfspace representation (Ax <= b)
        
        Args:
            A: (m, d) inequality matrix
            b: (m,) inequality vector
            
        Returns:
            Vertices of the polytope
        """
        # This would use scipy.spatial.HalfspaceIntersection in practice
        logger.warning("Halfspace to vertices conversion requires scipy.spatial")
        return np.array([])
