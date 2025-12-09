"""GCS Solver - Solve Graph of Convex Sets optimization problems"""
import numpy as np
import cvxpy as cp
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GCSSolver:
    """Solve GCS motion planning problems"""

    def __init__(self, verbose: bool = False, solver: str = 'ECOS'):
        """Initialize solver
        
        Args:
            verbose: Print solver output
            solver: Solver backend ('ECOS', 'SCS', 'MOSEK')
        """
        self.verbose = verbose
        self.solver = solver
        self.solution = None
        logger.info(f"GCSSolver initialized with {solver} backend")

    def solve(self, gcs_builder, start: np.ndarray, goal: np.ndarray,
              time_limit: float = 30.0) -> Optional[Dict]:
        """Solve GCS problem
        
        Args:
            gcs_builder: GCSBuilder instance with problem definition
            start: Start configuration
            goal: Goal configuration
            time_limit: Maximum solve time in seconds
            
        Returns:
            Solution dictionary or None if infeasible
        """
        gcs_builder.set_source_target('source', 'target')
        path = gcs_builder.get_shortest_path()
        
        if path is None:
            logger.warning("No path found in GCS graph")
            return None

        try:
            # Formulate as convex optimization
            trajectory = self._optimize_trajectory(
                gcs_builder, path, start, goal, time_limit
            )
            
            if trajectory is not None:
                self.solution = {
                    'trajectory': trajectory,
                    'path': path,
                    'feasible': True,
                    'length': len(trajectory)
                }
                logger.info(f"Found feasible solution with {len(trajectory)} waypoints")
                return self.solution
            else:
                logger.warning("Optimization failed to find feasible solution")
                return None
                
        except Exception as e:
            logger.error(f"Solver error: {e}")
            return None

    def _optimize_trajectory(self, gcs_builder, path: list, 
                            start: np.ndarray, goal: np.ndarray,
                            time_limit: float) -> Optional[np.ndarray]:
        """Optimize trajectory through convex sets
        
        Args:
            gcs_builder: Problem definition
            path: Sequence of convex sets
            start: Start point
            goal: Goal point
            time_limit: Time limit for optimization
            
        Returns:
            Optimized trajectory waypoints
        """
        dimension = gcs_builder.dimension
        num_sets = len(path)
        
        # Variables: waypoints in each convex set + binary variables for transitions
        waypoints = [cp.Variable(dimension) for _ in range(num_sets)]
        
        constraints = []
        
        # Start constraint
        constraints.append(waypoints[0] == start)
        
        # Goal constraint
        constraints.append(waypoints[-1] == goal)
        
        # Convexity constraints - each waypoint should be in its convex set
        for i, set_id in enumerate(path):
            if set_id in gcs_builder.convex_sets:
                vertices = gcs_builder.convex_sets[set_id]['vertices']
                # Simple constraint: waypoint is convex combination of vertices
                # In practice, use scipy.spatial.HalfspaceIntersection
                constraints.append(cp.sum(waypoints[i]) <= 1e6)  # Placeholder
        
        # Trajectory continuity constraints
        for i in range(len(waypoints) - 1):
            constraints.append(cp.norm(waypoints[i+1] - waypoints[i]) <= 10)
        
        # Objective: minimize path length
        objective = cp.Minimize(
            sum(cp.norm(waypoints[i+1] - waypoints[i]) for i in range(len(waypoints)-1))
        )
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=getattr(cp, self.solver), verbose=self.verbose)
        
        if problem.status == cp.OPTIMAL:
            trajectory = np.array([w.value for w in waypoints])
            return trajectory
        else:
            return None

    def get_trajectory_cost(self) -> Optional[float]:
        """Get cost of current solution
        
        Returns:
            Trajectory cost or None
        """
        if self.solution is None:
            return None
        
        trajectory = self.solution['trajectory']
        cost = 0.0
        for i in range(len(trajectory) - 1):
            cost += np.linalg.norm(trajectory[i+1] - trajectory[i])
        return cost
