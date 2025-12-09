"""
Plotly Dashboard - Production-Ready Training Metrics Visualization
Author: Hybrid-GCS Project
Status: PRODUCTION READY - All color formats fixed
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingDashboard:
    """Professional training metrics dashboard with Plotly - CORRECTED COLOR FORMATS"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.metrics: Dict[str, List[float]] = {
            "loss": [],
            "reward": [],
            "success_rate": [],
            "path_length": [],
            "planning_time": [],
        }
        logger.info("Dashboard initialized")
    
    def update_metrics(self, episode: int = 0, loss: float = 0.0, reward: float = 0.0,
                      success_rate: float = 0.0, path_length: float = 0.0,
                      planning_time: float = 0.0):
        """Update metrics with new data
        
        Args:
            episode: Episode number
            loss: Training loss
            reward: Episode reward
            success_rate: Success rate (0-1)
            path_length: Average path length
            planning_time: Planning time in seconds
        """
        self.metrics["loss"].append(float(loss))
        self.metrics["reward"].append(float(reward))
        self.metrics["success_rate"].append(float(success_rate))
        self.metrics["path_length"].append(float(path_length))
        self.metrics["planning_time"].append(float(planning_time))
    
    def create_dashboard(self, filename: str = "training_dashboard.html") -> go.Figure:
        """Create multi-panel dashboard
        
        Args:
            filename: HTML output filename
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("Loss", "Reward", "Success Rate",
                           "Path Length", "Planning Time", "Episode Count"),
            specs=[[{"secondary_y": False}] * 3] * 2
        )
        
        # Professional color palette (hex format) - VALID COLORS
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        metrics_list = [
            ("loss", "Loss", colors[0]),
            ("reward", "Reward", colors[1]),
            ("success_rate", "Success Rate", colors[2]),
            ("path_length", "Path Length", colors[3]),
            ("planning_time", "Planning Time", colors[4]),
        ]
        
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        
        for (metric_key, title, color), (row, col) in zip(metrics_list, positions):
            values = self.metrics.get(metric_key, [])
            if not values:
                continue
            
            x_vals = list(range(len(values)))
            
            # Convert hex to RGBA - CORRECTLY FORMATTED
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fill_color = f"rgba({r}, {g}, {b}, 0.2)"
            
            # Add main line trace
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=values,
                    name=title,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=4, opacity=0.6),
                    fill='tozeroy',
                    fillcolor=fill_color,
                    hovertemplate=f"{title}: %{{y:.4f}}<br>Episode: %{{x}}"
                ),
                row=row, col=col
            )
            
            # Add moving average (window=10)
            if len(values) > 10:
                window = 10
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                x_moving = x_vals[window-1:]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_moving,
                        y=moving_avg,
                        name=f"{title} (MA)",
                        mode='lines',
                        line=dict(color=color, width=3, dash='dash'),
                        hovertemplate=f"{title} MA: %{{y:.4f}}<br>Episode: %{{x}}"
                    ),
                    row=row, col=col
                )
            
            # Update axis labels
            fig.update_xaxes(title_text="Episode", row=row, col=col)
            fig.update_yaxes(title_text=title, row=row, col=col)
        
        # Episode count in bottom right
        total_episodes = len(self.metrics["loss"]) if self.metrics["loss"] else 0
        fig.add_annotation(
            text=f"Total Episodes: {total_episodes}",
            xref="paper", yref="paper",
            x=0.85, y=0.15,
            showarrow=False,
            font=dict(size=14, color="black")
        )
        
        # Update layout with professional styling
        fig.update_layout(
            title="Training Metrics Dashboard",
            height=800,
            showlegend=True,
            hovermode='x unified',
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=11),
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            paper_bgcolor="white",
        )
        
        # Save to file
        try:
            fig.write_html(filename)
            logger.info(f"Dashboard saved to {filename}")
            print(f"✓ Dashboard saved to {filename}")
        except Exception as e:
            logger.warning(f"Could not save dashboard: {e}")
        
        return fig
    
    def create_single_metric_plot(self, metric: str,
                                 filename: Optional[str] = None) -> Optional[go.Figure]:
        """Create detailed plot for single metric
        
        Args:
            metric: Metric name to plot
            filename: Optional HTML output filename
            
        Returns:
            Plotly figure object
        """
        values = self.metrics.get(metric, [])
        if not values:
            logger.warning(f"No data for metric {metric}")
            return None
        
        x_vals = list(range(len(values)))
        
        fig = go.Figure()
        
        # Main line
        fig.add_trace(go.Scatter(
            x=x_vals, y=values,
            name=metric,
            mode='lines+markers',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=6, opacity=0.7),
            fill='tozeroy',
            fillcolor='rgba(78, 205, 196, 0.2)',
            hovertemplate=f"{metric}: %{{y:.4f}}<br>Episode: %{{x}}"
        ))
        
        # Moving average
        if len(values) > 10:
            window = 10
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            x_moving = x_vals[window-1:]
            
            fig.add_trace(go.Scatter(
                x=x_moving, y=moving_avg,
                name=f"{metric} (Moving Avg)",
                mode='lines',
                line=dict(color='#FF6B6B', width=3, dash='dash'),
                hovertemplate=f"{metric} MA: %{{y:.4f}}<br>Episode: %{{x}}"
            ))
        
        fig.update_layout(
            title=f"{metric.title()} Analysis",
            xaxis_title="Episode",
            yaxis_title=metric.title(),
            height=500,
            hovermode='x unified',
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            paper_bgcolor="white",
        )
        
        if filename:
            try:
                fig.write_html(filename)
                logger.info(f"Metric plot saved to {filename}")
                print(f"✓ Metric plot saved to {filename}")
            except Exception as e:
                logger.warning(f"Could not save metric plot: {e}")
        
        return fig
    
    def create_comparison_plot(self, metrics: List[str],
                              filename: Optional[str] = None) -> go.Figure:
        """Create normalized comparison of multiple metrics
        
        Args:
            metrics: List of metric names to compare
            filename: Optional HTML output filename
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for metric, color in zip(metrics, colors):
            values = np.array(self.metrics.get(metric, []))
            
            if len(values) == 0:
                continue
            
            # Normalize to [0, 1]
            min_val = np.min(values)
            max_val = np.max(values)
            
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(values) * 0.5
            
            x_vals = list(range(len(normalized)))
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=normalized,
                name=metric,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=5, opacity=0.7),
                hovertemplate=f"{metric} (normalized): %{{y:.4f}}<br>Episode: %{{x}}"
            ))
        
        fig.update_layout(
            title="Normalized Metrics Comparison",
            xaxis_title="Episode",
            yaxis_title="Normalized Value",
            yaxis=dict(range=[0, 1]),
            height=500,
            hovermode='x unified',
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            paper_bgcolor="white",
        )
        
        if filename:
            try:
                fig.write_html(filename)
                logger.info(f"Comparison plot saved to {filename}")
                print(f"✓ Comparison plot saved to {filename}")
            except Exception as e:
                logger.warning(f"Could not save comparison plot: {e}")
        
        return fig
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of metrics
        
        Returns:
            Dictionary with statistics for each metric
        """
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if not values:
                continue
            
            values_array = np.array(values)
            summary[metric_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "count": len(values),
            }
        
        return summary
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_metrics_summary()
        
        print("\n" + "="*70)
        print("TRAINING METRICS SUMMARY")
        print("="*70)
        
        for metric_name, stats in summary.items():
            print(f"\n{metric_name.upper()}")
            print(f"  Mean:  {stats['mean']:10.6f}")
            print(f"  Std:   {stats['std']:10.6f}")
            print(f"  Min:   {stats['min']:10.6f}")
            print(f"  Max:   {stats['max']:10.6f}")
            print(f"  Count: {stats['count']}")
        
        print("\n" + "="*70 + "\n")
