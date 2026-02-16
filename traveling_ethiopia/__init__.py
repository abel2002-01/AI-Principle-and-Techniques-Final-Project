"""
Traveling Ethiopia Search Problem
AI Principles and Techniques Course Project

This package implements various search algorithms for navigating
through Ethiopian cities represented as a state space graph.
"""

from .graph import StateSpaceGraph, WeightedGraph
from .search_strategies import SearchSolver
from .ucs import UniformCostSearch
from .astar import AStarSearch
from .minimax import MiniMaxSearch

__version__ = "1.0.0"
__all__ = [
    "StateSpaceGraph",
    "WeightedGraph", 
    "SearchSolver",
    "UniformCostSearch",
    "AStarSearch",
    "MiniMaxSearch"
]

