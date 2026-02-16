"""
A* Search Implementation

Question 3: Write a class that uses A* search to generate a path 
from the initial state "Addis Ababa" to the goal state "Moyale".
"""

import heapq
from typing import List, Optional, Dict, Set, Tuple, Callable
from .graph import WeightedGraph


class AStarSearch:
    """
    A* Search Algorithm implementation.
    
    A* is an informed search algorithm that combines:
    - g(n): Actual cost from start to current node
    - h(n): Heuristic estimate from current node to goal
    
    f(n) = g(n) + h(n)
    
    Properties:
    - Complete: Yes (if finite branching factor and step cost > 0)
    - Optimal: Yes (if heuristic is admissible and consistent)
    - Time Complexity: O(b^d) in worst case
    - Space Complexity: O(b^d) - keeps all nodes in memory
    
    For optimality:
    - Admissible: h(n) <= actual cost from n to goal (never overestimates)
    - Consistent: h(n) <= c(n, n') + h(n') for all successors n'
    """
    
    def __init__(self, graph: WeightedGraph, 
                 heuristic: Optional[Dict[str, float]] = None):
        """
        Initialize A* search.
        
        Args:
            graph: WeightedGraph with edge costs and optionally heuristics
            heuristic: Optional dictionary of heuristic values {node: h_value}
                      If None, uses heuristics stored in the graph
        """
        self.graph = graph
        self.external_heuristic = heuristic
    
    def get_heuristic(self, node: str, goal: str) -> float:
        """
        Get the heuristic value for a node.
        
        Args:
            node: Current node
            goal: Goal node
        
        Returns:
            Heuristic estimate from node to goal
        """
        # Try external heuristic first
        if self.external_heuristic and node in self.external_heuristic:
            return self.external_heuristic[node]
        
        # Then try graph's stored heuristics
        return self.graph.get_heuristic(node, goal)
    
    def search(self, initial_state: str, goal_state: str) -> Optional[Dict]:
        """
        Find the optimal path from initial_state to goal_state using A*.
        
        Args:
            initial_state: Starting node (e.g., "Addis Ababa")
            goal_state: Target node (e.g., "Moyale")
        
        Returns:
            Dictionary containing:
            - path: List of nodes in the optimal path
            - cost: Total path cost (g-value)
            - nodes_expanded: Number of nodes expanded
            - f_values: f-values along the path
            Or None if no path exists
        """
        if initial_state == goal_state:
            return {
                "path": [initial_state],
                "cost": 0,
                "nodes_expanded": 0,
                "f_values": [0]
            }
        
        # Priority queue: (f_value, counter, g_value, node, path)
        counter = 0
        h_start = self.get_heuristic(initial_state, goal_state)
        frontier: List[Tuple[float, int, float, str, List[str]]] = []
        heapq.heappush(frontier, (h_start, counter, 0, initial_state, [initial_state]))
        
        # Track explored nodes with their g-values
        explored: Dict[str, float] = {}
        
        # Track f-values for path reconstruction
        f_values_map: Dict[str, float] = {initial_state: h_start}
        
        nodes_expanded = 0
        
        while frontier:
            f_value, _, g_value, current, path = heapq.heappop(frontier)
            
            # Skip if we've found a better path to this node
            if current in explored and g_value >= explored[current]:
                continue
            
            nodes_expanded += 1
            explored[current] = g_value
            
            # Goal test when expanding
            if current == goal_state:
                # Reconstruct f-values for the path
                f_vals = [f_values_map.get(node, 0) for node in path]
                return {
                    "path": path,
                    "cost": g_value,
                    "nodes_expanded": nodes_expanded,
                    "f_values": f_vals
                }
            
            # Expand neighbors
            for neighbor, edge_cost in self.graph.get_weighted_neighbors(current):
                new_g = g_value + edge_cost
                
                # Only consider if we haven't found a better path
                if neighbor not in explored or new_g < explored[neighbor]:
                    h = self.get_heuristic(neighbor, goal_state)
                    new_f = new_g + h
                    
                    counter += 1
                    new_path = path + [neighbor]
                    f_values_map[neighbor] = new_f
                    heapq.heappush(frontier, (new_f, counter, new_g, neighbor, new_path))
        
        # No path found
        return None
    
    def search_with_trace(self, initial_state: str, 
                          goal_state: str) -> Optional[Dict]:
        """
        Search with detailed trace for debugging/visualization.
        
        Returns additional information about the search process.
        """
        if initial_state == goal_state:
            return {
                "path": [initial_state],
                "cost": 0,
                "nodes_expanded": 0,
                "expansion_order": [initial_state],
                "frontier_evolution": []
            }
        
        counter = 0
        h_start = self.get_heuristic(initial_state, goal_state)
        frontier = [(h_start, counter, 0, initial_state, [initial_state])]
        
        explored: Dict[str, float] = {}
        expansion_order = []
        frontier_evolution = []
        
        nodes_expanded = 0
        
        while frontier:
            # Record frontier state
            frontier_state = [(f, n, g) for f, _, g, n, _ in frontier]
            frontier_evolution.append({
                "step": nodes_expanded,
                "frontier": frontier_state
            })
            
            f_value, _, g_value, current, path = heapq.heappop(frontier)
            
            if current in explored and g_value >= explored[current]:
                continue
            
            nodes_expanded += 1
            explored[current] = g_value
            expansion_order.append({
                "node": current,
                "g": g_value,
                "h": self.get_heuristic(current, goal_state),
                "f": f_value
            })
            
            if current == goal_state:
                return {
                    "path": path,
                    "cost": g_value,
                    "nodes_expanded": nodes_expanded,
                    "expansion_order": expansion_order,
                    "frontier_evolution": frontier_evolution
                }
            
            for neighbor, edge_cost in self.graph.get_weighted_neighbors(current):
                new_g = g_value + edge_cost
                
                if neighbor not in explored or new_g < explored[neighbor]:
                    h = self.get_heuristic(neighbor, goal_state)
                    new_f = new_g + h
                    counter += 1
                    heapq.heappush(frontier, 
                        (new_f, counter, new_g, neighbor, path + [neighbor]))
        
        return None


class WeightedAStar(AStarSearch):
    """
    Weighted A* (WA*) Search.
    
    Uses f(n) = g(n) + w * h(n) where w > 1.
    
    This trades optimality for speed:
    - w = 1: Standard A* (optimal)
    - w > 1: Faster but solution cost <= w * optimal cost
    """
    
    def __init__(self, graph: WeightedGraph, 
                 weight: float = 1.5,
                 heuristic: Optional[Dict[str, float]] = None):
        """
        Initialize Weighted A*.
        
        Args:
            graph: WeightedGraph instance
            weight: Weight factor w >= 1
            heuristic: Optional heuristic dictionary
        """
        super().__init__(graph, heuristic)
        self.weight = max(1.0, weight)
    
    def search(self, initial_state: str, goal_state: str) -> Optional[Dict]:
        """Search with weighted heuristic."""
        if initial_state == goal_state:
            return {
                "path": [initial_state],
                "cost": 0,
                "nodes_expanded": 0,
                "weight_used": self.weight
            }
        
        counter = 0
        h_start = self.weight * self.get_heuristic(initial_state, goal_state)
        frontier = [(h_start, counter, 0, initial_state, [initial_state])]
        
        explored: Dict[str, float] = {}
        nodes_expanded = 0
        
        while frontier:
            f_value, _, g_value, current, path = heapq.heappop(frontier)
            
            if current in explored and g_value >= explored[current]:
                continue
            
            nodes_expanded += 1
            explored[current] = g_value
            
            if current == goal_state:
                return {
                    "path": path,
                    "cost": g_value,
                    "nodes_expanded": nodes_expanded,
                    "weight_used": self.weight
                }
            
            for neighbor, edge_cost in self.graph.get_weighted_neighbors(current):
                new_g = g_value + edge_cost
                
                if neighbor not in explored or new_g < explored[neighbor]:
                    h = self.weight * self.get_heuristic(neighbor, goal_state)
                    new_f = new_g + h
                    counter += 1
                    heapq.heappush(frontier,
                        (new_f, counter, new_g, neighbor, path + [neighbor]))
        
        return None


class IDAStarSearch:
    """
    Iterative Deepening A* (IDA*) Search.
    
    Combines A*'s heuristic guidance with iterative deepening's
    memory efficiency. Uses f-value as the cutoff.
    
    Properties:
    - Complete: Yes
    - Optimal: Yes (with admissible heuristic)
    - Space Complexity: O(bd) - linear!
    """
    
    def __init__(self, graph: WeightedGraph,
                 heuristic: Optional[Dict[str, float]] = None):
        self.graph = graph
        self.external_heuristic = heuristic
    
    def get_heuristic(self, node: str, goal: str) -> float:
        if self.external_heuristic and node in self.external_heuristic:
            return self.external_heuristic[node]
        return self.graph.get_heuristic(node, goal)
    
    def search(self, initial_state: str, goal_state: str) -> Optional[Dict]:
        """
        Perform IDA* search.
        
        Args:
            initial_state: Starting node
            goal_state: Target node
        
        Returns:
            Path information or None
        """
        if initial_state == goal_state:
            return {"path": [initial_state], "cost": 0, "iterations": 0}
        
        threshold = self.get_heuristic(initial_state, goal_state)
        path = [initial_state]
        iterations = 0
        
        while True:
            iterations += 1
            result = self._search_recursive(path, 0, threshold, goal_state)
            
            if isinstance(result, dict):
                result["iterations"] = iterations
                return result
            
            if result == float('inf'):
                return None  # No solution exists
            
            threshold = result  # Increase threshold to minimum f > old threshold
    
    def _search_recursive(self, path: List[str], g: float, 
                          threshold: float, goal: str) -> any:
        """
        Recursive helper for IDA*.
        
        Returns:
            - dict with solution if found
            - float('inf') if no solution
            - new threshold otherwise
        """
        node = path[-1]
        f = g + self.get_heuristic(node, goal)
        
        if f > threshold:
            return f
        
        if node == goal:
            return {"path": list(path), "cost": g}
        
        min_threshold = float('inf')
        
        for neighbor, edge_cost in self.graph.get_weighted_neighbors(node):
            if neighbor not in path:  # Avoid cycles
                path.append(neighbor)
                result = self._search_recursive(path, g + edge_cost, threshold, goal)
                
                if isinstance(result, dict):
                    return result
                
                min_threshold = min(min_threshold, result)
                path.pop()
        
        return min_threshold

