"""
Search Strategies: Breadth-First Search and Depth-First Search

Question 1.2: Write a class that takes the converted state space graph, 
initial state, goal state and a search strategy and returns the 
corresponding solution/path according to the given strategy.
"""

from collections import deque
from typing import List, Optional, Dict, Set, Tuple
from .graph import StateSpaceGraph


class SearchSolver:
    """
    A search solver that can use either BFS or DFS strategy.
    
    This class implements both Breadth-First Search and Depth-First Search
    algorithms for finding paths in an unweighted graph.
    
    Attributes:
        graph: The state space graph to search
        initial_state: Starting node for the search
        goal_state: Target node to find
    """
    
    def __init__(self, graph: StateSpaceGraph, 
                 initial_state: str, 
                 goal_state: str):
        """
        Initialize the search solver.
        
        Args:
            graph: StateSpaceGraph instance representing the search space
            initial_state: The starting node
            goal_state: The target node to reach
        """
        self.graph = graph
        self.initial_state = initial_state
        self.goal_state = goal_state
    
    def solve(self, strategy: str = "bfs") -> Optional[List[str]]:
        """
        Find a path from initial_state to goal_state using the specified strategy.
        
        Args:
            strategy: Either "bfs" for Breadth-First Search or 
                     "dfs" for Depth-First Search
        
        Returns:
            List of nodes representing the path from initial to goal state,
            or None if no path exists.
        
        Raises:
            ValueError: If an invalid strategy is specified
        """
        strategy = strategy.lower()
        
        if strategy == "bfs":
            return self._breadth_first_search()
        elif strategy == "dfs":
            return self._depth_first_search()
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Use 'bfs' or 'dfs'.")
    
    def _breadth_first_search(self) -> Optional[List[str]]:
        """
        Implement Breadth-First Search.
        
        BFS explores all neighbors at the current depth before moving to
        nodes at the next depth level. It uses a QUEUE (FIFO) data structure.
        
        Properties:
        - Complete: Yes (if branching factor is finite)
        - Optimal: Yes (for unweighted graphs)
        - Time Complexity: O(b^d)
        - Space Complexity: O(b^d)
        
        Returns:
            Path from initial to goal state, or None if not found.
        """
        if self.initial_state == self.goal_state:
            return [self.initial_state]
        
        # Use a queue (FIFO) - elements are added at the end, removed from front
        frontier: deque = deque()
        frontier.append(self.initial_state)
        
        # Track visited nodes to avoid cycles
        explored: Set[str] = set()
        
        # Track parent of each node to reconstruct path
        parent: Dict[str, Optional[str]] = {self.initial_state: None}
        
        while frontier:
            # Remove from front of queue (FIFO behavior)
            current = frontier.popleft()
            
            # Mark as explored
            explored.add(current)
            
            # Explore all neighbors
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in explored and neighbor not in parent:
                    parent[neighbor] = current
                    
                    # Goal test when generating
                    if neighbor == self.goal_state:
                        return self._reconstruct_path(parent, neighbor)
                    
                    # Add to end of queue
                    frontier.append(neighbor)
        
        # No path found
        return None
    
    def _depth_first_search(self) -> Optional[List[str]]:
        """
        Implement Depth-First Search.
        
        DFS explores as far as possible along each branch before backtracking.
        It uses a STACK (LIFO) data structure.
        
        Properties:
        - Complete: No (might get stuck in infinite paths)
        - Optimal: No
        - Time Complexity: O(b^m) where m is maximum depth
        - Space Complexity: O(bm)
        
        Returns:
            Path from initial to goal state, or None if not found.
        """
        if self.initial_state == self.goal_state:
            return [self.initial_state]
        
        # Use a stack (LIFO) - elements are added and removed from the same end
        frontier: List[str] = []
        frontier.append(self.initial_state)
        
        # Track visited nodes to avoid cycles
        explored: Set[str] = set()
        
        # Track parent of each node to reconstruct path
        parent: Dict[str, Optional[str]] = {self.initial_state: None}
        
        while frontier:
            # Remove from end of list (LIFO behavior - stack pop)
            current = frontier.pop()
            
            # Goal test when expanding
            if current == self.goal_state:
                return self._reconstruct_path(parent, current)
            
            # Mark as explored
            explored.add(current)
            
            # Explore all neighbors (reversed to maintain intuitive order)
            for neighbor in reversed(self.graph.get_neighbors(current)):
                if neighbor not in explored and neighbor not in parent:
                    parent[neighbor] = current
                    # Add to end of list (stack push)
                    frontier.append(neighbor)
        
        # No path found
        return None
    
    def _reconstruct_path(self, parent: Dict[str, Optional[str]], 
                          goal: str) -> List[str]:
        """
        Reconstruct the path from initial state to goal state.
        
        Args:
            parent: Dictionary mapping each node to its parent
            goal: The goal node
        
        Returns:
            List of nodes from initial state to goal state
        """
        path = []
        current: Optional[str] = goal
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        # Reverse to get path from initial to goal
        path.reverse()
        return path
    
    def get_all_paths(self, strategy: str = "bfs", 
                      max_paths: int = 10) -> List[List[str]]:
        """
        Find multiple paths from initial to goal state.
        
        Args:
            strategy: Search strategy to use
            max_paths: Maximum number of paths to find
        
        Returns:
            List of paths found
        """
        paths = []
        
        if strategy == "bfs":
            paths = self._bfs_all_paths(max_paths)
        elif strategy == "dfs":
            paths = self._dfs_all_paths(max_paths)
        
        return paths
    
    def _bfs_all_paths(self, max_paths: int) -> List[List[str]]:
        """Find all paths using BFS."""
        if self.initial_state == self.goal_state:
            return [[self.initial_state]]
        
        paths = []
        queue: deque = deque()
        queue.append([self.initial_state])
        
        while queue and len(paths) < max_paths:
            path = queue.popleft()
            current = path[-1]
            
            if current == self.goal_state:
                paths.append(path)
                continue
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in path:  # Avoid cycles within this path
                    new_path = path + [neighbor]
                    queue.append(new_path)
        
        return paths
    
    def _dfs_all_paths(self, max_paths: int) -> List[List[str]]:
        """Find all paths using DFS."""
        if self.initial_state == self.goal_state:
            return [[self.initial_state]]
        
        paths = []
        stack: List[List[str]] = [[self.initial_state]]
        
        while stack and len(paths) < max_paths:
            path = stack.pop()
            current = path[-1]
            
            if current == self.goal_state:
                paths.append(path)
                continue
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in path:
                    new_path = path + [neighbor]
                    stack.append(new_path)
        
        return paths
    
    def visualize_search(self, strategy: str = "bfs") -> Dict[str, any]:
        """
        Visualize the search process by returning exploration order.
        
        Args:
            strategy: Search strategy to visualize
        
        Returns:
            Dictionary with exploration statistics
        """
        exploration_order = []
        nodes_expanded = 0
        
        if strategy.lower() == "bfs":
            frontier = deque([self.initial_state])
        else:
            frontier = [self.initial_state]
        
        explored = set()
        parent = {self.initial_state: None}
        
        while frontier:
            if strategy.lower() == "bfs":
                current = frontier.popleft()
            else:
                current = frontier.pop()
            
            exploration_order.append(current)
            nodes_expanded += 1
            explored.add(current)
            
            if current == self.goal_state:
                break
            
            neighbors = self.graph.get_neighbors(current)
            if strategy.lower() == "dfs":
                neighbors = reversed(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in explored and neighbor not in parent:
                    parent[neighbor] = current
                    if strategy.lower() == "bfs":
                        frontier.append(neighbor)
                    else:
                        frontier.append(neighbor)
        
        path = self._reconstruct_path(parent, self.goal_state) if self.goal_state in parent else None
        
        return {
            "strategy": strategy.upper(),
            "exploration_order": exploration_order,
            "nodes_expanded": nodes_expanded,
            "path": path,
            "path_length": len(path) if path else None,
            "path_found": path is not None
        }


class IterativeDeepeningDFS:
    """
    Iterative Deepening Depth-First Search (IDDFS).
    
    Combines the space efficiency of DFS with the optimality of BFS.
    """
    
    def __init__(self, graph: StateSpaceGraph, 
                 initial_state: str, 
                 goal_state: str):
        self.graph = graph
        self.initial_state = initial_state
        self.goal_state = goal_state
    
    def search(self, max_depth: int = 100) -> Optional[List[str]]:
        """
        Search with iterative deepening.
        
        Args:
            max_depth: Maximum depth to search
        
        Returns:
            Path from initial to goal, or None if not found
        """
        for depth in range(max_depth):
            result = self._depth_limited_search(depth)
            if result is not None:
                return result
        return None
    
    def _depth_limited_search(self, limit: int) -> Optional[List[str]]:
        """
        Perform depth-limited DFS.
        
        Args:
            limit: Maximum depth to explore
        
        Returns:
            Path if found within limit, None otherwise
        """
        return self._recursive_dls(self.initial_state, limit, [])
    
    def _recursive_dls(self, node: str, limit: int, 
                       path: List[str]) -> Optional[List[str]]:
        """Recursive helper for depth-limited search."""
        current_path = path + [node]
        
        if node == self.goal_state:
            return current_path
        
        if limit == 0:
            return None
        
        for neighbor in self.graph.get_neighbors(node):
            if neighbor not in current_path:  # Avoid cycles
                result = self._recursive_dls(neighbor, limit - 1, current_path)
                if result is not None:
                    return result
        
        return None

