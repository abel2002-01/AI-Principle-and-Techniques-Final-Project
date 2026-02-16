"""
Uniform Cost Search (UCS) Implementation

Question 2.2: Write a program that uses uniform cost search to generate 
a path to Lalibela from Addis Ababa.

Question 2.3: Write a customized uniform cost search algorithm to visit 
multiple goal states while preserving local optimum.
"""

import heapq
from typing import List, Optional, Dict, Set, Tuple
from .graph import WeightedGraph


class UniformCostSearch:
    """
    Uniform Cost Search implementation for weighted graphs.
    
    UCS is an uninformed search algorithm that finds the lowest-cost path
    from a start node to a goal node. It's optimal and complete for 
    non-negative edge weights.
    
    Properties:
    - Complete: Yes (if step cost >= ε for some ε > 0)
    - Optimal: Yes
    - Time Complexity: O(b^(1+C*/ε)) where C* is optimal cost
    - Space Complexity: O(b^(1+C*/ε))
    """
    
    def __init__(self, graph: WeightedGraph):
        """
        Initialize UCS with a weighted graph.
        
        Args:
            graph: WeightedGraph instance with edge costs
        """
        self.graph = graph
    
    def search(self, initial_state: str, goal_state: str) -> Optional[Dict]:
        """
        Find the lowest-cost path from initial_state to goal_state.
        
        Args:
            initial_state: Starting node
            goal_state: Target node
        
        Returns:
            Dictionary containing:
            - path: List of nodes in the optimal path
            - cost: Total path cost
            - nodes_expanded: Number of nodes expanded during search
            Or None if no path exists
        """
        if initial_state == goal_state:
            return {
                "path": [initial_state],
                "cost": 0,
                "nodes_expanded": 0
            }
        
        # Priority queue: (cost, node, path)
        # Using counter to break ties consistently
        counter = 0
        frontier: List[Tuple[float, int, str, List[str]]] = []
        heapq.heappush(frontier, (0, counter, initial_state, [initial_state]))
        
        # Track explored nodes
        explored: Set[str] = set()
        
        # Track best cost to reach each node
        best_cost: Dict[str, float] = {initial_state: 0}
        
        nodes_expanded = 0
        
        while frontier:
            cost, _, current, path = heapq.heappop(frontier)
            
            # Skip if we've found a better path to this node
            if current in explored:
                continue
            
            nodes_expanded += 1
            
            # Goal test when expanding
            if current == goal_state:
                return {
                    "path": path,
                    "cost": cost,
                    "nodes_expanded": nodes_expanded
                }
            
            explored.add(current)
            
            # Expand neighbors
            for neighbor, edge_cost in self.graph.get_weighted_neighbors(current):
                if neighbor not in explored:
                    new_cost = cost + edge_cost
                    
                    # Only add if we found a better path
                    if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                        best_cost[neighbor] = new_cost
                        counter += 1
                        new_path = path + [neighbor]
                        heapq.heappush(frontier, (new_cost, counter, neighbor, new_path))
        
        # No path found
        return None
    
    def multi_goal_search(self, initial_state: str, 
                          goal_states: List[str]) -> Optional[Dict]:
        """
        Find an optimal path that visits all goal states.
        
        Question 2.3: Customized UCS to visit multiple goal states while
        preserving local optimum. Uses a greedy approach - from current
        position, find the nearest unvisited goal.
        
        Args:
            initial_state: Starting node
            goal_states: List of all goal states to visit
        
        Returns:
            Dictionary containing:
            - path: Complete path visiting all goals
            - cost: Total path cost
            - goals_order: Order in which goals were visited
            - segment_costs: Cost of each segment between goals
        """
        if not goal_states:
            return {
                "path": [initial_state],
                "cost": 0,
                "goals_order": [],
                "segment_costs": []
            }
        
        current = initial_state
        remaining_goals = set(goal_states)
        complete_path = [current]
        total_cost = 0
        goals_order = []
        segment_costs = []
        
        while remaining_goals:
            # Find nearest unvisited goal from current position
            nearest_goal = None
            nearest_result = None
            min_cost = float('inf')
            
            for goal in remaining_goals:
                result = self.search(current, goal)
                if result and result["cost"] < min_cost:
                    min_cost = result["cost"]
                    nearest_goal = goal
                    nearest_result = result
            
            if nearest_goal is None:
                # Cannot reach any remaining goals
                return None
            
            # Add this segment to the complete path
            # Skip the first node as it's already in the path
            complete_path.extend(nearest_result["path"][1:])
            total_cost += nearest_result["cost"]
            goals_order.append(nearest_goal)
            segment_costs.append(nearest_result["cost"])
            
            # Update state
            current = nearest_goal
            remaining_goals.remove(nearest_goal)
        
        return {
            "path": complete_path,
            "cost": total_cost,
            "goals_order": goals_order,
            "segment_costs": segment_costs
        }
    
    def multi_goal_search_optimal(self, initial_state: str, 
                                   goal_states: List[str]) -> Optional[Dict]:
        """
        Find the truly optimal path visiting all goals using dynamic programming.
        
        This is essentially solving TSP (Traveling Salesman Problem) which is
        NP-hard. For small number of goals, we can use exact methods.
        
        Uses bitmask DP approach for small goal counts (up to ~15 goals).
        
        Args:
            initial_state: Starting node
            goal_states: List of all goal states to visit
        
        Returns:
            Optimal path visiting all goals
        """
        n = len(goal_states)
        
        if n == 0:
            return {
                "path": [initial_state],
                "cost": 0,
                "goals_order": [],
                "segment_costs": []
            }
        
        # For too many goals, fall back to greedy
        if n > 12:
            print("Warning: Too many goals for optimal search, using greedy approach")
            return self.multi_goal_search(initial_state, goal_states)
        
        # Precompute all pairwise shortest paths
        all_nodes = [initial_state] + goal_states
        dist = {}
        path_cache = {}
        
        for i, src in enumerate(all_nodes):
            for j, dst in enumerate(all_nodes):
                if i != j:
                    result = self.search(src, dst)
                    if result:
                        dist[(src, dst)] = result["cost"]
                        path_cache[(src, dst)] = result["path"]
                    else:
                        dist[(src, dst)] = float('inf')
                        path_cache[(src, dst)] = None
        
        # DP with bitmask
        # dp[mask][i] = minimum cost to visit cities in mask, ending at goal i
        INF = float('inf')
        dp = [[INF] * n for _ in range(1 << n)]
        parent = [[(-1, -1)] * n for _ in range(1 << n)]
        
        # Initialize: starting from initial_state to each goal
        for i, goal in enumerate(goal_states):
            if dist[(initial_state, goal)] < INF:
                dp[1 << i][i] = dist[(initial_state, goal)]
        
        # Fill DP table
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == INF:
                    continue
                
                for next_goal in range(n):
                    if mask & (1 << next_goal):
                        continue
                    
                    new_mask = mask | (1 << next_goal)
                    new_cost = dp[mask][last] + dist[(goal_states[last], goal_states[next_goal])]
                    
                    if new_cost < dp[new_mask][next_goal]:
                        dp[new_mask][next_goal] = new_cost
                        parent[new_mask][next_goal] = (mask, last)
        
        # Find optimal ending point
        full_mask = (1 << n) - 1
        min_cost = INF
        last_goal = -1
        
        for i in range(n):
            if dp[full_mask][i] < min_cost:
                min_cost = dp[full_mask][i]
                last_goal = i
        
        if min_cost == INF:
            return None
        
        # Reconstruct goals order
        goals_order = []
        mask = full_mask
        current_goal = last_goal
        
        while current_goal != -1:
            goals_order.append(goal_states[current_goal])
            prev_mask, prev_goal = parent[mask][current_goal]
            mask = prev_mask
            current_goal = prev_goal
        
        goals_order.reverse()
        
        # Reconstruct complete path
        complete_path = [initial_state]
        segment_costs = []
        
        for goal in goals_order:
            prev = complete_path[-1]
            segment_path = path_cache[(prev, goal)]
            segment_cost = dist[(prev, goal)]
            complete_path.extend(segment_path[1:])
            segment_costs.append(segment_cost)
        
        return {
            "path": complete_path,
            "cost": min_cost,
            "goals_order": goals_order,
            "segment_costs": segment_costs
        }


class BidirectionalUCS:
    """
    Bidirectional Uniform Cost Search.
    
    Searches from both start and goal simultaneously,
    potentially reducing search space significantly.
    """
    
    def __init__(self, graph: WeightedGraph):
        self.graph = graph
    
    def search(self, initial_state: str, goal_state: str) -> Optional[Dict]:
        """
        Perform bidirectional UCS.
        
        Args:
            initial_state: Starting node
            goal_state: Target node
        
        Returns:
            Optimal path information or None
        """
        if initial_state == goal_state:
            return {"path": [initial_state], "cost": 0, "nodes_expanded": 0}
        
        # Forward search from initial_state
        counter_f = 0
        frontier_f = [(0, counter_f, initial_state, [initial_state])]
        explored_f: Dict[str, Tuple[float, List[str]]] = {}
        
        # Backward search from goal_state
        counter_b = 0
        frontier_b = [(0, counter_b, goal_state, [goal_state])]
        explored_b: Dict[str, Tuple[float, List[str]]] = {}
        
        best_cost = float('inf')
        best_path = None
        nodes_expanded = 0
        
        while frontier_f or frontier_b:
            # Expand from forward frontier
            if frontier_f:
                cost_f, _, node_f, path_f = heapq.heappop(frontier_f)
                
                if node_f not in explored_f or cost_f < explored_f[node_f][0]:
                    explored_f[node_f] = (cost_f, path_f)
                    nodes_expanded += 1
                    
                    # Check if we've met the backward search
                    if node_f in explored_b:
                        cost_b, path_b = explored_b[node_f]
                        total = cost_f + cost_b
                        if total < best_cost:
                            best_cost = total
                            best_path = path_f + path_b[-2::-1]  # Combine paths
                    
                    for neighbor, edge_cost in self.graph.get_weighted_neighbors(node_f):
                        if neighbor not in explored_f:
                            new_cost = cost_f + edge_cost
                            if new_cost < best_cost:
                                counter_f += 1
                                heapq.heappush(frontier_f, 
                                    (new_cost, counter_f, neighbor, path_f + [neighbor]))
            
            # Expand from backward frontier
            if frontier_b:
                cost_b, _, node_b, path_b = heapq.heappop(frontier_b)
                
                if node_b not in explored_b or cost_b < explored_b[node_b][0]:
                    explored_b[node_b] = (cost_b, path_b)
                    nodes_expanded += 1
                    
                    # Check if we've met the forward search
                    if node_b in explored_f:
                        cost_f, path_f = explored_f[node_b]
                        total = cost_f + cost_b
                        if total < best_cost:
                            best_cost = total
                            best_path = path_f + path_b[-2::-1]
                    
                    for neighbor, edge_cost in self.graph.get_weighted_neighbors(node_b):
                        if neighbor not in explored_b:
                            new_cost = cost_b + edge_cost
                            if new_cost < best_cost:
                                counter_b += 1
                                heapq.heappush(frontier_b,
                                    (new_cost, counter_b, neighbor, path_b + [neighbor]))
            
            # Early termination check
            if frontier_f and frontier_b:
                if frontier_f[0][0] + frontier_b[0][0] >= best_cost:
                    break
        
        if best_path:
            return {
                "path": best_path,
                "cost": best_cost,
                "nodes_expanded": nodes_expanded
            }
        return None

