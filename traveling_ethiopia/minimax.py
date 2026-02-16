"""
MiniMax Search Algorithm Implementation

Question 4: Assume an adversary joins the Traveling Ethiopia Search Problem.
The goal of the agent is to reach a state where it gains good quality of Coffee.
Write a class that shows how MiniMax search algorithm directs an agent to 
the best achievable destination.
"""

from typing import Dict, List, Optional, Tuple, Any
import math


class MiniMaxSearch:
    """
    MiniMax Search Algorithm for adversarial game playing.
    
    In the Traveling Ethiopia context:
    - MAX player (Agent): Wants to maximize coffee quality
    - MIN player (Adversary): Wants to minimize agent's coffee quality
    
    The algorithm explores the game tree and chooses moves assuming
    both players play optimally.
    
    Properties:
    - Complete: Yes (if tree is finite)
    - Optimal: Yes (against optimal opponent)
    - Time Complexity: O(b^m)
    - Space Complexity: O(bm) - linear with DFS
    """
    
    def __init__(self, game_tree: Dict[str, Any]):
        """
        Initialize MiniMax with a game tree.
        
        Args:
            game_tree: Dictionary representing the game tree structure.
                       Each node has:
                       - 'type': 'MAX', 'MIN', or None for terminal
                       - 'children': dict of child nodes or None
                       - 'utility': value for terminal nodes
        """
        self.game_tree = game_tree
        self.nodes_evaluated = 0
    
    def search(self, state: str = "root", depth: Optional[int] = None) -> Dict:
        """
        Find the best action using MiniMax search.
        
        Args:
            state: Starting state in the game tree
            depth: Optional depth limit (None for full search)
        
        Returns:
            Dictionary containing:
            - best_action: The recommended action (child state)
            - value: The minimax value
            - optimal_path: Path of optimal play
            - nodes_evaluated: Number of nodes evaluated
        """
        self.nodes_evaluated = 0
        
        node = self._get_node(state)
        if node is None:
            return {"error": f"State '{state}' not found"}
        
        if "utility" in node:
            return {
                "best_action": None,
                "value": node["utility"],
                "optimal_path": [state],
                "nodes_evaluated": 1
            }
        
        node_type = node.get("type", "MAX")
        
        if node_type == "MAX":
            value, action, path = self._max_value(state, node, depth or float('inf'))
        else:
            value, action, path = self._min_value(state, node, depth or float('inf'))
        
        return {
            "best_action": action,
            "value": value,
            "optimal_path": [state] + path,
            "nodes_evaluated": self.nodes_evaluated
        }
    
    def _get_node(self, state: str) -> Optional[Dict]:
        """Get a node from the game tree by state name."""
        if state == "root":
            return self.game_tree.get("root")
        
        # Search recursively for the state
        return self._find_node(self.game_tree.get("root"), state)
    
    def _find_node(self, node: Dict, target: str) -> Optional[Dict]:
        """Recursively find a node in the tree."""
        if node is None:
            return None
        
        children = node.get("children", {})
        if target in children:
            return children[target]
        
        for child_name, child_node in children.items():
            if isinstance(child_node, dict):
                result = self._find_node(child_node, target)
                if result:
                    return result
        
        return None
    
    def _max_value(self, state: str, node: Dict, 
                   depth: float) -> Tuple[float, Optional[str], List[str]]:
        """
        Compute the maximum value for a MAX node.
        
        Args:
            state: Current state name
            node: Current node dictionary
            depth: Remaining depth
        
        Returns:
            Tuple of (value, best_action, path_to_terminal)
        """
        self.nodes_evaluated += 1
        
        # Terminal test
        if "utility" in node:
            return node["utility"], None, []
        
        # Depth limit reached
        if depth <= 0:
            return self._evaluate(node), None, []
        
        children = node.get("children", {})
        if not children:
            return self._evaluate(node), None, []
        
        max_value = -math.inf
        best_action = None
        best_path = []
        
        for action, child_node in children.items():
            if "utility" in child_node:
                value = child_node["utility"]
                path = []
                self.nodes_evaluated += 1
            elif child_node.get("type") == "MIN":
                value, _, path = self._min_value(action, child_node, depth - 1)
            else:
                value, _, path = self._max_value(action, child_node, depth - 1)
            
            if value > max_value:
                max_value = value
                best_action = action
                best_path = [action] + path
        
        return max_value, best_action, best_path
    
    def _min_value(self, state: str, node: Dict,
                   depth: float) -> Tuple[float, Optional[str], List[str]]:
        """
        Compute the minimum value for a MIN node.
        
        Args:
            state: Current state name
            node: Current node dictionary
            depth: Remaining depth
        
        Returns:
            Tuple of (value, best_action, path_to_terminal)
        """
        self.nodes_evaluated += 1
        
        # Terminal test
        if "utility" in node:
            return node["utility"], None, []
        
        # Depth limit reached
        if depth <= 0:
            return self._evaluate(node), None, []
        
        children = node.get("children", {})
        if not children:
            return self._evaluate(node), None, []
        
        min_value = math.inf
        best_action = None
        best_path = []
        
        for action, child_node in children.items():
            if "utility" in child_node:
                value = child_node["utility"]
                path = []
                self.nodes_evaluated += 1
            elif child_node.get("type") == "MAX":
                value, _, path = self._max_value(action, child_node, depth - 1)
            else:
                value, _, path = self._min_value(action, child_node, depth - 1)
            
            if value < min_value:
                min_value = value
                best_action = action
                best_path = [action] + path
        
        return min_value, best_action, best_path
    
    def _evaluate(self, node: Dict) -> float:
        """
        Evaluation function for non-terminal nodes at depth limit.
        
        Used when we can't search to terminal states.
        """
        # Simple heuristic: average of child utilities if available
        if "utility" in node:
            return node["utility"]
        
        children = node.get("children", {})
        if not children:
            return 0.0
        
        values = []
        for child in children.values():
            if "utility" in child:
                values.append(child["utility"])
        
        return sum(values) / len(values) if values else 0.0


class AlphaBetaSearch(MiniMaxSearch):
    """
    Alpha-Beta Pruning optimization for MiniMax.
    
    Prunes branches that cannot affect the final decision,
    significantly reducing the number of nodes evaluated.
    
    Properties:
    - Same result as MiniMax
    - Time Complexity: O(b^(m/2)) in best case
    - Optimal move ordering can double effective search depth
    """
    
    def search(self, state: str = "root", depth: Optional[int] = None) -> Dict:
        """
        Find the best action using Alpha-Beta pruning.
        
        Args:
            state: Starting state
            depth: Depth limit
        
        Returns:
            Same as MiniMax but typically with fewer nodes evaluated
        """
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        
        node = self._get_node(state)
        if node is None:
            return {"error": f"State '{state}' not found"}
        
        if "utility" in node:
            return {
                "best_action": None,
                "value": node["utility"],
                "optimal_path": [state],
                "nodes_evaluated": 1,
                "pruned_branches": 0
            }
        
        node_type = node.get("type", "MAX")
        alpha = -math.inf
        beta = math.inf
        
        if node_type == "MAX":
            value, action, path = self._max_value_ab(
                state, node, alpha, beta, depth or float('inf'))
        else:
            value, action, path = self._min_value_ab(
                state, node, alpha, beta, depth or float('inf'))
        
        return {
            "best_action": action,
            "value": value,
            "optimal_path": [state] + path,
            "nodes_evaluated": self.nodes_evaluated,
            "pruned_branches": self.pruned_branches
        }
    
    def _max_value_ab(self, state: str, node: Dict, alpha: float, 
                      beta: float, depth: float) -> Tuple[float, Optional[str], List[str]]:
        """MAX node with alpha-beta pruning."""
        self.nodes_evaluated += 1
        
        if "utility" in node:
            return node["utility"], None, []
        
        if depth <= 0:
            return self._evaluate(node), None, []
        
        children = node.get("children", {})
        if not children:
            return self._evaluate(node), None, []
        
        max_value = -math.inf
        best_action = None
        best_path = []
        
        for action, child_node in children.items():
            if "utility" in child_node:
                value = child_node["utility"]
                path = []
                self.nodes_evaluated += 1
            elif child_node.get("type") == "MIN":
                value, _, path = self._min_value_ab(
                    action, child_node, alpha, beta, depth - 1)
            else:
                value, _, path = self._max_value_ab(
                    action, child_node, alpha, beta, depth - 1)
            
            if value > max_value:
                max_value = value
                best_action = action
                best_path = [action] + path
            
            # Alpha-beta pruning
            if max_value >= beta:
                self.pruned_branches += 1
                return max_value, best_action, best_path
            
            alpha = max(alpha, max_value)
        
        return max_value, best_action, best_path
    
    def _min_value_ab(self, state: str, node: Dict, alpha: float,
                      beta: float, depth: float) -> Tuple[float, Optional[str], List[str]]:
        """MIN node with alpha-beta pruning."""
        self.nodes_evaluated += 1
        
        if "utility" in node:
            return node["utility"], None, []
        
        if depth <= 0:
            return self._evaluate(node), None, []
        
        children = node.get("children", {})
        if not children:
            return self._evaluate(node), None, []
        
        min_value = math.inf
        best_action = None
        best_path = []
        
        for action, child_node in children.items():
            if "utility" in child_node:
                value = child_node["utility"]
                path = []
                self.nodes_evaluated += 1
            elif child_node.get("type") == "MAX":
                value, _, path = self._max_value_ab(
                    action, child_node, alpha, beta, depth - 1)
            else:
                value, _, path = self._min_value_ab(
                    action, child_node, alpha, beta, depth - 1)
            
            if value < min_value:
                min_value = value
                best_action = action
                best_path = [action] + path
            
            # Alpha-beta pruning
            if min_value <= alpha:
                self.pruned_branches += 1
                return min_value, best_action, best_path
            
            beta = min(beta, min_value)
        
        return min_value, best_action, best_path


class ExpectimaxSearch(MiniMaxSearch):
    """
    Expectimax Search for stochastic environments.
    
    Useful when the opponent doesn't play optimally or
    when there's randomness in the environment.
    
    Instead of MIN nodes, we have CHANCE nodes that compute
    expected values.
    """
    
    def _min_value(self, state: str, node: Dict,
                   depth: float) -> Tuple[float, Optional[str], List[str]]:
        """
        Override MIN to compute expected value (CHANCE node).
        
        Assumes uniform probability over children.
        """
        self.nodes_evaluated += 1
        
        if "utility" in node:
            return node["utility"], None, []
        
        if depth <= 0:
            return self._evaluate(node), None, []
        
        children = node.get("children", {})
        if not children:
            return self._evaluate(node), None, []
        
        total_value = 0.0
        count = 0
        all_paths = []
        
        for action, child_node in children.items():
            if "utility" in child_node:
                value = child_node["utility"]
                path = [action]
                self.nodes_evaluated += 1
            elif child_node.get("type") == "MAX":
                value, _, path = self._max_value(action, child_node, depth - 1)
                path = [action] + path
            else:
                value, _, path = self._min_value(action, child_node, depth - 1)
                path = [action] + path
            
            total_value += value
            count += 1
            all_paths.append((value, path))
        
        expected_value = total_value / count if count > 0 else 0.0
        
        # Return path with value closest to expected
        best_path = min(all_paths, key=lambda x: abs(x[0] - expected_value))[1]
        
        return expected_value, best_path[0] if best_path else None, best_path[1:] if len(best_path) > 1 else []


def demonstrate_coffee_search():
    """
    Demonstrate MiniMax search for the Coffee Quality problem.
    
    This function creates a game tree and shows how the agent
    should navigate to find the best coffee while an adversary
    tries to prevent it.
    """
    from .graph import create_adversary_graph_figure4
    
    # Create the adversary game tree
    game_tree = create_adversary_graph_figure4()
    
    print("=" * 60)
    print("TRAVELING ETHIOPIA - ADVERSARY COFFEE SEARCH")
    print("=" * 60)
    print("\nGame Tree Structure:")
    print("- Agent (MAX): Wants to reach highest quality coffee")
    print("- Adversary (MIN): Wants to minimize agent's coffee quality")
    print()
    
    # Run MiniMax search
    minimax = MiniMaxSearch(game_tree)
    result = minimax.search()
    
    print("MiniMax Search Result:")
    print(f"  Best action: {result['best_action']}")
    print(f"  Expected coffee quality: {result['value']}")
    print(f"  Optimal path: {' -> '.join(result['optimal_path'])}")
    print(f"  Nodes evaluated: {result['nodes_evaluated']}")
    print()
    
    # Run Alpha-Beta search
    alphabeta = AlphaBetaSearch(game_tree)
    result_ab = alphabeta.search()
    
    print("Alpha-Beta Search Result:")
    print(f"  Best action: {result_ab['best_action']}")
    print(f"  Expected coffee quality: {result_ab['value']}")
    print(f"  Optimal path: {' -> '.join(result_ab['optimal_path'])}")
    print(f"  Nodes evaluated: {result_ab['nodes_evaluated']}")
    print(f"  Branches pruned: {result_ab['pruned_branches']}")
    print()
    
    # Interpretation
    print("Interpretation:")
    print("-" * 40)
    if result['best_action'] == 'Sidamo':
        print("The agent should head to Sidamo region!")
        print("Even with adversarial conditions, Yirgacheffe offers")
        print("world-class coffee that makes this the optimal choice.")
    elif result['best_action'] == 'Jimma':
        print("The agent should head to Jimma!")
        print("As Ethiopia's coffee heartland, Jimma offers excellent quality.")
    elif result['best_action'] == 'Harar':
        print("The agent should head to Harar!")
        print("Famous for Harar Longberry coffee variety.")
    
    return result

