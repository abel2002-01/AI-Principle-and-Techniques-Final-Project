"""
Graph Data Structures for Traveling Ethiopia Search Problem

This module provides graph representations that can be used with
various search algorithms. It supports both unweighted graphs (for BFS/DFS)
and weighted graphs (for UCS/A*).

Question 1.1 & 2.1: Convert state space graphs into manageable data structures
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
import heapq


class StateSpaceGraph:
    """
    Unweighted graph representation for Question 1.
    
    Uses adjacency list representation which is efficient for
    sparse graphs like the Ethiopian cities network.
    
    The graph can be traversed using either a stack (DFS) or queue (BFS).
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.nodes: Set[str] = set()
    
    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(node)
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []
    
    def add_edge(self, from_node: str, to_node: str, bidirectional: bool = True) -> None:
        """
        Add an edge between two nodes.
        
        Args:
            from_node: Source node
            to_node: Destination node
            bidirectional: If True, adds edge in both directions (default)
        """
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        
        if to_node not in self.adjacency_list[from_node]:
            self.adjacency_list[from_node].append(to_node)
        
        if bidirectional and from_node not in self.adjacency_list[to_node]:
            self.adjacency_list[to_node].append(from_node)
    
    def get_neighbors(self, node: str) -> List[str]:
        """Get all neighbors of a node."""
        return self.adjacency_list.get(node, [])
    
    def get_nodes(self) -> Set[str]:
        """Get all nodes in the graph."""
        return self.nodes
    
    def to_stack(self) -> List[str]:
        """
        Convert graph nodes to a stack (LIFO) for DFS traversal.
        Returns list where last element is treated as top of stack.
        """
        return list(self.nodes)
    
    def to_queue(self) -> deque:
        """
        Convert graph nodes to a queue (FIFO) for BFS traversal.
        Returns deque for efficient popleft operations.
        """
        return deque(self.nodes)
    
    def __str__(self) -> str:
        """String representation of the graph."""
        result = "StateSpaceGraph:\n"
        for node in sorted(self.adjacency_list.keys()):
            neighbors = ", ".join(sorted(self.adjacency_list[node]))
            result += f"  {node} -> [{neighbors}]\n"
        return result
    
    def __repr__(self) -> str:
        return f"StateSpaceGraph(nodes={len(self.nodes)}, edges={sum(len(v) for v in self.adjacency_list.values())//2})"


class WeightedGraph(StateSpaceGraph):
    """
    Weighted graph representation for Questions 2 and 3.
    
    Extends StateSpaceGraph with edge weights (costs) and optional
    heuristic values for A* search.
    """
    
    def __init__(self):
        """Initialize an empty weighted graph."""
        super().__init__()
        self.weights: Dict[Tuple[str, str], float] = {}
        self.heuristics: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_edge(self, from_node: str, to_node: str, 
                 weight: float = 1.0, bidirectional: bool = True) -> None:
        """
        Add a weighted edge between two nodes.
        
        Args:
            from_node: Source node
            to_node: Destination node
            weight: Edge cost/weight
            bidirectional: If True, adds edge in both directions
        """
        super().add_edge(from_node, to_node, bidirectional)
        self.weights[(from_node, to_node)] = weight
        if bidirectional:
            self.weights[(to_node, from_node)] = weight
    
    def get_weight(self, from_node: str, to_node: str) -> float:
        """Get the weight/cost of an edge."""
        return self.weights.get((from_node, to_node), float('inf'))
    
    def set_heuristic(self, node: str, goal: str, value: float) -> None:
        """
        Set the heuristic value from a node to a specific goal.
        
        Args:
            node: Current node
            goal: Goal node
            value: Estimated cost to reach goal from node
        """
        self.heuristics[node][goal] = value
    
    def get_heuristic(self, node: str, goal: str) -> float:
        """Get the heuristic value from a node to a goal."""
        return self.heuristics.get(node, {}).get(goal, 0.0)
    
    def get_weighted_neighbors(self, node: str) -> List[Tuple[str, float]]:
        """Get neighbors with their edge weights."""
        neighbors = self.get_neighbors(node)
        return [(n, self.get_weight(node, n)) for n in neighbors]
    
    def to_priority_queue(self, start_node: str) -> List[Tuple[float, str]]:
        """
        Convert to priority queue format for UCS/A*.
        Returns list of (cost, node) tuples suitable for heapq.
        """
        pq = [(0, start_node)]
        heapq.heapify(pq)
        return pq
    
    def __str__(self) -> str:
        """String representation of the weighted graph."""
        result = "WeightedGraph:\n"
        for node in sorted(self.adjacency_list.keys()):
            neighbors_with_weights = []
            for neighbor in sorted(self.adjacency_list[node]):
                weight = self.get_weight(node, neighbor)
                neighbors_with_weights.append(f"{neighbor}({weight})")
            result += f"  {node} -> [{', '.join(neighbors_with_weights)}]\n"
        return result


# ============================================================
# Pre-defined graphs based on the Traveling Ethiopia problem
# ============================================================

def create_ethiopia_graph_figure1() -> StateSpaceGraph:
    """
    Create the state space graph from Figure 1.
    
    This represents the generic unweighted graph for BFS/DFS.
    Based on typical Ethiopian cities network.
    """
    graph = StateSpaceGraph()
    
    # Major Ethiopian cities and their connections
    # Central Ethiopia
    graph.add_edge("Addis Ababa", "Ambo")
    graph.add_edge("Addis Ababa", "Adama")
    graph.add_edge("Addis Ababa", "Debre Berhan")
    graph.add_edge("Addis Ababa", "Debre Markos")
    
    # Western route
    graph.add_edge("Ambo", "Nekemte")
    graph.add_edge("Nekemte", "Gimbi")
    graph.add_edge("Gimbi", "Assosa")
    graph.add_edge("Ambo", "Jimma")
    
    # Southwestern route
    graph.add_edge("Jimma", "Bonga")
    graph.add_edge("Bonga", "Mizan Teferi")
    graph.add_edge("Bonga", "Dawro")
    graph.add_edge("Dawro", "Wolaita Sodo")
    graph.add_edge("Wolaita Sodo", "Arba Minch")
    graph.add_edge("Jimma", "Wolaita Sodo")
    
    # Eastern route
    graph.add_edge("Adama", "Dire Dawa")
    graph.add_edge("Dire Dawa", "Harar")
    graph.add_edge("Harar", "Babile")
    graph.add_edge("Dire Dawa", "Chiro")
    graph.add_edge("Adama", "Matahara")
    graph.add_edge("Matahara", "Awash")
    
    # Northern route
    graph.add_edge("Debre Berhan", "Dessie")
    graph.add_edge("Dessie", "Woldia")
    graph.add_edge("Woldia", "Lalibela")
    graph.add_edge("Woldia", "Mekelle")
    graph.add_edge("Mekelle", "Axum")
    graph.add_edge("Axum", "Shire")
    
    # Northwestern route
    graph.add_edge("Debre Markos", "Bahir Dar")
    graph.add_edge("Bahir Dar", "Gondar")
    graph.add_edge("Gondar", "Axum")
    graph.add_edge("Gondar", "Metema")
    graph.add_edge("Bahir Dar", "Injibara")
    
    # Southern route
    graph.add_edge("Adama", "Assela")
    graph.add_edge("Assela", "Bale")
    graph.add_edge("Bale", "Goba")
    graph.add_edge("Goba", "Sof Oumer")
    graph.add_edge("Bale", "Dolo Odo")
    graph.add_edge("Arba Minch", "Konso")
    graph.add_edge("Konso", "Moyale")
    
    # Additional connections for completeness
    graph.add_edge("Assela", "Dodola")
    graph.add_edge("Dodola", "Bale")
    graph.add_edge("Awash", "Mille")
    graph.add_edge("Mille", "Semera")
    
    return graph


def create_ethiopia_graph_figure2() -> WeightedGraph:
    """
    Create the weighted state space graph from Figure 2.
    
    This represents the graph with backward costs for UCS.
    Weights approximate actual distances in km.
    """
    graph = WeightedGraph()
    
    # Central Ethiopia
    graph.add_edge("Addis Ababa", "Ambo", weight=114)
    graph.add_edge("Addis Ababa", "Adama", weight=99)
    graph.add_edge("Addis Ababa", "Debre Berhan", weight=130)
    graph.add_edge("Addis Ababa", "Debre Markos", weight=299)
    
    # Western route
    graph.add_edge("Ambo", "Nekemte", weight=211)
    graph.add_edge("Nekemte", "Gimbi", weight=145)
    graph.add_edge("Gimbi", "Assosa", weight=196)
    graph.add_edge("Ambo", "Jimma", weight=243)
    
    # Southwestern route
    graph.add_edge("Jimma", "Bonga", weight=117)
    graph.add_edge("Bonga", "Mizan Teferi", weight=147)
    graph.add_edge("Bonga", "Dawro", weight=226)
    graph.add_edge("Dawro", "Wolaita Sodo", weight=118)
    graph.add_edge("Wolaita Sodo", "Arba Minch", weight=166)
    graph.add_edge("Jimma", "Wolaita Sodo", weight=313)
    
    # Eastern route
    graph.add_edge("Adama", "Dire Dawa", weight=311)
    graph.add_edge("Dire Dawa", "Harar", weight=52)
    graph.add_edge("Harar", "Babile", weight=34)
    graph.add_edge("Dire Dawa", "Chiro", weight=105)
    graph.add_edge("Adama", "Matahara", weight=133)
    graph.add_edge("Matahara", "Awash", weight=25)
    
    # Northern route
    graph.add_edge("Debre Berhan", "Dessie", weight=189)
    graph.add_edge("Dessie", "Woldia", weight=120)
    graph.add_edge("Woldia", "Lalibela", weight=126)
    graph.add_edge("Woldia", "Mekelle", weight=261)
    graph.add_edge("Mekelle", "Axum", weight=247)
    graph.add_edge("Axum", "Shire", weight=63)
    
    # Northwestern route
    graph.add_edge("Debre Markos", "Bahir Dar", weight=265)
    graph.add_edge("Bahir Dar", "Gondar", weight=176)
    graph.add_edge("Gondar", "Axum", weight=306)
    graph.add_edge("Gondar", "Metema", weight=168)
    graph.add_edge("Bahir Dar", "Injibara", weight=99)
    
    # Southern route
    graph.add_edge("Adama", "Assela", weight=77)
    graph.add_edge("Assela", "Bale", weight=188)
    graph.add_edge("Bale", "Goba", weight=18)
    graph.add_edge("Goba", "Sof Oumer", weight=23)
    graph.add_edge("Bale", "Dolo Odo", weight=446)
    graph.add_edge("Arba Minch", "Konso", weight=90)
    graph.add_edge("Konso", "Moyale", weight=233)
    
    # Additional connections
    graph.add_edge("Assela", "Dodola", weight=110)
    graph.add_edge("Dodola", "Bale", weight=78)
    graph.add_edge("Awash", "Mille", weight=203)
    graph.add_edge("Mille", "Semera", weight=71)
    
    return graph


def create_ethiopia_graph_figure3() -> WeightedGraph:
    """
    Create the weighted state space graph from Figure 3 with heuristics.
    
    This includes heuristic values for A* search to Moyale.
    Heuristics are straight-line distance estimates.
    """
    graph = create_ethiopia_graph_figure2()
    
    # Heuristic values to Moyale (straight-line distance estimates in km)
    heuristics_to_moyale = {
        "Addis Ababa": 550,
        "Ambo": 600,
        "Adama": 500,
        "Debre Berhan": 620,
        "Debre Markos": 700,
        "Nekemte": 650,
        "Gimbi": 700,
        "Assosa": 750,
        "Jimma": 450,
        "Bonga": 400,
        "Mizan Teferi": 380,
        "Dawro": 350,
        "Wolaita Sodo": 300,
        "Arba Minch": 200,
        "Dire Dawa": 480,
        "Harar": 500,
        "Babile": 510,
        "Chiro": 450,
        "Matahara": 470,
        "Awash": 490,
        "Dessie": 650,
        "Woldia": 680,
        "Lalibela": 700,
        "Mekelle": 800,
        "Axum": 850,
        "Shire": 870,
        "Bahir Dar": 750,
        "Gondar": 780,
        "Metema": 800,
        "Injibara": 720,
        "Assela": 420,
        "Bale": 350,
        "Goba": 340,
        "Sof Oumer": 330,
        "Dolo Odo": 250,
        "Konso": 120,
        "Moyale": 0,
        "Dodola": 380,
        "Mille": 520,
        "Semera": 540
    }
    
    for node, h in heuristics_to_moyale.items():
        graph.set_heuristic(node, "Moyale", h)
    
    return graph


def create_adversary_graph_figure4() -> Dict[str, Any]:
    """
    Create the adversary game tree from Figure 4.
    
    This represents a MiniMax game tree where the agent seeks
    the best coffee quality destination while an adversary tries
    to prevent it.
    
    Returns a tree structure with utility values (coffee quality scores).
    """
    # Game tree structure for adversarial search
    # The values represent coffee quality scores at terminal states
    game_tree = {
        "root": {
            "type": "MAX",  # Agent's turn (maximizing)
            "children": {
                "Jimma": {
                    "type": "MIN",  # Adversary's turn (minimizing)
                    "children": {
                        "Jimma-Direct": {
                            "type": "MAX",
                            "children": {
                                "Jimma-Premium": {"utility": 95},  # Jimma is famous for coffee
                                "Jimma-Standard": {"utility": 85}
                            }
                        },
                        "Bonga": {
                            "type": "MAX", 
                            "children": {
                                "Bonga-Forest": {"utility": 90},
                                "Bonga-Regular": {"utility": 75}
                            }
                        }
                    }
                },
                "Harar": {
                    "type": "MIN",
                    "children": {
                        "Harar-Direct": {
                            "type": "MAX",
                            "children": {
                                "Harar-Premium": {"utility": 92},  # Harar longberry
                                "Harar-Standard": {"utility": 80}
                            }
                        },
                        "Dire Dawa": {
                            "type": "MAX",
                            "children": {
                                "Dire Dawa-Market": {"utility": 70},
                                "Dire Dawa-Local": {"utility": 65}
                            }
                        }
                    }
                },
                "Sidamo": {
                    "type": "MIN",
                    "children": {
                        "Yirgacheffe": {
                            "type": "MAX",
                            "children": {
                                "Yirgacheffe-Premium": {"utility": 98},  # World-famous
                                "Yirgacheffe-Standard": {"utility": 88}
                            }
                        },
                        "Sidamo-Local": {
                            "type": "MAX",
                            "children": {
                                "Sidamo-Washed": {"utility": 85},
                                "Sidamo-Natural": {"utility": 82}
                            }
                        }
                    }
                }
            }
        }
    }
    
    return game_tree

