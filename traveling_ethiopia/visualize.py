"""
Visualization utilities for the Traveling Ethiopia Search Problem.

Provides functions to visualize:
- Graph structures
- Search paths
- Algorithm comparisons
"""

from typing import List, Dict, Optional, Set
import os


def print_path(path: List[str], title: str = "Path") -> None:
    """
    Print a path in a formatted way.
    
    Args:
        path: List of nodes in the path
        title: Title to display
    """
    print(f"\n{title}:")
    print("-" * 40)
    if path:
        print(" -> ".join(path))
        print(f"Total steps: {len(path) - 1}")
    else:
        print("No path found!")


def print_weighted_path(result: Dict, title: str = "Weighted Path") -> None:
    """
    Print a weighted path with cost information.
    
    Args:
        result: Dictionary with path and cost information
        title: Title to display
    """
    print(f"\n{title}:")
    print("-" * 40)
    if result:
        path = result.get("path", [])
        cost = result.get("cost", 0)
        nodes_expanded = result.get("nodes_expanded", 0)
        
        print(f"Path: {' -> '.join(path)}")
        print(f"Total cost: {cost}")
        print(f"Nodes expanded: {nodes_expanded}")
        print(f"Path length: {len(path)} nodes")
    else:
        print("No path found!")


def print_graph(adjacency_list: Dict[str, List], title: str = "Graph") -> None:
    """
    Print a graph's adjacency list in a readable format.
    
    Args:
        adjacency_list: Dictionary of node -> neighbors
        title: Title to display
    """
    print(f"\n{title}:")
    print("-" * 40)
    for node in sorted(adjacency_list.keys()):
        neighbors = ", ".join(sorted(str(n) for n in adjacency_list[node]))
        print(f"  {node} -> [{neighbors}]")


def print_comparison(results: Dict[str, Dict], title: str = "Algorithm Comparison") -> None:
    """
    Print a comparison of different search algorithm results.
    
    Args:
        results: Dictionary of algorithm_name -> result_dict
        title: Title to display
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print("=" * 60)
    
    # Header
    print(f"{'Algorithm':<20} {'Cost':<10} {'Nodes':<10} {'Path Length':<12}")
    print("-" * 52)
    
    for algo_name, result in results.items():
        if result:
            cost = result.get("cost", "N/A")
            nodes = result.get("nodes_expanded", "N/A")
            path_len = len(result.get("path", [])) if result.get("path") else 0
            print(f"{algo_name:<20} {cost:<10} {nodes:<10} {path_len:<12}")
        else:
            print(f"{algo_name:<20} {'Failed':<10} {'-':<10} {'-':<12}")
    
    print("-" * 52)


def visualize_search_tree(graph, start: str, goal: str, 
                          explored: Set[str], path: List[str],
                          max_depth: int = 3) -> None:
    """
    Print a text-based visualization of the search tree.
    
    Args:
        graph: Graph object
        start: Starting node
        goal: Goal node
        explored: Set of explored nodes
        path: Solution path
        max_depth: Maximum depth to display
    """
    print("\nSearch Tree Visualization:")
    print("-" * 40)
    
    path_set = set(path) if path else set()
    
    def print_node(node: str, depth: int, prefix: str = "") -> None:
        if depth > max_depth:
            if graph.get_neighbors(node):
                print(f"{prefix}...")
            return
        
        # Determine node status
        if node == goal:
            marker = "[GOAL] "
        elif node == start:
            marker = "[START] "
        elif node in path_set:
            marker = "[PATH] "
        elif node in explored:
            marker = "[X] "
        else:
            marker = "[ ] "
        
        print(f"{prefix}{marker}{node}")
        
        neighbors = graph.get_neighbors(node)
        for i, neighbor in enumerate(neighbors):
            is_last = (i == len(neighbors) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            connector = "└── " if is_last else "├── "
            
            if neighbor not in explored or neighbor in path_set or neighbor == goal:
                print_node(neighbor, depth + 1, prefix + connector.replace("── ", ""))
    
    print_node(start, 0)


def create_ascii_map() -> str:
    """
    Create an ASCII art map of Ethiopia with major cities.
    
    Returns:
        ASCII art string
    """
    map_art = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                     TRAVELING ETHIOPIA MAP                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║                    ● Axum ─── ● Shire                            ║
    ║                    │                                              ║
    ║              ● Metema ── ● Gondar                                ║
    ║                          │                                        ║
    ║                    ● Bahir Dar ── ● Debre Markos                 ║
    ║                          │              │                         ║
    ║                    ● Injibara     ● Addis Ababa ●── Debre Berhan ║
    ║                                   │    │     │            │       ║
    ║    ● Assosa ── ● Gimbi ── ● Nekemte ● Ambo    ● Adama    ● Dessie║
    ║                                          │        │           │   ║
    ║                                    ● Jimma    ● Dire Dawa ● Woldia║
    ║                                    │    │         │           │   ║
    ║                              ● Bonga  ● Wolaita  ● Harar  ● Lalibela
    ║                              │         Sodo        │              ║
    ║                        ● Mizan Teferi    │      ● Babile          ║
    ║                                    ● Arba Minch                   ║
    ║                                          │                        ║
    ║                                    ● Konso                        ║
    ║                                          │                        ║
    ║                                    ● Moyale                       ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    return map_art


def print_multi_goal_result(result: Dict, title: str = "Multi-Goal Search") -> None:
    """
    Print the result of a multi-goal search.
    
    Args:
        result: Dictionary with multi-goal search results
        title: Title to display
    """
    print(f"\n{title}:")
    print("=" * 60)
    
    if not result:
        print("No solution found!")
        return
    
    print(f"Complete path: {' -> '.join(result['path'])}")
    print(f"Total cost: {result['cost']}")
    print()
    
    goals_order = result.get('goals_order', [])
    segment_costs = result.get('segment_costs', [])
    
    print("Goals visited in order:")
    print("-" * 40)
    
    cumulative_cost = 0
    for i, (goal, cost) in enumerate(zip(goals_order, segment_costs), 1):
        cumulative_cost += cost
        print(f"  {i}. {goal} (segment cost: {cost}, cumulative: {cumulative_cost})")


def generate_report(all_results: Dict) -> str:
    """
    Generate a comprehensive report of all search results.
    
    Args:
        all_results: Dictionary containing results from all questions
    
    Returns:
        Report string
    """
    report = []
    report.append("=" * 70)
    report.append("TRAVELING ETHIOPIA SEARCH PROBLEM - COMPLETE REPORT")
    report.append("AI Principles and Techniques Course Project")
    report.append("=" * 70)
    report.append("")
    
    # Question 1: BFS/DFS
    if "q1" in all_results:
        report.append("QUESTION 1: BFS and DFS Search")
        report.append("-" * 50)
        for strategy, result in all_results["q1"].items():
            if result:
                report.append(f"  {strategy.upper()}:")
                report.append(f"    Path: {' -> '.join(result['path'])}")
                report.append(f"    Path length: {len(result['path']) - 1} steps")
            else:
                report.append(f"  {strategy.upper()}: No path found")
        report.append("")
    
    # Question 2: UCS
    if "q2" in all_results:
        report.append("QUESTION 2: Uniform Cost Search")
        report.append("-" * 50)
        
        if "single_goal" in all_results["q2"]:
            result = all_results["q2"]["single_goal"]
            if result:
                report.append(f"  Path to Lalibela:")
                report.append(f"    Route: {' -> '.join(result['path'])}")
                report.append(f"    Total distance: {result['cost']} km")
        
        if "multi_goal" in all_results["q2"]:
            result = all_results["q2"]["multi_goal"]
            if result:
                report.append(f"  Multi-Goal Tour:")
                report.append(f"    Order: {' -> '.join(result['goals_order'])}")
                report.append(f"    Total distance: {result['cost']} km")
        report.append("")
    
    # Question 3: A* Search
    if "q3" in all_results:
        report.append("QUESTION 3: A* Search")
        report.append("-" * 50)
        result = all_results["q3"]
        if result:
            report.append(f"  Path to Moyale:")
            report.append(f"    Route: {' -> '.join(result['path'])}")
            report.append(f"    Total distance: {result['cost']} km")
            report.append(f"    Nodes expanded: {result['nodes_expanded']}")
        report.append("")
    
    # Question 4: MiniMax
    if "q4" in all_results:
        report.append("QUESTION 4: MiniMax Search (Adversarial)")
        report.append("-" * 50)
        result = all_results["q4"]
        if result:
            report.append(f"  Best coffee destination: {result['best_action']}")
            report.append(f"  Expected quality score: {result['value']}")
            report.append(f"  Optimal path: {' -> '.join(result['optimal_path'])}")
        report.append("")
    
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)

