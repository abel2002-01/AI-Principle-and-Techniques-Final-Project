#!/usr/bin/env python3
"""
Traveling Ethiopia Search Problem - Main Demo

This file demonstrates all search algorithms implemented for the
AI Principles and Techniques Course Project.

Questions covered:
1. BFS and DFS Search
2. Uniform Cost Search (single and multi-goal)
3. A* Search
4. MiniMax Search for adversarial scenarios
"""

from .graph import (
    StateSpaceGraph,
    WeightedGraph,
    create_ethiopia_graph_figure1,
    create_ethiopia_graph_figure2,
    create_ethiopia_graph_figure3,
    create_adversary_graph_figure4
)
from .search_strategies import SearchSolver, IterativeDeepeningDFS
from .ucs import UniformCostSearch, BidirectionalUCS
from .astar import AStarSearch, WeightedAStar, IDAStarSearch
from .minimax import MiniMaxSearch, AlphaBetaSearch, ExpectimaxSearch
from .visualize import (
    print_path,
    print_weighted_path,
    print_comparison,
    print_multi_goal_result,
    create_ascii_map,
    generate_report
)


def demo_question1():
    """
    Question 1: BFS and DFS Search
    
    1.1: Convert Figure 1 into a manageable data structure
    1.2: Implement search with BFS and DFS strategies
    """
    print("\n" + "=" * 70)
    print("QUESTION 1: BFS AND DFS SEARCH")
    print("=" * 70)
    
    # Create the state space graph (Q1.1)
    print("\n1.1 Creating state space graph from Figure 1...")
    graph = create_ethiopia_graph_figure1()
    print(graph)
    
    # Define search parameters
    initial_state = "Addis Ababa"
    goal_state = "Axum"
    
    print(f"\n1.2 Searching for path from '{initial_state}' to '{goal_state}'")
    
    # Create solver
    solver = SearchSolver(graph, initial_state, goal_state)
    
    # BFS Search
    print("\n--- Breadth-First Search (BFS) ---")
    bfs_path = solver.solve(strategy="bfs")
    print_path(bfs_path, "BFS Path")
    
    # Get visualization info
    bfs_info = solver.visualize_search(strategy="bfs")
    print(f"Nodes expanded: {bfs_info['nodes_expanded']}")
    print(f"Exploration order (first 10): {bfs_info['exploration_order'][:10]}...")
    
    # DFS Search
    print("\n--- Depth-First Search (DFS) ---")
    dfs_path = solver.solve(strategy="dfs")
    print_path(dfs_path, "DFS Path")
    
    dfs_info = solver.visualize_search(strategy="dfs")
    print(f"Nodes expanded: {dfs_info['nodes_expanded']}")
    print(f"Exploration order (first 10): {dfs_info['exploration_order'][:10]}...")
    
    # Compare the two strategies
    print("\n--- Comparison ---")
    print(f"BFS path length: {len(bfs_path) if bfs_path else 'N/A'}")
    print(f"DFS path length: {len(dfs_path) if dfs_path else 'N/A'}")
    print(f"BFS finds shortest path: {len(bfs_path) <= len(dfs_path) if bfs_path and dfs_path else 'N/A'}")
    
    # Try different goal
    print("\n--- Additional Test: Addis Ababa to Moyale ---")
    solver2 = SearchSolver(graph, "Addis Ababa", "Moyale")
    bfs_path2 = solver2.solve(strategy="bfs")
    dfs_path2 = solver2.solve(strategy="dfs")
    print_path(bfs_path2, "BFS Path to Moyale")
    print_path(dfs_path2, "DFS Path to Moyale")
    
    return {
        "bfs": {"path": bfs_path, "nodes_expanded": bfs_info['nodes_expanded']},
        "dfs": {"path": dfs_path, "nodes_expanded": dfs_info['nodes_expanded']}
    }


def demo_question2():
    """
    Question 2: Uniform Cost Search
    
    2.1: Convert Figure 2 into a data structure with weights
    2.2: UCS from Addis Ababa to Lalibela
    2.3: Multi-goal UCS to visit all specified cities
    """
    print("\n" + "=" * 70)
    print("QUESTION 2: UNIFORM COST SEARCH")
    print("=" * 70)
    
    # Create weighted graph (Q2.1)
    print("\n2.1 Creating weighted state space graph from Figure 2...")
    graph = create_ethiopia_graph_figure2()
    print(graph)
    
    # Create UCS solver
    ucs = UniformCostSearch(graph)
    
    # Q2.2: Single goal search to Lalibela
    print("\n2.2 Finding optimal path from Addis Ababa to Lalibela...")
    result = ucs.search("Addis Ababa", "Lalibela")
    print_weighted_path(result, "UCS Path to Lalibela")
    
    # Q2.3: Multi-goal search
    print("\n2.3 Multi-goal search visiting all specified cities...")
    goal_states = [
        "Axum", "Gondar", "Lalibela", "Babile",
        "Jimma", "Bale", "Sof Oumer", "Arba Minch"
    ]
    
    print(f"Goal cities: {', '.join(goal_states)}")
    
    # Greedy approach (preserving local optimum)
    print("\n--- Greedy Multi-Goal Search (Local Optimum) ---")
    multi_result = ucs.multi_goal_search("Addis Ababa", goal_states)
    print_multi_goal_result(multi_result, "Greedy Multi-Goal Result")
    
    # Optimal approach (if feasible)
    print("\n--- Optimal Multi-Goal Search (TSP-like) ---")
    try:
        optimal_result = ucs.multi_goal_search_optimal("Addis Ababa", goal_states)
        print_multi_goal_result(optimal_result, "Optimal Multi-Goal Result")
        
        # Compare
        if multi_result and optimal_result:
            savings = multi_result['cost'] - optimal_result['cost']
            print(f"\nOptimal saves: {savings} km ({100*savings/multi_result['cost']:.1f}%)")
    except Exception as e:
        print(f"Could not compute optimal solution: {e}")
    
    return {
        "single_goal": result,
        "multi_goal": multi_result
    }


def demo_question3():
    """
    Question 3: A* Search
    
    Find optimal path from Addis Ababa to Moyale using A* with heuristics.
    """
    print("\n" + "=" * 70)
    print("QUESTION 3: A* SEARCH")
    print("=" * 70)
    
    # Create graph with heuristics
    print("\nCreating weighted graph with heuristics from Figure 3...")
    graph = create_ethiopia_graph_figure3()
    
    # Create A* solver
    astar = AStarSearch(graph)
    
    # Search from Addis Ababa to Moyale
    print("\nFinding optimal path from Addis Ababa to Moyale...")
    result = astar.search("Addis Ababa", "Moyale")
    print_weighted_path(result, "A* Path to Moyale")
    
    if result:
        print("\nStep-by-step f-values along the path:")
        for node, f_val in zip(result['path'], result['f_values']):
            h = graph.get_heuristic(node, "Moyale")
            g = f_val - h
            print(f"  {node}: g={g:.0f}, h={h:.0f}, f={f_val:.0f}")
    
    # Compare with other algorithms
    print("\n--- Comparison with other algorithms ---")
    
    # UCS (no heuristic)
    ucs = UniformCostSearch(graph)
    ucs_result = ucs.search("Addis Ababa", "Moyale")
    
    # Weighted A* (faster but suboptimal)
    wastar = WeightedAStar(graph, weight=1.5)
    wastar_result = wastar.search("Addis Ababa", "Moyale")
    
    # IDA* (memory efficient)
    idastar = IDAStarSearch(graph)
    idastar_result = idastar.search("Addis Ababa", "Moyale")
    
    comparison = {
        "UCS": ucs_result,
        "A*": result,
        "Weighted A* (w=1.5)": wastar_result,
        "IDA*": idastar_result
    }
    
    print_comparison(comparison, "Search Algorithm Comparison (Addis Ababa -> Moyale)")
    
    return result


def demo_question4():
    """
    Question 4: MiniMax Search
    
    Adversarial search for finding the best coffee destination
    when an adversary tries to prevent it.
    """
    print("\n" + "=" * 70)
    print("QUESTION 4: MINIMAX SEARCH (ADVERSARIAL)")
    print("=" * 70)
    
    # Create adversary game tree
    print("\nCreating adversary game tree from Figure 4...")
    game_tree = create_adversary_graph_figure4()
    
    print("\nGame Setup:")
    print("-" * 50)
    print("- Agent (MAX): Wants to maximize coffee quality")
    print("- Adversary (MIN): Tries to minimize agent's reward")
    print("- Terminal states have coffee quality scores (0-100)")
    print()
    
    # Print game tree structure
    print("Coffee Regions and Quality Scores:")
    root = game_tree['root']
    for region, region_data in root['children'].items():
        print(f"\n{region}:")
        for subregion, sub_data in region_data['children'].items():
            if 'children' in sub_data:
                for dest, dest_data in sub_data['children'].items():
                    if 'utility' in dest_data:
                        print(f"    {dest}: Quality = {dest_data['utility']}")
    
    # MiniMax Search
    print("\n--- MiniMax Search ---")
    minimax = MiniMaxSearch(game_tree)
    mm_result = minimax.search()
    
    print(f"Best action: {mm_result['best_action']}")
    print(f"Guaranteed coffee quality: {mm_result['value']}")
    print(f"Optimal play sequence: {' -> '.join(mm_result['optimal_path'])}")
    print(f"Nodes evaluated: {mm_result['nodes_evaluated']}")
    
    # Alpha-Beta Search
    print("\n--- Alpha-Beta Search ---")
    alphabeta = AlphaBetaSearch(game_tree)
    ab_result = alphabeta.search()
    
    print(f"Best action: {ab_result['best_action']}")
    print(f"Guaranteed coffee quality: {ab_result['value']}")
    print(f"Optimal play sequence: {' -> '.join(ab_result['optimal_path'])}")
    print(f"Nodes evaluated: {ab_result['nodes_evaluated']}")
    print(f"Branches pruned: {ab_result['pruned_branches']}")
    
    # Expectimax (for comparison)
    print("\n--- Expectimax Search (Probabilistic Opponent) ---")
    expectimax = ExpectimaxSearch(game_tree)
    exp_result = expectimax.search()
    
    print(f"Best action: {exp_result['best_action']}")
    print(f"Expected coffee quality: {exp_result['value']:.1f}")
    print(f"Recommended path: {' -> '.join(exp_result['optimal_path'])}")
    
    # Analysis
    print("\n--- Analysis ---")
    print(f"MiniMax assumes worst-case: adversary plays optimally")
    print(f"Expectimax assumes average-case: adversary plays randomly")
    print(f"")
    print(f"For a rational adversary, follow MiniMax: go to {mm_result['best_action']}")
    print(f"For an unpredictable adversary, Expectimax might be better")
    
    return mm_result


def main():
    """Run all demonstrations."""
    print(create_ascii_map())
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  TRAVELING ETHIOPIA SEARCH PROBLEM - AI PRINCIPLES PROJECT  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Collect all results
    all_results = {}
    
    # Question 1
    all_results["q1"] = demo_question1()
    
    # Question 2
    all_results["q2"] = demo_question2()
    
    # Question 3
    all_results["q3"] = demo_question3()
    
    # Question 4
    all_results["q4"] = demo_question4()
    
    # Generate and print report
    print("\n")
    report = generate_report(all_results)
    print(report)
    
    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()

