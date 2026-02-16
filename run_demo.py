#!/usr/bin/env python3
"""
Traveling Ethiopia Search Problem - Demo Runner

Run this script to see all search algorithms in action.

Usage:
    python run_demo.py [--question N]

Examples:
    python run_demo.py           # Run all questions
    python run_demo.py --question 1   # Run only Question 1
    python run_demo.py --question 3   # Run only Question 3
"""

import sys
import argparse

# Add package to path
sys.path.insert(0, '.')

from traveling_ethiopia.graph import (
    StateSpaceGraph,
    WeightedGraph,
    create_ethiopia_graph_figure1,
    create_ethiopia_graph_figure2,
    create_ethiopia_graph_figure3,
    create_adversary_graph_figure4
)
from traveling_ethiopia.search_strategies import SearchSolver, IterativeDeepeningDFS
from traveling_ethiopia.ucs import UniformCostSearch, BidirectionalUCS
from traveling_ethiopia.astar import AStarSearch, WeightedAStar, IDAStarSearch
from traveling_ethiopia.minimax import MiniMaxSearch, AlphaBetaSearch, ExpectimaxSearch
from traveling_ethiopia.visualize import (
    print_path,
    print_weighted_path,
    print_comparison,
    print_multi_goal_result,
    create_ascii_map,
    generate_report
)


def run_question1():
    """Question 1: BFS and DFS Search"""
    print("\n" + "=" * 70)
    print("QUESTION 1: BFS AND DFS SEARCH")
    print("=" * 70)
    
    # Create the state space graph
    print("\n1.1 Creating state space graph from Figure 1...")
    graph = create_ethiopia_graph_figure1()
    
    print("\nGraph Structure (Adjacency List):")
    print(graph)
    
    # Define search parameters
    initial_state = "Addis Ababa"
    goal_state = "Axum"
    
    print(f"\n1.2 Searching for path from '{initial_state}' to '{goal_state}'")
    
    solver = SearchSolver(graph, initial_state, goal_state)
    
    # BFS Search
    print("\n" + "-" * 50)
    print("BREADTH-FIRST SEARCH (BFS)")
    print("-" * 50)
    bfs_path = solver.solve(strategy="bfs")
    
    if bfs_path:
        print(f"Path found: {' -> '.join(bfs_path)}")
        print(f"Path length: {len(bfs_path) - 1} edges")
    else:
        print("No path found!")
    
    bfs_info = solver.visualize_search(strategy="bfs")
    print(f"Nodes expanded: {bfs_info['nodes_expanded']}")
    
    # DFS Search
    print("\n" + "-" * 50)
    print("DEPTH-FIRST SEARCH (DFS)")
    print("-" * 50)
    dfs_path = solver.solve(strategy="dfs")
    
    if dfs_path:
        print(f"Path found: {' -> '.join(dfs_path)}")
        print(f"Path length: {len(dfs_path) - 1} edges")
    else:
        print("No path found!")
    
    dfs_info = solver.visualize_search(strategy="dfs")
    print(f"Nodes expanded: {dfs_info['nodes_expanded']}")
    
    # Comparison
    print("\n" + "-" * 50)
    print("COMPARISON")
    print("-" * 50)
    print(f"BFS path length: {len(bfs_path) - 1 if bfs_path else 'N/A'} edges")
    print(f"DFS path length: {len(dfs_path) - 1 if dfs_path else 'N/A'} edges")
    print(f"BFS is optimal for unweighted graphs: finds shortest path")
    print(f"DFS uses less memory but may not find shortest path")
    
    return {"bfs": bfs_path, "dfs": dfs_path}


def run_question2():
    """Question 2: Uniform Cost Search"""
    print("\n" + "=" * 70)
    print("QUESTION 2: UNIFORM COST SEARCH")
    print("=" * 70)
    
    # Create weighted graph
    print("\n2.1 Creating weighted graph from Figure 2...")
    graph = create_ethiopia_graph_figure2()
    
    print("\nWeighted Graph Structure:")
    print(graph)
    
    ucs = UniformCostSearch(graph)
    
    # Q2.2: Single goal search
    print("\n" + "-" * 50)
    print("2.2 UCS: ADDIS ABABA TO LALIBELA")
    print("-" * 50)
    
    result = ucs.search("Addis Ababa", "Lalibela")
    
    if result:
        print(f"Optimal path: {' -> '.join(result['path'])}")
        print(f"Total cost: {result['cost']} km")
        print(f"Nodes expanded: {result['nodes_expanded']}")
    else:
        print("No path found!")
    
    # Q2.3: Multi-goal search
    print("\n" + "-" * 50)
    print("2.3 MULTI-GOAL UCS: VISIT ALL SPECIFIED CITIES")
    print("-" * 50)
    
    goal_states = [
        "Axum", "Gondar", "Lalibela", "Babile",
        "Jimma", "Bale", "Sof Oumer", "Arba Minch"
    ]
    
    print(f"Goal cities to visit: {', '.join(goal_states)}")
    
    multi_result = ucs.multi_goal_search("Addis Ababa", goal_states)
    
    if multi_result:
        print(f"\nComplete tour path:")
        print(f"  {' -> '.join(multi_result['path'])}")
        print(f"\nTotal distance: {multi_result['cost']} km")
        print(f"\nVisit order (with segment costs):")
        for i, (goal, cost) in enumerate(zip(multi_result['goals_order'], 
                                              multi_result['segment_costs']), 1):
            print(f"  {i}. {goal}: {cost} km")
    else:
        print("Could not find path visiting all goals!")
    
    return {"single_goal": result, "multi_goal": multi_result}


def run_question3():
    """Question 3: A* Search"""
    print("\n" + "=" * 70)
    print("QUESTION 3: A* SEARCH")
    print("=" * 70)
    
    # Create graph with heuristics
    print("\nCreating weighted graph with heuristics from Figure 3...")
    graph = create_ethiopia_graph_figure3()
    
    astar = AStarSearch(graph)
    
    print("\n" + "-" * 50)
    print("A* SEARCH: ADDIS ABABA TO MOYALE")
    print("-" * 50)
    
    result = astar.search("Addis Ababa", "Moyale")
    
    if result:
        print(f"Optimal path: {' -> '.join(result['path'])}")
        print(f"Total cost: {result['cost']} km")
        print(f"Nodes expanded: {result['nodes_expanded']}")
        
        print("\nDetailed path with f-values:")
        print("-" * 40)
        for node, f_val in zip(result['path'], result['f_values']):
            h = graph.get_heuristic(node, "Moyale")
            g = f_val - h
            print(f"  {node:20} | g={g:6.0f} | h={h:6.0f} | f={f_val:6.0f}")
    else:
        print("No path found!")
    
    # Compare algorithms
    print("\n" + "-" * 50)
    print("ALGORITHM COMPARISON")
    print("-" * 50)
    
    ucs = UniformCostSearch(graph)
    ucs_result = ucs.search("Addis Ababa", "Moyale")
    
    print(f"{'Algorithm':<20} {'Cost':<12} {'Nodes Expanded':<15}")
    print("-" * 47)
    print(f"{'UCS':<20} {ucs_result['cost']:<12} {ucs_result['nodes_expanded']:<15}")
    print(f"{'A*':<20} {result['cost']:<12} {result['nodes_expanded']:<15}")
    
    if ucs_result['nodes_expanded'] > result['nodes_expanded']:
        savings = ucs_result['nodes_expanded'] - result['nodes_expanded']
        pct = 100 * savings / ucs_result['nodes_expanded']
        print(f"\nA* expanded {savings} fewer nodes ({pct:.1f}% reduction)")
    
    return result


def run_question4():
    """Question 4: MiniMax Search"""
    print("\n" + "=" * 70)
    print("QUESTION 4: MINIMAX SEARCH (ADVERSARIAL)")
    print("=" * 70)
    
    # Create game tree
    print("\nCreating adversary game tree from Figure 4...")
    game_tree = create_adversary_graph_figure4()
    
    print("\nGame Setup:")
    print("-" * 50)
    print("The agent wants to find the best coffee in Ethiopia.")
    print("An adversary tries to prevent the agent from reaching")
    print("high-quality coffee destinations.")
    print()
    print("Player roles:")
    print("  - MAX (Agent): Maximizes coffee quality score")
    print("  - MIN (Adversary): Minimizes agent's coffee quality")
    print()
    
    # Show coffee regions
    print("Ethiopian Coffee Regions and Quality Scores:")
    print("-" * 50)
    root = game_tree['root']
    for region, region_data in root['children'].items():
        print(f"\n{region} Region:")
        for subregion, sub_data in region_data['children'].items():
            if 'children' in sub_data:
                for dest, dest_data in sub_data['children'].items():
                    if 'utility' in dest_data:
                        print(f"    └─ {dest}: Quality = {dest_data['utility']}/100")
    
    # MiniMax Search
    print("\n" + "-" * 50)
    print("MINIMAX SEARCH")
    print("-" * 50)
    
    minimax = MiniMaxSearch(game_tree)
    mm_result = minimax.search()
    
    print(f"Best action for agent: {mm_result['best_action']}")
    print(f"Guaranteed coffee quality: {mm_result['value']}/100")
    print(f"Optimal play: {' -> '.join(mm_result['optimal_path'])}")
    print(f"Nodes evaluated: {mm_result['nodes_evaluated']}")
    
    # Alpha-Beta Search
    print("\n" + "-" * 50)
    print("ALPHA-BETA PRUNING")
    print("-" * 50)
    
    alphabeta = AlphaBetaSearch(game_tree)
    ab_result = alphabeta.search()
    
    print(f"Best action for agent: {ab_result['best_action']}")
    print(f"Guaranteed coffee quality: {ab_result['value']}/100")
    print(f"Nodes evaluated: {ab_result['nodes_evaluated']}")
    print(f"Branches pruned: {ab_result['pruned_branches']}")
    
    if mm_result['nodes_evaluated'] > ab_result['nodes_evaluated']:
        savings = mm_result['nodes_evaluated'] - ab_result['nodes_evaluated']
        print(f"\nAlpha-Beta saved {savings} node evaluations!")
    
    # Interpretation
    print("\n" + "-" * 50)
    print("INTERPRETATION")
    print("-" * 50)
    
    best_region = mm_result['best_action']
    best_score = mm_result['value']
    
    print(f"The agent should head to: {best_region}")
    print()
    
    if best_region == "Sidamo":
        print("Sidamo/Yirgacheffe is Ethiopia's most famous coffee region.")
        print("Even against an adversary, the agent can guarantee premium quality!")
    elif best_region == "Jimma":
        print("Jimma is the historical heartland of Ethiopian coffee.")
        print("A solid choice with consistent quality.")
    elif best_region == "Harar":
        print("Harar produces unique dry-processed coffee beans.")
        print("Known for the distinctive Harar Longberry variety.")
    
    print(f"\nWith optimal play, the agent is guaranteed coffee quality of {best_score}/100")
    
    return mm_result


def main():
    """Main function to run demonstrations."""
    parser = argparse.ArgumentParser(
        description="Traveling Ethiopia Search Problem - Demo Runner"
    )
    parser.add_argument(
        "--question", "-q",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific question (1-4). Omit to run all."
    )
    args = parser.parse_args()
    
    # Print header
    print(create_ascii_map())
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  TRAVELING ETHIOPIA SEARCH PROBLEM  ".center(68) + "#")
    print("#" + "  AI Principles and Techniques - Course Project  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    results = {}
    
    if args.question:
        # Run specific question
        if args.question == 1:
            results['q1'] = run_question1()
        elif args.question == 2:
            results['q2'] = run_question2()
        elif args.question == 3:
            results['q3'] = run_question3()
        elif args.question == 4:
            results['q4'] = run_question4()
    else:
        # Run all questions
        results['q1'] = run_question1()
        results['q2'] = run_question2()
        results['q3'] = run_question3()
        results['q4'] = run_question4()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

