# Traveling Ethiopia Search Problem
## AI Principles and Techniques - Course Project

Addis Ababa University, Institute of Technology  
School of Information Science and Engineering  
Artificial Intelligence Graduate Program

---

## Project Overview

This project implements various AI search algorithms for the "Traveling Ethiopia" problem, simulating navigation between Ethiopian cities. The project covers:

- **Question 1**: BFS and DFS Search
- **Question 2**: Uniform Cost Search (single and multi-goal)
- **Question 3**: A* Search with heuristics
- **Question 4**: MiniMax Search for adversarial scenarios
- **Question 5**: ROS2/Gazebo robot simulation (Bonus)

## Project Structure

```
AI-Principles-Final-Project/
├── README.md
├── requirements.txt
├── run_demo.py                 # Main demo runner
├── traveling_ethiopia/         # Core Python package
│   ├── __init__.py
│   ├── graph.py               # Graph data structures (Q1.1, Q2.1)
│   ├── search_strategies.py   # BFS, DFS implementation (Q1.2)
│   ├── ucs.py                 # Uniform Cost Search (Q2.2, Q2.3)
│   ├── astar.py               # A* Search (Q3)
│   ├── minimax.py             # MiniMax Search (Q4)
│   ├── visualize.py           # Visualization utilities
│   └── main.py                # Package demo runner
└── ros2_navigation/           # ROS2 Gazebo package (Q5)
    ├── README.md
    ├── package.xml
    ├── CMakeLists.txt
    ├── urdf/
    │   └── three_wheel_robot.urdf.xacro
    ├── worlds/
    │   └── ethiopia_cities.world
    ├── launch/
    │   ├── robot_launch.py
    │   └── world_launch.py
    ├── scripts/
    │   └── navigation_node.py
    └── config/
        └── navigation_params.yaml
```

## Requirements

```bash
pip install networkx matplotlib numpy
```

For Question 5 (ROS2 simulation):
```bash
# ROS 2 Jazzy (or later) + Gazebo Sim (gz sim) + ros_gz_bridge
# Install ROS 2, Gazebo Sim, and ros_gz_bridge for your distro
```

## Quick Start

### Run all demos:

```bash
cd AI-Principles-Final-Project
python run_demo.py
```

### Run specific question:

```bash
python run_demo.py --question 1  # BFS/DFS only
python run_demo.py --question 2  # UCS only
python run_demo.py --question 3  # A* only
python run_demo.py --question 4  # MiniMax only
```

---

## Question Answers

### Question 1: BFS and DFS Search

**1.1 Data Structure Conversion**

The state space graph is converted to an adjacency list representation:

```python
from traveling_ethiopia.graph import create_ethiopia_graph_figure1

graph = create_ethiopia_graph_figure1()
# Graph stored as: {node: [neighbor1, neighbor2, ...]}
```

**1.2 Search Implementation**

```python
from traveling_ethiopia.search_strategies import SearchSolver

solver = SearchSolver(graph, "Addis Ababa", "Axum")
bfs_path = solver.solve(strategy="bfs")  # Uses Queue (FIFO)
dfs_path = solver.solve(strategy="dfs")  # Uses Stack (LIFO)
```

**Sample Output:**
```
BFS Path: Addis Ababa -> Debre Markos -> Bahir Dar -> Gondar -> Axum
DFS Path: Addis Ababa -> Debre Berhan -> Dessie -> Woldia -> Mekelle -> Axum
```

### Question 2: Uniform Cost Search

**2.2 Single Goal UCS**

```python
from traveling_ethiopia.ucs import UniformCostSearch
from traveling_ethiopia.graph import create_ethiopia_graph_figure2

graph = create_ethiopia_graph_figure2()
ucs = UniformCostSearch(graph)
result = ucs.search("Addis Ababa", "Lalibela")
# Returns: path, cost (565 km), nodes_expanded
```

**2.3 Multi-Goal UCS**

```python
goals = ["Axum", "Gondar", "Lalibela", "Babile", "Jimma", "Bale", "Sof Oumer", "Arba Minch"]
result = ucs.multi_goal_search("Addis Ababa", goals)
# Uses greedy nearest-neighbor approach for local optimum
```

### Question 3: A* Search

```python
from traveling_ethiopia.astar import AStarSearch
from traveling_ethiopia.graph import create_ethiopia_graph_figure3

graph = create_ethiopia_graph_figure3()  # Includes heuristics
astar = AStarSearch(graph)
result = astar.search("Addis Ababa", "Moyale")
```

**Sample Output:**
```
Path: Addis Ababa -> Ambo -> Jimma -> Wolaita Sodo -> Arba Minch -> Konso -> Moyale
Cost: 1159 km
Nodes expanded: 30 (vs 40 for UCS - 25% reduction!)
```

### Question 4: MiniMax Search

```python
from traveling_ethiopia.minimax import MiniMaxSearch, AlphaBetaSearch
from traveling_ethiopia.graph import create_adversary_graph_figure4

game_tree = create_adversary_graph_figure4()
minimax = MiniMaxSearch(game_tree)
result = minimax.search()
```

**Sample Output:**
```
Best action: Jimma
Guaranteed coffee quality: 90/100
Optimal path: root -> Jimma -> Bonga -> Bonga-Forest
```

### Question 5: ROS2 Robot Simulation (Bonus)

**5.1 Robot Design**

Three-wheel differential drive robot with:
- 2 drive wheels + 1 caster wheel
- Sensors: LiDAR, IMU (gyroscope), RGB camera
- Gazebo physics integration

**5.2 World File**

Ethiopian cities positioned in Cartesian coordinates:
- Origin: Addis Ababa (0, 0)
- Scale: ~1 unit = 10 km
- Cities represented as skyline-style markers with labels and regional color coding

**5.3 Navigation**

```bash
# Recommended (all-in-one):
ros2 launch traveling_ethiopia_robot sim_launch.py

# Or split into two terminals:
ros2 launch traveling_ethiopia_robot world_launch.py
ros2 launch traveling_ethiopia_robot robot_launch.py \
    initial_city:="Addis Ababa" goal_city:="Moyale" search_strategy:="bfs"
```

---

## Algorithm Comparison

| Algorithm | Complete | Optimal | Time | Space |
|-----------|----------|---------|------|-------|
| BFS | Yes | Yes* | O(b^d) | O(b^d) |
| DFS | No | No | O(b^m) | O(bm) |
| UCS | Yes | Yes | O(b^(C*/ε)) | O(b^(C*/ε)) |
| A* | Yes | Yes** | O(b^d) | O(b^d) |
| MiniMax | Yes | Yes*** | O(b^m) | O(bm) |

\* For unweighted graphs  
\** With admissible heuristic  
\*** Against optimal opponent

---

## Sample Results

### Path from Addis Ababa to Axum

| Algorithm | Path Length | Nodes Expanded |
|-----------|-------------|----------------|
| BFS | 4 edges | 34 |
| DFS | 5 edges | 34 |
| UCS | 1046 km | 38 |
| A* | 1046 km | 28 |

### Multi-Goal Tour (8 cities)

Starting from Addis Ababa, visiting all goal cities:
- Total distance: 4781 km
- Visit order optimized using nearest-neighbor heuristic

---

## Author

Name: Abel Adissu Tsegaye  
ID: GSR/7389/18  
Instructor: Dr. Natnael Argaw  
Course: AI Principles and Techniques - Regular  
Addis Ababa University, 2025/2026

## License

MIT License
