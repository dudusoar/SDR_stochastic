# VRP-Toolkit Architecture Map

**Last Updated:** 2026-01-04
**Version:** 0.1.0
**Status:** Evolving (Phase 3 - Extension)

---

## System Overview

VRP-Toolkit is a Python framework for solving Vehicle Routing Problems (VRP) and variants including VRPTW (with Time Windows), PDPTW (Pickup-Delivery with Time Windows), and CVRP (Capacitated VRP). Originating from academic research code, it has been systematically refactored into a reusable, teachable system emphasizing clear separation of concerns through a three-layer architecture.

The toolkit targets researchers, students, and practitioners who want to experiment with VRP algorithms without drowning in implementation complexity. It prioritizes **learning through interaction** (tutorials, playground) over dense documentation, and **code reusability** over paper-specific optimizations.

**Core philosophy:**
- **Educational first:** Tutorials and playground over API docs
- **Clean architecture:** Three-layer design (Problem/Algorithm/Data)
- **Reproducible research:** Seed control, experiment saving, contract testing
- **Interactive learning:** Streamlit playground for hands-on exploration

---

## Three-Layer Architecture

The toolkit enforces strict layer separation to enable:
- Algorithm-agnostic problem definitions
- Problem-independent algorithm implementations
- Easy extension with new problems/algorithms

### 1. Problem Layer (`vrp_toolkit/problems/`)

**Purpose:** Define VRP problems independent of solving algorithms

**Responsibilities:**
- Problem instance creation and validation
- Solution representation and feasibility checking
- Constraint definitions (time windows, capacity, battery)
- Instance/solution serialization

**Key classes:**
- `PDPTWInstance` - Pickup-Delivery Problem with Time Windows
- `PDPTWSolution` - Solution for PDPTW problems
- `VRPProblem` - Abstract interface for all problem types
- `VRPSolution` - Abstract interface for all solution types

**Key concepts:**
- **Node numbering:** Depot (0), Pickups (1-n), Deliveries (n+1-2n), Charging (optional)
- **Time windows:** `time_windows[node] = (earliest, latest)`
- **Demand:** `demands[node]` for capacity constraints
- **Distance matrix:** `distance_matrix[i][j]` for routing costs

**Dependencies:** None (problems are self-contained)

**Documentation:** See `maintain-data-structures` skill → `problem_layer.md`

### 2. Algorithm Layer (`vrp_toolkit/algorithms/`)

**Purpose:** Implement solving algorithms that work on generic VRP problems

**Responsibilities:**
- Algorithm implementations (ALNS, future: GA, Tabu Search)
- Destroy/repair operators for metaheuristics
- Configuration management (`ALNSConfig`, etc.)
- Solution construction and improvement

**Key classes:**
- `Solver` - Abstract base class with `solve(problem) → solution` interface
- `ConfigurableSolver` - Solver with configuration support
- `ALNSSolver` - Adaptive Large Neighborhood Search implementation
- `ALNSConfig` - ALNS parameters (temperature, cooling, operators)
- `RemovalOperators` - Destroy operators (Shaw, Random, Worst, SISR)
- `RepairOperators` - Repair operators (Greedy, Regret insertion)

**Key concepts:**
- **Unified interface:** All solvers implement `solve(problem, **kwargs) → solution`
- **Adapter pattern:** `PDPTWProblemAdapter` wraps specific instances for generic interface
- **Configuration-driven:** Algorithm parameters in dataclass configs
- **Metaheuristic structure:** Construct → Destroy → Repair → Accept → Repeat

**Dependencies:** Depends on Problem layer (consumes `VRPProblem`, produces `VRPSolution`)

**Documentation:** See `maintain-data-structures` skill → `algorithm_layer.md`

### 3. Data Layer (`vrp_toolkit/data/`)

**Purpose:** Generate, load, and transform problem data

**Responsibilities:**
- Synthetic data generation (random instances for testing)
- Real-world data loading (CSV, OSMnx street networks)
- Benchmark dataset integration (Solomon, Li & Lim - planned)
- Distance/time matrix computation

**Key classes:**
- `OrderGenerator` - Generate PDPTW order tables from demand data
- `DemandGenerator` - Generate synthetic customer-restaurant demand
- `RealMap` - Synthetic map with random coordinates
- `RealDataMap` - Load real map data from CSV
- `OSMnxNetworkLoader` - Load street networks from OpenStreetMap

**Key concepts:**
- **Pipeline:** Map → Demand → Orders → Instance
- **Reproducibility:** Seed control for synthetic generation
- **Real-world integration:** OSMnx for actual street networks
- **Flexibility:** Configurable parameters for problem characteristics

**Dependencies:** Depends on Problem layer (creates problem instances)

**Documentation:** See `maintain-data-structures` skill → `data_layer.md`

### 4. Visualization Layer (`vrp_toolkit/visualization/`)

**Purpose:** Visualize problems, solutions, and algorithm behavior

**Responsibilities:**
- Route visualization (2D maps with matplotlib)
- Convergence plots (cost vs. iteration)
- Algorithm diagnostics (operator usage, temperature)
- Interactive visualizations (Streamlit playground - in development)

**Key classes:**
- `PDPTWVisualizer` - Visualize PDPTW instances and solutions
- `ALNSVisualizer` - Visualize ALNS search behavior
- `MapVisualizer` - Visualize map data
- `DemandVisualizer` - Visualize demand patterns

**Key concepts:**
- **Matplotlib-based:** Static plots for publication
- **Interactive (planned):** Streamlit components for exploration
- **Multi-view:** Route maps, convergence, metrics dashboard

**Dependencies:** Depends on Problem and Algorithm layers

**Documentation:** See visualization module docstrings

### 5. Utils Layer (`vrp_toolkit/utils/`)

**Purpose:** Common utilities used across all layers

**Responsibilities:**
- Configuration file support (JSON/YAML)
- Input validation helpers
- Logging and debugging utilities

**Key classes:**
- `VRPConfig` - Hierarchical configuration container
- `ConfigLoader` - Load/save configs from files

**Dependencies:** None (utils are standalone)

---

## Module Guide

### `vrp_toolkit/problems/`

**Purpose:** VRP problem definitions and solution representations

**Files:**
- `pdptw.py` - Pickup-Delivery Problem with Time Windows (main implementation)
- `base.py` - Abstract interfaces (VRPProblem, VRPSolution)
- `__init__.py` - Module exports

**Public API:**
- `PDPTWInstance(order_table, ...)` - Create PDPTW problem from order data
- `PDPTWSolution(routes, instance)` - Create and validate solution
- `VRPProblem` - Interface for custom problem types
- `VRPSolution` - Interface for custom solution types

**Key workflows:**
- Load instance from CSV: `PDPTWInstance(order_table=pd.read_csv(...))`
- Create solution: `PDPTWSolution(routes, instance)`
- Validate solution: `solution.is_feasible()`
- Evaluate solution: `solution.objective_value`

### `vrp_toolkit/algorithms/`

**Purpose:** Solving algorithms for VRP problems

**Submodules:**
- `base.py` - Solver interfaces
- `alns/` - Adaptive Large Neighborhood Search
  - `solver.py` - ALNS implementation and config
  - `operators.py` - Removal and repair operators
- `genetic/` - Genetic algorithm (planned)
- `tabu/` - Tabu search (planned)

**Public API:**
- `ALNSSolver(config)` - ALNS solver
- `ALNSConfig(...)` - ALNS configuration dataclass
- `greedy_insertion_initial_solution(problem, ...)` - Initial solution generator
- `PDPTWProblemAdapter(instance)` - Adapter for generic interface
- `PDPTWSolutionAdapter(solution)` - Adapter for generic interface

**Key workflows:**
- Configure ALNS: `config = ALNSConfig(max_iterations=1000, ...)`
- Solve problem: `solution = ALNSSolver(config).solve(problem, num_vehicles=3, ...)`
- Access results: `solution.objective_value`, `solution.routes`

### `vrp_toolkit/data/`

**Purpose:** Data generation and loading

**Files:**
- `generators.py` - Order and demand generators
- `map.py` - Synthetic and real map data
- `osmnx_integration.py` - Real-world street network integration
- `benchmarks.py` - Benchmark datasets (planned)

**Public API:**
- `OrderGenerator(real_map, demand_table, ...).generate()` - Generate orders
- `DemandGenerator(num_customers, ...).generate()` - Generate demands
- `RealMap(num_customers, ...)` - Create synthetic map
- `RealDataMap(node_file, tt_matrix_file)` - Load real map data
- `OSMnxNetworkLoader.load_network(place_name)` - Load street network
- `OSMnxNetworkLoader.compute_distance_matrix(...)` - Network-based distances

**Key workflows:**
- Generate synthetic instance: `RealMap → DemandGenerator → OrderGenerator → PDPTWInstance`
- Load real data: `RealDataMap → PDPTWInstance`
- OSMnx integration: `load_network → compute_distance_matrix → create_pdptw_orders`

### `vrp_toolkit/visualization/`

**Purpose:** Visualization tools for problems and solutions

**Files:**
- `problem.py` - Problem visualizers
- `algorithm.py` - Algorithm visualizers
- `data.py` - Data visualizers

**Public API:**
- `PDPTWVisualizer(instance).visualize(solution, ax=...)` - Plot routes
- `plot_convergence(cost_history)` - Plot cost over iterations

**Key workflows:**
- Visualize solution: `PDPTWVisualizer(instance).visualize(solution)`
- Plot convergence: `plt.plot(solver.cost_history)`

### `vrp_toolkit/utils/`

**Purpose:** Common utilities

**Files:**
- `config.py` - Configuration system
- `validation.py` - Input validation (planned)

**Public API:**
- `VRPConfig(...)` - Hierarchical configuration
- `ConfigLoader.load(file_path)` - Load config from YAML/JSON
- `ConfigLoader.save(config, file_path)` - Save config to file

---

## Entry Points

### 1. Create PDPTW Problem from CSV

```python
import pandas as pd
from vrp_toolkit.problems.pdptw import PDPTWInstance

# Load order data
order_table = pd.read_csv("orders.csv")

# Create instance
instance = PDPTWInstance(order_table=order_table)
print(f"Instance: {instance.n} orders, {len(instance.indices)} nodes")
```

### 2. Generate Synthetic Problem

```python
from vrp_toolkit.data.generators import OrderGenerator, DemandGenerator
from vrp_toolkit.data.map import RealMap
from vrp_toolkit.problems.pdptw import PDPTWInstance

# Create synthetic map
real_map = RealMap(num_customers=20, num_restaurants=5, area_size=100, seed=42)

# Generate demands
demand_gen = DemandGenerator(num_customers=20, num_restaurants=5, seed=42)
demand_table = demand_gen.generate()

# Generate orders
order_gen = OrderGenerator(
    real_map=real_map,
    demand_table=demand_table,
    time_params={'time_window_length': 30, 'service_time': 5, 'extra_time': 10},
    robot_speed=1.0
)
order_table = order_gen.generate()

# Create instance
instance = PDPTWInstance(order_table=order_table)
```

### 3. Solve with ALNS

```python
from vrp_toolkit.algorithms.alns import ALNSSolver, ALNSConfig, greedy_insertion_initial_solution
from vrp_toolkit.algorithms.base import PDPTWProblemAdapter
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Configure ALNS
config = ALNSConfig(
    max_iterations=1000,
    start_temp=10.0,
    cooling_rate=0.95,
    segment_length=100
)

# Wrap instance in adapter
problem = PDPTWProblemAdapter(instance)

# Generate initial solution
initial_solution = greedy_insertion_initial_solution(
    problem=problem,
    num_vehicles=3,
    vehicle_capacity=1000,
    battery_capacity=10.0,
    battery_consume_rate=1.0
)

# Solve with ALNS
from vrp_toolkit.algorithms.alns import ALNS
alns = ALNS(
    initial_solution=initial_solution._solution,
    config=config,
    dist_matrix=instance.distance_matrix,
    battery_capacity=10.0
)
alns.run()

print(f"Best cost: {alns.best_solution.objective_value:.2f}")
```

### 4. Visualize Solution

```python
from vrp_toolkit.visualization.problem import PDPTWVisualizer
import matplotlib.pyplot as plt

# Create visualizer
visualizer = PDPTWVisualizer(instance)

# Plot solution
fig, ax = plt.subplots(figsize=(10, 8))
visualizer.visualize(alns.best_solution, ax=ax)
plt.title(f"Solution (Cost: {alns.best_solution.objective_value:.2f})")
plt.show()

# Plot convergence
plt.figure(figsize=(10, 4))
plt.plot(alns.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Total Cost')
plt.title('ALNS Convergence')
plt.grid(True, alpha=0.3)
plt.show()
```

### 5. Use OSMnx for Real-World Data

```python
from vrp_toolkit.data.osmnx_integration import OSMnxNetworkLoader

# Load street network
loader = OSMnxNetworkLoader()
G, nodes_gdf = loader.load_network(place_name="Purdue University, West Lafayette, IN, USA")

# Compute distance matrix
nodes = nodes_gdf.sample(20, random_state=42)  # Select 20 nodes
dist_matrix, time_matrix = loader.compute_distance_matrix(G, nodes)

# Create PDPTW instance from network nodes
order_table = loader.create_pdptw_orders_from_nodes(
    nodes=nodes,
    num_orders=5,
    time_params={'time_window_length': 30, 'service_time': 5},
    seed=42
)

instance = PDPTWInstance(order_table=order_table)
```

---

## Data Flows

### Primary Flow: Data → Problem → Solution → Visualization

```
┌──────────────┐
│  Data Layer  │  Generate/load order data
└──────┬───────┘
       │ creates
       ▼
┌──────────────┐
│Problem Layer │  PDPTWInstance (order_table)
└──────┬───────┘
       │ consumed by
       ▼
┌──────────────┐
│Algorithm     │  ALNSSolver.solve(instance)
│  Layer       │    ├─ greedy initial solution
└──────┬───────┘    ├─ ALNS destroy/repair loop
       │            └─ return best solution
       ▼
┌──────────────┐
│Visualization │  PDPTWVisualizer.visualize(solution)
│   Layer      │
└──────────────┘
```

### ALNS Algorithm Pipeline

```
Input: PDPTWInstance
   ↓
Step 1: Generate Initial Solution
   ├─ greedy_insertion_initial_solution()
   └─ Returns feasible starting routes
   ↓
Step 2: ALNS Iterations (max_iterations times)
   ├─ Destroy: Remove requests
   │   ├─ Shaw removal (relatedness-based)
   │   ├─ Random removal
   │   ├─ Worst removal (cost-based)
   │   └─ SISR removal (paper-specific)
   ├─ Repair: Reinsert requests
   │   ├─ Greedy insertion (best position)
   │   └─ Regret insertion (opportunity cost)
   ├─ Evaluate: Calculate objective value
   └─ Accept: Simulated annealing criterion
       ├─ Always accept if better
       ├─ Accept worse with probability exp(-Δ/T)
       └─ Update temperature: T = T * cooling_rate
   ↓
Output: Best solution found
```

### Configuration Flow

```
User Input
   ↓
Python dict / UI sliders
   ↓
ALNSConfig(max_iterations=1000, start_temp=10.0, ...)
   ↓
ALNSSolver.__init__(config)
   ↓
ALNS.run() → uses config parameters
```

**Detailed flows:** See `.claude/docs/data_flows.md` (to be created)

---

## Key Abstractions

### VRPProblem Interface

Abstract interface enabling algorithm-independent problem definitions:

```python
class VRPProblem(ABC):
    @abstractmethod
    def get_num_nodes(self) -> int: ...

    @abstractmethod
    def get_distance(self, i: int, j: int) -> float: ...

    @abstractmethod
    def get_time(self, i: int, j: int) -> float: ...
```

**Purpose:** Solvers work with `VRPProblem` interface, not specific problem types.

### VRPSolution Interface

Abstract interface for solutions:

```python
class VRPSolution(ABC):
    @abstractmethod
    def get_routes(self) -> List[List[int]]: ...

    @abstractmethod
    def objective_value(self) -> float: ...

    @abstractmethod
    def is_feasible(self) -> bool: ...
```

**Purpose:** Standard solution querying across all problem types.

### Solver Interface

All algorithms implement this:

```python
class Solver(ABC):
    @abstractmethod
    def solve(self, problem: VRPProblem, **kwargs) -> VRPSolution: ...
```

**Purpose:** Unified API for all solving algorithms.

### Adapter Pattern

Bridges old specific interfaces with new generic ones:

- `PDPTWProblemAdapter`: `PDPTWInstance` → `VRPProblem`
- `PDPTWSolutionAdapter`: `PDPTWSolution` → `VRPSolution`

**Purpose:** Backward compatibility during architecture migration.

---

## Module Dependencies

```
       Data
        ↓
     Problem ←─────────┐
        ↓              │
    Algorithm          │
        ↓              │
   Visualization ──────┘

   Utils (used by all layers)
```

**Dependency rules:**
- ✅ Higher layers can depend on lower layers
- ❌ Lower layers MUST NOT depend on higher layers
- ✅ All layers can use Utils

**Current status:** No circular dependencies ✅

**Detailed graph:** See `.claude/docs/module_dependencies.md` (to be created)

---

## Extension Guide

### Adding a New Problem Type (e.g., CVRP)

1. Create `vrp_toolkit/problems/cvrp.py`
2. Define `CVRPInstance` class implementing `VRPProblem`
3. Define `CVRPSolution` class implementing `VRPSolution`
4. Implement required methods (get_num_nodes, get_distance, etc.)
5. Add to `vrp_toolkit/problems/__init__.py`
6. Create tutorial `tutorials/XX_cvrp_problems.ipynb`
7. Update ARCHITECTURE_MAP.md (this file)

### Adding a New Algorithm (e.g., Genetic Algorithm)

1. Create `vrp_toolkit/algorithms/genetic/`
2. Create `solver.py` with `GeneticSolver(ConfigurableSolver)`
3. Create `config.py` with `GAConfig` dataclass
4. Implement `solve(problem, **kwargs) → solution`
5. Add to `vrp_toolkit/algorithms/__init__.py`
6. Create unit tests in `tests/unit/algorithms/genetic/`
7. Create tutorial `tutorials/XX_genetic_algorithm.ipynb`
8. Update ARCHITECTURE_MAP.md

### Adding a New Operator (e.g., 2-opt)

1. Add method to `vrp_toolkit/algorithms/alns/operators.py`
2. Add to `RemovalOperators` or `RepairOperators` class
3. Integrate into ALNS operator selection logic
4. Create unit test in `tests/unit/algorithms/alns/test_operators.py`
5. Document in algorithm layer data structures

---

## Quick Reference

| Task | Code |
|------|------|
| Load instance | `instance = PDPTWInstance(order_table=pd.read_csv("data.csv"))` |
| Generate instance | `instance = generate_synthetic_instance(num_orders=20, seed=42)` |
| Configure ALNS | `config = ALNSConfig(max_iterations=1000, start_temp=10.0)` |
| Solve | `solution = ALNSSolver(config).solve(problem, num_vehicles=3, ...)` |
| Get cost | `cost = solution.objective_value` |
| Check feasibility | `is_ok = solution.is_feasible()` |
| Visualize | `PDPTWVisualizer(instance).visualize(solution)` |
| Plot convergence | `plt.plot(alns.cost_history)` |
| Load OSMnx network | `G, nodes = OSMnxNetworkLoader().load_network(place_name="...")` |

---

## Tutorials & Learning Resources

**Recommended learning path:**

1. **01_quickstart.ipynb** - Basic workflow (generate → solve → visualize)
2. **02_real_world_maps.ipynb** - OSMnx integration for real street networks
3. **03_custom_problems.ipynb** - Creating custom PDPTW instances
4. **04_problem_variants.ipynb** - Understanding VRP/CVRP/PDP/PDPTW differences
5. **05_sensitivity_analysis.ipynb** - Parameter sensitivity experiments
6. **06_custom_algorithms.ipynb** - Implementing custom heuristics
7. **07_data_generation.ipynb** - Synthetic data generation techniques

**Interactive learning:**
- **Playground (in development):** Streamlit-based interactive exploration
- **Design philosophy:** See `playground/VISION.md`

---

## Project Status

**Current Phase:** Phase 3 - Extension (25% complete)

**Completed:**
- ✅ Phase 1: All 9 files migrated from research code
- ✅ Phase 2: Architecture refactored (three-layer design)
- ✅ Test suite: 40/40 ALNS tests passing
- ✅ Tutorials: 7 comprehensive notebooks
- ✅ OSMnx integration: Real-world street network support

**In Progress:**
- 🚧 Playground: Streamlit interactive learning environment
- 🚧 Documentation: Architecture docs and skill system

**Planned:**
- 🔮 Additional algorithms (Genetic Algorithm, Tabu Search)
- 🔮 More problem variants (VRP, CVRP)
- 🔮 Benchmark datasets (Solomon, Li & Lim)
- 🔮 Package publication (PyPI)

---

## Related Documentation

- **Data Structures:** `maintain-data-structures` skill for detailed class documentation
- **Tutorials:** `tutorials/` directory for step-by-step guides
- **Playground:** `playground/VISION.md` for interactive learning philosophy
- **Migration History:** `MIGRATION_LOG.md` for refactoring details
- **Task Tracking:** `TASK_BOARD.md` for current/planned work
- **Skills Reference:** `SKILLS.md` for automation skills

---

**Last Updated:** 2026-01-04
**Maintained by:** `maintain-architecture-map` skill
