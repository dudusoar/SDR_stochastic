# VRP Toolkit: From Research Code to Reusable Framework

> **Last Updated**: 2026-01-04
> **Status**: Phase 3 - Extension (In Progress)

## 📋 Project Overview

This repository contains the **paper-code** research code (formerly SDR_stochastic) and its transformation into the **vrp-toolkit** - a reusable framework for Vehicle Routing Problems (VRP) with Pickup and Delivery Time Windows (PDPTW).

### 🎯 Vision
Transform research code from academic papers into a **reusable, teachable VRP/PDPTW solving framework**.

### 📂 Repository Structure

```
.
├── paper-code/             # Original research code (legacy)
│   ├── data/              # Input data and distance matrices
│   ├── docs/              # Documentation and task files
│   ├── results/           # Experimental results and outputs
│   ├── demands.py         # Demand generation
│   ├── instance.py        # Problem instance definition
│   ├── operators.py       # ALNS operators
│   ├── order_info.py      # Order information handling
│   ├── real_map.py        # Real-world map integration
│   ├── sensitivity_test.ipynb  # Sensitivity analysis notebook
│   ├── sensitivity_test.py     # Sensitivity analysis script
│   ├── solution.py        # Solution representation
│   ├── solvers.py         # Solver implementations
│   └── test.ipynb         # Main test notebook
│
├── vrp-toolkit/           # NEW: Reusable framework package
│   ├── vrp_toolkit/       # Main Python package
│   │   ├── problems/      # Problem definitions (PDPTW, VRP, CVRP)
│   │   ├── algorithms/    # Solving algorithms (ALNS, GA, etc.)
│   │   ├── data/         # Data generation and loading
│   │   ├── visualization/ # Plotting and visualization
│   │   └── utils/         # Common utilities
│   ├── tutorials/         # Jupyter notebook tutorials (7 total)
│   ├── tests/             # Unit tests
│   ├── pyproject.toml     # Package configuration
│   └── README.md          # Package-specific documentation
│
└── .claude/               # Project management and automation tools
   ├── CLAUDE.md           # Project overview and guidelines
   ├── MIGRATION_LOG.md    # Detailed migration history
   ├── TASK_BOARD.md       # Task tracking and progress
   └── skills/             # 11 custom skills for workflow automation
```

## 🚀 Quick Start

### Installation

The main reusable framework is in the `vrp-toolkit/` directory:

```bash
# Navigate to the vrp-toolkit package
cd vrp-toolkit

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode with all dependencies
uv pip install -e .

# Alternative: Install specific dependencies
uv add numpy pandas matplotlib networkx osmnx
uv add --dev pytest black ruff jupyter
```

### Run the Quickstart Tutorial

```bash
# Start Jupyter notebook from the vrp-toolkit directory
jupyter notebook tutorials/01_quickstart.ipynb
```

### For Research Code Access
The original research code remains in `paper-code/` for reference and comparison.

## 🏗️ Architecture Design

### Three-Layer Architecture

1. **Problem Layer** (`vrp_toolkit/problems/`)
   - Defines problem instances independent of solving algorithms
   - `Instance`, `Solution`, `Node` classes

2. **Algorithm Layer** (`vrp_toolkit/algorithms/`)
   - Implements solving algorithms with common `Solver.solve(instance) -> Solution` interface
   - Adaptive Large Neighborhood Search (ALNS) implemented

3. **Data Layer** (`vrp_toolkit/data/`)
   - Data generation, loading, and OSMnx integration
   - Synthetic data generators and real-world map support

## 📊 Project Progress

### Phase 1: Minimal Migration ✅ Complete
- **Files migrated:** 9/9
- **Package installable:** `pip install -e .`
- **Quickstart tutorial:** Working

### Phase 2: Architecture Refactoring ✅ Complete
- **Three-layer architecture:** Problem/Algorithm/Data
- **Unified Solver interface:** VRPProblem, VRPSolution, Solver base classes
- **Test suite:** 40/40 ALNS tests passing
- **Skills system:** 11 custom skills for project automation
- **Documentation:** Comprehensive data structure references

### Phase 3: Extension 🚀 In Progress
- **OSMnx integration:** Real-world street network support (complete)
- **Tutorial system:** 7 comprehensive tutorials (complete)
- **Real-world examples:** OSMnx-based map integration tutorials
- **Current focus:** Additional algorithms, benchmark suite, PyPI publication

## 🛠️ Features

### Current Implementation
- **ALNS Algorithm**: Adaptive Large Neighborhood Search for PDPTW
- **PDPTW Problem**: Pickup and Delivery with Time Windows
- **Real-world Integration**: OSMnx for street network data and distance matrices
- **Data Generators**: Synthetic and real-world map data
- **Visualization**: Route plotting and solution analysis
- **Tutorial System**: 7 comprehensive Jupyter notebooks covering all features

### Planned Features
- Genetic Algorithm implementation
- Additional VRP variants (CVRP, VRPTW, etc.)
- Benchmark suite with standard instances
- Web-based visualization interface

## 📚 Tutorials

**Complete tutorial series (7 notebooks):**

1. **`01_quickstart.ipynb`** - Basic usage and problem solving
2. **`02_real_world_maps.ipynb`** - OSMnx integration with real street networks
3. **`03_custom_problems.ipynb`** - Creating custom PDPTW instances
4. **`04_problem_variants.ipynb`** - VRP, CVRP, PDP, PDPTW comparison
5. **`05_sensitivity_analysis.ipynb`** - Parameter sensitivity analysis
6. **`06_custom_algorithms.ipynb`** - Implementing custom heuristics
7. **`07_data_generation.ipynb`** - Synthetic data generation workflows

## 🔧 Development

### Project Management
This project uses **Claude Code** with 11 custom skills for automated workflows:

- **`build-session-context`** - Extract project status from logs for token-efficient session startup
- **`migrate-module`** - Guide file migration from paper-code to vrp-toolkit with refactoring
- **`update-migration-log`** - Log migration entries and progress to MIGRATION_LOG.md
- **`integrate-road-network`** - Integrate real-world street networks using OSMnx
- **`log-debug-issue`** - Track bugs and debugging processes in DEBUG_LOG.md
- **`update-task-board`** - Sync TASK_BOARD.md based on evidence from all logs
- **`maintain-data-structures`** - Reference for data structures (Problem/Algorithm/Data layers)
- **`git-log`** - Generate commit messages and maintain GIT_LOG.md
- **`manage-python-env`** - UV package manager reference and environment setup
- **`manage-skills`** - Audit, check compliance, and maintain skills documentation
- **`create-tutorial`** - Create high-quality, progressive learning tutorials

### Code Style
- Type hints where helpful (not everywhere)
- Docstrings for public APIs only
- Comments only for non-obvious logic
- Prefer clarity over cleverness

## 📖 Research Context

### Related Paper
**"Two-stage stochastic fleet and battery sizing with routing optimization for sidewalk delivery robots"** (Du, 2025)
- Problem: PDPTW with battery constraints
- Method: ALNS with SISR removal operator
- Benchmark: Purdue campus data

### Future Integration
Additional VRP research papers can be integrated into this framework.

## 🤝 Contributing

1. **For Research Code**: Add to `paper-code/` directory
2. **For Framework Extensions**: Add to `vrp-toolkit/` following the three-layer architecture
3. **For Tutorials**: Add Jupyter notebooks to `vrp-toolkit/tutorials/`

### Development Workflow
```bash
# Start work session with project context
# (Use Claude Code with build-session-context skill)

# Migrate new research code to reusable framework
# (Use migrate-module skill)

# Update progress documentation automatically
# (Use update-task-board and update-migration-log skills)
```

## 📄 License

Research code may have its own licensing terms. The vrp-toolkit framework is intended for academic and educational use.

## 🙏 Acknowledgments

- Original research code authors
- OpenStreetMap contributors (for OSMnx integration)
- ALNS algorithm community

---

**Note**: This repository contains both legacy research code (`paper-code/`) and the new reusable framework (`vrp-toolkit/`). The framework is actively developed as a template for transforming academic research into reusable tools.