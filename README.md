# VRP Toolkit: From Research Code to Reusable Framework

> **Last Updated**: 2025-12-30  
> **Status**: Phase 1 Complete - All 9 files migrated

## 📋 Project Overview

This repository contains the **SDR_stochastic** research code and its transformation into the **vrp-toolkit** - a reusable framework for Vehicle Routing Problems (VRP) with Pickup and Delivery Time Windows (PDPTW).

### 🎯 Vision
Transform research code from academic papers into a **reusable, teachable VRP/PDPTW solving framework**.

### 📂 Repository Structure

```
.
├── SDR_stochastic/           # Original research code (legacy)
│   ├── archive/             # Archived versions
│   └── new version/         # Latest research implementation
│
├── vrp-toolkit/             # NEW: Reusable framework package
│   ├── vrp_toolkit/         # Main Python package
│   │   ├── problems/        # Problem definitions (PDPTW, VRP, CVRP)
│   │   ├── algorithms/      # Solving algorithms (ALNS, GA, etc.)
│   │   ├── data/           # Data generation and loading
│   │   ├── visualization/  # Plotting and visualization
│   │   └── utils/          # Common utilities
│   ├── tutorials/          # Jupyter notebooks
│   ├── examples/           # Standalone Python scripts
│   └── tests/              # Unit tests
│
├── tutorials/              # Educational tutorials (cross-linked)
├── papers/                # Related academic papers
└── .claude/               # Project management and automation
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to the vrp-toolkit package
cd vrp-toolkit

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode
pip install -e .

# Or install dependencies directly
uv add numpy pandas matplotlib networkx
uv add --dev pytest black ruff jupyter
```

### Run the Quickstart Tutorial

```python
# Open the interactive tutorial
jupyter notebook tutorials/01_quickstart.ipynb
```

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

## 📊 Migration Status

**Complete ✅** (All 9 files migrated)
- [x] `instance.py` → `vrp_toolkit/problems/pdptw.py`
- [x] `solution.py` → `vrp_toolkit/problems/pdptw.py`
- [x] `solvers.py` → `vrp_toolkit/algorithms/alns/solver.py`
- [x] `operators.py` → `vrp_toolkit/algorithms/alns/operators.py`
- [x] `order_info.py` → `vrp_toolkit/data/generators.py`
- [x] `demands.py` → `vrp_toolkit/data/generators.py`
- [x] `real_map.py` → `vrp_toolkit/data/map.py`
- [x] `test.ipynb` → `tutorials/01_quickstart.ipynb`
- [x] `sensitivity_test.ipynb` → `tutorials/05_sensitivity_analysis.ipynb`

## 🛠️ Features

### Current Implementation
- **ALNS Algorithm**: Adaptive Large Neighborhood Search for PDPTW
- **PDPTW Problem**: Pickup and Delivery with Time Windows
- **Data Generators**: Synthetic and real-world map data (OSMnx integration)
- **Visualization**: Route plotting and solution analysis
- **Tutorials**: Step-by-step educational notebooks

### Planned Features
- Genetic Algorithm implementation
- Additional VRP variants (CVRP, VRPTW, etc.)
- Benchmark suite with standard instances
- Web-based visualization interface

## 📚 Tutorials

1. **`01_quickstart.ipynb`** - Basic usage and problem solving
2. **`05_sensitivity_analysis.ipynb`** - Parameter sensitivity analysis

## 🔧 Development

### Project Management
This project uses **Claude Code** with custom skills for automated workflows:

- **`session-start`** - Begin work session and check status
- **`migrate-module`** - Migrate files from research code to framework
- **`update-progress`** - Log completed work and update documentation
- **`data-structures`** - Reference for all data structures
- **`osmnx-integration`** - Real-world map integration guide
- **`git-workflow`** - Git operations reference
- **`uv-management`** - UV package manager reference

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

1. **For Research Code**: Add to `SDR_stochastic/` directory
2. **For Framework Extensions**: Add to `vrp-toolkit/` following the three-layer architecture
3. **For Tutorials**: Add Jupyter notebooks to `tutorials/`

### Development Workflow
```bash
# Start work session
# (Use Claude Code with session-start skill)

# Migrate new research code
# (Use migrate-module skill)

# Update progress documentation
# (Use update-progress skill)
```

## 📄 License

Research code may have its own licensing terms. The vrp-toolkit framework is intended for academic and educational use.

## 🙏 Acknowledgments

- Original research code authors
- OpenStreetMap contributors (for OSMnx integration)
- ALNS algorithm community

---

**Note**: This repository contains both legacy research code (`SDR_stochastic/`) and the new reusable framework (`vrp-toolkit/`). The framework is actively developed as a template for transforming academic research into reusable tools.