# Project Structure

## Root Directory
```
SDR_stochastic/
├── .claude/                    # Claude configuration and skills
│   ├── CLAUDE.md              # Project overview
│   ├── MIGRATION_LOG.md       # Migration history
│   └── skills/                # 7 custom skills
├── .git/                      # Git repository
├── .serena/                   # Serena configuration
├── papers/                    # Research papers (if any)
├── SDR_stochastic/            **Original source code**
│   └── new version/
│       ├── data/              # Data files
│       ├── instance.py        # PDPTW instance class
│       ├── solution.py        # Solution class
│       ├── solvers.py         # ALNS solver
│       ├── operators.py       # Removal/repair operators
│       ├── order_info.py      # Order information
│       ├── real_map.py        # Real map integration
│       ├── demands.py         # Demand generation
│       ├── test.ipynb         # Test notebook
│       └── sensitivity_test.ipynb  # Sensitivity analysis
└── vrp-toolkit/              **New package structure**
    ├── vrp_toolkit/          # Main package
    │   ├── problems/         # Problem definitions
    │   │   ├── __init__.py
    │   │   └── pdptw.py     # Migrated: instance.py + solution.py
    │   ├── algorithms/       # Solving algorithms
    │   │   ├── __init__.py
    │   │   └── alns/        # ALNS implementation
    │   │       └── __init__.py
    │   ├── data/            # Data generation and loading
    │   │   └── __init__.py
    │   ├── visualization/   # Plotting and visualization
    │   │   └── __init__.py
    │   └── utils/           # Common utilities
    │       └── __init__.py
    ├── tutorials/           # Jupyter notebooks (planned)
    ├── examples/           # Standalone scripts (planned)
    ├── benchmarks/         # Benchmark datasets (planned)
    ├── tests/             # Unit tests (planned)
    ├── main.py            # Entry point script
    ├── pyproject.toml     # Package configuration
    ├── README.md          # Project documentation
    └── test_pdptw_migration.py  # Test for migrated code
```

## Three-Layer Architecture

### 1. Problem Layer (`vrp_toolkit/problems/`)
- Defines problem instances independent of solving algorithms
- Core classes: `PDPTWInstance`, `Solution`
- Handles data validation, feasibility checking
- **Current status:** `pdptw.py` migrated (instance.py + solution.py)

### 2. Algorithm Layer (`vrp_toolkit/algorithms/`)
- Implements solving algorithms with common `Solver.solve(instance) -> Solution` interface
- **ALNS module:** `alns/` (to migrate solvers.py, operators.py)
- **Future:** Genetic Algorithm, Tabu Search
- **Current status:** Empty (needs migration)

### 3. Data Layer (`vrp_toolkit/data/`)
- Data generation, loading, and OSMnx integration
- **To migrate:** `order_info.py`, `real_map.py`, `demands.py`
- **Planned:** `generators.py`, `osmnx_integration.py`, `benchmarks.py`
- **Current status:** Empty (needs migration)

## File Migration Mapping

| Original File | New Location | Status |
|--------------|--------------|--------|
| `instance.py` | `problems/pdptw.py` | ✅ Migrated |
| `solution.py` | `problems/pdptw.py` | ✅ Migrated |
| `solvers.py` | `algorithms/alns/solver.py` | ❌ Pending |
| `operators.py` | `algorithms/alns/operators.py` | ❌ Pending |
| `order_info.py` | `data/generators.py` | ❌ Pending |
| `real_map.py` | `data/map.py` | ❌ Pending |
| `demands.py` | `data/generators.py` | ❌ Pending |
| `test.ipynb` | `tutorials/01_quickstart.ipynb` | ❌ Pending |
| `sensitivity_test.ipynb` | `tutorials/05_sensitivity_analysis.ipynb` | ❌ Pending |

## Entry Points
- `main.py`: Command-line interface (planned)
- `tutorials/*.ipynb`: Educational notebooks
- `examples/*.py`: Usage examples

## Data Files
- Original data in `SDR_stochastic/new version/data/`
- Benchmark datasets to be added to `benchmarks/`
- Generated data for testing in `tests/data/` (planned)