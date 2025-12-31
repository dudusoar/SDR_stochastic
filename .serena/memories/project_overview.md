# Project Overview

## Project Purpose
Transform research code from academic papers into a reusable, teachable VRP/PDPTW solving framework.

## Core Objectives
1. Decouple from specific papers - Make algorithms generalizable
2. Enable real-world usage - Integrate OSMnx for real map data
3. Educational focus - Clear tutorials and examples
4. Research asset - Display on personal website as "paper + code + demo"
5. Extensibility - Easy to add new algorithms and problem variants

## Design Principles
- Minimal viable clarity over perfection
- Tutorial-first documentation
- Quick start within 5 minutes
- Template-based for reuse across projects
- No over-engineering
- No endless refactoring

## Related Papers
- **Main Paper:** SDR Stochastic Delivery Robot paper
  - Problem: PDPTW with battery constraints
  - Method: ALNS with SISR removal operator
  - Benchmark: Purdue campus data

## Current Status (as of 2025-12-30)
- **Phase:** Phase 1 - Initial Setup
- **Migration Progress:** 2/9 files completed (instance.py, solution.py migrated to pdptw.py)
- **Next Steps:** Continue file migration, create quickstart tutorial, write README

## Directory Structure
```
vrp-toolkit/
├── vrp_toolkit/           # Main package
│   ├── problems/         # Problem definitions (PDPTW, VRP, CVRP)
│   ├── algorithms/       # Solving algorithms (ALNS, GA)
│   ├── data/            # Data generation and loading
│   ├── visualization/   # Plotting and visualization
│   └── utils/           # Common utilities
├── tutorials/           # Jupyter notebooks (PRIMARY FOCUS)
├── examples/           # Standalone Python scripts
├── benchmarks/         # Benchmark datasets
└── tests/              # Unit tests
```

## Source Code Location
- **Original:** `SDR_stochastic/new version/` (9 files to migrate)
- **New:** `vrp-toolkit/`

## Success Metrics
- Short-term: Someone can `pip install` and run quickstart in 5 min
- Medium-term: Real map example works, paper results reproducible
- Long-term: 2+ algorithms implemented, used by external researchers