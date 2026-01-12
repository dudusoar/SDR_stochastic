# PDPTW with Battery Constraints for Sidewalk Delivery Robots

> **Research Code** for "Two-stage stochastic fleet and battery sizing with routing optimization for sidewalk delivery robots" (Du, 2025)

## Problem

**Pickup and Delivery Problem with Time Windows (PDPTW)** with battery capacity constraints for autonomous sidewalk delivery robots. The problem involves:
- Pickup-delivery pairs with time window constraints
- Limited battery capacity requiring charging station visits
- Multi-vehicle fleet routing optimization
- Real-world campus delivery scenario

## Method

**Adaptive Large Neighborhood Search (ALNS)** with custom operators:
- **SISR (Stochastic Insertion with Sequential Removal)** - Novel removal operator for PDPTW
- Shaw removal, random removal, worst removal
- Greedy insertion and regret-k insertion
- Simulated annealing acceptance criterion
- Adaptive operator weight adjustment

**Key Innovation**: SISR operator combines stochastic insertion with sequential removal to efficiently explore the solution space while maintaining solution quality.

## Dataset

**Purdue University Campus** real-world data:
- 51 nodes (restaurants, customer locations, depot, charging stations)
- Distance and time matrices from campus map
- Realistic delivery demand patterns
- Located in `data/purdue_node_info.csv` and `data/tt_matrix.csv`

## Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib networkx
```

### Run Test Experiments
```bash
# Verify all components work
cd tests
python run_all_tests.py  # Should pass 4/4 tests in ~6 seconds
```

### Run Full Optimization
```python
from real_map import RealDataMap
from demands import DemandGenerator
from order_info import OrderGenerator
from instance import PDPTWInstance
from solvers import greedy_insertion_init, ALNS

# Load real campus data
real_map = RealDataMap('data/purdue_node_info.csv', 'data/tt_matrix.csv')

# Generate demand
demand_gen = DemandGenerator(...)
order_gen = OrderGenerator(real_map, demand_gen.demand_table, ...)

# Create PDPTW instance
instance = PDPTWInstance(order_gen.order_table)

# Solve with ALNS
initial_solution = greedy_insertion_init(instance, ...)
alns = ALNS(initial_solution, ...)
best_solution = alns.run()
```

See `test.ipynb` for complete examples.

## Repository Structure

```
paper-code/
├── data/
│   ├── purdue_node_info.csv    # Campus node coordinates and types
│   └── tt_matrix.csv            # Time/distance matrix (51×51)
├── results/
│   └── sensitivity_analysis_*.csv  # Experimental results
├── tests/
│   ├── run_all_tests.py         # Quick validation (4 tests)
│   └── test_*.py                # Individual component tests
├── real_map.py                  # Map data loading and distance calculation
├── demands.py                   # Stochastic demand generation
├── order_info.py                # Order table generation with time windows
├── instance.py                  # PDPTWInstance class definition
├── solution.py                  # PDPTWSolution class and feasibility checking
├── operators.py                 # ALNS removal and repair operators (SISR here)
├── solvers.py                   # ALNS main algorithm
├── test.ipynb                   # Main testing notebook
└── sensitivity_test.ipynb       # Sensitivity analysis experiments
```

## Experimental Results

Sensitivity analysis results available in `results/`:
- **Average order count** impact on solution quality
- **Number of vehicles** impact on routing efficiency
- Statistical analysis across multiple random seeds

Run `sensitivity_test.ipynb` to reproduce experiments.

## Validation

All core components have been tested and verified:
```bash
cd tests
python run_all_tests.py
```

**Test Results** (2026-01-09):
- ✅ Data Layer (RealMap + DemandGenerator)
- ✅ Order Generation (time windows, partner pairing)
- ✅ Instance & Solution (feasibility checking)
- ✅ Solver (greedy initial + ALNS optimization)

See `tests/TEST_SUMMARY.md` for detailed results.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{du2025twostage,
  title={Two-stage stochastic fleet and battery sizing with routing optimization for sidewalk delivery robots},
  author={Du, Yuchen},
  year={2025}
}
```

## Related Framework

This research code is being transformed into a reusable **VRP Toolkit** framework. See the [`vrp-toolkit/`](../vrp-toolkit/) directory for the generalized implementation with:
- Modular three-layer architecture
- Support for multiple VRP variants
- Real-world OSMnx integration
- Comprehensive tutorials

## License

This code is provided for academic and research purposes. Please cite the paper if you use this code.

## Contact

For questions about the research or code implementation, please open an issue in this repository.

---

**Status**: ✅ Fully tested and validated (4/4 tests passing)
**Last Updated**: 2026-01-09
