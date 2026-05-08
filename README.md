# Two-Stage Stochastic Fleet and Battery Sizing for Sidewalk Delivery Robots

[![Paper](https://img.shields.io/badge/Paper-TRE%202025-blue)](https://doi.org/10.1016/j.tre.2025.104220)
[![ALNS](https://img.shields.io/badge/Algorithm-ALNS-green)](https://doi.org/10.1016/j.tre.2025.104220)
[![PDPTW](https://img.shields.io/badge/Problem-PDPTW-orange)](https://doi.org/10.1016/j.tre.2025.104220)

This repository contains the research code and reproducible experiment assets for:

**Du, Y., Yang, H., Chow, J. Y. J., & Le, T. V. (2025)**  
*Two-stage stochastic fleet and battery sizing with routing optimization for sidewalk delivery robots*  
Transportation Research Part E, 201, 104220.  
https://doi.org/10.1016/j.tre.2025.104220

## Overview

The paper studies strategic resource planning for sidewalk delivery robot systems under stochastic demand. The model combines:

- fleet sizing
- spare battery provisioning
- operational pickup-delivery routing with soft time windows and battery constraints

The first stage chooses fleet and battery resources before demand is realized. The second stage solves routing operations for each realized demand scenario using a customized ALNS heuristic.

## Repository Scope

This repository is now intentionally scoped to the TRE paper code only. The reusable `vrp-toolkit` framework has been split into a separate local project and is no longer maintained inside this repository.

```text
VRP-heuristics/
├── assets/
│   ├── Battery-Swapping-Insertion.jpg
│   ├── Purdue-Campus.jpg
│   ├── Routing-Example.jpg
│   └── Second-Stage-Framework.png
├── paper-code/
│   ├── data/
│   ├── results/
│   ├── tests/
│   ├── demands.py
│   ├── instance.py
│   ├── operators.py
│   ├── order_info.py
│   ├── real_map.py
│   ├── solution.py
│   └── solvers.py
├── requirements.txt
└── README.md
```

## Method

The operational routing subproblem is a pickup and delivery problem with time windows and battery constraints. The implementation includes:

- SISR removal
- Shaw removal
- random removal
- worst removal
- greedy insertion
- regret-k insertion
- simulated annealing acceptance
- adaptive operator weight updates
- explicit battery-swapping insertion after route optimization

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
cd paper-code\tests
python run_all_tests.py
```

The validation suite covers:

- data loading and demand generation
- order generation
- PDPTW instance and solution evaluation
- greedy initialization and ALNS optimization

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{du2025two,
  title   = {Two-stage stochastic fleet and battery sizing with routing optimization for sidewalk delivery robots},
  author  = {Du, Yuchen and Yang, Hai and Chow, Joseph Y. J. and Le, Tho V.},
  journal = {Transportation Research Part E},
  volume  = {201},
  pages   = {104220},
  year    = {2025},
  doi     = {10.1016/j.tre.2025.104220}
}
```

## License

This code is provided for academic and research purposes. Please cite the paper if you use it in research.
