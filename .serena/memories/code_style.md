# Code Style and Conventions

## General Principles
- **Clarity over cleverness:** Prefer readable code over clever optimizations
- **Minimal documentation:** Docstrings for public APIs only, comments only for non-obvious logic
- **Gradual improvement:** Migrate code first, improve documentation later

## Python Version
- Target Python 3.11+
- Use type hints where helpful (not required everywhere)

## Naming Conventions
- **Classes:** PascalCase (e.g., `PDPTWInstance`, `ALNSSolver`)
- **Functions/Methods:** snake_case (e.g., `greedy_insertion_init`, `calculate_distance`)
- **Variables:** snake_case (e.g., `distance_matrix`, `time_windows`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `NODE_TYPE_DEPOT`, `COL_ID`)
- **Private methods:** Leading underscore (e.g., `_extract_demands`)

## Type Hints
- Use for function signatures and class attributes when helpful
- Example from migrated code:
```python
from typing import List, Tuple, Dict, Optional, Union, Any

class PDPTWInstance:
    order_table: pd.DataFrame
    robot_speed: float
    n: int
    indices: List[int]
    demands: List[float]
    time_windows: List[Tuple[float, float]]
```

## Documentation
### Docstrings
- Use triple double quotes (`"""`)
- Include for all public classes and methods
- Follow Google style or numpy style (observe existing code)
- Include parameter descriptions and return values

### Comments
- Use sparingly, only for non-obvious logic
- Prefer English comments in migrated code (original code has Chinese comments)
- Avoid commented-out code

## Import Organization
1. Standard library imports
2. Third-party imports
3. Local imports
- Use absolute imports within package
- Example:
```python
import numpy as np
import pandas as pd
from typing import List, Tuple
from vrp_toolkit.problems.pdptw import PDPTWInstance
```

## Formatting Standards
- **Line length:** 100 characters (configured in black and ruff)
- **Indentation:** 4 spaces (no tabs)
- **Quotes:** Double quotes for docstrings, single quotes for strings (or follow black)
- **Trailing commas:** Use where appropriate

## Error Handling
- Use exceptions for error conditions
- Provide informative error messages
- Avoid silent failures

## Testing Style
- Focus on integration tests over unit tests
- Test main workflows, not every function
- Keep tests runnable in <10 seconds total
- Use pytest fixtures for shared setup

## Migration-Specific Guidelines
- Preserve original algorithm logic during migration
- Improve documentation and type hints gradually
- Extract hardcoded values to configuration
- Decouple problem definition from algorithm implementation

## Example Comparison
### Original Code (Chinese comments, minimal docs)
```python
def _extract_demands(self):
    demands = [0] * (max(self.indices) + 1)
    for _, row in self.order_table.iterrows():
        demands[row['ID']] = row['Demand']
    return demands
```

### Migrated Code (English docs, type hints)
```python
def _extract_demands(self) -> List[float]:
    """Extract demand values from order table.
    
    Returns:
        List of demand values indexed by node ID.
    """
    demands = [0.0] * (max(self.indices) + 1)
    for _, row in self.order_table.iterrows():
        demands[row['ID']] = row['Demand']
    return demands
```