# Technology Stack

## Core Dependencies
- **Python:** >=3.11
- **Numerical Computing:** numpy>=1.24.0, pandas>=2.0.0
- **Visualization:** matplotlib>=3.7.0
- **Graph Networks:** networkx>=3.0

## Optional Dependencies
### OSMnx Integration (for real-world maps)
- osmnx>=1.6.0
- geopandas>=0.14.0
- folium>=0.15.0

### Development Tools
- pytest>=7.0.0 (testing)
- black>=23.0.0 (code formatting)
- ruff>=0.1.0 (linting)
- ipython>=8.0.0, jupyter>=1.0.0 (notebooks)

## Build System
- **Package Builder:** hatchling
- **Package Manager:** uv (recommended), pip

## Platform Support
- Primary: Windows (current development environment)
- Should work on Linux/macOS

## Version Control
- Git

## Project Configuration
- `pyproject.toml` for package metadata and dependencies
- `uv.lock` for deterministic dependency resolution (if using uv)

## Code Analysis Tools
- **Ruff:** Linting with 100 line length
- **Black:** Code formatting with 100 line length
- **Pytest:** Testing framework

## Virtual Environment
- **Recommended:** uv venv (fast environment creation)
- **Alternative:** python -m venv

## Package Installation
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv add numpy pandas matplotlib networkx
uv add --dev pytest black ruff jupyter
uv add osmnx geopandas folium  # for real-world map integration

# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,osmnx]  # after pyproject.toml is ready
```