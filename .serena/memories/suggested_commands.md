# Suggested Commands for Development

## Environment Setup

### Using UV (Recommended)
```bash
# Initialize virtual environment
uv venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Unix/macOS
source .venv/bin/activate

# Install core dependencies
uv add numpy pandas matplotlib networkx

# Install development dependencies
uv add --dev pytest black ruff jupyter

# Install OSMnx integration (optional)
uv add osmnx geopandas folium

# Install package in development mode
uv pip install -e .
```

### Using pip
```bash
# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/macOS

# Install package with dependencies
pip install -e .[dev,osmnx]
```

## Development Workflow

### Code Formatting
```bash
# Format all Python files
black vrp_toolkit/ tests/ examples/

# Check formatting without applying
black --check vrp_toolkit/

# Format specific file
black vrp_toolkit/problems/pdptw.py
```

### Linting
```bash
# Run ruff linter
ruff check vrp_toolkit/

# Fix automatically fixable issues
ruff check --fix vrp_toolkit/

# Check specific directory
ruff check vrp_toolkit/problems/
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest test_pdptw_migration.py

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=vrp_toolkit

# Run tests in specific directory
pytest tests/
```

### Package Management
```bash
# Update dependencies
uv sync

# Add new dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Remove dependency
uv remove package_name

# List installed packages
uv pip list
```

## Git Operations

### Common Git Commands
```bash
# Check status
git status

# View changes
git diff

# Stage files
git add <file_or_directory>

# Commit with message
git commit -m "feat: description"

# View commit history
git log --oneline -10

# Create new branch
git checkout -b feature/name

# Switch branch
git checkout main

# Merge branch
git merge feature/name

# Push to remote
git push origin main
```

### Migration-Specific Commits
```bash
# Migration commit format
git commit -m "feat(migration): migrate instance.py to problems/pdptw.py"

# Progress update commit
git commit -m "chore: update progress (3/9 files migrated)"

# Documentation update
git commit -m "docs: update README with quickstart guide"
```

## Running the Code

### Execute Main Script
```bash
python main.py
```

### Run Jupyter Notebooks
```bash
# Start Jupyter server
jupyter notebook

# Or use specific notebook
jupyter notebook tutorials/01_quickstart.ipynb
```

### Interactive Python
```bash
# Start IPython
ipython

# Import package
import vrp_toolkit
from vrp_toolkit.problems.pdptw import PDPTWInstance
```

## Build and Distribution

### Build Package
```bash
# Build distribution
python -m build

# Install from local build
pip install dist/vrp_toolkit-*.tar.gz
```

### Clean Build Artifacts
```bash
# Remove build directories
rm -rf build/ dist/ *.egg-info/

# Or on Windows
rmdir /s build dist *.egg-info
```

## System Utilities (Windows)

### File Operations
```cmd
# List files
dir

# Change directory
cd path\to\directory

# Create directory
mkdir directory_name

# Remove directory
rmdir /s directory_name

# Copy files
copy source destination

# Move files
move source destination
```

### Process Management
```cmd
# Find Python processes
tasklist | findstr python

# Kill process
taskkill /PID process_id /F

# Check Python version
python --version
```

## Troubleshooting

### Common Issues
```bash
# If uv not found, install with:
pip install uv

# If virtual environment not activating (Windows):
.venv\Scripts\activate.bat

# If package not found after installation:
pip install -e .  # Reinstall in development mode

# If import errors:
python -c "import sys; print(sys.path)"  # Check Python path
```

### Dependency Conflicts
```bash
# Update lock file
uv lock --upgrade

# Recreate virtual environment
uv venv --clean

# Check dependency tree
uv pip show package_name
```

## Quick Reference Table

| Task | Command |
|------|---------|
| Activate venv (Win) | `.venv\Scripts\activate` |
| Activate venv (Unix) | `source .venv/bin/activate` |
| Format code | `black vrp_toolkit/` |
| Lint code | `ruff check vrp_toolkit/` |
| Run tests | `pytest` |
| Install package | `uv pip install -e .` |
| Add dependency | `uv add package_name` |
| Git status | `git status` |
| Git commit | `git commit -m "message"` |
| Run notebook | `jupyter notebook` |
| Run main script | `python main.py` |