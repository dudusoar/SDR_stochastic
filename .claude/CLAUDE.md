# VRP Toolkit - Claude Context Document

**Last Updated:** 2026-01-01
**Status:** Phase 2 - Refactoring (in progress)

---

## 🎯 Project Vision

Transform research code from academic papers into a **reusable, teachable VRP/PDPTW solving framework**.

### Core Objectives
1. **Decouple from specific papers** - Make algorithms generalizable
2. **Enable real-world usage** - Integrate OSMnx for real map data
3. **Educational focus** - Clear tutorials and examples
4. **Research asset** - Display on personal website as "paper + code + demo"
5. **Extensibility** - Easy to add new algorithms and problem variants

### Design Principles
- ✅ Minimal viable clarity over perfection
- ✅ Tutorial-first documentation
- ✅ Quick start within 5 minutes
- ✅ Template-based for reuse across projects
- ❌ No over-engineering
- ❌ No endless refactoring

---

## 📂 Project Structure

```
vrp-toolkit/
├── .claude/
│   ├── CLAUDE.md               # This file - Project overview
│   ├── MIGRATION_LOG.md        # Detailed migration history
│   └── skills/                 # Custom skills (7 total)
│       ├── session-start/
│       ├── migrate-module/
│       ├── update-progress/
│       ├── data-structures/
│       ├── osmnx-integration/
│       ├── git-workflow/
│       └── uv-management/
│
├── vrp_toolkit/                # Main package
│   ├── problems/               # Problem definitions (PDPTW, VRP, CVRP)
│   ├── algorithms/             # Solving algorithms
│   │   ├── alns/              # Adaptive Large Neighborhood Search
│   │   ├── ga/                # Genetic Algorithm (future)
│   │   └── base.py            # Common solver interface
│   ├── data/                   # Data generation and loading
│   │   ├── generators.py      # Synthetic data generators
│   │   ├── osmnx_integration.py  # Real-world map integration
│   │   └── benchmarks.py      # Standard benchmark datasets
│   ├── visualization/          # Plotting and visualization
│   └── utils/                  # Common utilities
│
├── tutorials/                  # Jupyter notebooks (PRIMARY FOCUS)
├── examples/                   # Standalone Python scripts
├── benchmarks/                 # Benchmark datasets
└── tests/                      # Unit tests
```

---

## 🏗️ Architecture Design

### Three-Layer Architecture

1. **Problem Layer** (`vrp_toolkit/problems/`)
   - Defines problem instances independent of solving algorithms
   - See **data-structures skill** → problem_layer.md for details

2. **Algorithm Layer** (`vrp_toolkit/algorithms/`)
   - Implements solving algorithms with common `Solver.solve(instance) -> Solution` interface
   - See **data-structures skill** → algorithm_layer.md for details

3. **Data Layer** (`vrp_toolkit/data/`)
   - Data generation, loading, and OSMnx integration
   - See **data-structures skill** → data_layer.md for details

**For detailed data structure documentation**, use the **data-structures** skill.

---

## 📋 Migration Plan from SDR_stochastic

### Source Code Location
- **Original:** `/Users/yuchendu/Desktop/Github/heuristic in VRP/SDR_stochastic/new version/`
- **New:** `/Users/yuchendu/Desktop/Github/heuristic in VRP/vrp-toolkit/`

### File Mapping

| Original File | New Location | Refactoring Needed |
|--------------|--------------|-------------------|
| `instance.py` | `vrp_toolkit/problems/pdptw.py` | Extract generic parts |
| `solution.py` | `vrp_toolkit/problems/pdptw.py` | Keep solution class |
| `solvers.py` | `vrp_toolkit/algorithms/alns/solver.py` | Extract ALNS core |
| `operators.py` | `vrp_toolkit/algorithms/alns/operators.py` | Modularize operators |
| `order_info.py` | `vrp_toolkit/data/generators.py` | Rename to OrderGenerator |
| `real_map.py` | `vrp_toolkit/data/map.py` | Keep as-is initially |
| `demands.py` | `vrp_toolkit/data/generators.py` | Merge with generators |
| `test.ipynb` | `tutorials/01_quickstart.ipynb` | Clean up for tutorial |
| `sensitivity_test.ipynb` | `tutorials/05_sensitivity_analysis.ipynb` | Polish |

**Total: 9 files to migrate**

### Migration Phases

**Phase 1: Minimal Migration** (CURRENT)
- [x] Create directory structure
- [x] Create CLAUDE.md and MIGRATION_LOG.md
- [x] Create 7 custom skills for workflow automation
- [x] Copy core files with minimal changes
- [x] Create basic README and quickstart tutorial
- [ ] Make it installable (`pip install -e .`)

**Phase 2: Refactoring**
- [ ] Separate problem definition from algorithm
- [ ] Create unified Solver interface
- [ ] Add configuration file support
- [ ] Improve visualization

**Phase 3: Extension**
- [ ] OSMnx integration
- [ ] Add second algorithm (GA or TabuSearch)
- [ ] Benchmark suite
- [ ] Website project page content

---

## 🛠️ Skills Reference

We have created **7 custom skills** to automate common workflows. These skills are located in `.claude/skills/` and packaged as `.skill` files (located in `skills-packages/`).

### Workflow Skills

#### 1. **session-start** - Begin Work Session
**When to use:** Starting work, returning after a break, or checking project status

**What it does:**
- Reads and summarizes CLAUDE.md current status
- Shows migration progress (X/9 files completed)
- Displays in-progress tasks and next priorities
- Shows recent Git commits
- Suggests specific next action

**Trigger:** Say "start work", "project status", or "what should I do next?"

---

#### 2. **migrate-module** - Migrate Code Files
**When to use:** Migrating files from SDR_stochastic to vrp-toolkit

**What it does:**
- Guides through 5-step migration workflow
- Identifies and extracts hardcoded values
- Decouples architecture layers
- Adds documentation and tests
- Validates migration completion

**Trigger:** Say "migrate [filename]" or "migrate instance.py"

**References:**
- File mapping: `.claude/skills/migrate-module/references/migration_map.md`
- Architecture: `.claude/skills/migrate-module/references/architecture.md`

---

#### 3. **update-progress** - Log Completed Work
**When to use:** After completing a migration or task

**What it does:**
- Updates CLAUDE.md status sections
- Logs migration details to MIGRATION_LOG.md
- Updates progress percentages
- Records issues and resolutions
- Suggests next task

**Trigger:** Say "update progress" or "log this migration"

**Updates:**
- CLAUDE.md → Completed tasks, In Progress, Next Steps
- MIGRATION_LOG.md → Detailed migration entries

---

#### 4. **osmnx-integration** - Real-World Map Integration
**When to use:** Creating instances with actual street networks

**What it does:**
- Guides through 8-step OSMnx integration
- Loads street networks from OpenStreetMap
- Maps locations to network nodes
- Computes network-based distances
- Creates PDPTW instances with real data

**Trigger:** Say "integrate OSMnx", "use real map", or "create Purdue campus instance"

**References:**
- Examples: `.claude/skills/osmnx-integration/references/osmnx_examples.md` (10 examples)
- Troubleshooting: `.claude/skills/osmnx-integration/references/troubleshooting.md`

---

### Reference Skills

#### 5. **data-structures** - Data Structure Reference
**When to use:** Need to understand data structures without reading code

**What it provides:**
- Problem layer structures (Instance, Solution, Node)
- Algorithm layer structures (Solver, ALNS, Operators)
- Data layer structures (OSMnx, distance matrices)
- Runtime formats (routes as lists, time windows as tuples)

**Trigger:** Automatic when other skills need structure info, or say "what is [structure name]?"

**References:**
- Problem layer: `problem_layer.md` (~300 lines)
- Algorithm layer: `algorithm_layer.md` (~350 lines)
- Data layer: `data_layer.md` (~300 lines)
- Runtime formats: `runtime_formats.md` (~400 lines)

**Value:** Saves 50-70% tokens by avoiding repeated code reading

---

### Utility Skills

#### 6. **git-workflow** - Git Operations Reference
**When to use:** Performing Git operations, committing, branching, or managing repository

**What it provides:**
- Common Git command patterns (status, commit, branch, merge)
- VRP project-specific workflows (feature development, migration commits)
- Conventional commit message formats
- Token-saving shortcuts table
- Troubleshooting common Git issues

**Trigger:** Say "how to commit?", "git workflow", or automatic when performing Git operations

**Key Patterns:**
- Feature development workflow
- Migration commit format: `feat(migration): migrate [file]`
- Progress update commits: `chore: update progress (X/9)`
- Includes commit message template with Claude Code attribution

**Value:** Saves tokens by avoiding Git documentation lookups

---

#### 7. **uv-management** - UV Package Manager Reference
**When to use:** Setting up Python environment, installing packages, managing dependencies

**What it provides:**
- UV installation and project initialization
- Virtual environment creation and activation
- Package management (add, remove, update)
- VRP-specific pyproject.toml template
- Comparison with pip/venv workflows
- Troubleshooting dependency issues

**Trigger:** Say "setup environment", "install packages", or "create venv with uv"

**VRP Project Setup:**
```bash
uv init vrp-toolkit
uv venv
source .venv/bin/activate
uv add numpy pandas matplotlib networkx
uv add --dev pytest black ruff jupyter
uv add osmnx geopandas
```

**Value:** Fast environment setup (10-100x faster than pip), includes complete pyproject.toml template

---

### Skills Workflow

Typical daily workflow using skills:

```
1. session-start
   ↓ Shows status and suggests next task
   ↓
2. migrate-module
   ↓ Executes migration (auto-uses data-structures as needed)
   ↓
3. update-progress
   ↓ Logs completion and updates docs
   ↓
4. Back to session-start for next task
```

For OSMnx work:
```
1. osmnx-integration
   ↓ Creates real-world instance (auto-uses data-structures)
   ↓
2. update-progress
   ↓ Logs the work done
```

---

## 🔧 Development Guidelines

### Code Style
- Use type hints where helpful (not everywhere)
- Docstrings for public APIs only
- Comments only for non-obvious logic
- Prefer clarity over cleverness

### Testing Strategy
- Focus on integration tests over unit tests
- Test main workflows, not every function
- Keep tests runnable in <10 seconds total

### Documentation Priority
1. **Tutorials** (most important) - Show how to use
2. **README** - Quick overview and installation
3. **Docstrings** - API reference
4. **Website docs** - Project showcase

### Configuration Management
- Use dataclasses or simple dicts for configs
- YAML files for complex scenarios
- Avoid config classes with 50 parameters

---

## 📊 Current Status

### Completed ✅
- [x] Directory structure created
- [x] CLAUDE.md initial version
- [x] MIGRATION_LOG.md template
- [x] Created 7 custom skills (session-start, migrate-module, update-progress, data-structures, osmnx-integration, git-workflow, uv-management)
- [x] Migrate instance.py and solution.py to vrp_toolkit/problems/pdptw.py
- [x] Migrate solvers.py and operators.py to vrp_toolkit/algorithms/alns/
- [x] Migrate order_info.py and demands.py to vrp_toolkit/data/generators.py
- [x] Migrate real_map.py to vrp_toolkit/data/map.py
- [x] Migrate test.ipynb to tutorials/01_quickstart.ipynb
- [x] Migrate sensitivity_test.ipynb to tutorials/05_sensitivity_analysis.ipynb
- [x] Create comprehensive README documentation (root and package)
- [x] Create basic `pyproject.toml`
- [x] Make package installable (`pip install -e .`)
- [x] Test installation and package import
- [x] Test quickstart tutorial execution
- [x] Fix import issues in generators.py and add missing DemandGenerator class
- [x] Analyze coupling between PDPTWInstance and ALNS classes
- [x] Design unified Solver interface for problem-algorithm separation (VRPProblem, VRPSolution, Solver base classes)
- [x] Adapt ALNS to use new Solver interface (ALNSSolver class)
- [x] Update quickstart tutorial to use new architecture

### In Progress 🚧
- None

### Next Steps 📋
1. Refactor PDPTWInstance to remove algorithm-specific logic
2. Add configuration file support
3. Improve visualization
4. Create comprehensive test suite for new architecture

### Blockers 🚫
- None currently

---

## 🎓 Research Context

### Related Papers
- **Main Paper:** SDR Stochastic Delivery Robot paper
  - Problem: PDPTW with battery constraints
  - Method: ALNS with SISR removal operator
  - Benchmark: Purdue campus data

### Future Papers to Integrate
- (List other papers that should become part of this toolkit)

---

## 💡 Design Decisions & Rationale

### Why "vrp-toolkit" not "sdr-solver"?
- Generic name allows expansion beyond one paper
- Signals reusability
- Better for website portfolio

### Why tutorials > documentation?
- Research code users learn by example
- Lower barrier to entry
- Faster to create and maintain

### Why OSMnx integration?
- Real-world credibility
- Demonstrates practical value
- Unique feature for academic tools

### Why custom skills?
- Automate repetitive workflows
- Ensure consistency across sessions
- Save tokens by caching knowledge
- Make complex tasks accessible

---

## 🤝 Working with Claude

### For New Sessions

**Quick Start:**
```
You: "start work"
→ session-start skill triggers
→ See current status and next task
```

**Specific Task:**
```
You: "migrate instance.py"
→ migrate-module skill triggers
→ Follows standard workflow
```

**Need Help:**
```
You: "how do routes work?"
→ data-structures skill triggers
→ Shows runtime format documentation
```

### Updating This File

**When updating CLAUDE.md:**
1. Update "Last Updated" date at top
2. Move tasks between Completed/In Progress/Next Steps
3. Add new design decisions as they arise
4. Keep file focused on project-level info (not detailed workflows)

**When to use MIGRATION_LOG.md instead:**
- Detailed migration entries
- Issue tracking and resolutions
- Progress statistics
- File-specific notes

**Let skills handle:**
- Step-by-step workflows → Use skills instead
- Data structure details → data-structures skill
- Troubleshooting → Skill reference docs

### Anti-Patterns to Avoid
- ❌ Don't let Claude refactor everything at once
- ❌ Don't add features not in roadmap without discussion
- ❌ Don't over-abstract before having 2+ use cases
- ❌ Don't write documentation before code works
- ❌ Don't duplicate content between CLAUDE.md and skills

---

## 📚 External Resources

### Key Dependencies
- Python 3.8+
- NumPy, Pandas - Data manipulation
- Matplotlib - Visualization
- OSMnx - Map data (for real-world integration)
- Folium - Interactive maps (future)

### Benchmark Sources
- Solomon instances (VRPTW)
- Li & Lim instances (PDPTW)
- Custom: Purdue campus data

### Inspiration Projects
- OR-Tools (Google) - Professional but complex
- VRPy - Simple but limited
- ALNS-Framework - Academic but outdated

---

## 🎯 Success Metrics

### Short-term (Phase 1)
- [ ] Someone can `pip install` and run quickstart in 5 min
- [x] README clearly explains what this is
- [x] At least 1 working tutorial
- [x] All 9 files successfully migrated

### Medium-term (Phase 2)
- [ ] Real map example works
- [ ] Paper results are reproducible
- [ ] Code is on personal website

### Long-term (Phase 3)
- [ ] 2+ algorithms implemented
- [ ] Used by at least one external researcher
- [ ] Template applied to another project

---

**Remember:** This is a research asset, not a startup product. Good enough > Perfect.

**Use skills for workflows. Keep this file for project overview.**
