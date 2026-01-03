# VRP Toolkit - Claude Context Document

**Last Updated:** 2026-01-03
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
│       ├── build-session-context/
│       ├── migrate-module/
│       ├── update-migration-log/
│       ├── maintain-data-structures/
│       ├── integrate-road-network/
│       ├── git-log/
│       └── manage-python-env/
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
   - See **maintain-data-structures skill** → problem_layer.md for details

2. **Algorithm Layer** (`vrp_toolkit/algorithms/`)
   - Implements solving algorithms with common `Solver.solve(instance) -> Solution` interface
   - See **maintain-data-structures skill** → algorithm_layer.md for details

3. **Data Layer** (`vrp_toolkit/data/`)
   - Data generation, loading, and OSMnx integration
   - See **maintain-data-structures skill** → data_layer.md for details

**For detailed data structure documentation**, use the **maintain-data-structures** skill.

---

## 📋 Migration from SDR_stochastic

**Objective:** Transform research code into reusable toolkit

**Status:** 9/9 files migrated, Phase 2 refactoring 90% complete

**For detailed migration guide, see:**
- Technical guide: [migrate-module skill](skills/migrate-module/) → `references/MIGRATION_GUIDE.md`
- File mapping and refactoring guidelines
- Migration phases and workflow
- Common patterns and issues

**Migration tracking:**
- Detailed history: [MIGRATION_LOG.md](MIGRATION_LOG.md)
- Task progress: [TASK_BOARD.md](TASK_BOARD.md)

---

## 🛠️ Skills Reference

We have created **10 custom skills** to automate common workflows. Skills are located in `.claude/skills/` as source directories.

**All Skills:**
1. **build-session-context** - Extract project status from logs for token-efficient session startup
2. **migrate-module** - Guide file migration from SDR_stochastic to vrp-toolkit with refactoring
3. **update-migration-log** - Log migration entries and progress to MIGRATION_LOG.md
4. **integrate-road-network** - Integrate real-world street networks using OSMnx
5. **log-debug-issue** - Track bugs and debugging processes in DEBUG_LOG.md
6. **update-task-board** - Sync TASK_BOARD.md based on evidence from all logs
7. **maintain-data-structures** - Reference for data structures (Problem/Algorithm/Data layers)
8. **git-log** - Generate commit messages and maintain GIT_LOG.md
9. **manage-python-env** - UV package manager reference and environment setup
10. **manage-skills** - Audit, check compliance, and maintain skills documentation

**For detailed documentation, see [SKILLS.md](SKILLS.md)**

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

**Phase:** Phase 2 - Refactoring (90% complete)

**Progress:**
- ✅ Phase 1: Minimal Migration (100% complete)
- 🚧 Phase 2: Refactoring (90% complete - test suite in progress)
- ⏳ Phase 3: Extension (not started)

**Current Focus:** Creating comprehensive test suite for new architecture

**For detailed task tracking, see [TASK_BOARD.md](TASK_BOARD.md)**

### Quick Summary

| Category | Status |
|----------|--------|
| All 9 files migrated | ✅ Complete |
| Architecture refactored | ✅ Complete |
| Skills system (10 skills) | ✅ Complete |
| Documentation structure | ✅ Complete |
| Testing suite | 🚧 In Progress |

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
→ build-session-context skill triggers
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
→ maintain-data-structures skill triggers
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
- Data structure details → maintain-data-structures skill
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
