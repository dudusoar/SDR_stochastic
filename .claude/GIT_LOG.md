# Git Log - VRP Toolkit Project

*Comprehensive record of all commits with detailed information.*

## Recent Commits (newest first)

### 2026-01-01 - feat(architecture): implement unified Solver interface for problem-algorithm separation
**Hash:** 4569fa12cac9a0815799ae86372d7a90a7f115cf
**Author:** YuChen Du
**Date:** 2026-01-01 00:54:35 -0500

**Changes:**
- Implemented unified Solver interface architecture
- Created VRPProblem, VRPSolution, Solver abstract base classes
- Updated ALNSSolver to use new interface pattern

**Files modified:**
- .claude/CLAUDE.md
- .claude/MIGRATION_LOG.md
- vrp-toolkit/tutorials/01_quickstart.ipynb
- vrp-toolkit/vrp_toolkit/algorithms/__init__.py
- vrp-toolkit/vrp_toolkit/algorithms/alns/__init__.py
- vrp-toolkit/vrp_toolkit/algorithms/alns/solver.py
- vrp-toolkit/vrp_toolkit/algorithms/base.py

---

### 2025-12-31 - feat(setup): make package installable with pyproject.toml
**Hash:** 95900142f0e48b119cefd53a4e0f14b15e5b8e1a
**Author:** YuChen Du
**Date:** 2025-12-31 15:23:35 -0500

**Changes:**
- Created pyproject.toml for package installation
- Made vrp-toolkit package installable with pip
- Updated git-workflow skill documentation

**Files modified:**
- .claude/CLAUDE.md
- .claude/skills/git-workflow/SKILL.md
- vrp-toolkit/pyproject.toml

---

### 2025-12-30 - docs: add comprehensive README and organize skill packages
**Hash:** 0287f904fdea8caa26a1294f80c3432948c2a3c7
**Author:** YuChen Du
**Date:** 2025-12-30 21:32:20 -0500

**Changes:**
- Created comprehensive README documentation
- Organized skill packages into .skill files
- Updated project documentation structure

**Files modified:**
- .claude/CLAUDE.md
- .claude/MIGRATION_LOG.md
- README.md
- skills-packages/data-structures.skill
- skills-packages/git-workflow.skill
- skills-packages/migrate-module.skill
- skills-packages/osmnx-integration.skill
- skills-packages/session-start.skill
- skills-packages/update-progress.skill
- skills-packages/uv-management.skill
- vrp-toolkit/README.md

---

### 2025-12-30 - feat: complete migration of all 9 files to vrp-toolkit architecture
**Hash:** e0001d0add5cd365cd4afd951e1ce14b9cc344a6
**Author:** YuChen Du
**Date:** 2025-12-30 21:15:50 -0500

**Changes:**
- Completed migration of all 9 files from SDR_stochastic
- All files successfully migrated to vrp-toolkit architecture
- Migration phase 1 completed

---

### 2025-12-30 - fix: add SDR_stochastic source code as regular files
**Hash:** 71424e3c1daa9fef53acbfe0efe340577f2de29f
**Author:** Yuchen Du
**Date:** 2025-12-30 15:09:07 -0500

**Changes:**
- Added SDR_stochastic source code as regular files in repository
- Preserved original research code structure

---

### 2025-12-30 - feat: initialize VRP toolkit project with 7 custom skills
**Hash:** fd17a79534678d3d80e5dc383c5b631562e934c7
**Author:** Yuchen Du
**Date:** 2025-12-30 14:51:45 -0500

**Changes:**
- Initialized VRP toolkit project structure
- Created 7 custom skills for workflow automation
- Set up project documentation and CLAUDE.md

---

## Archive

*Older commits will be moved here when "Recent Commits" section grows beyond 20 entries.*

**Last Updated:** 2026-01-03
*This file is maintained by the git-workflow skill.*