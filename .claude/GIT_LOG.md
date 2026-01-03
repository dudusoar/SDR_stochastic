# Git Log - VRP Toolkit Project

*Comprehensive record of all commits with detailed information.*

## Recent Commits (newest first)

### 2026-01-03 - feat(paper): restructure paper code into organized directory
**Hash:** 53372aa517a4abd770843400b9e2f1a5b238efd6
**Author:** YuChen Du
**Date:** 2026-01-03 15:13:38 -0500

**Changes:**
- Create 'paper code/' directory to organize original research code
- Move all paper-related files (demands.py, instance.py, operators.py, etc.)
- Include original data files (Purdue campus datasets)
- Preserve sensitivity analysis results and documentation
- Maintain original structure for reproducibility

**Files modified:**
- paper code/data/purdue_node_info.csv
- paper code/data/tt_matrix.csv
- paper code/demands.py
- paper code/docs/task.xlsx
- paper code/instance.py
- paper code/operators.py
- paper code/order_info.py
- paper code/process
- paper code/real_map.py
- paper code/results/sensitivity_analysis_average_order_20240922_150308.csv
- paper code/results/sensitivity_analysis_average_order_20240922_163157.csv
- paper code/results/sensitivity_analysis_num_vehicles_20240923.csv
- paper code/results/sensitivity_analysis_num_vehicles_20240924.csv
- paper code/results/sensitivity_analysis_num_vehicles_20240925.csv
- paper code/sensitivity_test.ipynb
- paper code/sensitivity_test.py
- paper code/solution.py
- paper code/solvers.py
- paper code/test.ipynb

---

### 2026-01-03 - chore(skills): standardize skill names and update documentation
**Hash:** 35c258840ebd5c7aa0a4b0953a79a5ccb7a4c842
**Author:** YuChen Du
**Date:** 2026-01-03 15:05:57 -0500

**Changes:**
- Update SKILL.md files to have consistent name fields matching directory names
- Fix cross-references between skills using new standardized names
- Update CLAUDE.md skill documentation for consistency
- Log skill name standardization work in MIGRATION_LOG.md

**Files modified:**
- .claude/CLAUDE.md
- .claude/DEBUG_LOG.md
- .claude/GIT_LOG.md
- .claude/MIGRATION_LOG.md
- .claude/skills/build-session-context/SKILL.md
- .claude/skills/build-session-context/references/claude_md_structure.md
- .claude/skills/git-log/SKILL.md
- .claude/skills/git-workflow/SKILL.md
- .claude/skills/integrate-road-network/SKILL.md
- .claude/skills/integrate-road-network/references/osmnx_examples.md
- .claude/skills/integrate-road-network/references/troubleshooting.md
- .claude/skills/log-debug-issue/SKILL.md
- .claude/skills/maintain-data-structures/SKILL.md
- .claude/skills/maintain-data-structures/references/algorithm_layer.md
- .claude/skills/maintain-data-structures/references/data_layer.md
- .claude/skills/maintain-data-structures/references/problem_layer.md
- .claude/skills/maintain-data-structures/references/runtime_formats.md
- .claude/skills/manage-python-env/SKILL.md
- .claude/skills/session-start/SKILL.md
- .claude/skills/update-migration-log/SKILL.md
- .claude/skills/update-migration-log/references/update_templates.md
- .claude/skills/update-progress/SKILL.md
- .claude/skills/update-progress/references/update_templates.md
- .claude/skills/update-task-board/SKILL.md

---

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
*This file is maintained by the git-log skill.*