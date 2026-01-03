# VRP Toolkit - Skills Reference

**Last Updated:** 2026-01-03

Complete reference for all skills in the VRP Toolkit project.

---

## Overview

We have created **10 custom skills** to automate common workflows. These skills are located in `.claude/skills/` as source directories.

**Categories:**
- **Workflow Skills** (6) - Task execution and project management
- **Reference Skills** (1) - Knowledge base and documentation
- **Utility Skills** (2) - Development tools
- **Meta Skills** (1) - Skill management

---

## Workflow Skills

### 1. build-session-context
**Begin Work Session**

**When to use:** Starting work, returning after a break, or checking project status

**What it does:**
- Extracts key information from 7 project files (CLAUDE.md, TASK_BOARD.md, MIGRATION_LOG.md, DEBUG_LOG.md, GIT_LOG.md, SKILLS_LOG.md, Git status)
- Generates token-efficient context summary (under 1000 tokens)
- Shows migration progress, task status, recent activity
- Displays active blockers and next priorities
- Suggests context-aware next action

**Key Sources:**
- **CLAUDE.md** - Project phase and high-level overview
- **TASK_BOARD.md** - Detailed task tracking (Completed/In Progress/Next/Blockers)
- **MIGRATION_LOG.md** - Migration history and progress
- **DEBUG_LOG.md** - Active issues and recent resolutions
- **GIT_LOG.md** - Recent commits and development activity
- **SKILLS_LOG.md** - Recent skill changes (optional)

**Trigger:** Say "start work", "project status", or "what should I do next?"

**Value:** Prevents token overflow from searching multiple files; provides instant project context

**Location:** `.claude/skills/build-session-context/`

---

### 2. migrate-module
**Migrate Code Files**

**When to use:** Migrating files from SDR_stochastic to vrp-toolkit

**What it does:**
- Guides through 5-step migration workflow
- Identifies and extracts hardcoded values
- Decouples architecture layers
- Adds documentation and tests
- Validates migration completion

**Trigger:** Say "migrate [filename]" or "migrate instance.py"

**Key Documentation:**
- **Comprehensive guide:** `references/MIGRATION_GUIDE.md` - Complete technical migration guide with source locations, file mapping (9 files), migration phases, refactoring patterns, and common issues/solutions
- **Quick references:** `references/migration_map.md` (file mappings), `references/architecture.md` (three-layer architecture)

**Focus:** Migration architecture and process (task progress tracking handled by update-task-board)

**Location:** `.claude/skills/migrate-module/`

---

### 3. update-migration-log
**Migration History Logger**

**When to use:** After completing a file migration

**What it does:**
- Logs detailed migration entries to MIGRATION_LOG.md
- Records issues and solutions
- Tracks verification steps
- Updates migration progress statistics

**Trigger:** Say "log migration" or "update migration log"

**Manages:**
- MIGRATION_LOG.md → Detailed migration history

**Does NOT modify:** CLAUDE.md, TASK_BOARD.md (handled by other skills)

**Location:** `.claude/skills/update-migration-log/`

---

### 4. integrate-road-network
**Real-World Map Integration**

**When to use:** Creating instances with actual street networks

**What it does:**
- Guides through 8-step OSMnx integration
- Loads street networks from OpenStreetMap
- Maps locations to network nodes
- Computes network-based distances
- Creates PDPTW instances with real data

**Trigger:** Say "integrate OSMnx", "use real map", or "create Purdue campus instance"

**References:**
- Examples: `.claude/skills/integrate-road-network/references/osmnx_examples.md` (10 examples)
- Troubleshooting: `.claude/skills/integrate-road-network/references/troubleshooting.md`

**Location:** `.claude/skills/integrate-road-network/`

---

### 5. log-debug-issue
**Problem & Bug Tracking**

**When to use:** Recording issues, bugs, and debugging processes with solutions

**What it does:**
- Logs problems and bugs with structured templates
- Tracks debugging processes and solutions
- Maintains DEBUG_LOG.md with active/resolved issues
- Identifies recurring patterns for prevention

**Trigger:** Say "log bug", "debug issue", or when encountering problems during development

**Integration:**
- Works with update-task-board to convert active issues to blockers
- Provides context for git-log commit messages
- Supplies data for build-session-context status summary

**Location:** `.claude/skills/log-debug-issue/`

---

### 6. update-task-board
**Task Board Manager**

**When to use:** After completing tasks, when syncing project status

**What it does:**
- Reads logs to assess actual progress (MIGRATION_LOG.md, DEBUG_LOG.md, GIT_LOG.md)
- Updates TASK_BOARD.md (Completed/In Progress/Next Steps/Blockers)
- Ensures consistency between logs and task tracking
- Identifies discrepancies between planned and actual work

**Trigger:** Say "update tasks", "sync status", or after completing work

**Manages:**
- TASK_BOARD.md → Project task tracking

**Reads:**
- MIGRATION_LOG.md, DEBUG_LOG.md, GIT_LOG.md for evidence

**Does NOT modify:** CLAUDE.md, MIGRATION_LOG.md (handled by other skills)

**Location:** `.claude/skills/update-task-board/`

---

## Reference Skills

### 7. maintain-data-structures
**Data Structure Reference**

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

**Location:** `.claude/skills/maintain-data-structures/`

---

## Utility Skills

### 8. git-log
**Commit Message Generator & Git Log Maintenance**

**When to use:** Generating appropriate commit messages and maintaining git log documentation

**What it provides:**
- Commit message generation following Conventional Commits format
- Git log documentation maintenance (GIT_LOG.md)
- Analysis of changes to determine commit type and scope
- Integration with other skills for consistent tracking
- Quick reference for common git commands

**Trigger:** Say "commit changes", "generate commit message", or when preparing to commit

**Key Features:**
- Structured commit message templates for VRP project
- Automatic extraction of commit information for GIT_LOG.md
- Support for project-specific scopes (architecture, migration, setup, etc.)
- Integration with update-task-board and log-debug-issue for context

**Value:** Ensures consistent commit history and comprehensive change documentation

**Location:** `.claude/skills/git-log/`

---

### 9. manage-python-env
**UV Package Manager Reference**

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

**References:**
- Troubleshooting: `references/troubleshooting.md`
- Advanced usage: `references/advanced.md`
- Migration from pip: `references/migration.md`

**Location:** `.claude/skills/manage-python-env/`

---

## Meta Skills

### 10. manage-skills
**Skills Management & Compliance**

**When to use:** Managing skills through compliance checking, audit tracking, and documentation synchronization

**What it does:**
- Audits skills directory vs SKILLS.md consistency
- Checks skill compliance (independence, size, structure)
- Updates SKILLS.md index
- Records changes in SKILLS_LOG.md
- Performs periodic skill health checks

**Trigger:** Say "audit skills", "check compliance", "update skills index", or when adding/modifying skills

**Key Features:**
- Automated compliance checking (independence, size ≤500 lines, structure)
- Directory synchronization detection
- Change logging and tracking
- Skill split/merge/rename procedures

**Scripts:**
- `audit_skills.py` - Compare directory with SKILLS.md
- `check_compliance.py` - Validate skill standards

**References:**
- Compliance checklist: `references/compliance_checklist.md`
- Update procedures: `references/update_procedures.md`

**Value:** Ensures skills stay independent, focused, and well-documented as project grows

**Location:** `.claude/skills/manage-skills/`

---

## Skills Workflow

### Typical Daily Workflow

```
1. build-session-context
   ↓ Shows status and suggests next task
   ↓
2. migrate-module (or other task)
   ↓ Executes migration (auto-uses maintain-data-structures as needed)
   ↓
3. update-migration-log
   ↓ Logs completion and updates docs
   ↓
4. Back to build-session-context for next task
```

### OSMnx Integration Workflow

```
1. integrate-road-network
   ↓ Creates real-world instance (auto-uses maintain-data-structures)
   ↓
2. update-migration-log
   ↓ Logs the work done
```

### Debugging Workflow

```
1. Encounter issue
   ↓
2. log-debug-issue
   ↓ Records problem and investigation
   ↓
3. Fix issue
   ↓
4. update-task-board
   ↓ Updates status if blocker
   ↓
5. git-log
   ↓ Creates commit with fix
```

---

## Quick Reference

| Skill | Primary Use | Trigger Phrase |
|-------|-------------|----------------|
| build-session-context | Start work session | "start work" |
| migrate-module | Migrate files | "migrate [file]" |
| update-migration-log | Log completed work | "update progress" |
| integrate-road-network | Add real maps | "integrate OSMnx" |
| log-debug-issue | Track bugs | "log bug" |
| update-task-board | Sync task status | "update tasks" |
| maintain-data-structures | Understand data | Auto-triggered |
| git-log | Create commits | "commit changes" |
| manage-python-env | Setup environment | "setup environment" |
| manage-skills | Maintain skills | "audit skills" |

---

## Skill Dependencies

**Skills that work together:**
- **build-session-context** uses: update-task-board, git-log, log-debug-issue (for status)
- **migrate-module** uses: maintain-data-structures (for reference)
- **integrate-road-network** uses: maintain-data-structures (for reference)
- **update-migration-log** updates: CLAUDE.md, MIGRATION_LOG.md
- **update-task-board** reads: MIGRATION_LOG.md, DEBUG_LOG.md, GIT_LOG.md
- **log-debug-issue** integrates with: update-task-board, git-log
- **manage-skills** manages: All skills, SKILLS.md, SKILLS_LOG.md

---

## For More Information

- **Skill source code:** `.claude/skills/[skill-name]/`
- **Skill changes:** `.claude/SKILLS_LOG.md`
- **Compliance standards:** `.claude/skills/manage-skills/references/compliance_checklist.md`
- **Project overview:** `.claude/CLAUDE.md`
