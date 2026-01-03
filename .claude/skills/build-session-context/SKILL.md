---
name: build-session-context
description: Build concise session context by extracting key information from project logs (CLAUDE.md, MIGRATION_LOG.md, DEBUG_LOG.md, GIT_LOG.md) to provide token-efficient project status. Use when beginning work, returning after break, or needing quick project overview without reading full files.
---

# Build Session Context

Quickly resume work by extracting key information from project logs and presenting a concise context summary for efficient session startup.

## Goal

Extract essential information from multiple project documentation files and present a **concise, token-efficient summary** that enables quick context recovery without reading entire files.

## Workflow

### Step 1: Extract Key Information from Each File

Read each file and extract only the most critical information:

**1. CLAUDE.md (Project Overview)**
- **Phase:** Current project phase
- **Last Updated:** Date of last update
- **Completed Tasks:** Count and 2-3 most recent
- **In Progress Tasks:** List (limit to 3 most important)
- **Next Steps:** Top 2-3 priorities
- **Blockers:** Any active blockers

**2. MIGRATION_LOG.md (Migration History)**
- **Progress:** X/9 files completed (percentage)
- **Recent Migrations:** Last 2-3 entries (date, title, status)
- **Index:** Check index for quick overview if available

**3. DEBUG_LOG.md (Problem Tracking)**
- **Active Issues:** Count and brief descriptions
- **Recent Resolutions:** Last 2-3 resolved issues
- **Urgent Blockers:** Any issues marked as critical

**4. GIT_LOG.md (Change History)**
- **Recent Commits:** Last 3-5 commits (hash, message)
- **Activity Pattern:** Type of recent work (features, fixes, docs)
- **Last Commit Date:** When was last work done

**5. Git Status (Current Workspace)**
- **Current Branch:** Which branch is active
- **Uncommitted Changes:** Any staged/unstaged changes
- **Untracked Files:** New files not yet in git

### Step 2: Generate Concise Context Summary

Format the extracted information into a compact summary:

```markdown
## 🚀 Session Start - VRP Toolkit
**Phase:** [Phase] | **Last Updated:** [Date] | **Branch:** [branch]

## 📊 Status Snapshot
- **Migration:** X/9 files (XX%) | **Active Issues:** [count]
- **Completed:** [count] tasks | **In Progress:** [count] tasks
- **Blockers:** [None/list]

## 🔍 Recent Activity
**Git:** [Last 3 commits, one line each]
**Migrations:** [Last 2 migration titles]
**Issues:** [Active issue count, recent resolution]

## 🎯 Current Focus
1. [Primary in-progress task]
2. [Secondary in-progress task]

## 📋 Immediate Next Steps
1. [Highest priority next action]
2. [Secondary next action]

## ⚠️ Active Blockers (if any)
- [Blocker 1]
- [Blocker 2]
```

**Token Optimization Guidelines:**
- Keep total summary under 1000 tokens if possible
- Use abbreviations: e.g., "ALNS" not "Adaptive Large Neighborhood Search"
- Limit lists to 3-5 items maximum
- Use bullet points over paragraphs
- Prioritize recent information over historical

### Step 3: Suggest Context-Aware Next Action

Based on the summary, recommend the most appropriate next step:

**Decision Matrix:**
- **Blockers present** → Address most critical blocker first
- **Uncommitted changes** → Review and commit before new work
- **Active issues in DEBUG_LOG.md** → Continue debugging if in progress
- **Migration in progress** → Continue with current migration
- **No active tasks** → Start highest priority from "Next Steps"
- **Long break (>3 days)** → Review recent commits first

**Be specific and actionable:**
- Instead of: "Work on migration"
- Use: "Continue migrating `operators.py` to `vrp_toolkit/algorithms/alns/operators.py`"

### Step 4: Offer Skill Integration

After presenting summary, suggest relevant skills:

**Common integrations:**
- `update-task-board` - Update task status if work completed
- `git-log` - Commit uncommitted changes
- `log-debug-issue` - Document issues encountered
- `update-migration-log` - Log completed work
- `migrate-module` - Continue file migration

## Special Cases & Optimization

### Long Log Files
When logs are very long (e.g., MIGRATION_LOG.md > 500 lines):
- Rely on index section if available
- Read only first/last few entries
- Use `grep` or search for recent dates
- Extract statistics rather than details

### Missing Log Files
If a log file doesn't exist:
- Note its absence in summary
- Suggest creating it if needed
- Proceed with available information

### First Session After Break
If last update > 3 days ago:
- Emphasize git history to rebuild context
- Highlight any stale in-progress tasks
- Suggest verifying environment/dependencies

### High Token Count Situation
If summary is too long:
- Further trim lists (2 items instead of 3)
- Remove less critical sections
- Use more abbreviations
- Focus only on current/next actions

## Integration with New Skills

**update-task-board:** Session-start uses update-task-board's parsing logic for consistent task tracking.

**git-log:** Git history comes from GIT_LOG.md maintained by git-log.

**log-debug-issue:** Issue status comes from DEBUG_LOG.md maintained by log-debug-issue.

**update-migration-log:** Migration progress comes from MIGRATION_LOG.md maintained by update-migration-log.

## Example Output

```
## 🚀 Session Start - VRP Toolkit
**Phase:** Phase 2 - Refactoring | **Last Updated:** 2026-01-01 | **Branch:** main

## 📊 Status Snapshot
- **Migration:** 9/9 files (100%) | **Active Issues:** 0
- **Completed:** 15 tasks | **In Progress:** 2 tasks
- **Blockers:** None

## 🔍 Recent Activity
**Git:** 4569fa1 feat(architecture): unified Solver interface
        9590014 feat(setup): make package installable
        0287f90 docs: comprehensive README
**Migrations:** Phase 2: Test Suite Enhancement
                Phase 2: Visualization System Improvement
**Issues:** 0 active, last resolved: ImportError in generators.py

## 🎯 Current Focus
1. Creating comprehensive test suite for new architecture
2. Testing ALNSSolver with other VRPProblem implementations

## 📋 Immediate Next Steps
1. Complete test suite with edge cases
2. Run integration tests to verify backward compatibility

## ⚠️ Active Blockers
None - ready to proceed
```

## Reference

For detailed file structures:
- CLAUDE.md structure: [claude_md_structure.md](references/claude_md_structure.md)
- Log file formats: See respective skill documentation
