# Update Templates and Patterns

This guide shows how to update CLAUDE.md and MIGRATION_LOG.md after completing tasks.

## Updating CLAUDE.md

### 1. Update "Last Updated" Date

Always update the date at the top of CLAUDE.md:

```markdown
**Last Updated:** 2025-12-30  ← Change to current date
**Status:** Phase 1 - Initial Setup
```

### 2. Move Tasks from "In Progress" to "Completed"

When a task is finished:

**Before:**
```markdown
### Completed ✅
- [x] Directory structure created
- [x] CLAUDE.md initial version

### In Progress 🚧
- [ ] Migrate core ALNS algorithm
- [ ] Create quickstart tutorial
```

**After:**
```markdown
### Completed ✅
- [x] Directory structure created
- [x] CLAUDE.md initial version
- [x] Migrate core ALNS algorithm  ← Moved from In Progress

### In Progress 🚧
- [ ] Create quickstart tutorial
```

### 3. Update Phase Status (if needed)

If completing a task finishes a phase:

```markdown
**Status:** Phase 1 - Initial Setup  → **Status:** Phase 2 - Refactoring
```

Common phase transitions:
- Phase 1 → Phase 2: When all minimal migration tasks are complete
- Phase 2 → Phase 3: When refactoring is done and ready for extensions

### 4. Add New Tasks to "In Progress" or "Next Steps"

If new work is identified, add it:

```markdown
### Next Steps 📋
1. Copy core files from SDR_stochastic
2. Create basic `pyproject.toml`
3. Add unit tests for migrated modules  ← New task discovered
```

### 5. Update Blockers (if resolved or new ones found)

```markdown
### Blockers 🚫
- None currently
```

Or if there's a blocker:

```markdown
### Blockers 🚫
- Missing OSMnx dependency documentation
```

## Updating MIGRATION_LOG.md

### 1. Update Progress Summary

Calculate and update the numbers:

```markdown
**Total Files:** 9
**Completed:** 2  ← Increment
**In Progress:** 0
**Remaining:** 7  ← Decrement

### Completion Rate: 22%  ← Update percentage
```

Formula: `Completion Rate = (Completed / Total Files) * 100`

### 2. Add Migration Entry

Add a new entry at the **top** of the Migration History section (most recent first):

```markdown
## Migration History

### 2025-12-30 15:45 - instance.py → pdptw.py

**Status:** ✅ Completed

**Source:** `/Users/yuchendu/Desktop/Github/heuristic in VRP/SDR_stochastic/new version/instance.py`
**Destination:** `vrp_toolkit/problems/pdptw.py`

**Refactoring Done:**
- [x] Extracted hardcoded values to parameters
- [x] Decoupled from paper-specific logic
- [x] Added docstrings
- [x] Updated imports
- [x] Created test case
- [x] Verified functionality

**Issues Encountered:**
- Circular import between Instance and Solution classes

**Resolution:**
- Moved Solution class to same file as Instance
- Used forward references in type hints

**Notes:**
- Battery constraint handling is well-designed
- Minimal changes needed to original structure

---

### [Previous entry...]
```

### 3. Update Files Remaining Checklist

Mark the file as completed:

```markdown
## Files Remaining to Migrate

1. [x] `instance.py` → `vrp_toolkit/problems/pdptw.py`  ← Mark as done
2. [ ] `solution.py` → `vrp_toolkit/problems/pdptw.py`
3. [ ] `solvers.py` → `vrp_toolkit/algorithms/alns/solver.py`
...
```

### 4. Add to Common Issues (if applicable)

If you encountered a new issue that others might face, add it:

```markdown
## Common Issues & Solutions

### Issue: Circular Imports
**Solution:** Move related classes to same file or use `TYPE_CHECKING`

### Issue: Battery Constraint Validation
**Solution:** Add validation method to Instance class with clear error messages
```

### 5. Update "Last Updated" Date

At the bottom of MIGRATION_LOG.md:

```markdown
**Last Updated:** 2025-12-30  ← Update to current date
```

## Migration Entry Checklist

Use this to ensure complete logging:

- [ ] Updated progress summary (numbers and percentage)
- [ ] Added migration entry with timestamp
- [ ] Marked status (✅ Completed / 🚧 In Progress / ⚠️ Issues / ❌ Failed)
- [ ] Listed all refactoring done with checkmarks
- [ ] Documented any issues encountered
- [ ] Described resolution for issues
- [ ] Added notes/observations
- [ ] Updated files remaining checklist
- [ ] Updated last modified date

## Status Symbols

Use these consistently:

- ✅ **Completed** - Task finished successfully
- 🚧 **In Progress** - Currently working on it
- ⚠️ **Issues** - Completed but with known issues
- ❌ **Failed** - Could not complete, blocked
- 📋 **Planned** - Scheduled for future
- 🔍 **Review** - Needs review before marking complete

## Example: Complete Update Flow

**Scenario:** Just finished migrating `instance.py`

**Step 1:** Update CLAUDE.md
- Change "Last Updated" to today
- Move "Migrate core ALNS algorithm" from "In Progress" to "Completed"
- Add "Create unit tests for Instance class" to "Next Steps"

**Step 2:** Update MIGRATION_LOG.md
- Change "Completed: 0" to "Completed: 1"
- Change "Remaining: 9" to "Remaining: 8"
- Update "Completion Rate: 0%" to "Completion Rate: 11%"
- Add new entry for instance.py with all details
- Mark `instance.py` as [x] in Files Remaining list
- Update "Last Updated" date

**Step 3:** Save both files
