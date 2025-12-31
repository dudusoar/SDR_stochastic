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

Add a new entry at the **top** of the Migration History section (most recent first). Use the detailed template below to capture comprehensive migration details:

```markdown
## Migration History

### YYYY-MM-DD HH:MM - [Original File] → [New Location]

**Status:** ✅ Completed / 🚧 In Progress / ⚠️ Issues / ❌ Failed  
**Time Spent:** [e.g., 45 minutes]  
**Migration Complexity:** [Low/Medium/High]

**Source:** `[path/to/original/file.py]`  
**Destination:** `[path/to/new/file.py]`

#### 📋 Migration Summary
- **Original Purpose:** [Brief description of original file's purpose]
- **Target Architecture Layer:** [Problem/Algorithm/Data/Visualization]
- **Key Changes Made:** [2-3 sentence overview]

#### 🔧 Specific Code Changes
**Added/Modified Functions/Classes:**
- [List specific functions/classes changed with brief descriptions]

**Code Snippets (Before/After):**
```python
# Before: [Brief description]
[Code snippet showing original implementation]

# After: [Brief description]
[Code snippet showing refactored implementation]
```

#### 🏗️ Architectural Refactoring
- [ ] **Extracted hardcoded values:** [Describe what was extracted]
- [ ] **Decoupled from paper-specific logic:** [Describe generalization]
- [ ] **Added docstrings:** [Type/style used]
- [ ] **Updated imports:** [Changes made to import statements]
- [ ] **Created test case:** [Test file/function names]
- [ ] **Verified functionality:** [How verification was done]

**Additional Architectural Improvements:**
- [ ] **Type hints:** [Coverage level]
- [ ] **Error handling:** [Improvements made]
- [ ] **Configuration:** [Parameterization added]
- [ ] **Performance optimizations:** [If any were made]

#### ⚠️ Issues Encountered & Solutions
**Issue 1: [Issue Title]**
- **Description:** [Detailed description of the issue]
- **Impact:** [How it affected migration]
- **Solution:** [Specific solution implemented]
- **Rationale:** [Why this solution was chosen]

[Add more issues as needed...]

#### ✅ Verification & Testing
**Tests Created:**
- [List test functions created]

**Verification Steps:**
1. [ ] **Code Compilation:** [Result]
2. [ ] **Import Test:** [Result]
3. [ ] **Type Check:** [Result]
4. [ ] **Runtime Test:** [Result]
5. [ ] **Integration Test:** [Result]

#### 📊 File Statistics
**Original File(s):**
- `[filename]`: [line count] lines

**New File:**
- `[filename]`: [line count] lines

**Changes:**
- **Lines Added:** [number] lines
- **Lines Modified:** [number] lines
- **Lines Removed:** [number] lines

#### 💡 Design Decisions & Rationale
1. **[Decision 1]:** [Rationale]
2. **[Decision 2]:** [Rationale]
3. **[Decision 3]:** [Rationale]

#### 🔮 Follow-up Tasks Identified
- [ ] [Task 1]
- [ ] [Task 2]
- [ ] [Task 3]

#### 📝 Notes & Observations
- [Any additional observations, surprises, or lessons learned]

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

Use this comprehensive checklist to ensure detailed and complete logging:

### Essential Updates
- [ ] Updated progress summary (numbers and percentage)
- [ ] Added migration entry with timestamp (YYYY-MM-DD HH:MM)
- [ ] Marked status (✅ Completed / 🚧 In Progress / ⚠️ Issues / ❌ Failed)
- [ ] Updated files remaining checklist
- [ ] Updated last modified date in both CLAUDE.md and MIGRATION_LOG.md

### Migration Entry Content
- [ ] **Basic Information:** Source, destination, time spent, complexity rating
- [ ] **Migration Summary:** Original purpose, target layer, key changes overview
- [ ] **Specific Code Changes:** Listed functions/classes modified with descriptions
- [ ] **Code Snippets:** Before/after examples showing key refactoring
- [ ] **Architectural Refactoring:** All refactoring checkboxes completed
- [ ] **Issues & Solutions:** Each issue with description, impact, solution, rationale
- [ ] **Verification & Testing:** Tests created and verification steps documented
- [ ] **File Statistics:** Line counts for original and new files
- [ ] **Design Decisions:** Key decisions with rationale explained
- [ ] **Follow-up Tasks:** Identified tasks for future work
- [ ] **Notes & Observations:** Lessons learned and observations

### Quality Checks
- [ ] **Specificity:** Used concrete examples, not vague descriptions
- [ ] **Context:** Explained why decisions were made, not just what was done
- [ ] **Completeness:** All major changes documented
- [ ] **Clarity:** Entry is readable and well-organized
- [ ] **Future Value:** Information would be useful 3-6 months from now

### Examples of Good vs Bad Logging
**Good:** "Extracted `battery_capacity=100` to constructor parameter with default value"
**Bad:** "Made battery configurable"

**Good:** "Moved matplotlib import inside plot_solution() to avoid top-level dependency"
**Bad:** "Fixed import issue"

**Good:** "Added column mapping system to support alternative data formats while maintaining backward compatibility"
**Bad:** "Generalized column names"

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
