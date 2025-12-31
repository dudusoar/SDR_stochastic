---
name: update-progress
description: Update project progress after completing tasks by modifying CLAUDE.md status sections and logging details to MIGRATION_LOG.md. Use when finishing a migration, completing a task, resolving a blocker, or when the user asks to update progress or log work completion.
---

# Update Progress

Systematically update project documentation after completing work to maintain accurate status tracking and detailed history.

## When to Use

Trigger this skill after:
- Completing a file migration
- Finishing any task listed in CLAUDE.md
- Resolving a blocker
- Discovering new tasks during work
- User explicitly asks to "update progress" or "log this work"

## Workflow

### Step 1: Gather Completion Information

Ask the user (or determine from context) detailed questions to create comprehensive migration logs:

#### 1. Basic Information
- **Task completed:** What specific task or migration was completed?
- **Files involved:** Which source file(s) and destination file(s)?
- **Time spent:** Approximately how long did it take? (e.g., "45 minutes")
- **Complexity rating:** Low/Medium/High based on refactoring difficulty

#### 2. Migration Summary
- **Original purpose:** What was the original file's purpose/functionality?
- **Target architecture layer:** Which layer does it belong to now? (Problem/Algorithm/Data/Visualization)
- **Key changes overview:** 2-3 sentence summary of what was accomplished

#### 3. Specific Code Changes
- **Modified functions/classes:** List specific functions, classes, or methods that were added or modified
- **Code snippets:** Key before/after code examples showing refactoring
- **Architectural improvements:** Beyond basic refactoring checklist:
  - Type hints added/improved?
  - Error handling enhancements?
  - Configuration parameterization?
  - Performance optimizations?

#### 4. Refactoring Details (Complete Checklist)
- [ ] **Extracted hardcoded values:** What specific values were parameterized?
- [ ] **Decoupled from paper-specific logic:** How was the code generalized?
- [ ] **Added docstrings:** What style was used? Which methods documented?
- [ ] **Updated imports:** How were imports changed? (absolute→relative, dependency management)
- [ ] **Created test case:** What tests were created? (function names, coverage)
- [ ] **Verified functionality:** How was verification done? (compilation, import, runtime tests)

#### 5. Issues & Solutions
For each issue encountered:
- **Description:** What exactly went wrong?
- **Impact:** How did it affect the migration?
- **Solution:** What specific fix was implemented?
- **Rationale:** Why was this solution chosen over alternatives?

#### 6. Verification & Testing
- **Tests created:** List test functions and what they verify
- **Verification steps:** Code compilation, import tests, type checks, runtime tests
- **Test results:** What passed/failed? Any remaining test requirements?

#### 7. File Statistics
- **Original file size:** Approximate line count of source file(s)
- **New file size:** Approximate line count of destination file
- **Changes:** Lines added, modified, removed

#### 8. Design Decisions & Rationale
- **Key decisions:** What architectural or implementation choices were made?
- **Rationale:** Why were these choices made? Trade-offs considered?

#### 9. Follow-up Tasks Identified
- **Immediate follow-up:** Tasks that should be done next
- **Future improvements:** Enhancements identified but not implemented
- **Documentation needs:** What documentation should be added?

#### 10. Notes & Observations
- **Surprises:** What was easier/harder than expected?
- **Lessons learned:** What would you do differently next time?
- **Observations:** Interesting patterns, code quality notes, etc.

### Step 2: Update CLAUDE.md

Follow these updates in order:

1. **Update "Last Updated" date**
   - Change to current date (YYYY-MM-DD)

2. **Move completed task**
   - Find task in "In Progress 🚧" section
   - Move to "Completed ✅" section
   - Keep checkbox format: `- [x] Task description`

3. **Update "In Progress" (if applicable)**
   - Remove completed task
   - Add next task if starting it immediately
   - If no tasks in progress, leave section empty

4. **Update "Next Steps" (if applicable)**
   - Add newly discovered tasks
   - Reorder by priority if needed
   - Remove task if just started (moved to In Progress)

5. **Update or clear "Blockers"**
   - If blocker was resolved, remove it
   - If new blocker found, add it
   - Change to "None currently" if empty

6. **Update Phase** (if milestone reached)
   - Check if phase transition occurred
   - Update status line if needed
   - Phase 1 → Phase 2: All minimal migrations done
   - Phase 2 → Phase 3: Refactoring complete

See [update_templates.md](references/update_templates.md) for detailed examples.

### Step 3: Update MIGRATION_LOG.md

**Only if the completed work was a file migration.**

1. **Update progress summary**
   - Increment "Completed" count
   - Decrement "Remaining" count
   - Recalculate "Completion Rate" percentage
   - Formula: `(Completed / Total Files) * 100`

2. **Add detailed migration entry**
   - Insert at top of "Migration History" (newest first)
   - Use current timestamp: `YYYY-MM-DD HH:MM`
   - Set status: ✅ Completed (or 🚧/⚠️/❌ if other)
   - Use the **detailed migration template** (see [update_templates.md](references/update_templates.md)):
     - Include basic information: time spent, complexity rating
     - Write migration summary with purpose and key changes
     - List specific code changes with function/class names
     - Include code snippets (before/after) when illustrative
     - Complete all architectural refactoring checkboxes
     - Document each issue with description, impact, solution, rationale
     - Detail verification steps and test results
     - Provide file statistics (line counts)
     - Explain design decisions and rationale
     - List follow-up tasks identified
     - Add notes and observations

3. **Update files remaining checklist**
   - Find the migrated file in the list
   - Change `[ ]` to `[x]`

4. **Add to common issues** (if applicable)
   - If new issue type encountered, add it
   - Include solution for future reference

5. **Update "Last Updated" date**
   - Change to current date at bottom of file

See [update_templates.md](references/update_templates.md) for entry template.

### Step 4: Verify and Commit

1. **Read back the changes**
   - Verify CLAUDE.md updates are correct
   - Verify MIGRATION_LOG.md entry is complete
   - Check for any formatting issues

2. **Suggest git commit** (optional)
   - Recommend committing the progress update
   - Suggest commit message like:
     - `"Update progress: completed migration of instance.py"`
     - `"Update progress: finished quickstart tutorial"`

## Quick Reference

### For Non-Migration Tasks

Update **only CLAUDE.md**:
1. Update "Last Updated" date
2. Move task from "In Progress" to "Completed"
3. Update other sections as needed

### For Migration Tasks

Update **both CLAUDE.md and MIGRATION_LOG.md**:
1. Update CLAUDE.md (all steps in Step 2)
2. Update MIGRATION_LOG.md (all steps in Step 3)

## Status Symbols Guide

Use these consistently in logs:
- ✅ **Completed** - Successfully finished
- 🚧 **In Progress** - Currently working
- ⚠️ **Issues** - Done but has known issues
- ❌ **Failed** - Could not complete
- 📋 **Planned** - Scheduled for future

## Integration with Other Skills

**After using this skill:**
- Suggest running `session-start` to see updated status
- If more work to do, suggest next priority task
- If migration complete (9/9), celebrate and suggest Phase 2 planning

**Before using this skill:**
- Should have used `migrate-module` or completed other work
- Have clear understanding of what was accomplished

## Example Scenarios

### Scenario 1: Completed File Migration

**User:** "I just finished migrating instance.py"

**Actions:**
1. Ask about refactoring details and any issues
2. Update CLAUDE.md:
   - Last Updated: Today's date
   - Move "Migrate instance.py" to Completed
3. Update MIGRATION_LOG.md:
   - Progress: 1/9 (11%)
   - Add detailed entry with all information
   - Mark instance.py as [x] in checklist
4. Suggest: "Progress updated! 1/9 files migrated. Next priority: Migrate solution.py?"

### Scenario 2: Non-Migration Task

**User:** "I created the quickstart tutorial"

**Actions:**
1. Update CLAUDE.md only:
   - Last Updated: Today's date
   - Move "Create quickstart tutorial" to Completed
   - Add to Next Steps: "Test quickstart tutorial with fresh install"
2. Suggest: "Progress updated! Want to start on the next task?"

### Scenario 3: Blocker Resolved

**User:** "I fixed the circular import issue"

**Actions:**
1. Update CLAUDE.md:
   - Last Updated: Today's date
   - Remove blocker from Blockers section
   - Change to "None currently"
2. Update MIGRATION_LOG.md (if was migration-related):
   - Add resolution to the issue section
   - Add to Common Issues if helpful
3. Suggest: "Blocker cleared! Ready to continue with the migration?"

## Tips for Effective Logging

### Core Principles
- **Be specific and concrete:** "Extracted `battery_capacity=100` to constructor parameter with default value" not "made battery configurable"
- **Include context and rationale:** Explain why decisions were made, not just what was done. "Chose to keep classes together because..." vs "Moved classes"
- **Document the unexpected:** Note what was easier/harder than expected, surprises, and lessons learned
- **Think future value:** What information would help you or someone else 3-6 months from now?

### Detailed Logging Guidelines
- **Code changes:** List specific functions/methods modified, not just "refactored code"
- **Examples matter:** Include before/after code snippets for key refactoring patterns
- **Issue documentation:** For each issue: description, impact, solution, rationale
- **Quantify:** Use line counts, time estimates, complexity ratings when possible
- **Structure:** Follow the detailed template sections for comprehensive coverage

### Common Pitfalls to Avoid
- ❌ **Vague:** "Improved performance" → ✅ "Reduced distance matrix calculation from O(n²) to O(n log n)"
- ❌ **Generic:** "Fixed bugs" → ✅ "Fixed index out of bounds error in `_extract_demands()` method"
- ❌ **Minimal:** "Added tests" → ✅ "Added `test_instance_creation()`, `test_solution_feasibility()`, covering 85% of critical paths"
- ❌ **Assumptive:** "As expected" → ✅ "Validated assumption: column mapping adds <5% overhead"

### Quality Checklist
- [ ] Would this entry help you understand the changes 6 months from now?
- [ ] Could someone new to the project understand what was done and why?
- [ ] Are there concrete examples illustrating key refactoring?
- [ ] Is the entry organized and easy to scan?
- [ ] Are follow-up tasks clearly identified?

## Reference

For detailed update patterns and templates, see [update_templates.md](references/update_templates.md).
