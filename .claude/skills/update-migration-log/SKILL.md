---
name: update-migration-log
description: Update project progress after completing tasks by modifying CLAUDE.md status sections and logging details to MIGRATION_LOG.md. Use when finishing a migration, completing a task, resolving a blocker, or when the user asks to update progress or log work completion.
---

# Update Progress

Quickly update project documentation after completing work to maintain accurate status tracking.

## When to Use

Use this skill after:
- Completing a file migration
- Finishing any task in CLAUDE.md
- Resolving a blocker
- User asks to "update progress"

## Quick Workflow

### 1. Update CLAUDE.md
- Update **"Last Updated"** date at top
- Move completed task from **"In Progress"** to **"Completed"** section
- Update **"Next Steps"** if needed
- Update **"Blockers"** section

### 2. Update MIGRATION_LOG.md (for migrations only)
- Update progress summary numbers
- **Maintain index** (add new entry to index section at top)
- Add migration entry at top of "Migration History" section
- Mark file as completed in checklist
- Update **"Last Updated"** date

### 3. Verify and Commit
- Check updates are correct
- Consider committing with progress message

## Migration Entry Template

For detailed migration logging, use this template (also in MIGRATION_LOG.md):

```markdown
### YYYY-MM-DD - [Original File] → [New Location]

**Status:** ✅ Completed
**Time Spent:** [e.g., 45 minutes]
**Complexity:** Low/Medium/High

**Source:** `[path/to/original/file.py]`
**Destination:** `[path/to/new/file.py]`

#### 📋 Migration Summary
- **Original Purpose:** [Brief description]
- **Target Architecture Layer:** [Problem/Algorithm/Data/Visualization]
- **Key Changes Made:** [2-3 sentence overview]

#### 🔧 Key Changes
- **Extracted hardcoded values:** [What was parameterized]
- **Decoupled from paper logic:** [How generalized]
- **Added/improved:** [Docstrings, type hints, etc.]

#### ⚠️ Issues & Solutions
**Issue 1:** [Description]
- **Solution:** [Fix implemented]
- **Impact:** [Effect on migration]

#### ✅ Verification
- [ ] Code compilation
- [ ] Import tests
- [ ] Runtime tests (if environment available)

#### 📝 Notes
- [Any important observations]
```

## Index Maintenance

To keep MIGRATION_LOG.md manageable as it grows, maintain an index at the top of the file:

### 1. Check for Index Section
If MIGRATION_LOG.md doesn't have an "## Index" section after the "Migration Progress Summary", add one:

```markdown
---

## Index

*Recent migration entries (newest first):*

- [2026-01-01] Phase 2: Test Suite Enhancement for New Architecture (✅ Completed)
- [2025-12-31] Validation test and bug fixes (✅ Completed)
- [2025-12-30] README creation and git push (✅ Completed)
```

### 2. Update Index with New Entry
When adding a new migration entry:

1. **Extract title:** Use the first line of the migration entry (e.g., "### 2026-01-01 - Phase 2: Test Suite Enhancement for New Architecture")
2. **Extract date:** From the title (YYYY-MM-DD)
3. **Extract status:** From the "**Status:**" line in the entry
4. **Add to index:** Insert at the top of the index list (newest first)

**Index entry format:**
```markdown
- [YYYY-MM-DD] [Brief title] ([status symbol] [Status])
```

**Example:**
```markdown
- [2026-01-01] Phase 2: Test Suite Enhancement for New Architecture (✅ Completed)
```

### 3. Keep Index Manageable
- **Limit size:** Keep only the last 10-15 entries in the main index
- **Archive old entries:** If the log grows very large, consider moving older entries to a separate "Archive Index" section
- **Link to entries:** Use Markdown anchor links if needed (e.g., `[2026-01-01](#2026-01-01---phase-2-test-suite-enhancement-for-new-architecture)`)

### 4. Automated Index Updates
When using this skill, the AI should:
1. Check if index exists, create if missing
2. Add new entry to top of index
3. Trim index if it exceeds 15 entries
4. Update the MIGRATION_LOG.md file

## Quick Reference

### Status Symbols
- ✅ **Completed** - Successfully finished
- 🚧 **In Progress** - Currently working
- ⚠️ **Issues** - Done but has known issues
- ❌ **Failed** - Could not complete

### Integration with Other Skills
- After updating: Suggest `build-session-context` to see updated status
- For migrations: Use `migrate-module` skill first
- For setup: Use `manage-python-env` for environment

### Tips
- Be specific: "Extracted `battery_capacity=100` to parameter" not "made configurable"
- Include context: Why decisions were made
- Note surprises: What was easier/harder than expected

## Reference
For detailed templates, see [update_templates.md](references/update_templates.md) or check existing entries in MIGRATION_LOG.md.
