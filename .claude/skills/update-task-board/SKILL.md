---
name: update-task-board
description: Update CLAUDE.md task status and read logs to maintain accurate project tracking. Use when tasks are completed, when progress is made, or when needing to sync project status across documentation files. Maintains coherent task tracking across all project logs.
---

# Task Manager

Keep CLAUDE.md task status synchronized with actual project progress by reading various log files and updating task tracking accordingly.

## When to Use

Use this skill when:
- Completing a task and needing to update CLAUDE.md
- Starting a new task and wanting to track it properly
- Syncing project status across multiple log files
- Reviewing overall project progress
- Preparing project status reports
- Identifying inconsistencies between logs and task status

## Workflow

### Step 1: Read Current Status
Read all relevant project documentation to understand current state:

**Files to read:**
1. **CLAUDE.md** - Current task status (Completed, In Progress, Next Steps, Blockers)
2. **MIGRATION_LOG.md** - Migration progress and recent entries
3. **DEBUG_LOG.md** - Active and resolved issues
4. **GIT_LOG.md** - Recent commits and changes
5. **Git status** - Uncommitted changes and current branch

### Step 2: Analyze Progress
Compare documented task status with actual progress:

**Check for:**
- **Completed tasks** in CLAUDE.md that have corresponding completion evidence in logs
- **In-progress tasks** that may be completed based on log entries
- **New tasks** mentioned in logs but not tracked in CLAUDE.md
- **Blockers** that may be resolved based on DEBUG_LOG.md entries
- **Migration progress** updates needed based on MIGRATION_LOG.md

### Step 3: Update Task Status
Update CLAUDE.md sections based on analysis:

**Sections to update:**
- **Completed ✅:** Move tasks from "In Progress" here when evidence shows completion
- **In Progress 🚧:** Add new tasks being worked on, remove completed ones
- **Next Steps 📋:** Update based on project priorities and recent progress
- **Blockers 🚫:** Add new blockers, remove resolved ones
- **Current Status:** Update phase if milestones reached
- **Last Updated:** Always update to current date

### Step 4: Verify Consistency
Ensure all documentation files tell a consistent story:

**Cross-reference:**
- Migration entries in MIGRATION_LOG.md should match completed migration tasks
- Resolved issues in DEBUG_LOG.md should match fixed blockers
- Recent commits in GIT_LOG.md should relate to completed tasks
- No contradictions between different status indicators

## CLAUDE.md Parsing Guide

### Task Status Detection
CLAUDE.md uses specific sections for task tracking:

**Completed section:**
```markdown
### Completed ✅
- [x] Task description
```

**In Progress section:**
```markdown
### In Progress 🚧
- [ ] Task description
```

**Next Steps section:**
```markdown
### Next Steps 📋
1. Task description
```

**Blockers section:**
```markdown
### Blockers 🚫
- Description of blocker
```

### Update Patterns

**Marking task as completed:**
1. Find task in "In Progress 🚧" section
2. Move to "Completed ✅" section
3. Change `[ ]` to `[x]`
4. Add completion date or reference if appropriate

**Adding new task:**
1. Determine if task is "In Progress" or "Next Steps"
2. Add to appropriate section with correct formatting
3. Include brief description and context

**Resolving blocker:**
1. Find blocker in "Blockers 🚫" section
2. Remove or mark as resolved
3. Add note about resolution if appropriate

## Log File Integration

### Reading MIGRATION_LOG.md
**Key information to extract:**
- Total files migrated vs. completed (from progress summary)
- Recent migration entries (last 3-5)
- Any issues flagged in migration entries
- Completion percentage

**Action items:**
- Update migration progress in CLAUDE.md if changed
- Mark migration tasks as completed if new entries exist
- Add migration-related blockers if issues reported

### Reading DEBUG_LOG.md
**Key information to extract:**
- Active issues (status: Investigating, Needs Fix, Waiting)
- Recently resolved issues
- Common patterns that might indicate systemic issues

**Action items:**
- Add active issues to CLAUDE.md "Blockers" if significant
- Move resolved issues out of "Blockers" if previously listed
- Consider adding debugging tasks to "In Progress" if major investigation needed

### Reading GIT_LOG.md
**Key information to extract:**
- Recent commit messages (indicate completed work)
- Files modified in recent commits
- Patterns of activity (e.g., lots of documentation commits)

**Action items:**
- Use commit messages to identify completed tasks
- Cross-reference with CLAUDE.md tasks
- Update task status based on commit evidence

## Update Templates

### Task Completion Update
When a task is completed based on log evidence:

**CLAUDE.md update:**
```markdown
### Completed ✅
- [x] Implement unified Solver interface for problem-algorithm separation
  *Completed on 2026-01-01, see migration entry for details*
```

**Additional updates:**
- Update "Last Updated" date
- Remove from "In Progress" if present
- Update any related statistics (e.g., migration progress)

### New Task Addition
When a new task is identified from logs:

**CLAUDE.md update:**
```markdown
### In Progress 🚧
- [ ] Fix matplotlib import issues on Windows
  *Identified from DEBUG_LOG.md, investigating platform compatibility*
```

### Blocker Resolution
When a blocker is resolved:

**CLAUDE.md update:**
```markdown
### Blockers 🚫
- ~~Dependency conflict between numpy 1.24 and pandas 2.0~~ RESOLVED
  *Fixed by pinning numpy to 1.23.5, see DEBUG_LOG.md for details*
```

## Integration with Other Skills

**update-progress:** After migration tasks are completed, task-manager ensures CLAUDE.md reflects the completion.

**git-log:** Commit messages provide evidence for task completion that task-manager can use.

**log-debug-issue:** Active issues become blockers, resolved issues are removed from blockers.

**build-session-context:** Task-manager provides accurate current status for build-session-context to display.

**migrate-module:** When migration completes, task-manager updates CLAUDE.md task status.

## Automation Guidelines

When AI uses this skill, it should:

### Regular Status Sync
1. **Read all logs:** CLAUDE.md, MIGRATION_LOG.md, DEBUG_LOG.md, GIT_LOG.md
2. **Identify discrepancies:** Tasks marked incomplete but evidence shows completion
3. **Update CLAUDE.md:** Move tasks between sections, update status
4. **Maintain consistency:** Ensure all files tell the same story

### Evidence-Based Updates
- **Require evidence:** Only mark tasks complete if logs show completion
- **Cite sources:** Reference specific log entries when updating status
- **Be conservative:** When in doubt, leave task as in-progress

### Proactive Monitoring
- **Check for new work:** Recent commits may indicate unlogged tasks
- **Monitor blockers:** Active issues in DEBUG_LOG.md may need escalation
- **Track progress:** Migration percentage changes should update CLAUDE.md

## Usage Examples

### Example 1: Migration Completion
**Situation:** MIGRATION_LOG.md shows new entry for "instance.py migration completed"

**Task-manager actions:**
1. Read MIGRATION_LOG.md, find new completion entry
2. Check CLAUDE.md for "Migrate instance.py" task
3. Move task from "In Progress" to "Completed"
4. Update migration progress statistics
5. Update "Last Updated" date

### Example 2: Bug Resolution
**Situation:** DEBUG_LOG.md shows "ImportError: cannot import name 'RealMap'" marked as resolved

**Task-manager actions:**
1. Read DEBUG_LOG.md, find resolved issue
2. Check CLAUDE.md "Blockers" for related issue
3. Remove or mark blocker as resolved
4. Add note about fix

### Example 3: New Feature Development
**Situation:** GIT_LOG.md shows recent commits for "feat(visualization): add interactive map plotting"

**Task-manager actions:**
1. Read GIT_LOG.md, identify new feature commits
2. Check CLAUDE.md for corresponding task
3. If not tracked, add to "Completed" or "In Progress" as appropriate
4. Update task description based on commit details

## Maintenance Schedule

### Daily (during active development)
- Quick sync: Check for obvious completions/inconsistencies
- Update based on most recent work

### Weekly
- Comprehensive review of all logs
- Deep consistency check
- Update progress statistics

### Project Milestones
- Full status audit
- Generate progress reports
- Archive completed tasks if CLAUDE.md gets too long

## Troubleshooting

### Inconsistent Status
If logs contradict each other:
1. Check dates/timestamps
2. Look for incomplete log entries
3. Ask user for clarification
4. Default to most recent evidence

### Missing Log Files
If a log file doesn't exist:
1. Create minimal version if needed
2. Note absence in status report
3. Suggest creating it with relevant skill

### Ambiguous Task Boundaries
When it's unclear if a task is complete:
1. Look for explicit completion markers in logs
2. Check if all subtasks are done
3. When in doubt, leave in-progress with note
4. Ask user for clarification