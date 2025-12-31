# Task Completion Checklist

## After Completing Any Development Task

### 1. Code Quality Checks
```bash
# Format code
black vrp_toolkit/ tests/ examples/

# Run linter
ruff check vrp_toolkit/

# Fix auto-fixable issues
ruff check --fix vrp_toolkit/
```

### 2. Run Tests
```bash
# Run existing tests
pytest

# If new functionality added, write tests
# Test files should be in tests/ directory
```

### 3. Documentation Updates
- [ ] Update docstrings for new/changed public APIs
- [ ] Update README.md if functionality changed
- [ ] Update tutorials if examples changed
- [ ] Update CLAUDE.md status if applicable

### 4. Git Workflow
```bash
# Check what changed
git status
git diff

# Stage changes
git add <modified_files>

# Commit with descriptive message
git commit -m "feat: description of changes"

# Or for migrations:
git commit -m "feat(migration): migrate filename to new location"

# Push if ready
git push origin <branch>
```

## Migration-Specific Completion

### After Migrating a File
1. **Verify migration:**
   - [ ] File copied to new location
   - [ ] Imports updated to new structure
   - [ ] Hardcoded values extracted to configuration
   - [ ] Type hints added where helpful
   - [ ] Documentation improved

2. **Test migrated code:**
   ```bash
   # Run specific test for migrated file
   pytest test_pdptw_migration.py
   
   # Or create new test
   python -c "from vrp_toolkit.problems.pdptw import PDPTWInstance; print('Import successful')"
   ```

3. **Update progress tracking:**
   - [ ] Update CLAUDE.md: Move task from "In Progress" to "Completed"
   - [ ] Update migration count (X/9 files)
   - [ ] Log details in MIGRATION_LOG.md
   - [ ] Use `update-progress` skill if available

### After Completing a Tutorial
1. **Verify tutorial works:**
   - [ ] Run all notebook cells
   - [ ] Check for errors
   - [ ] Verify outputs are correct
   - [ ] Test in clean environment

2. **Update documentation:**
   - [ ] Add/update tutorial in tutorials/ directory
   - [ ] Update README with tutorial link
   - [ ] Update CLAUDE.md progress

## Code Review Checklist (Before Committing)

### General
- [ ] Code follows project style guide
- [ ] No debug prints or commented code left
- [ ] Meaningful variable/function names
- [ ] No hardcoded secrets or paths

### Functionality
- [ ] New code works as intended
- [ ] Edge cases handled
- [ ] Error messages are helpful
- [ ] No breaking changes to existing APIs

### Documentation
- [ ] Public functions/classes have docstrings
- [ ] Complex logic has comments
- [ ] README/tutorials updated if needed

### Testing
- [ ] Existing tests pass
- [ ] New tests added for new functionality
- [ ] Tests cover main use cases

## After Major Features or Phase Completion

### 1. Integration Testing
```bash
# Run full test suite
pytest

# Test installation in clean environment
cd /tmp
python -m venv test_env
test_env\Scripts\activate  # Windows
pip install ../vrp-toolkit/
python -c "import vrp_toolkit; print('Success')"
```

### 2. Update Project Status
- Update CLAUDE.md phase if moving to next phase
- Update migration progress percentage
- Document any architectural decisions made
- Log issues and resolutions in MIGRATION_LOG.md

### 3. Create Release (If Applicable)
```bash
# Update version in pyproject.toml
# Build package
python -m build

# Test installation from built package
pip install dist/vrp_toolkit-*.tar.gz
```

## Quick Reference

### Daily Workflow
1. `session-start` skill to check status
2. Work on task (use `migrate-module` for migrations)
3. Run quality checks (black, ruff, pytest)
4. Commit changes with descriptive message
5. `update-progress` skill to log completion

### Common Issues to Check
- **Import errors:** Check module paths in new structure
- **Test failures:** Ensure tests updated for new API
- **Formatting issues:** Run black to fix
- **Linter warnings:** Address or document intentional violations

### When to Ask for Help
- Blocked for >30 minutes
- Architectural decision needed
- Uncertain about migration approach
- Dependencies conflicting