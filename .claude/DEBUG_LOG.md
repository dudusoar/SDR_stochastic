# Debug Log - VRP Toolkit Project

*Structured record of problems, bugs, and solutions encountered during development.*

## Active Issues 🚧
*Issues not yet resolved or needing follow-up.*

### Import errors in generators.py during initial testing
**Date Opened:** 2025-12-30
**Last Updated:** 2025-12-30
**Status:** Resolved (moved to Resolved Issues)

**Problem:**
Syntax errors and import issues in generators.py blocking initial tests

**Symptoms:**
- SyntaxError: escaped quotes (`\"`) causing parsing errors
- ImportError when trying to import OrderGenerator and DemandGenerator
- Plotting function with matplotlib dependencies causing issues

**Current Investigation:**
- **Hypothesis 1:** File encoding issues with escaped characters
  - Test: Manual inspection of file content
  - Result: Found multiple escaped quote characters that needed cleaning
- **Hypothesis 2:** Missing dependencies for plotting
  - Test: Simplified plot_instance() method to pass
  - Result: Unblocked imports while preserving interface

**Next Steps:**
1. ~~Clean up escaped characters in generators.py~~
2. ~~Add fallback constants for PDPTWInstance dependencies~~
3. ~~Simplify plotting function to avoid matplotlib dependency issues~~
4. Add proper matplotlib integration when environment is set up

---

## Resolved Issues ✅
*Problems that have been solved.*

### Import errors in generators.py during initial testing
**Date Opened:** 2025-12-30
**Date Resolved:** 2025-12-30
**Resolution Time:** ~30 minutes

**Problem:**
Syntax errors and import issues in generators.py blocking initial tests of migrated code

**Symptoms:**
- `SyntaxError: unexpected character after line continuation character`
- `ImportError: cannot import name 'OrderGenerator' from 'vrp_toolkit.data.generators'`
- Complex string formatting with escaped quotes causing parsing failures

**Root Cause:**
1. **Escaped quote characters:** File contained `\"` sequences that caused syntax errors
2. **Missing fallback constants:** Generators relied on constants from PDPTWInstance class but didn't have local fallbacks
3. **Matplotlib dependency at module level:** plot_instance() method had complex matplotlib code without proper import handling

**Solution:**
1. **Cleaned escaped quotes:** Replaced `\"` with regular double quotes in f-string literals
2. **Added fallback constants:** Defined local constants matching PDPTWInstance constants for standalone use
3. **Simplified plotting function:** Changed plot_instance() to just `pass` temporarily to avoid matplotlib dependency issues
4. **Fixed imports:** Updated data module `__init__.py` to properly export OrderGenerator and DemandGenerator

**Prevention:**
- **Code review for escaped characters:** Check for unnecessary escape sequences
- **Dependency isolation:** Keep matplotlib imports inside functions, not at module level
- **Fallback mechanisms:** Provide local constants for classes that may be used independently

**Lessons Learned:**
- File encoding issues can cause subtle syntax errors that aren't obvious from error messages
- When migrating research code, watch for platform-specific encoding problems
- It's better to simplify and temporarily disable non-essential functionality than to block core imports

**Files Modified:**
- vrp_toolkit/data/generators.py
- vrp_toolkit/data/__init__.py

---

### Unicode encoding issues in test output on Windows
**Date Opened:** 2025-12-30
**Date Resolved:** 2025-12-30
**Resolution Time:** ~15 minutes

**Problem:**
Test scripts using Unicode characters (✓, ❌) causing encoding errors on Windows with GBK codec

**Symptoms:**
- `UnicodeEncodeError: 'gbk' codec can't encode character '\u2713'`
- Test output failing on Windows but working on other platforms
- Inconsistent test results across development environments

**Root Cause:**
Windows default encoding (GBK) doesn't support certain Unicode characters used in test output formatting

**Solution:**
- Replaced Unicode checkmarks (✓) with ASCII `[OK]`
- Replaced Unicode cross marks (❌) with ASCII `[FAIL]`
- Used platform-agnostic ASCII characters for all test output

**Prevention:**
- Use ASCII characters for cross-platform compatibility in test output
- Consider platform encoding differences when designing output formatting
- Test on multiple platforms or use encoding-aware output methods

**Lessons Learned:**
- Always consider cross-platform compatibility for terminal output
- ASCII is safer than Unicode for basic status indicators
- Error messages should be checked on all target platforms

**Files Modified:**
- test_tutorial_migration.py
- test_sensitivity_migration.py
- test_map_migration.py
- test_generators_migration.py
- test_alns_migration.py
- test_pdptw_migration.py

---

## Common Patterns & Solutions 🔧
*Recurring issues and their solutions for quick reference.*

### Pattern 1: Import Chain Failures
**Symptoms:** `ImportError: cannot import name 'X' from 'Y'`, circular import warnings
**Cause:** Missing exports in `__init__.py` files, circular dependencies, or syntax errors in imported modules
**Solution:**
1. Check `__init__.py` exports for missing names
2. Use `from __future__ import annotations` for forward references
3. Move imports inside functions to break circular dependencies
4. Check for syntax errors in the imported module
**Example:** [Import errors in generators.py](#import-errors-in-generatorspy-during-initial-testing)

### Pattern 2: Platform Encoding Issues
**Symptoms:** `UnicodeEncodeError` with specific characters, works on some platforms but not others
**Cause:** Different default encodings across platforms (UTF-8 vs GBK vs CP1252)
**Solution:**
1. Use ASCII characters for cross-platform compatibility
2. Explicitly specify encoding when reading/writing files
3. Use `sys.stdout.reconfigure(encoding='utf-8')` if available
**Example:** [Unicode encoding issues in test output on Windows](#unicode-encoding-issues-in-test-output-on-windows)

### Pattern 3: Research Code Migration Issues
**Symptoms:** Hardcoded values, paper-specific logic, missing configuration
**Cause:** Academic code often contains assumptions and hardcoded parameters
**Solution:**
1. Extract hardcoded values to parameters with sensible defaults
2. Create configuration classes or dataclasses for parameter groups
3. Add type hints and docstrings for better maintainability
4. Preserve original functionality while making it configurable
**Example:** Multiple migrations in MIGRATION_LOG.md demonstrate this pattern

---

**Last Updated:** 2026-01-03
*This file is maintained by the debug-logger skill.*