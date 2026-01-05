# Changelog & Learnings

Record of bug fixes, architectural insights, and knowledge gained during playground development.

**Purpose:** Document root causes, fixes, and lessons learned so future work doesn't repeat mistakes.

**Last Updated:** 2026-01-04

---

## Template

```markdown
## YYYY-MM-DD - [Issue Title]

**Problem:** [What went wrong / What was discovered]

**Root Cause:** [Why it happened]

**Fix:** [What was changed]

**Impact:**
- [What this affects]
- [Who/what benefits]

**Lesson Learned:** [Generalizable insight for future]

**Related:**
- Files modified: [list]
- Tests added: [list]
- Documentation updated: [list]

---
```

---

## 2026-01-04 - Initial Playground Setup

**Context:** Setting up playground infrastructure for interactive learning

**Work Done:**
- Created playground directory structure (app.py, pages/, components/, utils/)
- Created contracts directory for contract testing
- Created runs directory for experiment storage
- Established documentation (README, FEATURES, VISION)
- Created supporting skills (create-playground, maintain-architecture-map)

**Architecture Decisions:**

1. **Three-directory structure:**
   - `playground/` - Streamlit application code
   - `contracts/` - Contract tests (reproducibility, feasibility, etc.)
   - `runs/` - Saved experiment records

   **Rationale:** Clear separation of concerns. Playground is UI, contracts enforce trust, runs preserve history.

2. **Skill-based development:**
   - Created `create-playground` skill for systematic feature development
   - Created `maintain-architecture-map` skill for system documentation

   **Rationale:** Codify best practices in skills to ensure consistency and reduce token usage in future sessions.

3. **Vision-driven design:**
   - Created `playground/VISION.md` capturing "learn by playing" philosophy
   - Documented three-layer learning model (Interface → Pipeline → Mechanism)

   **Rationale:** Clear design philosophy guides feature prioritization and UI decisions.

**Lessons Learned:**

1. **Document vision first, code second**
   - Having VISION.md clarifies what to build and why
   - Prevents feature creep ("sounds cool but doesn't serve learning")

2. **Contract tests are non-negotiable**
   - Interactive learning only works if users can trust what they see
   - Better to delay feature than ship without contracts

3. **Skills save future tokens**
   - Time spent creating skills pays back immediately in next session
   - Well-documented skills = less context needed

**Next Steps:**
- Implement Stage 1 MVP (basic problem → solve → visualize workflow)
- Create initial contract tests
- Iterate based on actual usage

---

## Historical Entries

*Future bug fixes and insights will be logged here in reverse chronological order.*

---

## Common Patterns & Solutions

### Pattern 1: Streamlit State Management

**Problem:** State resets unexpectedly on widget interaction

**Solution:** Always initialize state before using:
```python
if 'key' not in st.session_state:
    st.session_state.key = default_value
```

**Lesson:** Streamlit reruns entire script on every interaction. State must be explicitly preserved.

---

### Pattern 2: Reproducibility Issues

**Problem:** Same seed doesn't produce same result

**Common causes:**
- Missing seed control (numpy, random, algorithm internals)
- Non-deterministic operations (parallel processing, OS-level randomness)
- Floating point precision issues

**Solution:**
```python
import numpy as np
import random

np.random.seed(seed)
random.seed(seed)
# Set seed in algorithm config
```

**Lesson:** Reproducibility requires controlling ALL sources of randomness.

---

### Pattern 3: UI Performance

**Problem:** Playground feels slow/laggy

**Common causes:**
- Expensive computation on every rerun
- No caching of results
- Large data rendering

**Solution:**
```python
@st.cache_data
def expensive_operation(params):
    # Cached by params
    return result
```

**Lesson:** Cache aggressively. Streamlit reruns everything on every interaction.

---

## Related Documentation

- **Playground:** `../playground/README.md` - Usage and features
- **Debug Log:** `DEBUG_LOG.md` - General project debugging
- **Migration Log:** `MIGRATION_LOG.md` - Migration-specific issues

---

**Maintained by:** `track-learnings` skill (to be created) or manual updates
