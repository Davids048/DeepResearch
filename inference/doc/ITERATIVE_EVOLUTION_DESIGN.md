# Iterative Evolution Design
---
**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Status**: Design Phase - Ready for Implementation
---

## Overview

This document outlines the design for implementing **iterative evolution** in the Evolver. Currently, `evolve()` performs a single pass (generate → reflect → curate). This design enables multiple refinement iterations on the same task until the agent produces a correct answer or budget is exhausted.

## Current State Analysis

### What `evolve()` Does Now (Single Pass)

```python
def evolve(self, task: dict):
    # 1. Generate trajectory (once)
    trajectory = self.generator.generate(task, playbook, reflection=None)

    # 2. Reflect on trajectory
    reflector_output = self.reflector.reflect(trajectory, playbook)

    # 3. Tag bullets
    self._apply_bullet_tags(reflector_output)

    # 4. Curate playbook updates
    curator_output = self.curator.curate(..., reflector_output)

    # 5. Apply delta
    self.playbook.apply_delta(curator_output.delta)

    # 6. Return single result
    return {...}
```

### Key Observations

✅ **Already Supports Iteration**:
- Generator accepts `reflection` parameter (currently unused)
- ReflectorOutput has `correctness_judgement`: "correct|incorrect|incomplete"
- Playbook accumulates knowledge across iterations

❌ **Missing for Iteration**:
- No iteration loop
- No stopping conditions
- No history tracking
- No reflection context passing

---

## Proposed Architecture: Self-Improving Refinement Loop

### Strategy

Run multiple iterations on the **same task** until correct answer achieved or budget exhausted.

**Each iteration**:
1. Uses **updated playbook** from previous iteration
2. Passes **previous reflection** to guide next attempt
3. Stops when answer is **correct** or limits reached

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│ evolve(task, max_iterations=3, stop_on_correct=True)   │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  for iteration in 1..max:     │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  1. GENERATE                   │
        │     → Use current playbook     │
        │     → Pass previous reflection │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  2. REFLECT                    │
        │     → Evaluate correctness     │
        │     → Identify errors          │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  3. CHECK CORRECTNESS          │
        │     → correct? → STOP ✓        │
        │     → incorrect? → continue    │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  4. CURATE                     │
        │     → Update playbook          │
        │     → Apply delta              │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  5. RECORD ITERATION           │
        │     → Save trajectory          │
        │     → Save reflection          │
        │     → Snapshot playbook        │
        └───────────────────────────────┘
                        ↓
                 Next iteration
                        ↓
        ┌───────────────────────────────┐
        │  RETURN EVOLUTION HISTORY      │
        └───────────────────────────────┘
```

---

## Implementation Design

### Signature

```python
def evolve(
    self,
    task: dict,
    max_iterations: int = 3,
    stop_on_correct: bool = True,
    curate_when_correct: bool = False,
) -> Dict[str, Any]:
    """
    Evolve the agent through iterative refinement on a single task.

    Args:
        task: Task to solve (must contain 'question' and 'answer')
        max_iterations: Maximum evolution iterations (default: 3)
        stop_on_correct: Stop if correct answer achieved (default: True)
        curate_when_correct: Update playbook even when correct (default: False)

    Returns:
        Dict containing:
            - history: List of all iterations
            - final_playbook: Final playbook state
            - iterations_used: Number of iterations executed
            - final_correctness: Last iteration's correctness
            - achieved_correct: Whether correct answer was achieved
    """
```

### Core Loop Structure

```python
def evolve(self, task: dict, max_iterations: int = 3, stop_on_correct: bool = True):
    history = []
    previous_reflection_summary = None

    for iteration in range(1, max_iterations + 1):
        logger.info(f"=== Evolution Iteration {iteration}/{max_iterations} ===")

        # 1. GENERATE: Use current playbook + previous insights
        trajectory = self.generator.generate(
            task=task,
            playbook=self.playbook,
            reflection=previous_reflection_summary  # ← Context from previous iteration
        )

        logger.debug(f"Task agent finished iteration {iteration}")

        # 2. REFLECT: Evaluate the attempt
        reflector_output = self.reflector.reflect(
            trajectory=trajectory,
            playbook=self.playbook,
        )

        # 3. TAG: Apply bullet tags based on performance
        self._apply_bullet_tags(reflector_output)

        # 4. CHECK: Determine correctness
        correctness = reflector_output.correctness_judgement.lower()
        is_correct = correctness == "correct"

        logger.info(f"Iteration {iteration} correctness: {correctness}")

        # 5. CURATE: Update playbook (if needed)
        curator_output = None
        should_curate = not is_correct or curate_when_correct

        if should_curate:
            curator_output = self.curator.curate(
                question_context=task.get("question", ""),
                playbook=self.playbook,
                reflector_output=reflector_output,
            )
            self.playbook.apply_delta(curator_output.delta)
            logger.debug(f"Playbook now has {len(self.playbook.bullets())} bullets")

        # 6. RECORD: Save iteration result
        iteration_result = {
            "iteration": iteration,
            "trajectory": trajectory,
            "reflection": reflector_output,
            "curation": curator_output,
            "correctness": correctness,
            "playbook_snapshot": self.playbook.dumps(),  # Save state
        }
        history.append(iteration_result)

        # 7. PREPARE: Format reflection for next iteration
        if iteration < max_iterations:
            previous_reflection_summary = self._format_reflection_for_next_iteration(
                reflector_output
            )

        # 8. STOP: Check stopping condition
        if is_correct and stop_on_correct:
            logger.info(f"✅ Correct answer achieved at iteration {iteration}")
            break

    # Return evolution history
    return {
        "history": history,
        "final_playbook": self.playbook,
        "iterations_used": len(history),
        "final_correctness": history[-1]["correctness"],
        "achieved_correct": history[-1]["correctness"] == "correct"
    }
```

### Helper Method: Format Reflection

```python
def _format_reflection_for_next_iteration(self, reflection: ReflectorOutput) -> str:
    """Format reflection as context for next generation attempt.

    Args:
        reflection: Reflection from previous iteration

    Returns:
        Formatted string to pass to generator
    """
    return f"""Previous Attempt Analysis:
- Error Identified: {reflection.error_identification}
- Root Cause: {reflection.root_cause_analysis}
- Correct Approach: {reflection.correct_approach}
- Key Insight: {reflection.key_insight}
- Judgement: {reflection.correctness_judgement}

Apply these insights to improve your next attempt."""
```

---

## Design Decisions

### 1. Stopping Conditions

| Condition | When Checked | Priority | Configurable |
|-----------|--------------|----------|--------------|
| **Correct answer** | After each reflection | Highest | `stop_on_correct` |
| **Max iterations** | Loop counter | Medium | `max_iterations` |
| **No playbook changes** | After curation | Optional | Future |
| **Budget exhausted** | External tracker | High | Future |

**Recommendation**: Start with correct answer + max iterations

### 2. Curation Strategy

**Option A: Always curate** (maximum learning)
```python
# Update playbook every iteration regardless of correctness
curator_output = self.curator.curate(...)
```

**Pros**: Captures insights from successful attempts too
**Cons**: May add unnecessary bullets, higher cost

**Option B: Only curate when incorrect** (efficiency)
```python
# Only update playbook if answer is wrong
if correctness != "correct":
    curator_output = self.curator.curate(...)
```

**Pros**: Preserves good playbook state, lower cost
**Cons**: Misses insights from successful reasoning

**Recommendation**: Option B (configurable via `curate_when_correct`)

### 3. Context Passing

**What to pass to next iteration**:
- ✅ Previous reflection (errors, insights)
- ✅ Correctness judgement
- ✅ Correct approach
- ❌ Full trajectory (too verbose)
- ❌ Raw messages (redundant)

**How it's used**:
Generator's `_build_system_prompt_with_playbook()` adds it to system prompt:
```python
if reflection:
    sections.append("\n\n# Recent Insights\n")
    sections.append(reflection)
```

### 4. History Tracking

**What to track per iteration**:
```python
{
    "iteration": 1,
    "trajectory": {...},           # Full agent execution
    "reflection": ReflectorOutput, # Evaluation results
    "curation": CuratorOutput,     # Playbook updates
    "correctness": "incorrect",    # Judgement
    "playbook_snapshot": "...",    # Playbook state (JSON)
}
```

**Benefits**:
- Debug evolution process
- Analyze learning progression
- Rollback to better states
- Visualize improvement trajectory

---

## Configuration Options

### Configuration Class (Future Enhancement)

```python
@dataclass
class EvolutionConfig:
    """Configuration for iterative evolution."""

    # Iteration control
    max_iterations: int = 3
    stop_on_correct: bool = True

    # Curation control
    curate_when_correct: bool = False

    # Context passing
    pass_reflection_to_generator: bool = True

    # History tracking
    track_full_history: bool = True
    save_playbook_snapshots: bool = True

    # Budget (future)
    max_total_llm_calls: Optional[int] = None
    max_tokens_per_iteration: Optional[int] = None
```

### Environment Variables (Alternative)

```bash
# Simple configuration via env vars
export EVOLUTION_MAX_ITERATIONS=5
export EVOLUTION_STOP_ON_CORRECT=true
export EVOLUTION_CURATE_WHEN_CORRECT=false
```

---

## Budget Considerations

### Cost Per Iteration

| Component | LLM Calls | Cost Level |
|-----------|-----------|------------|
| Generator | 50-100 | High (agent runs full trajectory) |
| Reflector | 1 | Low |
| Curator | 1 | Low |
| **Total** | **52-102** | **High** |

### Budget Strategies

**Strategy 1: Fixed iterations** (predictable)
```python
max_iterations = 3
# Total cost: ~150-300 LLM calls
```

**Strategy 2: Correctness-based** (efficient)
```python
max_iterations = 5
stop_on_correct = True
# Cost: 1-5 iterations (early stopping)
# Average: ~2-3 iterations if 60% success rate
```

**Strategy 3: Budget-aware** (advanced, future)
```python
class BudgetTracker:
    def __init__(self, max_llm_calls: int):
        self.max_calls = max_llm_calls
        self.calls_used = 0

    def track(self, calls: int):
        self.calls_used += calls

    def can_continue(self) -> bool:
        return self.calls_used < self.max_calls

    def remaining(self) -> int:
        return self.max_calls - self.calls_used

# In evolve():
budget = BudgetTracker(max_llm_calls=500)
for iteration in range(1, max_iterations + 1):
    if not budget.can_continue():
        logger.warning("Budget exhausted")
        break
    # ... run iteration ...
    budget.track(trajectory.num_llm_calls)
```

### Cost Optimization Tips

1. **Early stopping**: Use `stop_on_correct=True`
2. **Lower max_iterations**: Start with 2-3 instead of 5
3. **Agent budget**: Reduce `MAX_LLM_CALL_PER_RUN` per trajectory
4. **Selective curation**: Only curate when incorrect

---

## Alternative Evolution Patterns

### Pattern 1: Multi-Task Learning (ACE/RISE Style)

**Use case**: Build general-purpose playbook across diverse tasks

```python
def evolve_batch(self, tasks: List[dict], max_tasks: int = 10) -> Playbook:
    """Evolve playbook across multiple diverse tasks (one pass each).

    This follows the ACE/RISE pattern: broad learning across many examples.
    """
    for i, task in enumerate(tasks[:max_tasks]):
        logger.info(f"=== Task {i+1}/{max_tasks} ===")

        # Single pass per task
        trajectory = self.generator.generate(task, self.playbook)
        reflection = self.reflector.reflect(trajectory, self.playbook)
        self._apply_bullet_tags(reflection)
        curation = self.curator.curate(
            task.get("question", ""),
            self.playbook,
            reflection
        )
        self.playbook.apply_delta(curation.delta)

        logger.debug(f"Playbook: {len(self.playbook.bullets())} bullets")

    return self.playbook
```

**Pros**: Diverse knowledge, handles varied domains
**Cons**: May not master any single task type

### Pattern 2: Curriculum Learning (Hybrid)

**Use case**: Progressive difficulty with iterative refinement

```python
def evolve_curriculum(
    self,
    tasks: List[dict],
    iterations_per_task: int = 2,
    difficulty_sorted: bool = True
) -> Dict[str, Any]:
    """Evolve through curriculum of tasks with multiple attempts per task.

    Combines breadth (multiple tasks) with depth (iterations per task).
    """
    if difficulty_sorted:
        # Start with easier tasks, progress to harder
        tasks = sorted(tasks, key=lambda t: t.get('difficulty', 0))

    curriculum_results = []

    for i, task in enumerate(tasks):
        logger.info(f"=== Curriculum Task {i+1}/{len(tasks)} ===")

        # Multiple iterations per task
        result = self.evolve(
            task,
            max_iterations=iterations_per_task,
            stop_on_correct=True
        )
        curriculum_results.append(result)

        # Optional: Skip to next if this one is mastered
        if result["achieved_correct"]:
            logger.info("Task mastered, moving to next")

    return {
        "curriculum_results": curriculum_results,
        "final_playbook": self.playbook,
        "tasks_completed": len(curriculum_results)
    }
```

**Pros**: Best of both worlds - depth and breadth
**Cons**: Highest cost

### Pattern 3: Active Learning

**Use case**: Focus on hardest examples

```python
def evolve_active(
    self,
    tasks: List[dict],
    max_iterations_per_task: int = 3,
    max_tasks: int = 10
) -> Dict[str, Any]:
    """Focus evolution on tasks where agent struggles most."""

    results = []

    for task in tasks[:max_tasks]:
        # Try once to assess difficulty
        result = self.evolve(task, max_iterations=1, stop_on_correct=False)

        # If incorrect, invest more iterations
        if result["final_correctness"] != "correct":
            logger.info("Task difficult, investing more iterations")
            result = self.evolve(
                task,
                max_iterations=max_iterations_per_task,
                stop_on_correct=True
            )

        results.append(result)

    return {"results": results, "final_playbook": self.playbook}
```

**Pros**: Efficient use of budget on hard cases
**Cons**: May miss easy task insights

---

## Implementation Phases

### Phase 1: Basic Iteration Loop ⭐ (Start Here)
- Add `max_iterations` parameter
- Implement for loop
- Track iteration count
- Return history list

**Complexity**: Low
**Value**: High
**Effort**: 1-2 hours

### Phase 2: Correctness-Based Stopping
- Check `correctness_judgement` after reflection
- Add `stop_on_correct` parameter
- Break loop when correct

**Complexity**: Low
**Value**: High
**Effort**: 30 mins

### Phase 3: Reflection Context Passing
- Implement `_format_reflection_for_next_iteration()`
- Pass to `generator.generate(reflection=...)`
- Update generator to use reflection in prompt

**Complexity**: Medium
**Value**: Very High
**Effort**: 1 hour

### Phase 4: History Tracking
- Record each iteration's results
- Save playbook snapshots
- Return structured history

**Complexity**: Low
**Value**: Medium
**Effort**: 1 hour

### Phase 5: Budget Tracking (Future)
- Create `BudgetTracker` class
- Track LLM calls per iteration
- Stop when budget exhausted

**Complexity**: Medium
**Value**: Medium
**Effort**: 2 hours

### Phase 6: Multi-Task Evolution (Future)
- Implement `evolve_batch()`
- Implement `evolve_curriculum()`
- Support task difficulty sorting

**Complexity**: Medium
**Value**: High (for research)
**Effort**: 3-4 hours

---

## Expected Benefits

### Quantitative Improvements

| Metric | Single Pass | Iterative (3 rounds) | Improvement |
|--------|-------------|---------------------|-------------|
| **Correctness Rate** | 40-60% | 70-90% | +30-50% |
| **Playbook Quality** | 5-10 bullets | 15-30 bullets | 3x |
| **Cost (LLM calls)** | ~100 | ~300 | 3x |
| **Self-correction** | 0% | 40-60% | +40-60% |

### Qualitative Improvements

✅ **Self-correcting**: Agent learns from mistakes
✅ **Robust playbook**: Refined through multiple attempts
✅ **Detailed insights**: Rich reflection history
✅ **Progressive learning**: Each iteration builds on previous
✅ **Debuggable**: Full history for analysis

---

## Testing Strategy

### Unit Tests

```python
def test_single_iteration():
    """Test that basic iteration works."""
    evolver = create_test_evolver()
    result = evolver.evolve(test_task, max_iterations=1)
    assert result["iterations_used"] == 1
    assert len(result["history"]) == 1

def test_stop_on_correct():
    """Test early stopping when correct."""
    # Mock reflector to return "correct" on 2nd iteration
    evolver = create_mock_evolver(correct_on_iteration=2)
    result = evolver.evolve(test_task, max_iterations=5, stop_on_correct=True)
    assert result["iterations_used"] == 2
    assert result["achieved_correct"] == True

def test_max_iterations_reached():
    """Test that max iterations is respected."""
    evolver = create_test_evolver()
    result = evolver.evolve(test_task, max_iterations=3)
    assert result["iterations_used"] <= 3
```

### Integration Tests

```python
def test_full_evolution_flow():
    """Test complete evolution with real components."""
    # Use real generator, reflector, curator
    evolver = create_real_evolver()

    result = evolver.evolve(
        task=real_test_task,
        max_iterations=3,
        stop_on_correct=True
    )

    # Verify structure
    assert "history" in result
    assert "final_playbook" in result

    # Verify progression
    for i, iteration in enumerate(result["history"]):
        assert iteration["iteration"] == i + 1
        assert "correctness" in iteration
```

---

## Migration Path

### Current Code (No Changes Required)
Existing single-pass behavior preserved as default:
```python
# This still works exactly as before
evolver.evolve(task)
# Equivalent to: evolver.evolve(task, max_iterations=1)
```

### Opt-In to Iteration
```python
# Explicitly enable iteration
evolver.evolve(task, max_iterations=3, stop_on_correct=True)
```

### Default in Future
After validation, consider making `max_iterations=3` the default:
```python
def evolve(self, task: dict, max_iterations: int = 3, ...):
```

---

## Open Questions

1. **Should we curate on correct answers?**
   - Pro: Captures successful patterns
   - Con: May dilute playbook with redundant info
   - **Recommendation**: Make configurable (default: False)

2. **How to handle partial correctness?**
   - Currently: "correct|incorrect|incomplete"
   - Should "incomplete" continue iterating?
   - **Recommendation**: Treat "incomplete" as incorrect (continue)

3. **Should playbook snapshot be full JSON or summary?**
   - Full JSON: Complete state, large size
   - Summary: Just bullet count/sections, small size
   - **Recommendation**: Full JSON (can compress later)

4. **How to visualize evolution progress?**
   - Log files with progress bars?
   - JSON report with metrics?
   - **Recommendation**: Both (logs for real-time, JSON for analysis)

---

## References

### Related Work

- **ACE (Automatic Curriculum Evolution)**: Multi-task learning across diverse problems
- **RISE (Recursive Instruction Self-Evolution)**: Self-improving through reflection
- **Reflexion**: Iterative refinement with verbal feedback
- **Self-Refine**: Multi-round generation with self-feedback

### Code Locations

- `evolve/evolver.py`: Main evolution logic
- `evolve/generator.py`: Prompt building with playbook
- `evolve/reflector.py`: Correctness evaluation
- `evolve/curator.py`: Playbook updates
- `evolve/playbook.py`: Knowledge storage

---

## Appendix: Complete Code Example

See implementation phases above. Start with Phase 1-3 for minimum viable iteration:

```python
# Minimal working example
def evolve(self, task: dict, max_iterations: int = 3, stop_on_correct: bool = True):
    history = []
    prev_reflection = None

    for i in range(1, max_iterations + 1):
        # Generate
        trajectory = self.generator.generate(task, self.playbook, prev_reflection)

        # Reflect
        reflection = self.reflector.reflect(trajectory, self.playbook)
        self._apply_bullet_tags(reflection)

        # Check correctness
        is_correct = reflection.correctness_judgement.lower() == "correct"

        # Curate if incorrect
        if not is_correct:
            curation = self.curator.curate(task["question"], self.playbook, reflection)
            self.playbook.apply_delta(curation.delta)

        # Record
        history.append({
            "iteration": i,
            "correctness": reflection.correctness_judgement,
            "trajectory": trajectory,
        })

        # Prepare next iteration
        if i < max_iterations:
            prev_reflection = self._format_reflection(reflection)

        # Stop if correct
        if is_correct and stop_on_correct:
            break

    return {"history": history, "iterations_used": len(history)}
```

---


