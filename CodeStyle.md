## Code Style

### Activate Virtual Environment Before Running Programs
- There is a `.venv` folder managed by `uv sync` in the project directory
- Activate it before running any programs: `source .venv/bin/activate`

### Control Flow Nesting Must Not Exceed 3 Levels, Function Recursion Must Not Exceed 2 Levels
- Control flow statements that require indentation (`for`, `while`, `if`, etc.) must not nest more than 3 levels when combined with each other

### No Defensive Programming
- Avoid unnecessary safety checks and validations
- Let code fail fast when preconditions are not met. Let exceptions bubble up naturally when appropriate
- Do NOT use try-catch. If you think it's absolutely necessary, ask the user first and only proceed with user approval
- **NEVER** use `dict.get()` with default values - use direct dictionary access: `config['key']`. If you think `dict.get()` is necessary, ask the user first and only proceed with user approval
- When a try-catch or `dict.get()` is approved by the user, add a comment explaining why it's necessary (e.g. `# try-catch approved: lock file may be corrupted, auto-delete and recover`). Without such a comment, the next session will remove it as a code style violation

### Self-Documenting Code Over Comments
- **Good code doesn't need comments** - if control flow is clear and variables are well-named, the code should be self-explanatory
- Reading the code should be easier than reading comments
- However, if your code is complex or unclear, comments are better than no documentation
- Prioritize: Clean Code > Commented Code > Uncommented Complex Code

### Minimize Code - Write Only What's Necessary to Complete the Task, Nothing More

### Import Organization
- **All imports must be placed at the beginning of the file** - imports are not allowed in the middle of code

### Ask User Before Execute
- AI-generated code is often low-quality and cannot be executed directly, so please ask the user before execution and only execute after the user agrees
- When you become confused or uncertain while coding, ask the user for guidance or which approach is better - don't just keep writing blindly


### Reuse Before Reinvent
- Before implementing new functionality, always search the codebase for existing code that can be reused or adapted
- Avoid duplicating logic that already exists elsewhere in the repository

### Minimize Token Usage
- For large copy-paste operations, use the search/replace tool instead of generating the entire content
- For file renaming, use `mv` command instead of generating a new file
- Maximize efficiency to save tokens and user's time

### When Uncertain, Ask First
- If you encounter something unclear or are unsure how to implement it, ask the user immediately - clarify everything before writing code

### NEVER Delete Without Explicit Confirmation
- **NEVER** delete files, directories, or data without explicitly listing what will be deleted and getting user confirmation
- When user says "delete garbage data", ASK which specific files/directories, don't assume

### NEVER Use `git checkout` to Revert Uncommitted Changes
- **NEVER** run `git checkout <file>` or `git restore <file>`
- Uncommitted changes will be **permanently lost** if you checkout/restore

### No Fake Unit Tests
- Fake UTs are meaningless unit tests, typically falling into two categories: paranoid assertions that can never fail, and tautological assertions
- To identify tautological assertions: ask yourself, if I intentionally break the code, would this assertion catch it?

### Unit Tests Must Test Real Code, Not Self-Defined Rules
- Unit tests should test actual functions/classes in the codebase as black boxes
- **WRONG**: Creating tests that define your own rules and then test those rules (self-entertainment)
- **CORRECT**: End-to-end tests that exercise the real function with real inputs and expected outputs
- Example of bad practice:
  ```python
  # BAD: Testing your own interpretation of rules, not the actual code
  class TestRule1_ShortNameMatch:
      def test_exact_match(self):
          # This tests YOUR understanding of rule 1, not the actual is_equal() function
          assert "mod11c3" in "mod11c3 modis terra"  # self-defined logic

  # GOOD: Test the actual function end-to-end
  class TestIsEqual:
      def test_positive_cases(self):
          result = {"short_name": "MOD11C3", "title": "MODIS Terra LST"}
          assert is_equal(result, "MOD11C3 MODIS/Terra LST")  # actual function
  ```

### Think Before You Code - No Bug Loops
- Before making changes, mentally verify that your solution addresses ALL requirements
- Do NOT create a cycle of bugA → bugB → bugC → bugA by rushing to fix one issue while breaking another
- When implementing a feature, list all constraints first, then design a solution that satisfies ALL of them simultaneously
- If you're uncertain whether your approach will work, ask the user before coding

### Define Variables Close to Their Usage
- Variable definitions should be placed immediately before where they are used
- Avoid inserting unrelated code between a variable's definition and its usage

### Inline Single-Use Variables
- Variables used only once should be inlined, unless the variable name adds semantic clarity
- If the expression is complex or non-obvious, a descriptive variable name serves as self-documentation

### Examples

```python
# non-informative comment
# Compute metrics
metrics = {
    "l1_error": torch.mean(l1_error).item(),
    "l2_error": torch.mean(l2_error).item(),
    "l1_error_weighted": torch.mean(weighted_l1).item(),
    "l2_error_weighted": torch.mean(weighted_l2).item(),
    "num_frames": len(l1_error),
}

# non-informative comment
def clear_conversation(self):
    """Clear conversation history."""
    self.conversations = []

# wrong: too many checks
[arg for arg in sys.argv[2:] if "=" in arg]
# correct: simple and pop out error
sys.argv[2:]

# totally wrong: defense programming using wrong parameters and user doees not even know
if torch.distributed.is_initialized():
    world_size = torch.distributed.get_world_size()
else:
    world_size = 1

# wrong: too many checks and hide the error
# correct way: this if is necessary, del it
if arg.startswith("--") and i + 1 < len(sys.argv):
    key = arg[2:]  # 去掉 '--'
    value = sys.argv[i + 1]

# wrong: useless if, complex control flow
if i == j:
    distance_matrix[i, j] = 0.0
else:
    lev_distance = distance.Levenshtein.distance(texts[i], texts[j])
    max_len = max(len(texts[i]), len(texts[j]))
    similarity = 1.0 - (lev_distance / max_len) if max_len > 0 else 1.0
    distance_matrix[i, j] = similarity
    distance_matrix[j, i] = similarity

# wrong: fallback default hides bugs - if sources is empty, something is wrong
return sources or ["BM25"]
# correct: let it fail so we can find the bug
return sources
```
