# Ruff and Pyright Quality Gate Learning Guide

This quality gate makes editor warnings reproducible in the terminal. Ruff checks source structure and formatting, Pyright applies the same project type configuration used by Pylance, and pytest proves runtime behavior. The value comes from keeping those responsibilities distinct while giving every developer one shared configuration.

## The 80/20 View

Five ideas explain most of this change:

1. `pyproject.toml` is the single authority for command-line and editor checks;
2. Ruff, Pyright, and pytest answer different questions;
3. useful type checking turns dynamic framework boundaries into explicit contracts;
4. protocols describe the small interface the application needs;
5. a quality gate is useful only when it is fast, repeatable, and honest about its limits.

The development dependencies and tool settings live in [`pyproject.toml`](../pyproject.toml). VS Code discovers the same project through [`.vscode/settings.json`](../.vscode/settings.json) and recommends the relevant extensions through [`.vscode/extensions.json`](../.vscode/extensions.json). Verified developer commands are recorded in [`README.md`](../README.md).

## 1. Keep one configuration authority

The development extra now installs all three local verification tools:

```toml
[project.optional-dependencies]
dev = [
    "pyright>=1.1.411,<2",
    "pytest>=9,<10",
    "ruff>=0.16.3,<0.17",
]
```

Ruff and Pyright are configured in the same file. Ruff targets Python 3.14 and selects import, correctness, modernization, and Ruff-specific rules. Pyright checks `src` and `tests`, resolves packages from `.venv`, targets Python 3.14, and uses `standard` type-checking mode.

The VS Code settings do not repeat `python.analysis.typeCheckingMode`. Pylance refuses that duplication when it already reads a `pyrightconfig.json` or `[tool.pyright]` section. Instead, the workspace selects the project interpreter and Ruff configuration, while `pyproject.toml` owns type-checking policy.

```text
pyproject.toml
   ├── Ruff CLI + Ruff extension
   └── Pyright CLI + Pylance extension
```

This removes “works in my editor” drift. A rule change made in the repository affects both local commands and supported editor integrations.

**Transferable lesson:** choose one repository-owned configuration source and make every frontend consume it rather than copying settings between tools.

## 2. Ruff, Pyright, and pytest prove different things

The quality gate has three independent layers:

| Tool | Main question | Example finding |
|---|---|---|
| Ruff lint | Is the Python source structurally clean and consistent? | unsorted imports, unused imports, obsolete typing imports |
| Ruff format | Does every file match one deterministic layout? | line wrapping and whitespace differences |
| Pyright/Pylance | Are values used according to their declared contracts? | passing arbitrary `object` to `list()` without proving it is iterable |
| pytest | Does executed behavior produce the expected result? | a tracked box becomes the correct immutable `Detection` |

No layer replaces another. Ruff can approve code with an incorrect return type. Pyright can approve code whose geometry algorithm is logically wrong. Pytest exercises chosen examples but does not inspect every type path or formatting rule.

The verified sequence is:

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pyright
.venv/bin/python -m pytest
```

The checks are intentionally separate commands. A failure immediately identifies whether the problem is lint, formatting, static typing, or runtime behavior.

**Transferable lesson:** combine complementary proof boundaries and preserve the name of the boundary that failed.

## 3. Static typing should strengthen dynamic boundaries

The original Pylance warning came from `_as_list()` in [`src/roundabout_ai/detector.py`](../src/roundabout_ai/detector.py). Ultralytics exposes tensor-like objects dynamically, so the adapter accepted `object`. The fallback then called `list(value)`, but an arbitrary object does not promise `__iter__`.

The repaired boundary validates that promise at runtime:

```python
if isinstance(value, Iterable):
    return list(value)
raise TypeError("model output is not iterable")
```

The `tolist()` branch applies the same rule to the converted result. `_number()` similarly accepts only values proven to be `int` or `float`. Box rows must be sequences with exactly four coordinates before they become `Detection.xyxy`.

These guards do more than silence Pyright. They turn an unclear downstream error into an immediate adapter error if a future model returns an unexpected shape or value. The public application type remains precise:

```text
dynamic model output
    -> validate iterable / numeric / four-coordinate shape
    -> Detection(class_id, label, confidence, xyxy, track_id)
```

The important boundary is not “make every third-party object perfectly typed.” It is “validate enough at the integration edge that internal code can rely on ordinary Python types.”

**Transferable lesson:** when static analysis cannot prove a dynamic value is safe, add a real runtime check at the boundary instead of hiding the diagnostic with an unchecked ignore.

## 4. Protocols describe the interface the project owns

The project does not need every OpenCV or Ultralytics method. It needs small slices:

- `VideoCaptureLike` in [`src/roundabout_ai/capture.py`](../src/roundabout_ai/capture.py) describes opening, reading, configuring, and releasing a camera;
- `ModelLike` in [`src/roundabout_ai/detector.py`](../src/roundabout_ai/detector.py) describes model names plus prediction and tracking calls.

These protocols support production integrations and lightweight test doubles. Positional-only parameters on `VideoCaptureLike.open()` and `.set()` reflect how the project calls those methods without imposing irrelevant parameter names on OpenCV or fakes. `ModelLike.names` is a read-only property accepting a mapping or sequence, which matches the information the adapter consumes without requiring ownership of the framework container.

Third-party stubs are sometimes broader than the project contract. `_opencv_capture_factory()` and the YOLO constructor use narrow `cast()` calls at the exact framework edge. A cast does not validate anything at runtime, so it is appropriate only where the actual library is known to provide the protocol and the mismatch comes from overloaded external type declarations. Dynamic model payloads still receive runtime checks before entering domain types.

The tests in [`tests/test_capture.py`](../tests/test_capture.py) and [`tests/test_detector.py`](../tests/test_detector.py) now declare the same frame and model contracts as production code. This makes the fakes proof of the adapter interface rather than loosely shaped objects that happen to work at runtime.

**Transferable lesson:** define the smallest application-owned protocol, cast only at a trusted framework construction edge, and validate untrusted payload values before conversion.

## 5. Editor convenience should reproduce a terminal command

Pylance provides immediate feedback while editing, but it is a VS Code extension rather than a project command. Pyright supplies the repeatable terminal check using the same `[tool.pyright]` policy. Ruff follows the same pattern: its extension gives live diagnostics, while `.venv/bin/ruff` is the portable command.

The workspace recommends, but does not silently install:

```text
charliermarsh.ruff
ms-python.python
ms-python.vscode-pylance
```

It also selects `${workspaceFolder}/.venv/bin/python`. This matters because static analysis must resolve the actual OpenCV, NumPy, PyYAML, PyTorch, Ultralytics, pytest, and project packages. Before `venvPath` and `venv` were configured, Pyright reported many missing imports and cascading invalid-type errors that were environment problems rather than code problems.

The caches and virtual environment remain local. `.ruff_cache/` is excluded in [`.gitignore`](../.gitignore), just like `.pytest_cache/` and `.venv/`.

**Transferable lesson:** treat an editor diagnostic as useful feedback, but require a repository command that reproduces it with the intended interpreter and dependencies.

## Execution Flow

Installation establishes one reproducible developer environment:

```text
pip install -e '.[dev]'
          │
          ├── runtime dependencies
          └── pytest + Ruff + Pyright
```

Both editor and terminal then converge on project configuration:

```text
                         pyproject.toml
                       /                \
                      v                  v
          Ruff extension / CLI    Pylance / Pyright CLI
                      \                  /
                       v                v
                         source + tests
                                │
                                v
                         pytest execution
```

For the original detector warning, the concrete path is:

```text
Pylance flags list(object)
        -> Pyright reproduces reportArgumentType
        -> adapter checks Iterable at runtime
        -> Pyright proves the narrowed branch
        -> detector tests prove expected conversion behavior
```

## What the Checks Prove

The verified quality gate currently reports:

```text
Ruff lint:    all checks passed
Ruff format:  16 files already formatted
Pyright:      0 errors, 0 warnings, 0 informations
pytest:       31 passed
```

Together these results prove that:

- imports and selected Python correctness rules satisfy the shared Ruff policy;
- source and tests have deterministic Ruff formatting;
- production and test code satisfy standard-mode type checking under Python 3.14;
- external packages resolve through the project virtual environment;
- dynamic detector values are checked before iterable and numeric conversion;
- OpenCV and model test doubles conform to the application protocols;
- all existing behavioral examples still pass after type-boundary changes.

They do not prove:

- that every possible Ruff rule is enabled;
- that Pyright `strict` mode would pass;
- that third-party type stubs perfectly describe native runtime behavior;
- that a `cast()` is true at runtime;
- model accuracy, tracker stability, or live crossing-count accuracy;
- behavior not represented by the 31 tests.

The quality gate protects developer feedback and code contracts. Phase-specific live acceptance criteria remain separate.

## Try It

### Run the complete local gate

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pyright
.venv/bin/python -m pytest
```

Predict first: all four commands should pass in the configured `.venv`. If Pyright reports missing imports, verify VS Code and `pyproject.toml` are using the same virtual environment before changing annotations.

### Experiment: let Pyright narrow a value

In a temporary edit to `_as_list()`, remove only the `isinstance(value, Iterable)` condition while leaving `return list(value)`. Run:

```bash
.venv/bin/pyright
```

Predict first: Pyright should reproduce the original `reportArgumentType` diagnostic because `object` does not guarantee iteration. Restore the condition and rerun; the diagnostic should disappear. The experiment changes no stored data and should not be committed.

### Experiment: separate lint from formatting

Temporarily reverse two standard-library imports in one file, then run:

```bash
.venv/bin/ruff check src tests
```

Predict first: Ruff should report `I001`. Running `ruff format` alone may not organize the imports because lint fixes and formatting have distinct responsibilities. Use this safe repair:

```bash
.venv/bin/ruff check --fix src tests
.venv/bin/ruff format src tests
```

Review the diff before keeping any automatic fix.

## Continuous-Learning Loop

Use this loop whenever an editor reports a new quality warning:

1. **Define the user-visible goal.** Make the warning reproducible and keep runtime behavior unchanged.
2. **Name the enabling concept.** Shared configuration plus the correct proof boundary.
3. **Implement the smallest useful behavior.** Add a truthful annotation, protocol, runtime guard, or deterministic formatting fix.
4. **Prove it at the cheapest meaningful boundary.** Run the focused tool first, then the complete local gate.
5. **Explain what failures revealed.** Separate environment resolution, external stub mismatch, internal contract errors, formatting, and behavioral regression.
6. **Record the transferable lesson.** Fix the source of uncertainty rather than merely suppressing its symptom.

The reusable pattern is:

```text
editor warning -> terminal reproduction -> classify the boundary
               -> smallest truthful fix -> static proof + runtime proof
```
