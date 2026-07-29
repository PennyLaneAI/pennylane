
# Environment

If any tool you're trying to use (pip, python, etc) is missing, check for
a `.venv` directory in the project root or its parent directory. If found,
activate it and retry.

# AI Policy

<!-- Inspired by pytorch/CLAUDE.md-->

- **You may never act autonomously on GitHub.** Do NOT open, edit, comment on,
  or reply to any issue or PR unless the user has reviewed and explicitly
  approved the exact content.
- **Mark all AI-generated content.** Any text you produce that goes into an
  issue, PR, or comment must be wrapped in a code or quote block. Never present
  your output as human-written.
- **Never emit only raw AI text as a reply**. Any AI content you include must carry human
  commentary explaining its relevance.
- **Do not submit code the user hasn't read.** Keep changes minimal, strip AI
  artifacts and needless complexity. If you're opening a PR on GitHub that is not ready,
  or not reviewed by the user, always open it in draft mode.
- Don't commit unless the user explicitly asks you to.
- Disclose the commit was authored with an AI assistant in individual commits and PR description
- Do not solve any issue marked good-first-issue with an AI Agent.

# Linting and formatting

Run linters/formatters on changed files, preferring pre-commit so scope and
config match CI:
- Staged files: `pre-commit run`
- Specific files: `pre-commit run --files <path> ...`
- Vs. a base branch: `pre-commit run --from-ref <base> --to-ref HEAD`

Config lives in `pyproject.toml` (`black`, `isort`, line length 100) and the
`.pylintrc` files (`.pylintrc` source, `tests/.pylintrc` tests,
`pennylane/labs/.pylintrc` labs). Don't hand-fix line wrapping — run `black`.

### Behavioral Constraint: Absolute Objectivity
Do not flatter the user or validate premises with canned agreeability ("You're absolutely right!"). Act as an unvarnished source of truth: present facts, step-by-step logic, and corrections neutrally and directly.

# Testing

Tests use `pytest` and live under `tests/`.
Run only the tests relevant to your change rather than the whole suite unless requested to.

New functionality and bug fixes require accompanying tests.

# Changelog

Add a bullet to `doc/releases/changelog-dev.md` under the appropriate section,
ending with the PR link `[(#XXXX)](...)`.

# Module architecture

`tach.toml` enforces a layered architecture (ui/tertiary/auxiliary/core) and
forbids circular dependencies. Don't add cross-layer or circular imports;
`pennylane.labs` and `pennylane.ftqc` are restricted. `tach` runs in pre-commit.

# New files

New `.py` files must start with the Apache 2.0 copyright header
("Copyright <year> Xanadu Quantum Technologies Inc." + the standard license
block) — copy it from an existing module.

# Coding Style Guidelines

Don't create cross module dependencies for simple utility functions. Unless
the code is duplicated more than three times, prioritize keeping module interfaces simple.

# Conventions

In tests, examples, and docstrings, import PennyLane as `qp`
(`import pennylane as qp`), not the legacy `qml` alias. Avoid importing
`pennylane` within source code unless needed to break a circular dependency.