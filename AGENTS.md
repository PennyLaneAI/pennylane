<!-- This file pulled in by every agent automatically. Keep it terse and minimal to 
avoid using too much context and making the agent worse. Write it for an agent, not a person.-->

# Environment

<!-- .venv is in gitignore. Need to tell it to not use glob tools.-->
1. Check `.venv` via `ls -ld .venv ../.venv`; no file-search/glob tools — may skip ignored dirs.
2. If found, run through its executables (e.g. `.venv/bin/python`, `.venv/bin/pre-commit`) or activate.
3. Only report no venv after shell check fails.

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
- Disclose that the work was authored with an AI assistant, both in the individual commits and in the PR description.
- Do not solve any issue marked good-first-issue with an AI Agent.
- Do not silence a pylint warning or mark a line as `pragma: no cover` without human approval.

# Testing
<!-- Place testing before linting, so testing runs before linting -->

Tests use `pytest` and live under `tests/`.
Run only the tests relevant to your change rather than the whole suite unless requested to.

New functionality and bug fixes require accompanying tests.

Docstring/code-example tests are collected by Sybil (see `conftest.py`); run them
by pointing pytest at the source file, e.g. `pytest pennylane/path/to/file.py`.

Validate new operator with `pennylane.ops.functions.assert_valid(op)`.

# Linting and formatting

<!--don't use pre-commit run as that fails in a sandboxed shell.-->
Run the configured tools directly on changed files. Run pylint first, then run
formatting (black, isort) once at the end.

<!--Order is important, as we dont want changes required for pylint to force black to rerun-->
<!--`--persistent=n` avoids writing a stats cache which a sandboxed shell can't access.-->
- Lint: `pylint -rn -sn --persistent=n --rcfile=.pylintrc <path> ...` for files
  under `pennylane/` (use `--rcfile=tests/.pylintrc` for files under `tests/` or
  `pennylane/labs/tests/`)
- Format: `black --config ./pyproject.toml <path> ...`
- Sort imports (only `pennylane/` and `tests/` files): `isort --settings-path ./pyproject.toml <path> ...`
- Module boundaries (repo-wide, no per-file mode): `tach check`

Config lives in `pyproject.toml` (`black`, `isort`, line length 100) and the
`.pylintrc` files (`.pylintrc` source, `tests/.pylintrc` tests,
`pennylane/labs/.pylintrc` labs).

# Changelog

Add a bullet to `doc/releases/changelog-dev.md` under the appropriate section,
ending with the PR link on the next line `  [(#XXXX)](...)`.

# Module architecture

`tach.toml` enforces a layered architecture (ui/tertiary/auxiliary/core) and
forbids circular dependencies. Don't add cross-layer or circular imports;
`pennylane.labs` and `pennylane.ftqc` are restricted. Run `tach check` to
verify.

# New files

New `.py` files must start with the Apache 2.0 copyright header
("Copyright <year> Xanadu Quantum Technologies Inc." + the standard license
block) — copy it from an existing module.

# Conventions

In tests, examples, and docstrings, import PennyLane as `qp`
(`import pennylane as qp`), not the legacy `qml` alias. Avoid importing
`pennylane` within source code unless needed to break a circular dependency.

Use framework-agnostic `pennylane.math` library, not NumPy.

Keep validation and sanity checks minimal and opt-in.