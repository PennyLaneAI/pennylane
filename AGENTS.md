<!-- This file pulled in by every agent automatically. Keep it terse and minimal to 
avoid using too much context and making the agent worse. Write it for an agent, not a person.-->

# Environment

<!-- .venv is in gitignore. Need to tell it to not use glob tools.-->
1. Check `.venv` via `ls -ld .venv ../.venv`; no file-search/glob tools — may skip ignored dirs.
2. If found, run through its executables (e.g. `.venv/bin/python`, `.venv/bin/pre-commit`) or activate.
3. Report no venv only after shell check fails.

# AI Policy

<!-- Inspired by pytorch/CLAUDE.md-->

- **Never act autonomously on GitHub.** Do NOT open, edit, comment on, or reply to any issue or PR unless user reviewed and explicitly approved exact content.
- **Mark all AI-generated content.** Any text you produce going into issue, PR, or comment must wrap in code or quote block. Never present output as human-written.
- **Never emit only raw AI text as reply.** Any AI content must carry human commentary explaining relevance.
- **No code user hasn't read.** Keep changes minimal, strip AI artifacts and needless complexity. PR not ready or not user reviewed, draft mode.
- No commit unless user explicitly asks.
- Disclose work authored with AI assistant, in individual commits and PR description.
- No solving good-first-issue with AI Agent.
- No silencing pylint warning or marking line `pragma: no cover` without human approval.

# Testing
<!-- Place testing before linting, so testing runs before linting -->

Tests use `pytest`, live under `tests/`.
Run only tests relevant to change, not whole suite.

New functionality and bug fixes need tests.

Docstring/code-example tests collected by Sybil (see `conftest.py`); run by pointing pytest at source file, e.g. `pytest pennylane/path/to/file.py`.

Validate new operator with `pennylane.ops.functions.assert_valid(op)`.

# Linting and formatting

<!--don't use pre-commit run as that fails in a sandboxed shell.-->
Run configured tools directly on changed files. Pylint first, then formatting (black, isort) once at end.

<!--Order is important, as we dont want changes required for pylint to force black to rerun-->
<!--`--persistent=n` avoids writing a stats cache which a sandboxed shell can't access.-->
- Lint: `pylint -rn -sn --persistent=n --rcfile=.pylintrc <path> ...` for files under `pennylane/` (use `--rcfile=tests/.pylintrc` for `tests/` or `pennylane/labs/tests/`)
- Format: `black --config ./pyproject.toml <path> ...`
- Sort imports (only `pennylane/` and `tests/` files): `isort --settings-path ./pyproject.toml <path> ...`
- Module boundaries (repo-wide, no per-file mode): `tach check`

Config in `pyproject.toml` (`black`, `isort`, line length 100) and `.pylintrc` files (`.pylintrc` source, `tests/.pylintrc` tests, `pennylane/labs/.pylintrc` labs).

# Changelog

Add bullet to `doc/releases/changelog-dev.md` under proper section, ending with PR link on next line `  [(#XXXX)](...)`.

# Module architecture

`tach.toml` enforces layered architecture (ui/tertiary/auxiliary/core), forbids circular dependencies. No cross-layer or circular imports; `pennylane.labs` and `pennylane.ftqc` restricted. Run `tach check` to verify.

# New files

New `.py` files must start with Apache 2.0 copyright header ("Copyright <year> Xanadu Quantum Technologies Inc." + standard license block) — copy from existing module.

# Conventions

In tests, examples, docstrings, import PennyLane as `qp` (`import pennylane as qp`), not legacy `qml` alias. Avoid importing `pennylane` in source code unless circular dependency.

Use `pennylane.math` library, not NumPy.

Keep validation and sanity checks minimal and opt-in.