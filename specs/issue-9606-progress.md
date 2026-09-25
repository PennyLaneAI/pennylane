# Spec Progress: issue-9606

## Phases
- [x] Phase 1: light spec file data fetched → findings.md
      Evidence: specs/9606.md read in full (39 lines) → copied verbatim to specs/issue-9606-findings.md; verified line-by-line against source read output.
- [x] Phase 2: Requirements extracted
      Evidence: User story, 6 acceptance criteria, 5 functional requirements, 5 non-functional requirements + out-of-scope extracted from findings.md lines 3–36 into specs/issue-9606-spec.md §Requirements.
- [x] Phase 3: Technical spec drafted
      Evidence: read pennylane/data/data_manager/__init__.py (557 lines — bug isolated at lines 202–217); grep of tests/data/data_manager/test_dataset_access.py (mock infra lines 155/196/407/499, pytestmark line 56); grep "wait_mock|as_completed|futures.wait" tests/ → only unused wait_mock_fixture:218; grep "_download_datasets|num_threads" tests/ → 0 hits (no existing concurrency tests); ls progress/ → subpackage confirmed; decisions.md absent. Spec §Technical Specification written incl. proposed change block + API contract.
      (Note: an intermediate edit dropped the Out-of-Scope section; it was restored immediately — logged below.)
- [x] Phase 4: Implementation plan written
      Evidence: 7 sub-tasks (complexities 2/3/2/2/1/1/1) with dependency chain 1→{2→3,4,5}→7 and 6 independent; complexity rationale given (test work dominates; core loop rewrite is ~10 lines); 5 risks with mitigations. Written from verified code/test/changelog line references gathered in Phase 3.
- [x] Phase 5: Test strategy written
      Evidence: 4 test types covered — unit (4 new + 2 unmodified-primitive checks), integration (test_load matrix + -m data run), E2E (manual reproducer + interactive smoke), edge cases (7 enumerated incl. num_threads=1, cap>datasets, pbar=None generator, empty-list upstream guard). Test commands sourced from AGENTS.md + interface-unit-tests.yml:674; execution deferred to implementer (sandbox denial logged in Errors).
- [x] Phase 6: Definition of Done written
      Evidence: 8 DoD items mapped to plan sub-tasks 1–7 (AC1–6↔ST1–5; lint↔ST7; changelog↔ST6); "API docs N/A" claim backed by post-write grep: `num_threads` occurs only in pennylane/data/data_manager/__init__.py (lines 170/202/243/257) and doc/**/*.rst has 0 hits for _download_datasets|num_threads. Rollout note added (ships in 0.46.0 dev; single-commit revert).
- [x] Post-Phase 6: decisions.md evaluation
      Evidence: NO entry written. Rationale: the chosen pattern (bulk submit + as_completed; rejected cpu_count cap) is a local scheduling fix whose alternatives are already durably recorded in specs/issue-9606-findings.md §"Alternatives considered"; no repo-wide or cross-feature ruling was introduced ("Would another team benefit?" → already answered by the findings file + code itself). Per skill: not every spec produces a decision entry.

## Verification status
- pytest execution: NOT possible in analyst sandbox (bash allowlist: ls/cat/find/rg/grep only).
  Implementer must run the Test Strategy commands and paste results as Evidence here.

## Errors
| Phase | Error | Attempt | Resolution |
|-------|-------|---------|------------|
| 3 | `.venv/bin/python -m pytest tests/data/data_manager/test_dataset_access.py -x -q` denied by sandbox bash policy (only ls/cat/find/rg/grep allowed) | 1 | Changed approach: verification evidence in Test Strategy/DoD recorded as explicit negatives; test execution deferred to `/implement-spec` phase where the implementer runs pytest. |
| 3 | `ls ... && wc -l ...` compound command denied (wc not in allowlist) | 1 | Re-ran as plain `ls pennylane/data/data_manager/progress/` (allowed) — confirmed `_default` is a subpackage dir, `_rich.py`, `__init__.py`. |
| 3 | Phase 3 append edit used the Out-of-Scope section as oldString and dropped it from spec.md | 1 | Immediately re-inserted the Out-of-Scope section ahead of `## Technical Specification`; verified by re-editing the heading. Net file state correct. |

## Decisions consulted (Pre-Phase 3)
- `decisions.md` at repo root: **does not exist** (`ls decisions.md` → no output). No cross-issue
  constraints to apply. Noted for Phase 6 evaluation.

## Implementation Progress
- [x] Sub-task 1: Rewrite `_download_datasets` executor loop — bulk `pool.submit` of all futures + `futures.as_completed` raising first exception (matches spec "Proposed change" block verbatim; diff confined to lines 202–218; no signature/return change)
      Evidence: `.venv/bin/python -m pytest tests/data/data_manager/test_dataset_access.py -q` → 60 passed (post-change); `git diff pennylane/data/data_manager/__init__.py` reviewed against spec contract.
- [x] Sub-task 2: `test_download_datasets_concurrency` — 5 mock downloads × 0.2 s, `num_threads=5` → `max_concurrent == 5` (Lock-guarded counter via `_concurrency_mock` factory) and `elapsed < 0.8 s`
      Evidence: pytest run → passed; repeated 3× stable (9 passed each run).
- [x] Sub-task 3: `test_download_datasets_thread_cap` — parametrized (5,5,5), (2,5,2) plus edge cases (1,5,1), (10,3,3), (5,1,1); asserts `max_concurrent == min(num_threads, len(datasets))`, `call_count == n`, exact `dest_paths` list
      Evidence: pytest run → all 5 parametrizations passed.
- [x] Sub-task 4: `test_download_datasets_raises_first_exception` — parametrized failing task first-completed (index 0) and last-completed (index 4); asserts `RuntimeError("boom")` propagates and all 5 tasks submitted (bulk-submit evidence). Existing `test_load_except` passes unmodified.
      Evidence: pytest run → passed; full file 69 passed includes unmodified `test_load_except`.
- [x] Sub-task 5: Progress-bar regression — existing `test_load` × `progress_bar ∈ {True, False}` passes unmodified; new `test_download_datasets_progress_tasks` asserts `pbar.add_task` called once per dest path (with correct relative-path descriptions) and each download receives a distinct `pbar_task` (MagicMock pbar passed directly to `_download_datasets`).
      Evidence: pytest run → passed.
- [x] Sub-task 6: Changelog bullet added at top of `<h3>Bug fixes 🐛</h3>` in `doc/releases/changelog-dev.md` with `[(#XXXX)]` placeholder PR link per AGENTS.md.
      Evidence: `git diff doc/releases/changelog-dev.md` → 4 lines added under Bug fixes.
- [x] Sub-task 7: Lint/format gate
      Evidence: `pylint -rn -sn --persistent=n --rcfile=.pylintrc pennylane/data/data_manager/__init__.py` → exit 0, no output; `pylint ... --rcfile=tests/.pylintrc tests/data/data_manager/test_dataset_access.py` → exit 0 (initial R0903 fixed by restructuring recorder class into closure factory — no warning silenced, per AGENTS.md); `black --config ./pyproject.toml --check` → "2 files would be left unchanged"; `isort --settings-path ./pyproject.toml` → test file clean, source file skipped by repo config (`skip = ["__init__.py"]`, expected); `tach check` → "✅ All modules validated!" (2 pre-existing unrelated WARNs).

## Test cycles
| Sub-task | Run | Result | Failures fixed |
|----------|-----|--------|----------------|
| baseline | pytest tests/data/data_manager/test_dataset_access.py | 25 failed + 10 errors → 60 passed after env fix | missing `h5py` in .venv (environment, not code) |
| 1 | pytest test_dataset_access.py | 60 passed | — |
| 2–5 | pytest -k download_datasets (×3) | 9 passed, stable | — |
| 2–5 | pytest test_dataset_access.py | 69 passed | — |
| 1–5 | pytest tests/data/data_manager/ | 158 passed | — |
| 1–5 | pytest tests/data/ -m data (CI-equivalent) | 821 passed, 1 skipped | — |
| 7 | pylint/black/isort/tach | all clean | R0903 via restructure (not suppression) |

## E2E verification (manual reproducer — network)
- Per-task timing trace, `basis="full"`, 5 bondlengths, `num_threads=5`: **all 5 downloads start at t≈0.00–0.01 s** (bulk submit — pre-fix they started serially). 4 finish ≈0.95 s; total 15.6 s bounded below by one 15.6 s outlier file (server-side latency), vs ≈19.3 s serial sum.
- Comparable-size files (2 datasets): `num_threads=1` → 0.90 s; `num_threads=5` → 0.49 s (~1.8× — 2-way concurrency, max possible with 2 files).
- Full `load()`: `num_threads=1` 18.80 s vs `num_threads=5` 15.19 s (fixed GraphQL overhead included; wall-clock floor = slowest single download).
- Identical results: sorted dataset file paths equal for `num_threads=1` vs `num_threads=5` → `True` (5 datasets).
- `load_interactive` smoke: covered by existing `TestLoadInteractive` suite (passes unmodified through the same `_download_datasets` path with default `num_threads=50`).

## Errors
| Phase | Error | Attempt | Resolution |
|-------|-------|---------|------------|
| 3 | `.venv/bin/python -m pytest tests/data/data_manager/test_dataset_access.py -x -q` denied by sandbox bash policy (only ls/cat/find/rg/grep allowed) | 1 | Changed approach: verification evidence in Test Strategy/DoD recorded as explicit negatives; test execution deferred to `/implement-spec` phase where the implementer runs pytest. |
| 3 | `ls ... && wc -l ...` compound command denied (wc not in allowlist) | 1 | Re-ran as plain `ls pennylane/data/data_manager/progress/` (allowed) — confirmed `_default` is a subpackage dir, `_rich.py`, `__init__.py`. |
| 3 | Phase 3 append edit used the Out-of-Scope section as oldString and dropped it from spec.md | 1 | Immediately re-inserted the Out-of-Scope section ahead of `## Technical Specification`; verified by re-editing the heading. Net file state correct. |
| Impl (baseline) | Baseline suite red: 25 failed + 10 errors, root cause `ModuleNotFoundError: No module named 'h5py'` (test fixtures call `Dataset.open` → h5py) | 1 | Environment fix, not code: `.venv/bin/python -m pip` unavailable (uv-managed venv, no pip) → `uv pip install --python .venv/bin/python h5py` (h5py 3.16.0; declared in pyproject dev extras line 110). Baseline then green: 60 passed. |
| Impl (sub-task 7) | pylint R0903 `too-few-public-methods` on `_ConcurrencyRecorder` test helper class | 1 | AGENTS.md forbids silencing pylint without human approval → restructured into `_concurrency_mock()` closure factory (matches existing mock style in file). Pylint clean, tests still pass. |
