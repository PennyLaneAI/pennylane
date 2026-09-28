# Spec: issue-9606

> Source: `specs/issue-9606-findings.md` (light spec `specs/9606.md`, 39 lines)
> GitHub issue: https://github.com/PennyLaneAI/pennylane/issues/9606

## Requirements

### User Story

As a user of `qml.data.load` who downloads multiple datasets (e.g. `basis="full"` or
list-valued parameters), I want the documented `num_threads` parameter to actually run
up to N downloads concurrently, so that large dataset sweeps finish in a fraction of
the current serial wall-clock time.

*Extracted from findings.md "Problem" + "Why now" (lines 3–7).*

### Acceptance Criteria

- [ ] With `num_threads=N`, up to N downloads run concurrently (pool size = `min(num_threads, len(dest_paths))` is actually utilized).
- [ ] A test with 5 mock downloads at 0.2 s each completes in ~0.2 s, not ~1.0 s.
- [ ] `max_concurrent` observed in a concurrency-tracking test equals `min(num_threads, len(datasets))`.
- [ ] Existing error handling preserved: the first exception raised by any download task propagates out of `_download_datasets` (and hence out of `load`).
- [ ] Progress-bar integration preserved: each dataset still gets its own `pbar_task`, and per-task updates flow through unchanged.
- [ ] No changes to `_download_dataset`, `_download_partial`, or `_download_full`.

*Extracted from findings.md "Desired outcome" (lines 9–12) and "Constraints" (lines 14–18).*

### Functional Requirements

- **FR1** — In `_download_datasets` (`pennylane/data/data_manager/__init__.py`), replace
  the per-iteration *submit-one-then-wait* loop with a **bulk submit** of all
  `_download_dataset` tasks to the existing `ThreadPoolExecutor`.
- **FR2** — Iterate `concurrent.futures.as_completed(futures)` over the submitted futures
  and `raise` the exception of the first completed future that has one (surfaces errors as
  soon as any task finishes, rather than waiting for all).
- **FR3** — Keep the executor construction unchanged:
  `futures.ThreadPoolExecutor(min(num_threads, len(dest_paths)))` — this is the self-limiting
  concurrency cap; no `os.cpu_count()` cap may be introduced.
- **FR4** — Keep the `with`-block context manager so `pool.shutdown(wait=True)` still runs;
  in-flight downloads are not cancelled when an error is raised (matches current semantics —
  the old code never cancelled anything either).
- **FR5** — `_download_datasets` keeps its exact signature and its `list[Path]` return value
  (`dest_paths`); no public API (`load`, `load_interactive`) changes.

*Derived from findings.md "Approach sketch" (lines 28–31).*

### Non-Functional Requirements

- **Performance**: wall-clock time for K downloads of duration D at `num_threads=N` drops from
  `K·D` to `⌈K/N⌉·D` (I/O-bound concurrency). No CPU-based thread cap (downloads are
  network-bound) — findings.md lines 18, 36.
- **Correctness / error semantics**: raise-on-first-exception preserved (findings.md line 15).
- **Compatibility**: progress bar (default `rich` if installed, else `DefaultProgress`) must
  keep working with per-task updates (findings.md line 16); no behavioral change for
  `num_threads=1`.
- **Dependencies**: none added — `concurrent.futures` is already imported
  (`__init__.py:21`, `from concurrent import futures`). Stdlib only.
- **Style**: must pass `pylint` (source rcfile), `black`, `isort` (line length 100), and
  `tach check` per repo `AGENTS.md`.

### Out of Scope (from findings.md lines 20–23)

- Changing S3 URL resolution or GraphQL queries.
- Modifying `load_interactive` beyond the existing `num_threads` passthrough (note:
  `load_interactive` does not currently pass `num_threads` — it uses the `load` default of 50;
  unchanged).
- Adding rate-limiting or retry logic.

## Technical Specification

### Root cause (verified in code)

`pennylane/data/data_manager/__init__.py:202-217` — the executor loop submits **one** future
and immediately blocks on `futures.wait(...)` before submitting the next:

```python
with futures.ThreadPoolExecutor(min(num_threads, len(dest_paths))) as pool:
    for url, dest_path, pbar_task in zip(dataset_urls, dest_paths, pbar_tasks, strict=True):
        futs = [pool.submit(_download_dataset, url, dest_path, attributes=attributes, ...)]
        for result in futures.wait(futs, return_when=futures.FIRST_EXCEPTION).done:
            if result.exception() is not None:
                raise result.exception()
```

Because `futs` always contains exactly one future and the loop waits on it before the next
`pool.submit`, pool size `min(num_threads, len(dest_paths))` (line 202) is allocated but never
utilized → fully serial. The bug is purely in the *scheduling loop*, not in the pool config.

### Files to Modify

| File | Change |
|------|--------|
| `pennylane/data/data_manager/__init__.py` | Replace the serial submit-wait loop in `_download_datasets` (lines 202–217) with bulk submit + `futures.as_completed` iteration. No other function touched. |
| `tests/data/data_manager/test_dataset_access.py` | Add concurrency-tracking unit tests for `_download_datasets`; keep existing `test_load`, `test_load_except`, `test_load_other_attributes` green. |
| `doc/releases/changelog-dev.md` | Add bug-fix bullet under `<h3>Bug fixes 🐛</h3>` (line 1581), ending with `  [(#XXXX)](...)` PR link per `AGENTS.md`. |

### Files to Create

| File | Purpose |
|------|---------|
| — none — | Fix is localized to one existing function; tests go in the existing test module. |

### Proposed change (contract for implementer)

```python
with futures.ThreadPoolExecutor(min(num_threads, len(dest_paths))) as pool:
    futs = [
        pool.submit(
            _download_dataset,
            url,
            dest_path,
            attributes=attributes,
            force=force,
            block_size=block_size,
            pbar_task=pbar_task,
        )
        for url, dest_path, pbar_task in zip(dataset_urls, dest_paths, pbar_tasks, strict=True)
    ]
    for fut in futures.as_completed(futs):
        if fut.exception() is not None:
            raise fut.exception()
```

Notes binding this contract:
- `futures` is already imported (`__init__.py:21` — `from concurrent import futures`);
  `as_completed` is accessed as `futures.as_completed`. No new imports.
- The list comprehension iterates `pbar_tasks` exactly once. In the `pbar is None` branch
  `pbar_tasks` is a one-shot generator (`__init__.py:200`) — the comprehension consumes it in
  one pass, same as the old outer `for` loop did. Do **not** convert it to a list unless the
  generator semantics are preserved elsewhere (not needed).
- `as_completed` yields futures in completion order; `fut.exception()` on an already-completed
  future never blocks. First observed exception is re-raised → satisfies "raise on first
  exception" (findings.md line 15).
- On raise, the `with` block exits → `ThreadPoolExecutor.shutdown(wait=True)` runs → already
  queued/running downloads finish before the exception propagates. This matches the old code
  (which also never cancelled); documented as accepted behavior, not a regression.

### API Contracts

No public API changes. Internal contract of the modified function:

#### `_download_datasets` (private, `pennylane.data.data_manager`)
Signature unchanged:
```python
def _download_datasets(
    data_name: str,
    folder_path: Path,
    dataset_urls: list[str],
    dataset_ids: list[str],
    attributes: Iterable[str] | None,
    force: bool,
    block_size: int,
    num_threads: int,
    pbar: progress.Progress | None,
) -> list[Path]
```
Behavioral contract (post-fix):
- Submits all `len(dataset_urls)` tasks to the pool before waiting on any.
- Concurrency = `min(num_threads, len(dest_paths))` (pool `max_workers`, line 202 — unchanged).
- Raises the first exception observed via `as_completed`; propagates out of `load()`.
- Returns `dest_paths` unchanged.

#### `qml.data.load(...)` (public)
Request/response unchanged; `num_threads: int = 50` (`__init__.py:243`) now behaves as
documented ("maximum number of threads to spawn while downloading files (1 thread per file)",
line 257). No docstring change required — the docstring already promises the fixed behavior.

### Database Changes
- Table: none.
- Migration required: no.
- Changes: none — pure in-memory scheduling change.

### External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| — none new — | | `concurrent.futures` is stdlib and already imported; `rich` (optional, existing) untouched. |

### Codebase evidence (Phase 3 lookups)

- `grep "def test_|class |patch\(|_download|progress|pbar" tests/data/data_manager/test_dataset_access.py` → existing download-mock infra: `mock_download_dataset` fixture (line 407), `head_mock` (196), `graphql_mock` (155), `test_load_except` (499), `pytestmark = pytest.mark.data` (56).
- `grep "wait_mock|as_completed|futures\.wait" tests/` → only hit is the **unused** `wait_mock_fixture` (test_dataset_access.py:218). Nothing patches `futures.wait`, so replacing it with `as_completed` breaks no existing test. (Dead fixture left in place — out of scope.)
- `grep "_download_datasets|num_threads" tests/` → **no existing test** exercises concurrency or `_download_datasets` directly; new tests required (AC #2/#3).
- `ls pennylane/data/data_manager/progress/` → subpackage (`__init__.py`, `_default/`, `_rich.py`) exporting `Progress`/`Task`; `pbar.add_task(...)` API confirmed at `__init__.py:195-198` and exercised by `test_load(progress_bar=True)`.
- CI: `.github/workflows/interface-unit-tests.yml:674` runs the `data` marker → new tests must be marker-clean.
- `decisions.md`: absent at repo root → no binding cross-issue constraints.

## Implementation Plan

| # | Sub-task | Complexity (1–5) | Depends On |
|---|----------|------------------|------------|
| 1 | Rewrite `_download_datasets` executor loop in `pennylane/data/data_manager/__init__.py:202-217`: bulk `pool.submit` of all futures into a list, then `for fut in futures.as_completed(futs): if fut.exception(): raise fut.exception()`. No signature/return changes; no other function touched. | 2 | — |
| 2 | Add concurrency unit test in `tests/data/data_manager/test_dataset_access.py`: monkeypatch `_download_dataset` with a recorder that (a) sleeps 0.2 s, (b) tracks `max_concurrent` via a `threading.Lock`-guarded counter. Call `_download_datasets` directly with 5 synthetic urls/ids, `pbar=None`, `num_threads=5`; assert `max_concurrent == 5` and `elapsed < 0.8 s` (serial would be ~1.0 s). | 3 | 1 |
| 3 | Extend sub-task 2's test as a parametrized pair: `(num_threads=5, n=5, expect 5)` and `(num_threads=2, n=5, expect 2)` — pins AC#3 `max_concurrent == min(num_threads, len(datasets))` and proves the cap is respected. | 2 | 2 |
| 4 | Add direct error-propagation test for `_download_datasets`: one mock task raises `RuntimeError("boom")` after a short sleep, others succeed; assert the exception propagates and that all tasks were submitted (bulk-submit semantics). Confirm existing `test_load_except` (line 499) still passes unmodified. | 2 | 1 |
| 5 | Progress-bar regression check: run existing `test_load` parametrization with `progress_bar=True` (already covers `pbar.add_task` + per-task updates through the fixed loop); optionally assert `pbar.add_task` call count == number of datasets using a `MagicMock` pbar passed directly to `_download_datasets`. | 1 | 1 |
| 6 | Add changelog bullet to `doc/releases/changelog-dev.md` under `<h3>Bug fixes 🐛</h3>` (line 1581): one-line description + `  [(#XXXX)](https://github.com/PennyLaneAI/pennylane/pull/XXXX)` placeholder link per `AGENTS.md`. | 1 | — |
| 7 | Lint/format gate: `pylint -rn -sn --persistent=n --rcfile=.pylintrc pennylane/data/data_manager/__init__.py`, `pylint ... --rcfile=tests/.pylintrc tests/data/data_manager/test_dataset_access.py`, then `black --config ./pyproject.toml` + `isort --settings-path ./pyproject.toml` on both files, then `tach check`. | 1 | 1–5 |

**Complexity rationale**: the core change is a ~10-line localized rewrite of one loop (2); the
test work dominates because concurrency assertions must be deterministic, not merely
timing-based (3+2+2+1); changelog and lint are mechanical repo conventions (1+1). Total:
7 sub-tasks, all small-to-medium; no cross-module blast radius (`tach` layer untouched —
change stays inside `pennylane/data`).

### Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Timing-based assertion (`elapsed < 0.8 s`) flakes on loaded CI runners | medium | Make the deterministic `max_concurrent == min(num_threads, n)` counter the primary assertion (AC#3); keep the elapsed bound loose (< 0.8 s vs 1.0 s serial) and sleeps small (0.2 s). |
| Error path waits for in-flight downloads before raising (`shutdown(wait=True)`) — user-perceived "slow failure" | low | Accepted: matches pre-fix semantics (old code never cancelled either) and the findings.md constraint only requires *raise on first exception*, not cancellation. Documented in spec notes; no `cancel_futures` added (would change semantics beyond scope). |
| Mock not intercepting `_download_dataset` (module-attribute lookup) | low | Existing `mock_download_dataset` fixture (test line 407) already proves `monkeypatch.setattr(pennylane.data.data_manager, "_download_dataset", ...)` intercepts the call site inside `_download_datasets`; reuse that exact mechanism. |
| One-shot `pbar_tasks` generator double-consumed by refactor | low | Proposed change consumes it exactly once inside the list comprehension; sub-task 5's MagicMock-pbar test exercises the `pbar is not None` branch; `test_load(progress_bar=True/False)` covers both branches end-to-end. |
| `num_threads` ≤ 0 passed by a caller → `ThreadPoolExecutor` ValueError | low | Pre-existing behavior (pool construction at line 202 unchanged); out of scope to add validation — noted, no regression introduced. |

## Test Strategy

### Test Command

```bash
# Focused (primary loop during implementation):
.venv/bin/python -m pytest tests/data/data_manager/test_dataset_access.py -q

# Whole data-manager suite:
.venv/bin/python -m pytest tests/data/data_manager/ -q

# CI-equivalent marker selection for this area (matches interface-unit-tests.yml "data" job):
.venv/bin/python -m pytest tests/ -m data -q
```

> NEEDS VERIFICATION (execution, not existence): the analyst sandbox restricts bash to
> read-only inspection (`ls`/`cat`/`find`/`rg`/`grep`) — pytest could not be executed here
> (see progress.md Errors, Phase 3). The commands above are taken verbatim from repo
> `AGENTS.md` (venv + pytest conventions) and the CI config
> (`.github/workflows/interface-unit-tests.yml:674` → `pytest_markers: data`). The
> implementer must run them and record results as Evidence in progress.md.
> Baseline sanity before the change: the existing suite is expected green (tests import
> `pennylane as qp` per convention, marker `pytest.mark.data` registered — test file line 56).

### Unit Tests

- [ ] `test_download_datasets_concurrency` (new, sub-task 2): 5 mock downloads × 0.2 s,
  `num_threads=5` → `max_concurrent == 5` and `elapsed < 0.8 s` (AC#2, AC#3). Deterministic
  lock-guarded counter is the primary assertion.
- [ ] `test_download_datasets_thread_cap` (new, sub-task 3, parametrized with the above):
  `num_threads=2`, 5 datasets → `max_concurrent == 2` — pins
  `min(num_threads, len(datasets))` (AC#1/#3).
- [ ] `test_download_datasets_raises_first_exception` (new, sub-task 4): one failing task →
  exception propagates from `_download_datasets`; all 5 tasks were submitted (bulk-submit
  evidence). Existing `test_load_except` (line 499) must pass **unmodified** (Constraint:
  error handling preserved).
- [ ] `test_download_datasets_progress_tasks` (new, sub-task 5, optional-but-recommended):
  `MagicMock` pbar → `pbar.add_task` called once per dest path; each mock
  `_download_dataset` invocation receives a distinct `pbar_task` (Constraint: progress-bar
  integration preserved).
- [ ] Existing `test_download_dataset_full_or_partial` / `test_download_dataset_full_call`
  (lines 518/540) pass unmodified — download primitives untouched.

### Integration Tests

- [ ] Existing `test_load` (line 438) × `progress_bar ∈ {True, False}` ×
  `attributes ∈ {None, ["molecule"]}` passes unmodified — exercises the rewritten loop
  end-to-end through `load()` with mocked graphql/head/download.
- [ ] Existing `test_load_other_attributes` (line 475) passes unmodified.
- [ ] `-m data` marker run of the full `tests/data/` tree is green.

### E2E Scenarios

- [ ] Manual (not CI — needs network): run the issue reproducer —
  `qml.data.load("qchem", molname="H2", basis="full", bondlength=[...], num_threads=5,
  folder_path=tmp)` and compare wall-clock vs `num_threads=1`; expect multi-fold speedup
  with identical returned dataset list. Record timings in progress.md as Evidence.
- [ ] Interactive smoke (optional): `load_interactive()` still completes (uses default
  `num_threads=50` through the same path).

### Edge Cases

- [ ] `num_threads=1` → behaves serially, results correct (no regression for single-thread users).
- [ ] `num_threads > len(datasets)` → pool capped at `len(dest_paths)` via existing `min(...)`; all datasets downloaded.
- [ ] Single dataset (`len(dest_paths)==1`) → one future, completes normally.
- [ ] Exception in the **first-completed** vs **last-completed** task → both propagate (as_completed order-independent surfacing).
- [ ] `pbar=None` (one-shot `pbar_tasks` generator, `__init__.py:200`) → generator consumed exactly once in the list comprehension; no `StopIteration`/silently-empty bug.
- [ ] Empty dataset list → cannot reach `_download_datasets`; guarded upstream in `load()` (`__init__.py:347-351` raises `ValueError`). No new test needed; behavior unchanged.
- [ ] GIL sanity: mocks use `time.sleep` (releases GIL) — mirrors real I/O-bound downloads.

## Definition of Done
- [ ] All acceptance criteria satisfied (spec §Acceptance Criteria 1–6 ↔ plan sub-tasks 1–5; AC#6 "primitives untouched" verified by diff scope + unmodified tests lines 518/540).
- [ ] Unit test coverage ≥ 80% for changed files — repo has no CI coverage gate (no coverage config found in `pyproject.toml`/workflows); practical equivalent: the rewritten ~10-line block is exercised by ≥3 new tests (concurrency, cap, error) plus existing `test_load` matrix. Evidence = test run output in progress.md.
- [ ] Integration tests passing in CI — `-m data` job green on the PR (`.github/workflows/interface-unit-tests.yml:674`).
- [ ] API documentation updated — **N/A by design**: public API unchanged and `load`'s docstring (line 257) already documents the now-true behavior; no `.rst` references to `_download_datasets` internals (private). Verified by grep during Phase 3.
- [ ] No performance regressions — reproducer timing recorded (E2E scenario); `num_threads=1` path unchanged in cost; no CPU cap introduced.
- [ ] `pylint` (source + tests rcfiles), `black`, `isort`, `tach check` clean on changed files (plan sub-task 7).
- [ ] Changelog bullet present in `doc/releases/changelog-dev.md` Bug fixes section with PR link (plan sub-task 6; repo `AGENTS.md` convention).
- [ ] Code review approved (human; per repo AI policy, PR in draft until reviewed).

### Rollout
- Ships in the 0.46.0 dev release via `changelog-dev.md` — no feature flag, no migration,
  no config change. Revert = single-commit revert (change is confined to one function +
  tests). No downstream callers of `_download_datasets` besides `load()` (grep-verified:
  definition line 162, call sites lines 360/373 only).
