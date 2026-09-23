# 9606

## Problem
`qml.data.load` ignores the `num_threads` parameter and downloads datasets serially, making large dataset sweeps much slower than documented behavior.

## Why now
Users relying on parallel downloads (documented behavior) get serial performance. With `basis="full"` or list parameters, multiple datasets are needed and the lack of concurrency multiplies wait times linearly.

## Desired outcome
- With `num_threads=N`, up to N downloads run concurrently
- A test with 5 mock downloads at 0.2s each completes in ~0.2s, not ~1.0s
- `max_concurrent` in the reproducer equals `min(num_threads, len(datasets))`

## Constraints
- Must preserve existing error handling (raise on first exception)
- Must preserve progress bar integration (per-task `pbar_task` updates)
- Must not change the underlying download logic (`_download_dataset`, `_download_partial`, `_download_full`)
- No `os.cpu_count()` cap — downloads are I/O-bound, not CPU-bound

## Out of scope
- Changing S3 URL resolution or GraphQL queries
- Modifying `load_interactive` beyond existing `num_threads` passthrough
- Adding rate-limiting or retry logic

## Open questions
- None remaining — approach agreed on during brainstorm

## Approach sketch
In `_download_datasets` (`pennylane/data/data_manager/__init__.py:162`), submit **all** tasks to the `ThreadPoolExecutor` upfront instead of one-at-a-time with immediate waits. Then iterate `futures.as_completed()` to surface errors as soon as any task finishes, rather than waiting for all tasks to complete first. This allows the pool to run up to `min(num_threads, len(dest_paths))` downloads concurrently.

The key change: replace the serial submit-wait loop with a bulk submit of all futures followed by `as_completed` iteration that raises on the first exception.

### Alternatives considered
- **`futures.wait(..., ALL_COMPLETED)`**: Works but wastes time — on error, all remaining threads keep running before the exception is raised. `as_completed` surfaces errors sooner.
- **`pool.map()`**: Preserves order but is less flexible for mid-flight error handling and doesn't offer a clear advantage for this use case.
- **`os.cpu_count()` thread cap**: Rejected — downloads are I/O-bound (network latency, not CPU), so a CPU-based cap would hurt performance. The existing `min(num_threads, len(dest_paths))` is already self-limiting.

## Source
- GitHub Issue: https://github.com/PennyLaneAI/pennylane/issues/9606
