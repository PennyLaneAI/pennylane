from typing import MutableMapping, Tuple
import warnings
from pennylane.tape import QuantumTape
from pennylane.transforms import transform

_CACHED_EXECUTION_WITH_FINITE_SHOTS_WARNINGS = (
    "Cached execution with finite shots detected!\n"
    "Note that samples as well as all noisy quantities computed via sampling "
    "will be identical across executions. This situation arises where tapes "
    "are executed with identical operations, measurements, and parameters.\n"
    "To avoid this behaviour, provide 'cache=False' to the QNode or execution "
    "function."
)
"""str: warning message to display when cached execution is used with finite shots"""



@transform
def cache_transform(tape: QuantumTape, cache: MutableMapping):
    """Caches the result of ``tape`` using the provided ``cache``.

    .. note::

        This function makes use of :attr:`.QuantumTape.hash` to identify unique tapes.
    """

    def cache_hit_postprocessing(_results: Tuple[Tuple]) -> Tuple:
        result = cache[tape.hash]
        if result is not None:
            if tape.shots and getattr(cache, "_persistent_cache", True):
                warnings.warn(_CACHED_EXECUTION_WITH_FINITE_SHOTS_WARNINGS, UserWarning)
            return result

        raise RuntimeError(
            "Result for tape is missing from the execution cache. "
            "This is likely the result of a race condition."
        )

    if tape.hash in cache:
        return [], cache_hit_postprocessing

    def cache_miss_postprocessing(results: Tuple[Tuple]) -> Tuple:
        result = results[0]
        cache[tape.hash] = result
        return result

    # Adding a ``None`` entry to the cache indicates that a result will eventually be available for
    # the tape. This assumes that post-processing functions are called in the same order in which
    # the transforms are invoked. Otherwise, ``cache_hit_postprocessing()`` may be called before the
    # result of the corresponding tape is placed in the cache by ``cache_miss_postprocessing()``.
    cache[tape.hash] = None
    return [tape], cache_miss_postprocessing

