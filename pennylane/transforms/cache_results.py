from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, Tuple

from cachetools import LRUCache, Cache
from typing import Union, Tuple, Callable

import pennylane as qml


def cache_results(
    tape: QuantumScript, cache: Union[Cache, dict]
) -> Tuple[Tuple[QuantumScript], Callable[[ResultBatch], ResultBatch]]:
    def include_cached_results(results: ResultBatch) -> ResultBatch:
        """Place retrieved results in the cache and add results that should be retrieved
        from the cache.

        Args:
            results (ResultBatch): the results of the executions.

        Closure:
            cache (Cache, dict)
            tape (QuantumScript)

        Side Effects:
            populates the results into the cache.

        Returns:
            ResultBatch: the results for the tapes inputted

        """
        cache_value = cache[tape.hash]

        if cache_value is None:
            cache[tape.hash] = results[0]
            return results[0]
        return cache_value

    if tape.hash in cache:
        return tuple(), include_cached_results
    cache[tape.hash] = None
    return (tape,), include_cached_results
