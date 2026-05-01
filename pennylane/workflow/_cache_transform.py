# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the transform for caching the result of a ``tape``."""

import warnings
from collections.abc import MutableMapping

from pennylane.tape import QuantumScript
from pennylane.transforms import transform
from pennylane.typing import Result, ResultBatch

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
def _cache_transform(tape: QuantumScript, cache: MutableMapping):
    """Caches the result of ``tape`` using the provided ``cache``.

    .. note::

        This function makes use of :attr:`.QuantumTape.hash` to identify unique tapes.
    """

    def cache_hit_postprocessing(_results: ResultBatch) -> Result:
        result = cache[tape.hash]
        if result is not None:
            if tape.shots:
                warnings.warn(_CACHED_EXECUTION_WITH_FINITE_SHOTS_WARNINGS, UserWarning)
            return result

        raise RuntimeError(
            "Result for tape is missing from the execution cache. "
            "This is likely the result of a race condition."
        )

    if tape.hash in cache:
        return [], cache_hit_postprocessing

    def cache_miss_postprocessing(results: ResultBatch) -> Result:
        result = results[0]
        cache[tape.hash] = result
        return result

    # Adding a ``None`` entry to the cache indicates that a result will eventually be available for
    # the tape. This assumes that post-processing functions are called in the same order in which
    # the transforms are invoked. Otherwise, ``cache_hit_postprocessing()`` may be called before the
    # result of the corresponding tape is placed in the cache by ``cache_miss_postprocessing()``.
    cache[tape.hash] = None
    return [tape], cache_miss_postprocessing
