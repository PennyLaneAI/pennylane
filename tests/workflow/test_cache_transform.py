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
"""
Unit tests for the :func:`_cache_transform` transform function.
"""
# pylint: disable=protected-access,redefined-outer-name
from collections.abc import MutableMapping
from unittest.mock import MagicMock

import pytest

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.workflow import _cache_transform


@pytest.fixture
def tape() -> QuantumScript:
    """Returns a ``QuantumScript`` object."""
    return QuantumScript([], [qml.expval(qml.Z(0))])


@pytest.fixture
def cache() -> MutableMapping:
    """Returns an object which can be used as a cache."""
    return {}


@pytest.fixture
def transform_spy(mocker) -> MagicMock:
    """Returns a spy on the underlying ``_cache_transform()`` function."""
    return mocker.spy(qml.workflow._cache_transform, "_transform")


def test_cache_miss_before_cache_hit(tape, cache):
    """Tests that the "miss" post-processing function updates the cache so that
    calling the "hit" post-processing function afterwards returns the cached
    result for the tape.
    """
    miss_tapes, miss_fn = _cache_transform(tape, cache=cache)
    hit_tapes, hit_fn = _cache_transform(tape, cache=cache)

    assert miss_tapes
    assert not hit_tapes

    result = (1.23,)

    assert miss_fn((result,)) == result
    assert hit_fn(((),)) == result


def test_cache_hit_before_cache_miss(tape, cache):
    """Tests that a RuntimeError is raised if the "hit" post-processing function
    is called before the "miss" post-processing function for the same tape.
    """
    _cache_transform(tape, cache=cache)
    _, hit_fn = _cache_transform(tape, cache=cache)

    match = (
        r"Result for tape is missing from the execution cache\. "
        r"This is likely the result of a race condition\."
    )
    with pytest.raises(RuntimeError, match=match):
        hit_fn(((1.23,),))


def test_batch_of_different_tapes(cache):
    """Tests that the results of different tapes are not cached under the same key."""
    tape_1 = QuantumScript([], [qml.expval(qml.X(0))])
    tape_2 = QuantumScript([], [qml.expval(qml.Y(0))])
    tape_3 = QuantumScript([], [qml.expval(qml.Z(0))])

    batch_tapes, batch_fns = _cache_transform([tape_1, tape_2, tape_3], cache=cache)
    assert len(batch_tapes) == 3

    results = ((1.0,), (2.0,), (3.0,))
    assert batch_fns(results) == results


def test_batch_of_identical_tapes(cache):
    """Tests that the result of identical tapes are cached under the same key."""
    tape_1 = QuantumScript([], [qml.expval(qml.Z(0))])
    tape_2 = QuantumScript([], [qml.expval(qml.Z(0))])
    tape_3 = QuantumScript([], [qml.expval(qml.Z(0))])

    batch_tapes, batch_fns = _cache_transform([tape_1, tape_2, tape_3], cache=cache)
    assert len(batch_tapes) == 1

    result = (1.23,)
    assert batch_fns((result,)) == (result, result, result)


def test_finite_shots_with_persistent_cache_warning():
    """Tests that a UserWarning is emitted if a cache hit occurs for a tape with
    finite shots that uses a cache.
    """
    tape = QuantumScript([], [qml.expval(qml.Z(0))], shots=1)

    batch_tapes, batch_fns = _cache_transform([tape, tape], cache={})
    assert len(batch_tapes) == 1

    with pytest.warns(UserWarning, match=r"Cached execution with finite shots detected!"):
        batch_fns(((1.23,),))
