# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for compiler hint utilities.
"""

import pytest

import pennylane as qp
from pennylane.capture.hint import process_hints

_SUPPORTED = frozenset({"num-iters"})


def test_process_hints_keeps_canonical():
    """Exact supported keys should be returned unchanged."""
    assert process_hints({"num-iters": 10}, _SUPPORTED) == {"num-iters": 10}


@pytest.mark.parametrize(
    "provided",
    (
        "num_iters",
        "numiters",
        "Num-Iters",
        "num-iter",
        "nub-iters",
        "num-itters",
    ),
)
def test_process_hints_canonicalizes_typos(provided):
    """Close misspellings should be remapped to the canonical key."""
    assert process_hints({provided: 10}, _SUPPORTED) == {"num-iters": 10}


def test_process_hints_ignores_unknown():
    """Unrecognized keys should be dropped."""
    assert process_hints({"identity": True}, _SUPPORTED) == {}
    assert process_hints({"num-iters": 10, "identity": True}, _SUPPORTED) == {"num-iters": 10}


def test_process_hints_rejects_conflicting_aliases():
    """Two keys that map to the same canonical name should raise."""
    with pytest.raises(ValueError, match=r"Multiple hint keys map to 'num-iters'"):
        process_hints({"num-iters": 1, "num_iters": 2}, _SUPPORTED)


@pytest.mark.parametrize(
    "make_loop",
    (
        lambda: qp.for_loop(3),
        lambda: qp.while_loop(lambda i: i < 3),
    ),
    ids=("for_loop", "while_loop"),
)
class TestLoopHintProcessing:
    """Hint processing wired into for_loop and while_loop."""

    def test_typo_on_hinted_body_is_canonicalized(self, make_loop):
        """A typo'd key on a HintedCallable body should still set the hint."""

        @qp.hint({"num_iters": 10})
        def body(i):  # pylint: disable=unused-argument
            return i + 1

        loop = make_loop()(body)
        assert loop.num_iters_hint == 10

    def test_typo_on_apply_hint_is_canonicalized(self, make_loop):
        """Applying a typo'd hint to a loop callable should canonicalize it."""

        def body(i):  # pylint: disable=unused-argument
            return i + 1

        loop = qp.hint({"num_iters": 10})(make_loop()(body))
        assert loop.num_iters_hint == 10

    def test_unknown_hint_on_body_is_ignored(self, make_loop):
        """Unrecognized hint keys on the body should be ignored."""

        @qp.hint({"identity": True})
        def body(i):  # pylint: disable=unused-argument
            return i + 1

        loop = make_loop()(body)
        assert loop.num_iters_hint is None

    def test_valid_hint_on_body(self, make_loop):
        """A correctly spelled hint on the body should be accepted."""

        @qp.hint({"num-iters": 10})
        def body(i):  # pylint: disable=unused-argument
            return i + 1

        loop = make_loop()(body)
        assert loop.num_iters_hint == 10

    def test_valid_apply_hint(self, make_loop):
        """Applying a correctly spelled hint should set ``num_iters_hint``."""

        def body(i):  # pylint: disable=unused-argument
            return i + 1

        loop = qp.hint({"num-iters": 7})(make_loop()(body))
        assert loop.num_iters_hint == 7
