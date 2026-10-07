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
from pennylane.capture.hint import HintedCallable, apply_hint, process_hints

_SUPPORTED = frozenset({"num-iters"})


class TestProcessHints:
    """Unit tests for :func:`~pennylane.capture.hint.process_hints`."""

    def test_keeps_canonical(self):
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
    def test_canonicalizes_typos(self, provided):
        """Close misspellings should be remapped to the canonical key."""
        assert process_hints({provided: 10}, _SUPPORTED) == {"num-iters": 10}

    def test_ignores_unknown(self):
        """Unrecognized keys should be dropped."""
        assert process_hints({"identity": True}, _SUPPORTED) == {}
        assert process_hints({"num-iters": 10, "identity": True}, _SUPPORTED) == {"num-iters": 10}

    def test_rejects_conflicting_aliases(self):
        """Two keys that map to the same canonical name should raise."""
        with pytest.raises(ValueError, match=r"Multiple hint keys map to 'num-iters'"):
            process_hints({"num-iters": 1, "num_iters": 2}, _SUPPORTED)

    def test_empty_hints(self):
        """An empty hints dict should round-trip to empty."""
        assert process_hints({}, _SUPPORTED) == {}


class TestHintAPI:
    """Tests for :func:`~.hint`, :class:`~HintedCallable`, and :func:`~apply_hint`."""

    def test_hint_wraps_callable(self):
        """``qp.hint`` should wrap a plain callable in ``HintedCallable``."""

        def f(x):
            return x + 1

        hinted = qp.hint({"identity": True})(f)
        assert isinstance(hinted, HintedCallable)
        assert hinted.hints == {"identity": True}
        assert hinted.f is f
        assert hinted(3) == 4

    def test_hint_as_decorator(self):
        """``qp.hint`` should work as a decorator."""

        @qp.hint({"num-iters": 5})
        def f(x):
            return x

        assert isinstance(f, HintedCallable)
        assert f.hints == {"num-iters": 5}

    def test_stacking_hints(self):
        """Applying hints twice should merge dictionaries (later keys win)."""

        @qp.hint({"a": 1})
        @qp.hint({"b": 2, "a": 0})
        def f(x):
            return x

        assert f.hints == {"a": 1, "b": 2}

    def test_repr(self):
        """``HintedCallable`` should expose the wrapped function and hints."""

        def f(x):
            return x

        hinted = qp.hint({"num-iters": 3})(f)
        text = repr(hinted)
        assert "HintedCallable" in text
        assert "num-iters" in text

    def test_apply_hint_unsupported_type(self):
        """Unsupported types should raise ``NotImplementedError``."""
        with pytest.raises(NotImplementedError, match="No registered way to apply compiler hints"):
            apply_hint(3, {"num-iters": 1})
