# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for callable utilities."""

from functools import partial

from pennylane._callable_utils import apply_partial_args, unwrap_partial


def sample_fn(x, y, z=1, scale=1):
    """Sample function with positional and keyword arguments."""
    return scale * (x + y + z)


class TestUnwrapPartial:
    """Tests for ``unwrap_partial``."""

    def test_non_partial_callable(self):
        """A non-partial callable is returned unchanged with no bound arguments."""
        assert unwrap_partial(sample_fn) == (sample_fn, (), {})

    def test_positional_binding(self):
        """A partial with positional bindings returns its bound arguments."""
        assert unwrap_partial(partial(sample_fn, 1)) == (sample_fn, (1,), {})

    def test_keyword_binding(self):
        """A partial with keyword bindings returns its bound keywords."""
        fn, args, kwargs = unwrap_partial(partial(sample_fn, z=3))

        assert fn is sample_fn
        assert args == ()
        assert kwargs == {"z": 3}

    def test_nested_partial_bindings(self):
        """Nested partials preserve positional order and keyword override semantics."""
        fixed = partial(partial(sample_fn, 1, z=2), 3, z=4, scale=5)

        assert unwrap_partial(fixed) == (sample_fn, (1, 3), {"z": 4, "scale": 5})

    def test_partial_without_keyword_bindings(self):
        """A partial without keyword bindings returns an empty keyword dictionary."""
        fixed = partial(sample_fn, 1)

        assert unwrap_partial(fixed) == (sample_fn, (1,), {})


class TestApplyPartialArgs:
    """Tests for ``apply_partial_args``."""

    def test_no_bound_values_returns_same_callable(self):
        """No bound values leaves the original callable unchanged."""
        assert apply_partial_args(sample_fn, (), {}) is sample_fn

    def test_applies_bound_values(self):
        """Bound values are supplied before call-time values."""
        wrapped = apply_partial_args(sample_fn, (1,), {"z": 3})

        assert wrapped(2) == 6

    def test_call_time_keywords_override_bound_keywords(self):
        """Call-time keywords override partial-bound keywords."""
        wrapped = apply_partial_args(sample_fn, (1,), {"z": 3, "scale": 4})

        assert wrapped(2, z=5) == 32
