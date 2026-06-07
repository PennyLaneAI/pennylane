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
"""Unit tests for functools.partial unwrapping utilities."""

from functools import partial

from pennylane._functools_partial import bind_functools_partial, unwrap_functools_partial


def multiply(x, y, z=1):
    return x * y * z


class TestUnwrapFunctoolsPartial:
    """Tests for unwrap_functools_partial."""

    def test_non_partial_callable(self):
        """A plain callable is returned unchanged."""
        assert unwrap_functools_partial(multiply) == (multiply, (), {})

    def test_single_positional_binding(self):
        """Positional bindings from a single partial are collected."""
        bound = partial(multiply, 2)
        assert unwrap_functools_partial(bound) == (multiply, (2,), {})

    def test_single_keyword_binding(self):
        """Keyword bindings from a single partial are collected."""
        bound = partial(multiply, z=3)
        core, args, kwargs = unwrap_functools_partial(bound)
        assert core is multiply
        assert args == ()
        assert kwargs == {"z": 3}

    def test_nested_partials(self):
        """Nested partial bindings are merged with inner bindings first."""
        bound = partial(partial(multiply, 2), z=3)
        assert unwrap_functools_partial(bound) == (multiply, (2,), {"z": 3})


class TestBindFunctoolsPartial:
    """Tests for bind_functools_partial."""

    def test_no_bindings_returns_same_callable(self):
        """Without bindings the original callable is returned."""
        assert bind_functools_partial(multiply, (), {}) is multiply

    def test_applies_collected_bindings(self):
        """Bound arguments are supplied when the wrapper is called."""
        wrapped = bind_functools_partial(multiply, (2,), {"z": 3})
        assert wrapped(4) == 24

    def test_call_time_kwargs_override_bindings(self):
        """Keyword arguments provided at call time override partial bindings."""
        wrapped = bind_functools_partial(multiply, (), {"z": 3})
        assert wrapped(2, 4, z=5) == 40
