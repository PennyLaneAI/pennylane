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
"""Tests for Operator2's metaclass."""

import pytest

from pennylane.core.operator.capture import contains_abstract_type
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractWires, Wires

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax]


class TestContainsAbstractTypeHelper:
    """Tests the 'contains_abstract_type' helper."""

    @pytest.mark.parametrize(
        "val",
        [
            AbstractArray((1,), float),
            AbstractWires(2),
            [0.5, AbstractArray((1,), int)],
            {"a": AbstractWires(1)},
            (1.0, [AbstractArray((1,), float)]),
        ],
    )
    def test_abstract_leaves_are_flagged(self, val):
        """Tests that pytrees with at least one abstract value return True."""

        assert contains_abstract_type(val)

    @pytest.mark.parametrize(
        "val",
        [
            0.5,
            [0.5, 1.0],
            {"a": 1, "b": 2},
            Wires([0, 1, 2]),
            (1.0, 2.0),
        ],
    )
    def test_concrete_leaves_are_not_flagged(self, val):
        """Tests that pytrees containing purely concrete values are not falgged."""

        assert not contains_abstract_type(val)

    def test_jax_tracer_is_not_flagged(self):
        """Tests that JAX tracers are not considered 'abstract'."""

        captured = {}

        def f(x):
            captured["is_abstract"] = contains_abstract_type(x)

        _ = jax.make_jaxpr(f)(0.5)
        assert captured["is_abstract"] is False
