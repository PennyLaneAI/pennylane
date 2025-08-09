# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Tests for compatibility between the math module and catalyst.
"""
import pytest

import pennylane as qml


@pytest.mark.external
def test_catalyst_integration():
    """Test that scatter_element_add can be used with catalyst by specifying indices_are_sorted and unique_indices."""

    jnp = pytest.importorskip("jax.numpy")

    @qml.qjit
    def f(x, y):
        return qml.math.scatter_element_add(
            x, ((0, 1), (0, 1)), y, indices_are_sorted=True, unique_indices=True
        )

    x0 = jnp.zeros((2, 2))
    y = jnp.array([1.0, 2.0])

    out = f(x0, y)
    expected = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    assert qml.math.allclose(out, expected)
