# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for capturing vjp into jaxpr.
"""

import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

jnp = pytest.importorskip("jax.numpy")


class TestErrors:

    def test_error_on_big_argnum(self):
        """Test that an error is raised if the max argnums is bigger than the number of args."""

        def f(x, y):
            return x + y

        with pytest.raises(ValueError, match="Differentiating with respect to argnums"):
            qml.jvp(f, (0.5, 1.2), (1.0, 1.0), argnums=2)

    def test_error_on_bad_h(self):
        """Test that an error is raised on a bad h value."""

        with pytest.raises(ValueError, match="Invalid h value"):
            qml.jvp(lambda x: x * 2, (0.5,), (1.0,), h="something")

    def test_error_on_bad_method(self):
        """Test that an error is raised on a bad method."""

        with pytest.raises(ValueError, match="Got unrecognized method"):
            qml.jvp(lambda x: x**2, (0.5,), (1.0,), method="param-shift")

    def test_error_wrong_number_tangents(self):
        """Test that an error is raised for the wrong number of tangents."""

        with pytest.raises(
            TypeError, match="number of tangent and number of differentiable parameters"
        ):
            qml.jvp(lambda x: x**2, (0.5,), (1.0, 1.0))

    def test_error_wrong_dtype_tangents(self):
        """Test that an error is raised if the tangent is of the wrong dtype."""

        with pytest.raises(TypeError, match="dtypes must be equal"):
            qml.jvp(lambda x: x**2, (0.5,), (1,))

    def test_error_wrong_shape_tangents(self):
        """Test that an error is raised if the tangent has the wrong shape."""

        with pytest.raises(ValueError, match="params and tangent shapes"):
            qml.jvp(lambda x: x**2, (jnp.array(0.5),), (jnp.array([1.0, 1.0]),))
