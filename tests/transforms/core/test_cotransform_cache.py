# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Tests for the CotransformCache object.
"""
import pytest

import pennylane as qml
from pennylane.transforms.core import TransformContainer
from pennylane.transforms.core.cotransform_cache import CotransformCache


def test_classical_jacobian_error_if_not_in_cache():
    """Test a ValueError is raised if we request the classical jacobian for a transform not in the cotransform cache."""

    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(2 * x, 0)
        qml.RY(x * y, 0)
        return qml.expval(qml.Z(0)), qml.expval(qml.X(0))

    container = TransformContainer(qml.transforms.merge_rotations)
    x, y = qml.numpy.array(0.5), qml.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})
    with pytest.raises(ValueError, match=r"Could not find"):
        cc.get_classical_jacobian(container, 0)


def test_no_classical_jacobian_if_no_cotransform():
    """Test that the classical jacobian is None if the transform does not have a cotransform."""

    @qml.transforms.split_non_commuting
    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(2 * x, 0)
        qml.RY(x * y, 0)
        return qml.expval(qml.Z(0)), qml.expval(qml.X(0))

    container = c.transform_program[-1]
    x, y = qml.numpy.array(0.5), qml.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})
    assert cc.get_classical_jacobian(container, 0) is None


def test_simple_classical_jacobian():
    """Test the calculation of a simple classical jacobian."""

    @qml.gradients.param_shift
    @qml.transforms.split_non_commuting
    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(2 * x, 0)
        qml.RY(x * y, 0)
        return qml.expval(qml.Z(0)), qml.expval(qml.X(0))

    ps_container = c.transform_program[-1]
    x, y = qml.numpy.array(0.5), qml.numpy.array(3.0)

    a = CotransformCache(c, (x, y), {})
    for i in range(2):
        x_jac, y_jac = a.get_classical_jacobian(ps_container, i)
        assert qml.math.allclose(x_jac, qml.numpy.array([2.0, 3.0]))
        assert qml.math.allclose(y_jac, qml.numpy.array([0.0, 0.5]))


@pytest.mark.jax
@pytest.mark.parametrize(
    "argnums, trainable_params",
    [
        (0, [{0, 2}, {0, 2}]),
        (None, [{0, 2}, {0, 2}]),
        ([0], [{0, 2}, {0, 2}]),
        (
            1,
            [
                {
                    1,
                },
                {
                    1,
                },
            ],
        ),
        (
            [1],
            [
                {
                    1,
                },
                {
                    1,
                },
            ],
        ),
        ([0, 1], [{0, 1, 2}, {0, 1, 2}]),
    ],
)
def test_simple_argnums(argnums, trainable_params):
    """Test a simple calculation of the argnums."""

    import jax

    @qml.transforms.split_non_commuting
    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(x[0], 0)
        qml.RX(y, 0)
        qml.RY(x[1], 0)
        return qml.expval(qml.Z(0)), qml.expval(qml.X(0))

    c = qml.gradients.param_shift(c, argnums=argnums)

    ps_container = c.transform_program[-1]
    x, y = jax.numpy.array([0.5, 0.7]), jax.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})

    assert cc.get_argnums(ps_container) == trainable_params


@pytest.mark.jax
def test_no_jax_argnum_error():
    """Test that an error is raised in the jax interface is used with argnum"""

    import jax

    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(x[0], 0)
        qml.RX(y, 0)
        qml.RY(x[1], 0)
        return qml.expval(qml.Z(0))

    c = qml.gradients.param_shift(c, argnum=[0])

    ps_container = c.transform_program[-1]
    x, y = jax.numpy.array([0.5, 0.7]), jax.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})

    with pytest.raises(
        qml.exceptions.QuantumFunctionError, match="argnum does not work with the Jax interface"
    ):
        cc.get_argnums(ps_container)


@pytest.mark.jax
def test_no_argnums_if_no_classical_cotransform():
    """Test argnums is None if there is no classical cotransform."""
    import jax

    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(x[0], 0)
        qml.RX(y, 0)
        qml.RY(x[1], 0)
        return qml.expval(qml.Z(0))

    c = qml.transforms.merge_rotations(c)

    container = c.transform_program[-1]
    x, y = jax.numpy.array([0.5, 0.7]), jax.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})

    assert cc.get_argnums(container) is None


def test_no_argnums_nonjax_interface():
    """Test that the trainable params are None if the interface isn't jax."""

    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(x[0], 0)
        qml.RX(y, 0)
        qml.RY(x[1], 0)
        return qml.expval(qml.Z(0))

    c = qml.gradients.param_shift(c, argnum=[0])

    ps_container = c.transform_program[-1]
    x, y = qml.numpy.array([0.5, 0.7]), qml.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})
    assert cc.get_argnums(ps_container) is None
