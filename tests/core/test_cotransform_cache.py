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

import pennylane as qp
from pennylane.core.transforms import BoundTransform
from pennylane.core.transforms.cotransform_cache import CotransformCache, _get_interface


def test_classical_jacobian_error_if_not_in_cache():
    """Test a ValueError is raised if we request the classical jacobian for a transform not in the cotransform cache."""

    @qp.qnode(qp.device("default.qubit"))
    def c(x, y):
        qp.RX(2 * x, 0)
        qp.RY(x * y, 0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(0))

    container = BoundTransform(qp.transforms.merge_rotations)
    x, y = 0.5, 3.0

    cc = CotransformCache(c, (x, y), {})
    with pytest.raises(ValueError, match=r"Could not find"):
        cc.get_classical_jacobian(container, 0)


def test_no_classical_jacobian_if_no_cotransform():
    """Test that the classical jacobian is None if the transform does not have a cotransform."""

    @qp.transforms.split_non_commuting
    @qp.qnode(qp.device("default.qubit"))
    def c(x, y):
        qp.RX(2 * x, 0)
        qp.RY(x * y, 0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(0))

    container = c.compile_pipeline[-1]
    x, y = 0.5, 3.0

    cc = CotransformCache(c, (x, y), {})
    assert cc.get_classical_jacobian(container, 0) is None


@pytest.mark.all_interfaces
def test_simple_classical_jacobian(interface):
    """Test the calculation of a simple classical jacobian."""

    @qp.transforms.split_non_commuting
    @qp.qnode(qp.device("default.qubit"), interface=interface)
    def c(x, y):
        qp.RX(2 * x, 0)
        qp.RY(x * y, 0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(0))

    c = qp.gradients.param_shift(c, argnums=[0, 1])
    ps_container = c.compile_pipeline[-1]
    like = qp.math.Interface(interface).get_like()
    x, y = (qp.math.asarray(value, like=like, requires_grad=True) for value in (0.5, 3.0))

    a = CotransformCache(c, (x, y), {})
    for i in range(2):
        x_jac, y_jac = a.get_classical_jacobian(ps_container, i)
        assert qp.math.allclose(x_jac, [2.0, 3.0])
        assert qp.math.allclose(y_jac, [0.0, 0.5])


@pytest.mark.torch
@pytest.mark.parametrize("trainable_index, expected", [(0, [2.0, 3.0]), (1, [0.0, 0.5])])
def test_classical_jacobian_excludes_constant_torch_arguments(trainable_index, expected):
    """Constant arguments contribute no input Jacobian axes."""
    import torch

    @qp.transforms.split_non_commuting
    @qp.qnode(qp.device("default.qubit"), interface="torch")
    def circuit(x, y):
        qp.RX(2 * x, 0)
        qp.RY(x * y, 0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(0))

    circuit = qp.gradients.param_shift(circuit)
    transform = circuit.compile_pipeline[-1]
    args = tuple(
        torch.tensor(value, requires_grad=index == trainable_index)
        for index, value in enumerate((0.5, 3.0))
    )
    cache = CotransformCache(circuit, args, {})

    for tape_index in range(2):
        (jacobian,) = cache.get_classical_jacobian(transform, tape_index)
        assert qp.math.allclose(jacobian, expected)
    assert not args[1 - trainable_index].requires_grad


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

    @qp.transforms.split_non_commuting
    @qp.qnode(qp.device("default.qubit"))
    def c(x, y):
        qp.RX(x[0], 0)
        qp.RX(y, 0)
        qp.RY(x[1], 0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(0))

    c = qp.gradients.param_shift(c, argnums=argnums)

    ps_container = c.compile_pipeline[-1]
    x, y = jax.numpy.array([0.5, 0.7]), jax.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})

    assert cc.get_argnums(ps_container) == trainable_params


@pytest.mark.jax
def test_no_jax_argnum_error():
    """Test that an error is raised in the jax interface is used with argnum"""

    import jax

    @qp.qnode(qp.device("default.qubit"))
    def c(x, y):
        qp.RX(x[0], 0)
        qp.RX(y, 0)
        qp.RY(x[1], 0)
        return qp.expval(qp.Z(0))

    c = qp.gradients.param_shift(c, argnum=[0])

    ps_container = c.compile_pipeline[-1]
    x, y = jax.numpy.array([0.5, 0.7]), jax.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})

    with pytest.raises(
        qp.exceptions.QuantumFunctionError, match="argnum does not work with the Jax interface"
    ):
        cc.get_argnums(ps_container)


@pytest.mark.jax
def test_no_argnums_if_no_classical_cotransform():
    """Test argnums is None if there is no classical cotransform."""
    import jax

    @qp.qnode(qp.device("default.qubit"))
    def c(x, y):
        qp.RX(x[0], 0)
        qp.RX(y, 0)
        qp.RY(x[1], 0)
        return qp.expval(qp.Z(0))

    c = qp.transforms.merge_rotations(c)

    container = c.compile_pipeline[-1]
    x, y = jax.numpy.array([0.5, 0.7]), jax.numpy.array(3.0)

    cc = CotransformCache(c, (x, y), {})

    assert cc.get_argnums(container) is None


@pytest.mark.torch
def test_no_argnums_nonjax_interface():
    """Test that the trainable params are None if the interface isn't jax."""

    @qp.qnode(qp.device("default.qubit"))
    def c(x, y):
        qp.RX(x[0], 0)
        qp.RX(y, 0)
        qp.RY(x[1], 0)
        return qp.expval(qp.Z(0))

    c = qp.gradients.param_shift(c, argnum=[0])

    ps_container = c.compile_pipeline[-1]
    x = qp.math.asarray([0.5, 0.7], like="torch", requires_grad=True)
    y = qp.math.asarray(3.0, like="torch", requires_grad=True)

    cc = CotransformCache(c, (x, y), {})
    assert cc.get_argnums(ps_container) is None


@pytest.mark.parametrize("use_keyword_argument", [False, True])
def test_auto_rejects_autograd_interface(use_keyword_argument):
    """Test that automatic cotransform inference rejects Autograd tensors."""

    @qp.qnode(qp.device("default.qubit"))
    def circuit(angle):
        qp.RX(angle, 0)
        return qp.expval(qp.Z(0))

    angle = qp.numpy.array(0.5)
    args = () if use_keyword_argument else (angle,)
    kwargs = {"angle": angle} if use_keyword_argument else {}
    with pytest.raises(ValueError, match="'autograd' is not a valid Interface"):
        _get_interface(circuit, args, kwargs)
