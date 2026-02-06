# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the adjoint_metric_tensor function.
"""

import numpy as onp

# pylint: disable=protected-access
import pytest

import pennylane as qp
from pennylane import numpy as np

fixed_pars = [-0.2, 0.2, 0.5, 0.3, 0.7]


def fubini_ansatz0(params, wires=None):
    qp.RX(params[0], wires=wires[0])
    qp.RY(fixed_pars[0], wires=wires[0])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RZ(params[1], wires=wires[0])
    qp.CNOT(wires=[wires[0], wires[1]])


def fubini_ansatz1(params, wires=None):
    qp.RX(fixed_pars[1], wires=wires[0])
    for i, wire in enumerate(wires):
        qp.Rot(*params[0][i], wires=wire)
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.adjoint(qp.RY(fixed_pars[1], wires=wires[0]))
    qp.CNOT(wires=[wires[1], wires[2]])
    for i, wire in enumerate(wires):
        qp.Rot(*params[1][i], wires=wire)
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.RX(fixed_pars[2], wires=wires[1])


def fubini_ansatz2(params, wires=None):
    params0 = params[0]
    params1 = params[1]
    qp.RX(fixed_pars[1], wires=wires[0])
    qp.Rot(*fixed_pars[2:5], wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RY(params0, wires=wires[0])
    qp.RY(params0, wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.adjoint(qp.RX(params1, wires=wires[0]))
    qp.RX(params1, wires=wires[1])


def fubini_ansatz3(params, wires=None):
    params0 = params[0]
    params1 = params[1]
    params2 = params[2]
    qp.RX(fixed_pars[1], wires=wires[0])
    qp.RX(fixed_pars[3], wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.RX(params0, wires=wires[0])
    qp.RX(params0, wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.CNOT(wires=[wires[2], wires[0]])
    qp.RY(params1, wires=wires[0])
    qp.RY(params1, wires=wires[1])
    qp.RY(params1, wires=wires[2])
    qp.RZ(params2, wires=wires[0])
    qp.RZ(params2, wires=wires[1])
    qp.RZ(params2, wires=wires[2])


def fubini_ansatz4(params00, params_rest, wires=(0, 1, 2, 3)):
    params01 = params_rest[0]
    params10 = params_rest[1]
    params11 = params_rest[2]
    qp.RY(fixed_pars[3], wires=wires[0])
    qp.RY(fixed_pars[2], wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.RY(fixed_pars[4], wires=wires[0])
    qp.RX(params00, wires=wires[0])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RX(params01, wires=wires[1])
    qp.RZ(params10, wires=wires[1])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RZ(params11, wires=wires[1])


def fubini_ansatz5(params, wires=None):
    fubini_ansatz4(params[0], [params[0], params[1], params[1]], wires=wires)


def fubini_ansatz6(params, wires=None):
    fubini_ansatz4(params[0], [params[0], params[1], -params[1]], wires=wires)


def fubini_ansatz7(params0, params1, wires=None):
    fubini_ansatz4(params0[0], [params0[1], params1[0], params1[1]], wires=wires)


def fubini_ansatz8(x, wires=None):
    qp.RX(fixed_pars[0], wires=wires[0])
    qp.RX(x, wires=wires[0])


def fubini_ansatz9(params, wires=None):
    params0 = params[0]
    params1 = params[1]
    qp.RX(fixed_pars[1], wires=[wires[0]])
    qp.RY(fixed_pars[3], wires=[wires[0]])
    qp.RZ(fixed_pars[2], wires=[wires[0]])
    qp.RX(fixed_pars[2], wires=[wires[1]])
    qp.RY(fixed_pars[2], wires=[wires[1]])
    qp.RZ(fixed_pars[4], wires=[wires[1]])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RX(fixed_pars[0], wires=[wires[0]])
    qp.RY(fixed_pars[1], wires=[wires[0]])
    qp.RZ(fixed_pars[3], wires=[wires[0]])
    qp.RX(fixed_pars[1], wires=[wires[1]])
    qp.RY(fixed_pars[2], wires=[wires[1]])
    qp.RZ(fixed_pars[0], wires=[wires[1]])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RX(params0, wires=[wires[0]])
    qp.RX(params0, wires=[wires[1]])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RY(fixed_pars[4], wires=[wires[1]])
    qp.RY(params1, wires=[wires[0]])
    qp.RY(params1, wires=[wires[1]])
    qp.CNOT(wires=[wires[0], wires[1]])
    qp.RX(fixed_pars[2], wires=[wires[1]])


def fubini_ansatz10(weights, wires=None):
    # pylint: disable=unused-argument
    qp.templates.BasicEntanglerLayers(weights, wires=[wires[0], wires[1]])


B = np.array(
    [
        [
            [0.73, 0.49, 0.04],
            [0.29, 0.45, 0.59],
            [0.64, 0.06, 0.26],
        ],
        [
            [0.93, 0.14, 0.46],
            [0.31, 0.83, 0.79],
            [0.25, 0.40, 0.16],
        ],
    ],
    requires_grad=True,
)
fubini_ansatze_tape = [fubini_ansatz0, fubini_ansatz1, fubini_ansatz8]
fubini_params_tape = [
    (np.array([0.3434, -0.7245345], requires_grad=True),),
    (B,),
    (np.array(-0.1735, requires_grad=True),),
]

fubini_ansatze = [
    fubini_ansatz0,
    fubini_ansatz1,
    fubini_ansatz2,
    fubini_ansatz3,
    fubini_ansatz4,
    fubini_ansatz5,
    fubini_ansatz6,
    fubini_ansatz7,
    fubini_ansatz8,
    fubini_ansatz9,
    fubini_ansatz10,
]

fubini_params = [
    (np.array([0.3434, -0.7245345], requires_grad=True),),
    (B,),
    (np.array([-0.1111, -0.2222], requires_grad=True),),
    (np.array([-0.1111, -0.2222, 0.4554], requires_grad=True),),
    (
        np.array(-0.1735, requires_grad=True),
        np.array([-0.1735, -0.2846, -0.2846], requires_grad=True),
    ),
    (np.array([-0.1735, -0.2846], requires_grad=True),),
    (np.array([-0.1735, -0.2846], requires_grad=True),),
    (
        np.array([-0.1735, -0.2846], requires_grad=True),
        np.array([0.9812, -0.1492], requires_grad=True),
    ),
    (np.array(-0.1735, requires_grad=True),),
    (np.array([-0.1111, 0.3333], requires_grad=True),),
    (np.array([[0.21, 9.29], [-0.2, 0.12], [0.3, -2.1]], requires_grad=True),),
]


def autodiff_metric_tensor(ansatz, num_wires):
    """Compute the metric tensor by full state vector
    differentiation via autograd."""
    dev = qp.device("default.qubit", wires=num_wires)

    @qp.qnode(dev)
    def qnode(*params):
        ansatz(*params, wires=dev.wires)
        return qp.state()

    def mt(*params):
        state = qnode(*params)

        def rqnode(*params):
            return np.real(qnode(*params))

        def iqnode(*params):
            return np.imag(qnode(*params))

        rjac = qp.jacobian(rqnode)(*params)
        ijac = qp.jacobian(iqnode)(*params)

        if isinstance(rjac, tuple):
            out = []
            for rc, ic in zip(rjac, ijac):
                c = rc + 1j * ic
                psidpsi = np.tensordot(np.conj(state), c, axes=([0], [0]))
                out.append(
                    np.real(
                        np.tensordot(np.conj(c), c, axes=([0], [0]))
                        - np.tensordot(np.conj(psidpsi), psidpsi, axes=0)
                    )
                )
            return tuple(out)

        jac = rjac + 1j * ijac
        psidpsi = np.tensordot(np.conj(state), jac, axes=([0], [0]))
        return np.real(
            np.tensordot(np.conj(jac), jac, axes=([0], [0]))
            - np.tensordot(np.conj(psidpsi), psidpsi, axes=0)
        )

    return mt


@pytest.mark.parametrize("ansatz, params", list(zip(fubini_ansatze_tape, fubini_params_tape)))
class TestAdjointMetricTensorTape:
    """Test the adjoint method for the metric tensor when calling it directly on
    a tape.
    """

    num_wires = 3
    interfaces = ["auto", "autograd"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("interface", interfaces)
    def test_correct_output_tape_autograd(self, ansatz, params, interface):
        """Test that the output is correct when using Autograd and
        calling the adjoint metric tensor directly on a tape."""
        expected = autodiff_metric_tensor(ansatz, 3)(*params)
        dev = qp.device("default.qubit")

        wires = ("a", "b", "c")

        @qp.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, wires)
            return qp.expval(qp.PauliZ(wires[0]))

        circuit(*params)

        mt = qp.adjoint_metric_tensor(circuit)(*params)
        assert qp.math.allclose(mt, expected)

        tape = qp.workflow.construct_tape(circuit)(*params)
        met_tens = qp.adjoint_metric_tensor(tape)
        expected = qp.math.reshape(expected, qp.math.shape(met_tens))
        assert qp.math.allclose(met_tens, expected)

    @pytest.mark.jax
    @pytest.mark.skip("JAX does not support forward pass execution of the metric tensor.")
    def test_correct_output_tape_jax(self, ansatz, params):
        """Test that the output is correct when using JAX and
        calling the adjoint metric tensor directly on a tape."""

        import jax

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        j_params = tuple(jax.numpy.array(p) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface="jax")
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        circuit(*j_params)
        tape = qp.workflow.construct_tape(circuit)(*j_params)
        met_tens = qp.adjoint_metric_tensor(tape)
        expected = qp.math.reshape(expected, qp.math.shape(met_tens))
        assert qp.math.allclose(met_tens, expected)

        mt = qp.adjoint_metric_tensor(circuit)(*j_params)
        assert qp.math.allclose(mt, expected)

    interfaces = ["auto", "torch"]

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", interfaces)
    def test_correct_output_tape_torch(self, ansatz, params, interface):
        """Test that the output is correct when using Torch and
        calling the adjoint metric tensor directly on a tape."""

        import torch

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        t_params = tuple(torch.tensor(p, requires_grad=True) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        circuit(*t_params)
        mt = qp.adjoint_metric_tensor(circuit)(*t_params)
        assert qp.math.allclose(mt, expected)

        tape = qp.workflow.construct_tape(circuit)(*t_params)
        met_tens = qp.adjoint_metric_tensor(tape)
        expected = qp.math.reshape(expected, qp.math.shape(met_tens))
        assert qp.math.allclose(met_tens.detach().numpy(), expected)

    interfaces = ["auto", "tf"]

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", interfaces)
    def test_correct_output_tape_tf(self, ansatz, params, interface):
        """Test that the output is correct when using TensorFlow and
        calling the adjoint metric tensor directly on a tape."""

        import tensorflow as tf

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        t_params = tuple(tf.Variable(p) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        with tf.GradientTape():
            circuit(*t_params)
            tape = qp.workflow.construct_tape(circuit)(*t_params)
            mt = qp.adjoint_metric_tensor(tape)

        with tf.GradientTape():
            mt = qp.adjoint_metric_tensor(circuit)(*t_params)
        assert qp.math.allclose(mt, expected)

        expected = qp.math.reshape(expected, qp.math.shape(mt))
        assert qp.math.allclose(mt, expected)


class TestAdjointMetricTensorQNode:
    """Test the adjoint method for the metric tensor when calling it on
    a QNode.
    """

    num_wires = 3
    interfaces = ["auto", "autograd"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("ansatz, params", list(zip(fubini_ansatze, fubini_params)))
    @pytest.mark.parametrize("interface", interfaces)
    def test_correct_output_qnode_autograd(self, ansatz, params, interface):
        """Test that the output is correct when using Autograd and
        calling the adjoint metric tensor on a QNode."""
        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        mt = qp.adjoint_metric_tensor(circuit)(*params)

        if isinstance(mt, tuple):
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qp.math.allclose(mt, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("ansatz, params", list(zip(fubini_ansatze, fubini_params)))
    def test_correct_output_qnode_jax(self, ansatz, params):
        """Test that the output is correct when using JAX and
        calling the adjoint metric tensor on a QNode."""

        import jax

        jax.config.update("jax_enable_x64", True)

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        j_params = tuple(jax.numpy.array(p) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface="jax")
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        mt = qp.adjoint_metric_tensor(circuit, argnums=list(range(len(j_params))))(*j_params)

        if isinstance(mt, tuple):
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qp.math.allclose(mt, expected)

    interfaces = ["auto", "torch"]

    @pytest.mark.torch
    @pytest.mark.parametrize("ansatz, params", list(zip(fubini_ansatze, fubini_params)))
    @pytest.mark.parametrize("interface", interfaces)
    def test_correct_output_qnode_torch(self, ansatz, params, interface):
        """Test that the output is correct when using Torch and
        calling the adjoint metric tensor on a QNode."""

        import torch

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        t_params = tuple(torch.tensor(p, requires_grad=True, dtype=torch.float64) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        mt = qp.adjoint_metric_tensor(circuit)(*t_params)

        if isinstance(mt, tuple):
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qp.math.allclose(mt, expected)

    interfaces = ["auto", "tf"]

    @pytest.mark.tf
    @pytest.mark.parametrize("ansatz, params", list(zip(fubini_ansatze, fubini_params)))
    @pytest.mark.parametrize("interface", interfaces)
    def test_correct_output_qnode_tf(self, ansatz, params, interface):
        """Test that the output is correct when using TensorFlow and
        calling the adjoint metric tensor on a QNode."""

        import tensorflow as tf

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        t_params = tuple(tf.Variable(p, dtype=tf.float64) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        with tf.GradientTape():
            mt = qp.adjoint_metric_tensor(circuit)(*t_params)

        if isinstance(mt, tuple):
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qp.math.allclose(mt, expected)


diff_fubini_ansatze = [
    fubini_ansatz0,
    fubini_ansatz2,
    fubini_ansatz10,
]

diff_fubini_params = [
    fubini_params[0],
    fubini_params[2],
    fubini_params[10],
]


@pytest.mark.parametrize("ansatz, params", list(zip(diff_fubini_ansatze, diff_fubini_params)))
class TestAdjointMetricTensorDifferentiability:
    """Test the differentiability of the adjoint method for the metric
    tensor when calling it on a QNode.
    """

    num_wires = 3

    @pytest.mark.autograd
    def test_autograd(self, ansatz, params):
        """Test that the derivative is correct when using Autograd and
        calling the adjoint metric tensor on a QNode."""
        exp_fn = autodiff_metric_tensor(ansatz, self.num_wires)
        expected = qp.jacobian(exp_fn)(*params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface="autograd")
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        mt_jac = qp.jacobian(qp.adjoint_metric_tensor(circuit))(*params)

        if isinstance(mt_jac, tuple):
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt_jac, expected))
        else:
            assert qp.math.allclose(mt_jac, expected)

    @pytest.mark.jax
    def test_correct_output_qnode_jax(self, ansatz, params):
        """Test that the derivative is correct when using JAX and
        calling the adjoint metric tensor on a QNode."""

        import jax

        expected = qp.jacobian(autodiff_metric_tensor(ansatz, self.num_wires))(*params)
        j_params = tuple(jax.numpy.array(p) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface="jax")
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        mt_fn = qp.adjoint_metric_tensor(circuit)
        argnums = list(range(len(params)))
        mt_jac = jax.jacobian(mt_fn, argnums=argnums)(*j_params)

        if isinstance(mt_jac, tuple):
            if not isinstance(expected, tuple) and len(mt_jac) == 1:
                expected = (expected,)
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt_jac, expected))
        else:
            assert qp.math.allclose(mt_jac, expected)

    @pytest.mark.torch
    def test_correct_output_qnode_torch(self, ansatz, params):
        """Test that the derivative is correct when using Torch and
        calling the adjoint metric tensor on a QNode."""

        import torch

        expected = qp.jacobian(autodiff_metric_tensor(ansatz, self.num_wires))(*params)
        t_params = tuple(torch.tensor(p, requires_grad=True, dtype=torch.float64) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface="torch")
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        mt_fn = qp.adjoint_metric_tensor(circuit)
        mt_jac = torch.autograd.functional.jacobian(mt_fn, *t_params)

        if isinstance(mt_jac, tuple):
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt_jac, expected))
        else:
            assert qp.math.allclose(mt_jac, expected)

    @pytest.mark.tf
    def test_correct_output_qnode_tf(self, ansatz, params):
        """Test that the derivative is correct when using TensorFlow and
        calling the adjoint metric tensor on a QNode."""

        import tensorflow as tf

        expected = qp.jacobian(autodiff_metric_tensor(ansatz, self.num_wires))(*params)
        t_params = tuple(tf.Variable(p, dtype=tf.float64) for p in params)
        dev = qp.device("default.qubit", wires=self.num_wires)

        @qp.qnode(dev, interface="tf")
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires)
            return qp.expval(qp.PauliZ(0))

        with tf.GradientTape() as t:
            mt = qp.adjoint_metric_tensor(circuit)(*t_params)

        mt_jac = t.jacobian(mt, t_params)
        if isinstance(mt_jac, tuple):
            if not isinstance(expected, tuple) and len(mt_jac) == 1:
                expected = (expected,)
            assert all(qp.math.allclose(_mt, _exp) for _mt, _exp in zip(mt_jac, expected))
        else:
            assert qp.math.allclose(mt_jac, expected)


def test_error_finite_shots():
    """Test that an error is raised if the device has a finite number of shots set."""
    with qp.queuing.AnnotatedQueue() as q:
        qp.RX(0.2, wires=0)
        qp.RY(1.9, wires=1)
    tape = qp.tape.QuantumScript.from_queue(q, shots=1)

    with pytest.raises(ValueError, match="The adjoint method for the metric tensor"):
        qp.adjoint_metric_tensor(tape)


def test_works_with_state_prep():
    """Test that a state preparation operation is respected."""
    dev = qp.device("default.qubit")

    # Some random normalized state, no particular relevance
    init_state = onp.array([0.16769259, 0.71277864, 0.54562903, 0.4075718])

    def ansatz(angles, wires):
        qp.StatePrep(init_state, wires=wires)
        qp.Hadamard(wires[0])
        qp.RX(angles[0], wires=wires[0])
        qp.S(wires[1])
        qp.RY(angles[1], wires=wires[1])

    @qp.qnode(dev)
    def circuit(angles):
        ansatz(angles, wires=[0, 1])
        return qp.expval(qp.Z(0) @ qp.X(1))

    angles = np.random.uniform(size=(2,), requires_grad=True)
    qfim = qp.adjoint_metric_tensor(circuit)(angles)
    autodiff_qfim = autodiff_metric_tensor(ansatz, 2)(angles)
    assert onp.allclose(qfim, autodiff_qfim)
