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
Tests for the Fourier spectrum transform.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.argmap import ArgMap
from pennylane.fourier.spectrum import (
    spectrum,
    _join_spectra,
    _get_spectrum,
    _get_and_validate_classical_jacobian,
)
from pennylane.transforms import classical_jacobian

def circuit_0(a):
    [qml.RX(a, wires=0) for i in range(4)]
    return qml.expval(qml.PauliZ(0))

def circuit_1(a, b):
    qml.RZ(-a / 3, wires=0)
    qml.RX(a / 5, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(b * 2, wires=1)
    qml.RZ(-b, wires=1)
    return qml.expval(qml.PauliZ(0))

def circuit_2(x):
    [qml.RX(x[i], wires=0) for i in range(3)]
    return qml.expval(qml.PauliZ(0))

def circuit_3(x, y):
    [qml.RX(0.1*(i+1)*x[i], wires=0) for i in range(3)]
    for i in range(2):
        [qml.RY((i+j)*y[i, j], wires=1) for j in range(2)]
    return qml.expval(qml.PauliZ(0))

def circuit_4(x, y):
    perm_4 = ([2, 0, 1], [1, 2, 0, 3])
    for i in perm_4[0]:
        qml.RX(1.2*(i+1)*x[i], wires=0)
    for j in perm_4[1]:
        qml.RY(y[j // 2, j % 2], wires=1)
    return qml.expval(qml.PauliZ(0))

def circuit_5(x, y, z):
    [qml.RX(i*x[i], wires=0) for i in range(3)]
    qml.RZ(y[0, 1] - y[1, 0], wires=1)
    qml.RY(z[0] + 0.2 * z[1], wires=1)
    return qml.expval(qml.PauliZ(0))

circuits = [circuit_0, circuit_1, circuit_2, circuit_3, circuit_4, circuit_5]

a = 0.812
b = -5.231
x = np.array([0.1, -1.9, 0.7])
y = np.array([[0.4, 5.5], [1.6, 5.1]])
z = np.array([-1.9, -0.1, 0.49, 0.24])
all_args = [(a,), (a, b), (x,), (x, y), (x, y), (x, y, z)]

interfaces = [('tf', 'tensorflow'), ('torch',)*2, ('autograd', 'pennylane'), ('jax',)*2]

class TestHelpers:
    @pytest.mark.parametrize(
        "spectrum1, spectrum2, expected",
        [
            ({0, 1}, {0, 1}, [0, 1, 2]),
            ({0, 3}, {0, 5}, [0, 2, 3, 5, 8]),
            ({0, 1, 2}, {0, 1}, [0, 1, 2, 3]),
            ({0, 0.5}, {0, 1}, [0, 0.5, 1.0, 1.5]),
            ({0, 0.5}, {}, [0, 0.5]),
            ({0, 0.5}, {0}, [0, 0.5]),
            ({}, {0, 0.5}, [0, 0.5]),
            ({0}, {0, 0.5}, [0, 0.5]),
        ],
    )
    def test_join_spectra(self, spectrum1, spectrum2, expected, tol):
        """Test that spectra are joined correctly."""
        joined = _join_spectra(spectrum1, spectrum2)
        assert np.allclose(sorted(joined), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.RX(0.1, wires=0), [0, 1]),  # generator is a class
            (qml.RY(0.1, wires=0), [0, 1]),  # generator is a class
            (qml.RZ(0.1, wires=0), [0, 1]),  # generator is a class
            (qml.PhaseShift(0.5, wires=0), [0, 1]),  # generator is an array
            (qml.CRX(0.2, wires=[0, 1]), [0, 0.5, 1]),  # generator is an array
            (qml.ControlledPhaseShift(0.5, wires=[0, 1]), [0, 1]),  # generator is an array
        ],
    )
    def test_get_spectrum(self, op, expected, tol):
        """Test that the spectrum is correctly extracted from an operator."""
        spec = _get_spectrum(op)
        assert np.allclose(sorted(spec), expected, atol=tol, rtol=0)

    def test_get_spectrum_complains_no_generator(self):
        """Test that an error is raised if the operator has no generator defined."""

        # Observables have no generator attribute
        with pytest.raises(ValueError, match="Generator of operation"):
            _get_spectrum(qml.P(wires=0))

        # CNOT is an operation where generator is an abstract property
        with pytest.raises(ValueError, match="Generator of operation"):
            _get_spectrum(qml.CNOT(wires=[0, 1]))







class TestCircuits:
    """Tests that the spectrum is returned as expected."""

    @pytest.mark.parametrize("n_layers, n_qubits", [(1, 1), (2, 3), (4, 1)])
    def test_spectrum_grows_with_gates(self, n_layers, n_qubits):
        """Test that the spectrum grows linearly with the number of
        encoding gates if we use Pauli rotation encoding."""

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x):
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(x, wires=i)
                    qml.RY(0.4, wires=i)
            return qml.expval(qml.PauliZ(wires=0))

        res = spectrum(circuit)(0.1)
        expected_degree = n_qubits * n_layers
        assert np.allclose(res[0], range(-expected_degree, expected_degree + 1))

    def test_argnum(self):
        """Test that the spectrum contains the ids provided in encoding_gates, or
        all ids if encoding_gates is None."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(wires=0))
        x, y = [0.2, 0.1]

        res = spectrum(circuit, argnum=[0])(x, y)
        assert res == ArgMap({0: [-1.0, 0.0, 1.0]})

        res = spectrum(circuit, argnum=[0, 1])(x, y)
        assert res == ArgMap({0: [-1.0, 0.0, 1.0], 1: [-1.0, 0.0, 1.0]})

        res = spectrum(circuit)(x, y)
        assert res == ArgMap({0: [-1.0, 0.0, 1.0], 1: [-1.0, 0.0, 1.0]})

        with pytest.raises(ValueError, match="Could not compute Jacobian"):
            res = spectrum(circuit, argnum=[3])(x, y)

    def test_spectrum_changes_with_qnode_args(self):
        """Test that the spectrum changes per call if a qnode argument changes the
        circuit architecture."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, last_gate=False):
            qml.RX(x, wires=0)
            qml.RX(x, wires=1)
            if last_gate:
                qml.RX(x, wires=2)
            return qml.expval(qml.PauliZ(wires=0))

        x = 0.9
        res_true = spectrum(circuit, argnum=[0])(x, last_gate=True)
        assert np.allclose(res_true[0], range(-3, 4))

        res_false = spectrum(circuit, argnum=[0])(x, last_gate=False)
        assert np.allclose(res_false[0], range(-2, 3))

    def test_input_gates_not_of_correct_form(self):
        """Test that an error is thrown if gates marked as encoding gates
        are not single-parameter gates."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Rot(0.2, x, 0.4, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        with pytest.raises(ValueError, match="Can only consider one-parameter gates"):
            spectrum(circuit)(1.5)


def circuit(x, w):
    """Test circuit"""
    for l in range(2):
        for i in range(3):
            qml.RX(x[i], wires=0)
            qml.RY(w[l][i], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
    qml.RZ(x[0], wires=0)
    return qml.expval(qml.PauliZ(wires=0))


expected_result = ArgMap({
    (0, (0,)): [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    (0, (1,)): [-2.0, -1.0, 0.0, 1.0, 2.0],
    (0, (2,)): [-2.0, -1.0, 0.0, 1.0, 2.0],
})


class TestAutograd:

    @pytest.mark.parametrize("circuit, args", zip(circuits, all_args))
    def test_jacobian_validation(self, circuit, args):
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface="autograd")
        class_jac = classical_jacobian(qnode, argnum=list(range(len(args))))(*args)
        validated_class_jac = _get_and_validate_classical_jacobian(qnode, argnum=list(range(len(args))), args=args, kwargs={})
        if isinstance(class_jac, tuple):
            assert all((np.allclose(_jac, val_jac) for _jac, val_jac in zip(class_jac, validated_class_jac)))
        else:
            assert np.allclose(class_jac, validated_class_jac)

    def test_integration_autograd(self):
        """Test that the spectra of a circuit is calculated correctly
        in the autograd interface."""

        x = pnp.array([1.0, 2.0, 3.0])
        w = pnp.array([[-1, -2, -3], [-4, -5, -6]])

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="autograd")

        res = spectrum(qnode, argnum=0)(x, w)
        assert res
        assert res==expected_result

class TestTorch:

    @pytest.mark.parametrize("circuit, args", zip(circuits, all_args))
    def test_jacobian_validation(self, circuit, args):
        torch = pytest.importorskip("torch")
        args = tuple((torch.tensor(arg) for arg in args))
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface="torch")
        class_jac = classical_jacobian(qnode)(*args)
        validated_class_jac = _get_and_validate_classical_jacobian(qnode, argnum=list(range(len(args))), args=args, kwargs={})
        if isinstance(class_jac, tuple):
            assert all((np.allclose(_jac, val_jac) for _jac, val_jac in zip(class_jac, validated_class_jac)))
        else:
            assert np.allclose(class_jac, validated_class_jac)

    def test_integration_torch(self):
        """Test that the spectra of a circuit is calculated correctly
        in the torch interface."""

        torch = pytest.importorskip("torch")
        x = torch.tensor([1.0, 2.0, 3.0])
        w = torch.tensor([[-1, -2, -3], [-4, -5, -6]], dtype=float)

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="torch")

        res = spectrum(qnode, argnum=0)(x, w)
        assert res
        assert res==expected_result

class TestTensorflow:

    @pytest.mark.parametrize("circuit, args", zip(circuits, all_args))
    def test_jacobian_validation(self, circuit, args):
        tf = pytest.importorskip("tensorflow")
        args = tuple((tf.Variable(arg, dtype=tf.double) for arg in args))
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface="tf")
        class_jac = classical_jacobian(qnode)(*args)
        validated_class_jac = _get_and_validate_classical_jacobian(qnode, argnum=list(range(len(args))), args=args, kwargs={})
        if isinstance(class_jac, tuple):
            assert all((np.allclose(_jac, val_jac) for _jac, val_jac in zip(class_jac, validated_class_jac)))
        else:
            assert np.allclose(class_jac, validated_class_jac)

    def test_integration_tf(self):
        """Test that the spectra of a circuit is calculated correctly
        in the tf interface."""
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="tf")

        x = tf.Variable([1.0, 2.0, 3.0])
        w = tf.constant([[-1, -2, -3], [-4, -5, -6]], dtype=float)
        res = spectrum(qnode, argnum=[0])(x, w)

        assert res
        assert res==expected_result

    def test_integration_jax(self):
        """Test that the spectra of a circuit is calculated correctly
        in the jax interface."""

        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        x = jnp.array([1.0, 2.0, 3.0])
        w = [[-1, -2, -3], [-4, -5, -6]]

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="jax")

        res = spectrum(qnode, argnum=0)(x, w)

        assert res
        assert res==expected_result
