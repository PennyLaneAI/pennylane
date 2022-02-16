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
from collections import OrderedDict
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.fourier.qnode_spectrum import qnode_spectrum, _process_ids


def circuit_0(a):
    [qml.RX(a, wires=0) for i in range(4)]
    qml.Hadamard(0)
    return qml.expval(qml.PauliZ(0))


def circuit_1(a, b):
    qml.RZ(-a / 3, wires=0)
    qml.RX(a / 5, wires=1)
    qml.CRZ(0.3, wires=[1, 0])
    qml.CNOT(wires=[0, 1])
    qml.RY(b * 2, wires=1)
    qml.RZ(-b, wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_2(x):
    [qml.RX(x[i], wires=0) for i in range(3)]
    return qml.expval(qml.PauliZ(0))


def circuit_3(x, y):
    [qml.RX(0.1 * (i + 1) * x[i], wires=0) for i in range(3)]
    for i in range(2):
        [qml.RY((i + j) * y[i, j], wires=1) for j in range(2)]
    qml.Hadamard(0)
    return qml.expval(qml.PauliZ(0))


def circuit_4(x, y):
    perm_4 = ([2, 0, 1], [1, 2, 0, 3])
    for i, j in enumerate(perm_4[0]):
        qml.RX(1.2 * (i + 1) * x[j], wires=0)
    for j in perm_4[1]:
        qml.RY((j // 2 + (j % 2)) * y[j // 2, j % 2], wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_5(x, y, z):
    [qml.RX(i * x[i], wires=0) for i in range(3)]
    qml.RZ(y[0, 1] - y[1, 0], wires=1)
    qml.RY(z[0] + 0.2 * z[1], wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_6(x, y, z):
    [qml.RX(x[i] ** i, wires=0) for i in range(3)]
    qml.RZ(y[0, 1] / y[1, 0], wires=1)
    qml.RY(z[0] + 0.2 ** z[1], wires=1)
    return qml.expval(qml.PauliZ(0))


def circuit_7(a):
    qml.Hadamard(0)
    [qml.RX(qml.math.sin(a), wires=0) for i in range(4)]
    return qml.expval(qml.PauliZ(0))


def circuit_8(a, x):
    [qml.RX(a, wires=0) for i in range(4)]
    [qml.RX(x[i] * a, wires=1) for i in range(3)]
    return qml.expval(qml.PauliZ(0))


circuits_linear = [circuit_0, circuit_1, circuit_2, circuit_3, circuit_4, circuit_5]
expected_spectra = [
    OrderedDict([("a", {(): [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0]})]),
    OrderedDict(
        [
            ("a", {(): [-8 / 15, -5 / 15, -3 / 15, -2 / 15, 0, 2 / 15, 3 / 15, 5 / 15, 8 / 15]}),
            ("b", {(): [-3, -2, -1, 0, 1, 2, 3]}),
        ]
    ),
    OrderedDict([("x", {(0,): [-1.0, 0, 1.0], (1,): [-1.0, 0, 1.0], (2,): [-1.0, 0, 1.0]})]),
    OrderedDict(
        [
            ("x", {(0,): [-0.1, 0, 0.1], (1,): [-0.2, 0, 0.2], (2,): [-0.3, 0, 0.3]}),
            (
                "y",
                {
                    (0, 0): [0],
                    (0, 1): [-1.0, 0, 1.0],
                    (1, 0): [-1.0, 0, 1.0],
                    (1, 1): [-2.0, 0.0, 2.0],
                },
            ),
        ]
    ),
    OrderedDict(
        [
            ("x", {(2,): [-1.2, 0, 1.2], (0,): [-2.4, 0, 2.4], (1,): [-3.6, 0, 3.6]}),
            (
                "y",
                {
                    (0, 1): [-1.0, 0, 1.0],
                    (1, 0): [-1.0, 0, 1.0],
                    (0, 0): [0],
                    (1, 1): [-2.0, 0.0, 2.0],
                },
            ),
        ]
    ),
    OrderedDict(
        [
            ("x", {(0,): [0], (1,): [-1.0, 0, 1.0], (2,): [-2.0, 0, 2.0]}),
            ("y", {(0, 1): [-1.0, 0, 1.0], (1, 0): [-1.0, 0, 1.0], (0, 0): [0], (1, 1): [0]}),
            ("z", {(0,): [-1.0, 0, 1.0], (1,): [-0.2, 0, 0.2], (2,): [0], (3,): [0]}),
        ]
    ),
]

circuits_nonlinear = [circuit_6, circuit_7, circuit_8]

a = pnp.array(0.812, requires_grad=True)
b = pnp.array(-5.231, requires_grad=True)
x = pnp.array([0.1, -1.9, 0.7], requires_grad=True)
y = pnp.array([[0.4, 5.5], [1.6, 5.1]], requires_grad=True)
z = pnp.array([-1.9, -0.1, 0.49, 0.24], requires_grad=True)
all_args = [(a,), (a, b), (x,), (x, y), (x, y), (x, y, z)]
all_args_nonlinear = [(x, y, z), (a,), (a, x)]

# Test data for ``process_ids``
process_id_cases = [
    # circuit         input           expected output
    #       encoding_args, argnum  encoding_args, argnum
    # We will test that ellipses are filled in correctly in encoding_args, that
    # the argnum are compatible with the encoding_args and that encoding_args takes
    # precedence as input over argnum, i.e. if both are provided, argnum is overwritten
    (circuit_0, {"a"}, None, {"a": ...}, [0]),
    (circuit_0, None, 0, {"a": ...}, [0]),
    (circuit_0, None, -1, {"a": ...}, [-1]),
    (circuit_0, None, None, {"a": ...}, [0]),
    (circuit_0, {"a": [()]}, None, {"a": [()]}, [0]),
    (circuit_1, {"b"}, None, {"b": ...}, [1]),
    (circuit_1, None, None, {"a": ..., "b": ...}, [0, 1]),
    (circuit_1, {"a"}, [4], {"a": ...}, [0]),
    (circuit_2, {"x"}, None, {"x": ...}, [0]),
    (circuit_2, None, [0], {"x": ...}, [0]),
    (circuit_2, None, [-1], {"x": ...}, [-1]),
    (circuit_2, {"x": [(0,), (2,)]}, [0], {"x": [(0,), (2,)]}, [0]),
    (circuit_2, {"x": ...}, [0], {"x": ...}, [0]),
    (circuit_3, {"y"}, None, {"y": ...}, [1]),
    (circuit_3, None, [1], {"y": ...}, [1]),
    (
        circuit_3,
        OrderedDict({"y": [(0, 1), (1, 0)], "x": [(0,), (2,)]}),
        [0],
        {"x": [(0,), (2,)], "y": [(0, 1), (1, 0)]},
        [0, 1],
    ),
    (circuit_3, {"y": ..., "x": [(1,), (2,)]}, None, {"x": [(1,), (2,)], "y": ...}, [0, 1]),
    (circuit_4, {"y", "x"}, None, {"x": ..., "y": ...}, [0, 1]),
    (circuit_4, None, [1], {"y": ...}, [1]),
    (circuit_5, None, None, {"x": ..., "y": ..., "z": ...}, [0, 1, 2]),
    (circuit_5, {"y"}, None, {"y": ...}, [1]),
]
process_id_cases = [entry[:3] + (OrderedDict(entry[3]),) + entry[4:] for entry in process_id_cases]

process_id_cases_unknown_arg = [
    (circuit_0, {"b"}, None),
    (circuit_1, {"b", "c"}, [0]),
    (circuit_2, {"a", "x"}, [0, 1]),
    (circuit_3, {"xy", "x"}, None),
    (circuit_4, {"x", "z"}, None),
    (circuit_5, {"zy", "x"}, None),
]


class TestHelpers:
    @pytest.mark.parametrize(
        "circuit, enc_args, argnum, enc_args_exp, argnum_exp",
        process_id_cases,
    )
    def test_process_ids(self, circuit, enc_args, argnum, enc_args_exp, argnum_exp):
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev)
        encoding_args, argnum = _process_ids(enc_args, argnum, qnode)
        assert encoding_args == enc_args_exp
        assert all(np.issubdtype(type(num), int) for num in argnum)
        assert np.allclose(argnum, argnum_exp)

    @pytest.mark.parametrize(
        "circuit, enc_args, argnum",
        process_id_cases_unknown_arg,
    )
    def test_process_ids_unknown_arg(self, circuit, enc_args, argnum):
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev)
        with pytest.raises(ValueError, match="Not all names in"):
            _process_ids(enc_args, argnum, qnode)

    def test_process_ids_index_error(self):
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit_0, dev)
        with pytest.raises(IndexError, match="x"):
            _process_ids(None, [5], qnode)


class TestCircuits:
    """Tests that the spectrum is returned as expected."""

    @pytest.mark.parametrize(
        "circuit, args, expected",
        zip(circuits_linear, all_args, expected_spectra),
    )
    def test_various_circuits(self, circuit, args, expected):
        """Test the spectrum for some simple standard circuits."""
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev)
        spec = qnode_spectrum(qnode)(*args)
        assert spec.keys() == expected.keys()
        for outer_key in spec.keys():
            assert spec[outer_key].keys() == expected[outer_key].keys()
            for key in spec[outer_key]:
                assert np.allclose(spec[outer_key][key], expected[outer_key][key])

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

        res = qnode_spectrum(circuit)(0.1)
        expected_degree = n_qubits * n_layers
        assert list(res.keys()) == ["x"] and list(res["x"].keys()) == [()]
        assert np.allclose(res["x"][()], range(-expected_degree, expected_degree + 1))

    def test_argnum(self):
        """Test that the spectrum is computed for the arguments specified by ``argnum``."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(0.2 * y, wires=0)
            qml.RY(3 * y, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        x, y = pnp.array([0.2, 0.1], requires_grad=True)
        y_freq = [-3.2, -3.0, -2.8, -0.2, 0.0, 0.2, 2.8, 3.0, 3.2]

        res = qnode_spectrum(circuit, argnum=[0])(x, y)
        assert res == {"x": {(): [-1.0, 0.0, 1.0]}}

        res = qnode_spectrum(circuit, argnum=[0, 1])(x, y)
        assert res == {"x": {(): [-1.0, 0.0, 1.0]}, "y": {(): y_freq}}

        res = qnode_spectrum(circuit)(x, y)
        assert res == {"x": {(): [-1.0, 0.0, 1.0]}, "y": {(): y_freq}}

    def test_encoding_args(self):
        """Test that the spectrum is computed for the arguments
        specified by ``encoding_args``."""
        dev = qml.device("default.qubit", wires=2)
        z_0 = 2.1

        @qml.qnode(dev)
        def circuit(x, Y, z=z_0):
            qml.RX(z * x, wires=0)
            qml.RY(0.2 * Y[0, 1, 0], wires=0)
            qml.RY(3 * Y[0, 0, 0], wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        x = pnp.array(-1.5, requires_grad=True)
        Y = pnp.array([0.2, -1.2, 9.2, -0.2, 1.1, 4, -0.201, 0.8], requires_grad=True).reshape(
            (2, 2, 2)
        )
        z = 1.2

        res = qnode_spectrum(circuit, encoding_args={"x"})(x, Y, z=z)
        assert res == {"x": {(): [-z, 0.0, z]}}

        res = qnode_spectrum(circuit, encoding_args={"x"})(x, Y)
        assert res == {"x": {(): [-z_0, 0.0, z_0]}}

        res = qnode_spectrum(circuit, encoding_args={"x": ..., "Y": [(0, 0, 0), (1, 0, 1)]})(x, Y)
        assert res == {
            "x": {(): [-z_0, 0.0, z_0]},
            "Y": {(0, 0, 0): [-3.0, 0.0, 3.0], (1, 0, 1): [0.0]},
        }

    def test_spectrum_changes_with_qnode_args(self):
        """Test that the spectrum changes per call if a qnode keyword argument
        changes the circuit architecture."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, last_gate=False):
            qml.RX(x, wires=0)
            qml.RX(x, wires=1)
            if last_gate:
                qml.RX(x, wires=2)
            return qml.expval(qml.PauliZ(wires=0))

        x = 0.9
        res_true = qnode_spectrum(circuit, argnum=[0])(x, last_gate=True)
        assert np.allclose(res_true["x"][()], range(-3, 4))

        res_false = qnode_spectrum(circuit, argnum=[0])(x, last_gate=False)
        assert np.allclose(res_false["x"][()], range(-2, 3))

    def test_multi_par_error(self):
        """Test that an error is thrown if the spectrum of
        a multi-parameter gate that cannot be decomposed is requested."""
        dev = qml.device("default.qubit", wires=3)

        class nondecompRot(qml.Rot):
            @staticmethod
            def compute_decomposition(phi, theta, omega, wires):
                """Pseudo-decomposition: Just return the gate itself."""
                return [nondecompRot(phi, theta, omega, wires=wires)]

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            nondecompRot(0.2, x, 0.4, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        with pytest.raises(ValueError, match="Can only consider one-parameter gates"):
            qnode_spectrum(circuit)(1.5)

    @pytest.mark.parametrize(
        "measure_fn, measure_args",
        [(qml.state, ()), (qml.sample, (qml.PauliZ(0),)), (qml.var, (qml.PauliZ(0),))],
    )
    def test_wrong_return_type_error(self, measure_fn, measure_args):
        """Test that an error is thrown if the QNode has a ``MeasurementProcess``
        with an inadmissable ``return_type``."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return [qml.expval(qml.PauliX(1)), measure_fn(*measure_args)]

        with pytest.raises(ValueError, match=f"{measure_fn.__name__} is not supported"):
            qnode_spectrum(circuit)(1.5)


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


expected_result = {
    "x": {
        (0,): [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        (1,): [-2.0, -1.0, 0.0, 1.0, 2.0],
        (2,): [-2.0, -1.0, 0.0, 1.0, 2.0],
    }
}


class TestAutograd:
    def test_integration_autograd(self):
        """Test that the spectra of a circuit is calculated correctly
        in the autograd interface."""

        x = pnp.array([1.0, 2.0, 3.0], requires_grad=True)
        w = pnp.array([[-1, -2, -3], [-4, -5, -6]], dtype=float, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="autograd")

        res = qnode_spectrum(qnode, argnum=0)(x, w)
        assert res
        assert res == expected_result

    @pytest.mark.parametrize("circuit, args", zip(circuits_nonlinear, all_args_nonlinear))
    def test_nonlinear_error(self, circuit, args):
        """Test that an error is raised if non-linear
        preprocessing happens in a circuit."""
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface="autograd")
        with pytest.raises(ValueError, match="The Jacobian of the classical preprocessing"):
            qnode_spectrum(qnode)(*args)


class TestTorch:
    def test_integration_torch(self):
        """Test that the spectra of a circuit is calculated correctly
        in the torch interface."""

        torch = pytest.importorskip("torch")
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        w = torch.tensor([[-1, -2, -3], [-4, -5, -6]], dtype=float)

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="torch")

        res = qnode_spectrum(qnode, argnum=0)(x, w)
        assert res
        assert res == expected_result

    @pytest.mark.parametrize("circuit, args", zip(circuits_nonlinear, all_args_nonlinear))
    def test_nonlinear_error(self, circuit, args):
        """Test that an error is raised if non-linear
        preprocessing happens in a circuit."""
        torch = pytest.importorskip("torch")
        args = tuple(torch.tensor(arg) for arg in args)
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface="torch")
        with pytest.raises(ValueError, match="The Jacobian of the classical preprocessing"):
            qnode_spectrum(qnode)(*args)


class TestTensorflow:
    def test_integration_tf(self):
        """Test that the spectra of a circuit is calculated correctly
        in the tf interface."""

        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="tf")

        x = tf.Variable([1.0, 2.0, 3.0])
        w = tf.constant([[-1, -2, -3], [-4, -5, -6]], dtype=float)
        res = qnode_spectrum(qnode, argnum=[0])(x, w)

        assert res
        assert res == expected_result

    @pytest.mark.parametrize("circuit, args", zip(circuits_nonlinear, all_args_nonlinear))
    def test_nonlinear_error(self, circuit, args):
        """Test that an error is raised if non-linear
        preprocessing happens in a circuit."""
        tf = pytest.importorskip("tensorflow")
        args = tuple(tf.Variable(arg, dtype=np.float64) for arg in args)
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface="tf")
        with pytest.raises(ValueError, match="The Jacobian of the classical preprocessing"):
            qnode_spectrum(qnode)(*args)


class TestJax:
    def test_integration_jax(self):
        """Test that the spectra of a circuit is calculated correctly
        in the jax interface."""

        jax = pytest.importorskip("jax")

        x = jax.numpy.array([1.0, 2.0, 3.0])
        w = [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="jax")

        res = qnode_spectrum(qnode, argnum=0)(x, w)

        assert res
        assert res == expected_result

    @pytest.mark.parametrize("circuit, args", zip(circuits_nonlinear, all_args_nonlinear))
    def test_nonlinear_error(self, circuit, args):
        """Test that an error is raised if non-linear
        preprocessing happens in a circuit."""
        pytest.importorskip("jax")
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(circuit, dev, interface="jax")
        with pytest.raises(ValueError, match="The Jacobian of the classical preprocessing"):
            qnode_spectrum(qnode)(*args)
