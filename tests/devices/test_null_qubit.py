# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.plugin.NullQubit` device.
"""
import cmath

# pylint: disable=protected-access,cell-var-from-loop
import math

import pytest
import pennylane as qml
from pennylane import numpy as np, DeviceError
from pennylane.devices.null_qubit import NullQubit
from pennylane import Tracker

from collections import defaultdict


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def nullqubit_device(request):
    def _device(wires):
        return qml.device(
            "null.qubit", wires=wires, r_dtype=request.param[0], c_dtype=request.param[1]
        )

    return _device


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("null.qubit", wires=1, shots=1, analytic=True)


def test_dtype_errors():
    """Test that if an incorrect dtype is provided to the device then an error is raised."""
    with pytest.raises(DeviceError, match="Real datatype must be a floating point type."):
        qml.device("null.qubit", wires=1, r_dtype=np.complex128)
    with pytest.raises(
        DeviceError, match="Complex datatype must be a complex floating point type."
    ):
        qml.device("null.qubit", wires=1, c_dtype=np.float64)


def test_custom_op_with_matrix():
    """Test that a dummy op with a matrix is supported."""

    class DummyOp(qml.operation.Operation):
        num_wires = 1

        def compute_matrix(self):
            return np.eye(2)

    with qml.tape.QuantumTape() as tape:
        DummyOp(0)
        qml.state()

    dev = qml.device("null.qubit", wires=1)
    assert dev.execute(tape) == [0.0]


class TestApply:
    """Tests that operations and inverses of certain operations are applied correctly."""

    @pytest.mark.parametrize(
        "operation,input",
        [
            (
                qml.BasisState,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            ),
            (
                qml.QubitStateVector,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            ),
        ],
    )
    def test_apply_operation_state_preparation(self, nullqubit_device, operation, input):
        """Tests that the null.qubit does nothing regarding state initialization."""

        input = np.array(input)
        dev = nullqubit_device(wires=2)
        dev.reset()
        dev.apply([operation(input, wires=[0, 1])])
        assert dev._state == [0.0]

    test_data_single_wire_with_parameters = [
        (qml.PhaseShift, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.RX, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.RY, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.RZ, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.MultiRZ, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.Rot, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 2, math.pi / 4, math.pi / 8]),
        (
            qml.QubitUnitary,
            [1 / math.sqrt(5), 2 / math.sqrt(5)],
            [
                np.array(
                    [
                        [1j / math.sqrt(2), 1j / math.sqrt(2)],
                        [1j / math.sqrt(2), -1j / math.sqrt(2)],
                    ]
                )
            ],
        ),
        (qml.DiagonalQubitUnitary, [1 / math.sqrt(5), 2 / math.sqrt(5)], [np.array([-1, 1])]),
    ]

    @pytest.mark.parametrize(
        "op",
        [
            qml.SingleExcitation,
            qml.SingleExcitationPlus,
            qml.SingleExcitationMinus,
            qml.DoubleExcitation,
            qml.DoubleExcitationPlus,
            qml.DoubleExcitationMinus,
            qml.OrbitalRotation,
            qml.QubitSum,
            qml.QubitCarry,
        ],
    )
    def test_advanced_op(self, nullqubit_device, op):
        """Test qchem and arithmetic operations."""

        dev = nullqubit_device(wires=4)

        n_wires = op.num_wires
        n_params = op.num_params

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            if n_params == 0:
                op(wires=range(n_wires))
            elif n_params == 1:
                op(0.543, wires=range(n_wires))
            else:
                op([0.543] * n_params, wires=range(n_wires))
            return qml.state()

        assert circuit() == [0.0]


class TestExpval:
    """Tests that expectation values are properly (not) calculated."""

    @pytest.mark.parametrize(
        "operation,input",
        [
            (qml.PauliX, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.PauliY, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.PauliZ, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.Hadamard, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.Identity, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        ],
    )
    def test_expval_single_wire_no_parameters(self, nullqubit_device, operation, input):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        dev = nullqubit_device(wires=1)
        dev.reset()
        dev.apply([qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates())
        res = dev.expval(obs)
        assert res == [0.0]

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (qml.Hermitian, [1, 0], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [0, 1], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [1 / math.sqrt(2), -1 / math.sqrt(2)], [[1, 1j], [-1j, 1]]),
        ],
    )
    def test_expval_single_wire_with_parameters(self, nullqubit_device, operation, input, par):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        dev = nullqubit_device(wires=1)
        dev.reset()
        dev.apply([qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates())
        res = dev.expval(obs)

        assert res == [0.0]

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]],
            ),
        ],
    )
    def test_expval_two_wires_with_parameters(self, nullqubit_device, operation, input, par):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        dev = nullqubit_device(wires=2)
        dev.reset()
        dev.apply([qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates())
        res = dev.expval(obs)

        assert res == [0.0]


class TestVar:
    """Tests that variances are properly (not) calculated."""

    @pytest.mark.parametrize(
        "operation,input",
        [
            (qml.PauliX, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.PauliY, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.PauliZ, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.Hadamard, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.Identity, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        ],
    )
    def test_var_single_wire_no_parameters(self, nullqubit_device, operation, input):
        """Tests that variances are properly (not) calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        dev = nullqubit_device(wires=1)
        dev.reset()
        dev.apply([qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates())
        res = dev.var(obs)
        assert res == [0.0]

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (qml.Hermitian, [1, 0], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [0, 1], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [1 / math.sqrt(2), -1 / math.sqrt(2)], [[1, 1j], [-1j, 1]]),
        ],
    )
    def test_var_single_wire_with_parameters(self, nullqubit_device, operation, input, par):
        """Tests that variances are properly (not) calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        dev = nullqubit_device(wires=1)
        dev.reset()
        dev.apply([qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates())
        res = dev.var(obs)

        assert res == [0.0]

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]],
            ),
        ],
    )
    def test_var_two_wires_with_parameters(self, nullqubit_device, operation, input, par):
        """Tests that variances are properly (not) calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        dev = nullqubit_device(wires=2)
        dev.reset()
        dev.apply([qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates())
        res = dev.var(obs)

        assert res == [0.0]


class TestSample:
    """Tests that samples are properly (not) calculated."""

    def test_sample_values(self):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = qml.device("null.qubit", wires=2, shots=1000)

        dev.apply([qml.RX(1.5708, wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        assert s1 == [0.0]


class TestNullQubitIntegration:
    """Integration tests for null.qubit. These tests ensure it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("null.qubit", wires=1)
        cap = dev.capabilities()
        capabilities = {
            "model": "qubit",
            "supports_broadcasting": False,
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
            "returns_state": True,
            "passthru_devices": {
                "tf": "null.qubit",
                "torch": "null.qubit",
                "autograd": "null.qubit",
                "jax": "null.qubit",
            },
        }
        assert cap == capabilities

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit_state(self, nullqubit_device, r_dtype):
        """Test that the NullQubit plugin provides the correct state for a simple circuit"""

        p = 0.543

        dev = nullqubit_device(wires=1)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.state()

        assert circuit(p) == [0.0]

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit_expval(self, nullqubit_device, r_dtype):
        """Test that the NullQubit plugin provides the correct expval for a simple circuit"""

        p = 0.543

        dev = nullqubit_device(wires=1)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        assert circuit(p) == np.array([0.0], dtype=object)

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit_var(self, nullqubit_device, r_dtype):
        """Test that the NullQubit plugin provides the correct var for a simple circuit"""

        p = 0.543

        dev = nullqubit_device(wires=1)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliY(0))

        assert circuit(p) == np.array([0.0], dtype=object)

    def test_qubit_identity(self, nullqubit_device):
        """Test that the NullQubit plugin provides correct result for the Identity expectation"""

        p = 0.543

        @qml.qnode(nullqubit_device(wires=1), diff_method="parameter-shift")
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert circuit(p) == np.array([0.0], dtype=object)

    def test_nonzero_shots(self):
        """Test that the NullQubit plugin provides correct result for high shot number"""
        dev = qml.device("null.qubit", wires=1, shots=10**5)

        p = 0.543

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.all(runs == np.array([0.0], dtype=object))

    @pytest.mark.parametrize(
        "name,state",
        [
            ("PauliX", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            ("PauliY", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            ("PauliZ", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            ("Hadamard", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        ],
    )
    def test_supported_observable_single_wire_no_parameters(self, nullqubit_device, name, state):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        dev = nullqubit_device(wires=1)
        assert dev.supports_observable(name)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert circuit() == np.array([0.0], dtype=object)

    @pytest.mark.parametrize(
        "name,state,par",
        [
            ("Identity", [1 / math.sqrt(5), 2 / math.sqrt(5)], []),
            ("Hermitian", [1 / math.sqrt(5), 2 / math.sqrt(5)], [np.array([[1, 1j], [-1j, 1]])]),
        ],
    )
    def test_supported_observable_single_wire_with_parameters(
        self, nullqubit_device, name, state, par
    ):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        dev = nullqubit_device(wires=1)
        assert dev.supports_observable(name)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert circuit() == np.array([0.0], dtype=object)

    @pytest.mark.parametrize(
        "name,state,par",
        [
            (
                "Hermitian",
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])],
            ),
            (
                "Hermitian",
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [np.array([[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]])],
            ),
            (
                "Hermitian",
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])],
            ),
        ],
    )
    def test_supported_observable_two_wires_with_parameters(
        self, nullqubit_device, name, state, par
    ):
        """Tests supported observables on two wires with parameters."""

        obs = getattr(qml.ops, name)

        dev = nullqubit_device(wires=2)
        assert dev.supports_observable(name)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0, 1])
            return qml.expval(obs(*par, wires=[0, 1]))

        assert circuit() == np.array([0.0], dtype=object)

    @pytest.mark.parametrize(
        "method", ["best", "parameter-shift", "backprop", "finite-diff", "adjoint"]
    )
    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_diff_method(self, nullqubit_device, method, r_dtype):
        """Test that the NullQubit works with all, except for "device", diff_method options."""

        p = 0.543

        dev = nullqubit_device(wires=1)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method=method)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.state()

        assert circuit(p) == [0.0]

    @pytest.mark.parametrize(
        "method", ["best", "parameter-shift", "backprop", "finite-diff", "adjoint"]
    )
    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_diff_method_multi_results(self, nullqubit_device, method, r_dtype):
        """Test that the NullQubit works with all, except for "device", diff_method options."""

        p = 0.543

        dev = nullqubit_device(wires=4)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method=method)
        def circuit(x):
            for n in range(4):
                qml.RX(x, wires=n)
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        assert np.all(circuit(p) == np.array([0.0], dtype=object))


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test if tensor expectation values returns None"""

    def test_paulix_pauliy(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = nullqubit_device(wires=3)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == [0.0]

    def test_pauliz_identity(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = nullqubit_device(wires=3)

        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == [0.0]

    def test_pauliz_hadamard(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = nullqubit_device(wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == [0.0]

    def test_hermitian(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = nullqubit_device(wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == [0.0]

    def test_hermitian_hermitian(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = nullqubit_device(wires=3)

        A1 = np.array([[1, 2], [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.Hermitian(A1, wires=[0]) @ qml.Hermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == [0.0]

    def test_hermitian_identity_expectation(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = nullqubit_device(wires=2)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == [0.0]

    def test_hermitian_two_wires_identity_expectation(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = nullqubit_device(wires=3)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )
        Identity = np.array([[1, 0], [0, 1]])
        H = np.kron(np.kron(Identity, Identity), A)
        obs = qml.Hermitian(H, wires=[2, 1, 0])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == [0.0]


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Test if tensor variance returns None"""

    def test_paulix_pauliy(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = nullqubit_device(wires=3)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == [0.0]

    def test_pauliz_identity(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = nullqubit_device(wires=3)

        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == [0.0]

    def test_pauliz_hadamard(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = nullqubit_device(wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == [0.0]

    def test_hermitian(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = nullqubit_device(wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == [0.0]

    def test_hermitian_hermitian(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = nullqubit_device(wires=3)

        A1 = np.array([[1, 2], [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.Hermitian(A1, wires=[0]) @ qml.Hermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == [0.0]

    def test_hermitian_identity_expectation(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = nullqubit_device(wires=2)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == [0.0]

    def test_hermitian_two_wires_identity_expectation(self, nullqubit_device, theta, phi, varphi):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = nullqubit_device(wires=3)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )
        Identity = np.array([[1, 0], [0, 1]])
        H = np.kron(np.kron(Identity, Identity), A)
        obs = qml.Hermitian(H, wires=[2, 1, 0])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == [0.0]


@pytest.mark.parametrize("inverse", [True, False])
class TestApplyOps:
    """Tests special methods to apply gates in NullQubit."""

    state = np.arange(2**4, dtype=np.complex128).reshape((2, 2, 2, 2))
    dev = qml.device("null.qubit", wires=4)

    single_qubit_ops = [
        (dev._apply_x),
        (dev._apply_y),
        (dev._apply_z),
        (dev._apply_hadamard),
        (dev._apply_s),
        (dev._apply_t),
        (dev._apply_sx),
    ]
    two_qubit_ops = [
        (dev._apply_cnot),
        (dev._apply_swap),
        (dev._apply_cz),
    ]
    three_qubit_ops = [
        (dev._apply_toffoli),
    ]

    @pytest.mark.parametrize("method", single_qubit_ops)
    def test_apply_single_qubit_op(self, method, inverse):
        """Test if the application of single qubit operations is correct."""
        state_out = method(self.state, axes=[1], inverse=inverse)
        assert state_out == [0.0]

    @pytest.mark.parametrize("method", two_qubit_ops)
    def test_apply_two_qubit_op(self, method, inverse):
        """Test if the application of two qubit operations is correct."""
        state_out = method(self.state, axes=[0, 1])
        assert state_out == [0.0]

    @pytest.mark.parametrize("method", two_qubit_ops)
    def test_apply_two_qubit_op_reverse(self, method, inverse):
        """Test if the application of two qubit operations is correct when the applied wires are
        reversed."""
        state_out = method(self.state, axes=[2, 1])
        assert state_out == [0.0]

    @pytest.mark.parametrize("method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_smaller(self, method, inverse):
        """Test if the application of three qubit operations is correct when both control wires are
        smaller than the target wire."""
        state_out = method(self.state, axes=[0, 2, 3])
        assert state_out == [0.0]

    @pytest.mark.parametrize("method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_greater(self, method, inverse):
        """Test if the application of three qubit operations is correct when both control wires are
        greater than the target wire."""
        state_out = method(self.state, axes=[2, 1, 0])
        assert state_out == [0.0]

    @pytest.mark.parametrize("method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_split(self, method, inverse):
        """Test if the application of three qubit operations is correct when one control wire is smaller
        and one control wire is greater than the target wire."""
        state_out = method(self.state, axes=[3, 1, 2])
        assert state_out == [0.0]

    single_qubit_ops_param = [
        (dev._apply_phase, [1.0]),
    ]

    @pytest.mark.parametrize("method,par", single_qubit_ops_param)
    def test_apply_single_qubit_op_(self, method, par, inverse):
        """Test if the application of single qubit operations (with parameter) is correct."""
        state_out = method(self.state, axes=[1], parameters=par, inverse=inverse)
        assert state_out == [0.0]


class TestStateInitialization:
    """Unit tests for state initialization methods"""

    def test_state_vector_full_system(self, mocker):
        """Test applying a state vector to the full system"""
        state_wires = qml.wires.Wires(["a", "b", "c"])
        dev = NullQubit(wires=state_wires)
        state = np.array(
            [
                1 / math.sqrt(204),
                2 / math.sqrt(204),
                3 / math.sqrt(204),
                4 / math.sqrt(204),
                5 / math.sqrt(204),
                6 / math.sqrt(204),
                7 / math.sqrt(204),
                8 / math.sqrt(204),
            ]
        )

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert dev._state == [0.0]
        spy.assert_not_called()

    def test_basis_state_full_system(self, mocker):
        """Test applying a state vector to the full system"""
        state_wires = qml.wires.Wires(["a", "b", "c"])
        dev = NullQubit(wires=state_wires)
        state = np.array(
            [
                1 / math.sqrt(204),
                2 / math.sqrt(204),
                3 / math.sqrt(204),
                4 / math.sqrt(204),
                5 / math.sqrt(204),
                6 / math.sqrt(204),
                7 / math.sqrt(204),
                8 / math.sqrt(204),
            ]
        )

        spy = mocker.spy(dev, "_scatter")
        dev._apply_basis_state(state=state, wires=state_wires)

        assert dev._state == [0.0]
        spy.assert_not_called()


class TestOpCallIntegration:
    """Integration tests for operation call statistics."""

    single_qubit_ops = [
        (qml.PauliX, {"PauliX": 1}),
        (qml.PauliY, {"PauliY": 1}),
        (qml.PauliZ, {"PauliZ": 1}),
        (qml.Hadamard, {"Hadamard": 1}),
        (qml.S, {"S": 1}),
        (qml.T, {"T": 1}),
    ]
    two_qubit_ops = [
        (qml.CNOT, {"CNOT": 1}),
        (qml.SWAP, {"SWAP": 1}),
        (qml.CZ, {"CZ": 1}),
    ]

    @pytest.mark.parametrize("operation,expected", single_qubit_ops)
    def test_single_qubit_op(self, nullqubit_device, operation, expected):
        """Test if the application of single qubit operations, without parameters,
        is being accounted for."""

        dev = nullqubit_device(wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            operation(wires=[0])
            return qml.state()

        circuit()

        expected_dict = defaultdict(int, **expected)
        assert dev.operation_calls() == expected_dict

    @pytest.mark.parametrize("operation,expected", two_qubit_ops)
    def test_two_qubit_op(self, nullqubit_device, operation, expected):
        """Test if the application of two qubit operations, without parameters,
        is being accounted for."""

        dev = nullqubit_device(wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            operation(wires=[0, 1])
            return qml.state()

        circuit()

        expected_dict = defaultdict(int, **expected)
        assert dev.operation_calls() == expected_dict

    single_qubit_ops_par = [
        (qml.RX, [math.pi / 4], {"RX": 1}),
        (qml.RY, [math.pi / 4], {"RY": 1}),
        (qml.RZ, [math.pi / 4], {"RZ": 1}),
        (qml.MultiRZ, [math.pi / 2], {"MultiRZ": 1}),
        (qml.DiagonalQubitUnitary, [np.array([-1, 1])], {"DiagonalQubitUnitary": 1}),
    ]
    two_qubit_ops_par = [
        (qml.CRX, [math.pi / 2], {"CRX": 1}),
        (qml.CRY, [math.pi / 2], {"CRY": 1}),
        (qml.CRZ, [math.pi / 2], {"CRZ": 1}),
        (qml.MultiRZ, [math.pi / 2], {"MultiRZ": 1}),
        (qml.DiagonalQubitUnitary, [np.array([-1, 1, -1, 1])], {"DiagonalQubitUnitary": 1}),
        (qml.IsingXX, [math.pi / 2], {"IsingXX": 1}),
        (qml.IsingYY, [math.pi / 2], {"IsingYY": 1}),
        (qml.IsingZZ, [math.pi / 2], {"IsingZZ": 1}),
        (
            qml.QubitStateVector,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            {"QubitStateVector": 1},
        ),
        (
            qml.BasisState,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            {"BasisState": 1},
        ),
    ]

    @pytest.mark.parametrize("operation,input,expected", single_qubit_ops_par)
    def test_single_qubit_op_with_par(self, nullqubit_device, operation, input, expected):
        """Test if the application of single qubit operations, with parameters,
        is being accounted for."""

        dev = nullqubit_device(wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(input):
            operation(input, wires=[0])
            return qml.state()

        circuit(input)

        expected_dict = defaultdict(int, **expected)
        assert dev.operation_calls() == expected_dict

    @pytest.mark.parametrize("operation,input,expected", two_qubit_ops_par)
    def test_two_qubit_op_with_par(self, nullqubit_device, operation, input, expected):
        """Test if the application of two qubit operations, with parameters,
        is being accounted for."""

        dev = nullqubit_device(wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(input):
            operation(input, wires=[0, 1])
            return qml.state()

        circuit(input)

        expected_dict = defaultdict(int, **expected)
        assert dev.operation_calls() == expected_dict

    @pytest.mark.parametrize(
        "op,expected",
        [
            (qml.SingleExcitation, {"SingleExcitation": 1}),
            (qml.SingleExcitationPlus, {"SingleExcitationPlus": 1}),
            (qml.SingleExcitationMinus, {"SingleExcitationMinus": 1}),
            (qml.DoubleExcitation, {"DoubleExcitation": 1}),
            (qml.DoubleExcitationPlus, {"DoubleExcitationPlus": 1}),
            (qml.DoubleExcitationMinus, {"DoubleExcitationMinus": 1}),
            (qml.OrbitalRotation, {"OrbitalRotation": 1}),
            (qml.QubitSum, {"QubitSum": 1}),
            (qml.QubitCarry, {"QubitCarry": 1}),
        ],
    )
    def test_advanced_op(self, nullqubit_device, op, expected):
        """Test qchem and arithmetic operations."""
        n_wires = op.num_wires
        n_params = op.num_params

        dev = nullqubit_device(wires=4)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            if n_params == 0:
                op(wires=range(n_wires))
            elif n_params == 1:
                op(0.5, wires=range(n_wires))
            else:
                op([0.5] * n_params, wires=range(n_wires))
            return qml.state()

        circuit()
        expected_dict = defaultdict(int, **expected)
        assert dev.operation_calls() == expected_dict


class TestState:
    "Unit test for state and density_matrix operations."
    dev = qml.device("null.qubit", wires=3)

    @pytest.mark.parametrize(
        "measurement",
        [
            dev.state,
            dev.density_matrix(wires=[1]),
            dev.density_matrix(wires=[2, 0]),
            dev.density_matrix(wires=[2, 1, 0]),
        ],
    )
    def test_state_measurement(self, measurement):
        """Test that the NullQubit plugin provides correct state results for a simple circuit"""
        assert measurement == [0.0]


class TestTrackerIntegration:
    """Tests tracker integration behavior with 'null.qubit'."""

    def test_single_execution(self, nullqubit_device, mocker):
        """Test correct behavior with single circuit execution"""
        dev = nullqubit_device(wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        class callback_wrapper:
            @staticmethod
            def callback(totals=dict(), history=dict(), latest=dict()):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        with Tracker(circuit.device, callback=wrapper.callback) as tracker:
            circuit()
            circuit()

        assert tracker.totals == {"executions": 2, "batches": 2, "batch_len": 2}
        assert tracker.history == {
            "executions": [1, 1],
            "shots": [None, None],
            "batches": [1, 1],
            "batch_len": [1, 1],
        }
        assert tracker.latest == {"batches": 1, "batch_len": 1}

        _, kwargs_called = spy.call_args_list[-1]

        assert kwargs_called["totals"] == {"executions": 2, "batches": 2, "batch_len": 2}
        assert kwargs_called["history"] == {
            "executions": [1, 1],
            "shots": [None, None],
            "batches": [1, 1],
            "batch_len": [1, 1],
        }
        assert kwargs_called["latest"] == {"batches": 1, "batch_len": 1}

    def test_shots_execution(self, nullqubit_device, mocker):
        """Test that correct tracks shots."""
        dev = nullqubit_device(wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        class callback_wrapper:
            @staticmethod
            def callback(totals=dict(), history=dict(), latest=dict()):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        with Tracker(circuit.device, callback=wrapper.callback) as tracker:
            circuit(shots=10)
            circuit(shots=20)

        assert tracker.totals == {"executions": 2, "batches": 2, "batch_len": 2, "shots": 30}
        assert tracker.history == {
            "executions": [1, 1],
            "shots": [10, 20],
            "batches": [1, 1],
            "batch_len": [1, 1],
        }
        assert tracker.latest == {"batches": 1, "batch_len": 1}

        assert spy.call_count == 4

        _, kwargs_called = spy.call_args_list[-1]

        assert kwargs_called["totals"] == {
            "executions": 2,
            "batches": 2,
            "batch_len": 2,
            "shots": 30,
        }
        assert kwargs_called["history"] == {
            "executions": [1, 1],
            "shots": [10, 20],
            "batches": [1, 1],
            "batch_len": [1, 1],
        }
        assert kwargs_called["latest"] == {"batches": 1, "batch_len": 1}
