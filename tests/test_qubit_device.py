
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
Unit tests for the :mod:`pennylane` :class:`QubitDevice` class.
"""
import pytest
import numpy as np
import itertools
import functools

import pennylane as qml
from pennylane import QubitDevice, DeviceError
from pennylane.qnodes import QuantumFunctionError
from pennylane import expval, var, sample
from pennylane.plugins.default_qubit import I, X, Y, Z, S, Rotx, Roty, H, CNOT

mock_qubit_device_paulis = ["PauliX", "PauliY", "PauliZ"]

# pylint: disable=abstract-class-instantiated, no-self-use, redefined-outer-name, invalid-name

@pytest.fixture(scope="function")
def mock_qubit_device_with_operations(monkeypatch):
    """A mock instance of the abstract QubitDevice class with non-empty operations"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, 'operations', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'observables', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        yield QubitDevice()


@pytest.fixture(scope="function")
def mock_qubit_device_with_observables(monkeypatch):
    """A mock instance of the abstract QubitDevice class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, 'operations', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'observables', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        yield QubitDevice()


@pytest.fixture(scope="function")
def mock_qubit_device_supporting_paulis(monkeypatch):
    """A mock instance of the abstract QubitDevice class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, 'operations', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'observables', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        yield QubitDevice()


@pytest.fixture(scope="function")
def mock_qubit_device_supporting_paulis_and_inverse(monkeypatch):
    """A mock instance of the abstract QubitDevice class with non-empty operations
    and supporting inverses"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, 'operations', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'observables', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'short_name', 'MockQubitDevice')
        m.setattr(QubitDevice, '_capabilities', {"inverse_operations": True})
        yield QubitDevice()

@pytest.fixture(scope="function")
def mock_qubit_device_supporting_observables_and_inverse(monkeypatch):
    """A mock instance of the abstract QubitDevice class with non-empty operations
    and supporting inverses"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, 'operations', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'observables', mock_qubit_device_paulis + ['Hermitian'])
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        m.setattr(QubitDevice, '_capabilities', {"inverse_operations": True})
        yield QubitDevice()

mock_qubit_device_capabilities = {
    "measurements": "everything",
    "noise_models": ["depolarizing", "bitflip"],
}


@pytest.fixture(scope="function")
def mock_qubit_device_with_capabilities(monkeypatch):
    """A mock instance of the abstract QubitDevice class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, '_capabilities', mock_qubit_device_capabilities)
        yield QubitDevice()


@pytest.fixture(scope="function")
def mock_qubit_device_with_paulis_and_methods(monkeypatch):
    """A mock instance of the abstract QubitDevice class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, '_capabilities', mock_qubit_device_capabilities)
        m.setattr(QubitDevice, 'operations', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'observables', mock_qubit_device_paulis)
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        m.setattr(QubitDevice, 'expval', lambda self, x: 0)
        m.setattr(QubitDevice, 'var', lambda self, x: 0)
        m.setattr(QubitDevice, 'sample', lambda self, x: 0)
        m.setattr(QubitDevice, 'apply', lambda self, x: None)
        yield QubitDevice()


@pytest.fixture(scope="function")
def mock_qubit_device(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, '_capabilities', mock_qubit_device_capabilities)
        m.setattr(QubitDevice, 'operations', ["PauliY", "RX", "Rot"])
        m.setattr(QubitDevice, 'observables', ["PauliZ"])
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        m.setattr(QubitDevice, 'expval', lambda self, x: 0)
        m.setattr(QubitDevice, 'var', lambda self, x: 0)
        m.setattr(QubitDevice, 'sample', lambda self, x: 0)
        m.setattr(QubitDevice, 'apply', lambda self, x: None)
        yield QubitDevice()

class TestOperations:
    """Tests the logic related to operations"""

    def test_op_queue_accessed_outside_execution_context(self, mock_qubit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
                ValueError, match="Cannot access the operation queue outside of the execution context!"
        ):
            mock_qubit_device.op_queue

    def test_op_queue_is_filled_at_pre_measure(self, mock_qubit_device_with_paulis_and_methods, monkeypatch):
        """Tests that the op_queue is correctly filled when pre_measure is called and that accessing
           op_queue raises no error"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(1)),
            qml.sample(qml.PauliZ(2)),
        ]

        queue_at_pre_measure = []

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'pre_measure', lambda self: queue_at_pre_measure.extend(self.op_queue))
            mock_qubit_device_with_paulis_and_methods.execute(queue, observables)

        assert queue_at_pre_measure == queue

    def test_op_queue_is_filled_during_execution(self, mock_qubit_device_with_paulis_and_methods, monkeypatch):
        """Tests that the operations are properly applied and queued"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(1)),
            qml.sample(qml.PauliZ(2)),
        ]

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'apply', lambda self, op: call_history.append(op))
            mock_qubit_device_with_paulis_and_methods.execute(queue, observables)

        assert len(call_history) == 3
        assert isinstance(call_history[0], qml.PauliX)
        assert call_history[0].wires == [0] 

        assert isinstance(call_history[1], qml.PauliY)
        assert call_history[1].wires == [1]
 
        assert isinstance(call_history[2], qml.PauliZ)
        assert call_history[2].wires == [2]

    def test_unsupported_operations_raise_error(self, mock_qubit_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.Hadamard(wires=2),
        ]

        observables = [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(1)),
            qml.sample(qml.PauliZ(2)),
        ]

        with pytest.raises(DeviceError, match="Gate Hadamard not supported on device"):
            mock_qubit_device_with_paulis_and_methods.execute(queue, observables)


class TestObservables:
    """Tests the logic related to observables"""

    # pylint: disable=no-self-use, redefined-outer-name

    def test_obs_queue_accessed_outside_execution_context(self, mock_qubit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
                ValueError,
                match="Cannot access the observable value queue outside of the execution context!",
        ):
            mock_qubit_device.obs_queue

    def test_obs_queue_is_filled_at_pre_measure(self, mock_qubit_device_with_paulis_and_methods, monkeypatch):
        """Tests that the op_queue is correctly filled when pre_measure is called and that accessing
           op_queue raises no error"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(1)),
            qml.sample(qml.PauliZ(2)),
        ]

        queue_at_pre_measure = []

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'pre_measure', lambda self: queue_at_pre_measure.extend(self.obs_queue))
            mock_qubit_device_with_paulis_and_methods.execute(queue, observables)

        assert queue_at_pre_measure == observables

    def test_obs_queue_is_filled_during_execution(self, monkeypatch, mock_qubit_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        observables = [
            qml.expval(qml.PauliX(0)),
            qml.var(qml.PauliY(1)),
            qml.sample(qml.PauliZ(2)),
        ]

        # capture the arguments passed to dev methods
        expval_args = []
        var_args = []
        sample_args = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'expval', lambda self, *args: expval_args.extend(args))
            m.setattr(QubitDevice, 'var', lambda self, *args: var_args.extend(args))
            m.setattr(QubitDevice, 'sample', lambda self, *args: sample_args.extend(args))
            mock_qubit_device_with_paulis_and_methods.execute([], observables)

        assert len(expval_args) == 1
        assert isinstance(expval_args[0], qml.PauliX)
        assert expval_args[0].wires == [0] 

        assert len(var_args) == 1
        assert isinstance(var_args[0], qml.PauliY)
        assert var_args[0].wires == [1]
 
        assert len(sample_args) == 1
        assert isinstance(sample_args[0], qml.PauliZ)
        assert sample_args[0].wires == [2]

    def test_unsupported_observables_raise_error(self, mock_qubit_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [
            qml.expval(qml.Hadamard(0)),
            qml.var(qml.PauliZ(1)),
            qml.sample(qml.PauliZ(2)),
        ]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            mock_qubit_device_with_paulis_and_methods.execute(queue, observables)

    def test_unsupported_observable_return_type_raise_error(self, mock_qubit_device_with_paulis_and_methods):
        """Check that an error is raised if the return type of an observable is unsupported"""

        queue = [qml.PauliX(wires=0)]

        # Make a observable without specifying a return operation upon measuring
        obs = qml.PauliZ(0)
        obs.return_type = "SomeUnsupportedReturnType"
        observables = [obs]

        with pytest.raises(QuantumFunctionError, match="Unsupported return type specified for observable"):
            mock_qubit_device_with_paulis_and_methods.execute(queue, observables)


class TestParameters:
    """Test for checking device parameter mappings"""

    def test_parameters_accessed_outside_execution_context(self, mock_qubit_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        with pytest.raises(
                ValueError,
                match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            mock_qubit_device.parameters

    def test_parameters_available_at_pre_measure(self, mock_qubit_device, monkeypatch):
        """Tests that the parameter mapping is available when pre_measure is called and that accessing
           QubitDevice.parameters raises no error"""

        p0 = 0.54
        p1 = -0.32

        queue = [
            qml.RX(p0, wires=0),
            qml.PauliY(wires=1),
            qml.Rot(0.432, 0.123, p1, wires=2),
        ]

        parameters = {0: (0, 0), 1: (2, 3)}

        observables = [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(1)),
            qml.sample(qml.PauliZ(2)),
        ]

        p_mapping = {}

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "pre_measure", lambda self: p_mapping.update(self.parameters))
            mock_qubit_device.execute(queue, observables, parameters=parameters)

        assert p_mapping == parameters

THETA = np.linspace(0.11, 3, 5)
PHI = np.linspace(0.32, 3, 5)
VARPHI = np.linspace(0.02, 3, 5)

def ansatz(a, b, c):
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(c, wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

def tensor_product(observables):
    return functools.reduce(np.kron, observables)

@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorSample:
    """Tests for samples of tensor observables"""

    def test_paulix_tensor_pauliz(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving PauliX and PauliZ works correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return sample(qml.PauliX(0) @ qml.PauliZ(1))

        with monkeypatch.context() as m:
            s1 = circuit()

        # s1 should only contain 1
        assert np.allclose(s1, 1, atol=tol, rtol=0)

    def test_paulix_tensor_pauliy(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return sample(qml.PauliX(0) @  qml.PauliY(2))

        with monkeypatch.context() as m:
            s1 = circuit(theta, phi, varphi)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

        zero_state = np.zeros(2 ** 3)
        zero_state[0] = 1
        psi = zero_state
        psi = tensor_product([Rotx(theta), I, I]) @ zero_state
        psi = tensor_product([I, Rotx(phi), I]) @ psi
        psi = tensor_product([I, I, Rotx(varphi)]) @ psi
        psi = tensor_product([CNOT, I]) @ psi
        psi = tensor_product([I, CNOT]) @ psi

        # Diagonalize according to the observable
        psi = tensor_product([H, I, I]) @ psi
        psi = tensor_product([I, I, Z]) @ psi
        psi = tensor_product([I, I, S]) @ psi
        psi = tensor_product([I, I, H]) @ psi

        expected_probabilities = np.abs(psi) ** 2
        expected_probabilities = expected_probabilities.tolist()

        assert np.allclose(dev.probability(), expected_probabilities, atol=tol, rtol=0)

    def test_pauliz_tensor_hadamard(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving PauliZ and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return sample(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

        with monkeypatch.context() as m:
            s1 = circuit(theta, phi, varphi)

        zero_state = np.zeros(2 ** 3)
        zero_state[0] = 1
        psi = zero_state
        psi = tensor_product([Rotx(theta), I, I]) @ zero_state
        psi = tensor_product([I, Rotx(phi), I]) @ psi
        psi = tensor_product([I, I, Rotx(varphi)]) @ psi
        psi = tensor_product([CNOT, I]) @ psi
        psi = tensor_product([I, CNOT]) @ psi

        # Diagonalize according to the observable
        psi = tensor_product([I, Roty(-np.pi/4), I]) @ psi
        psi = tensor_product([I, I, Z]) @ psi
        psi = tensor_product([I, I, S]) @ psi
        psi = tensor_product([I, I, H]) @ psi

        expected_probabilities = np.abs(psi) ** 2
        expected_probabilities = expected_probabilities.tolist()

        assert np.allclose(dev.probability(), expected_probabilities, atol=tol, rtol=0)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

    def test_tensor_hermitian(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit(a, b, c):
            ansatz(a, b, c)
            return sample(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))

        with monkeypatch.context() as m:
            s1 = circuit(theta, phi, varphi)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A))
        assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))

        zero_state = np.zeros(2 ** 3)
        zero_state[0] = 1
        psi = zero_state
        psi = tensor_product([Rotx(theta), I, I]) @ zero_state
        psi = tensor_product([I, Rotx(phi), I]) @ psi
        psi = tensor_product([I, I, Rotx(varphi)]) @ psi
        psi = tensor_product([CNOT, I]) @ psi
        psi = tensor_product([I, CNOT]) @ psi

        # Diagonalize according to the observable
        eigvals, eigvecs = np.linalg.eigh(A)
        psi = tensor_product([I, eigvecs.conj().T]) @ psi

        expected_probabilities = np.abs(psi) ** 2
        expected_probabilities = expected_probabilities.tolist()

        assert np.allclose(dev.probability(), expected_probabilities, atol=tol, rtol=0)

