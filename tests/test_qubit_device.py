# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
from random import random

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import QubitDevice, DeviceError, QuantumFunctionError
from pennylane.measurements import Sample, Variance, Expectation, Probability, State
from pennylane.circuit_graph import CircuitGraph
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.measurements import state

mock_qubit_device_paulis = ["PauliX", "PauliY", "PauliZ"]
mock_qubit_device_rotations = ["RX", "RY", "RZ"]


# pylint: disable=abstract-class-instantiated, no-self-use, redefined-outer-name, invalid-name


@pytest.fixture(scope="function")
def mock_qubit_device(monkeypatch):
    """A function to create a mock device that mocks most of the methods except for e.g. probability()"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, "__abstractmethods__", frozenset())
        m.setattr(QubitDevice, "_capabilities", mock_qubit_device_capabilities)
        m.setattr(QubitDevice, "operations", ["PauliY", "RX", "Rot"])
        m.setattr(QubitDevice, "observables", ["PauliZ"])
        m.setattr(QubitDevice, "short_name", "MockDevice")
        m.setattr(QubitDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "apply", lambda self, *args, **kwargs: None)

        def get_qubit_device(wires=1):
            return QubitDevice(wires=wires)

        yield get_qubit_device


@pytest.fixture(scope="function")
def mock_qubit_device_extract_stats(monkeypatch):
    """A function to create a mock device that mocks the methods related to
    statistics (expval, var, sample, probability)"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, "__abstractmethods__", frozenset())
        m.setattr(QubitDevice, "_capabilities", mock_qubit_device_capabilities)
        m.setattr(QubitDevice, "operations", ["PauliY", "RX", "Rot"])
        m.setattr(QubitDevice, "observables", ["PauliZ"])
        m.setattr(QubitDevice, "short_name", "MockDevice")
        m.setattr(QubitDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "state", 0)
        m.setattr(QubitDevice, "density_matrix", lambda self, wires=None: 0)
        m.setattr(QubitDevice, "probability", lambda self, wires=None, *args, **kwargs: 0)
        m.setattr(QubitDevice, "apply", lambda self, x: x)

        def get_qubit_device(wires=1):
            return QubitDevice(wires=wires)

        yield get_qubit_device


@pytest.fixture(scope="function")
def mock_qubit_device_with_original_statistics(monkeypatch):
    """A function to create a mock device that mocks only basis methods and uses the original
    statistics related methods"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, "__abstractmethods__", frozenset())
        m.setattr(QubitDevice, "_capabilities", mock_qubit_device_capabilities)
        m.setattr(QubitDevice, "operations", ["PauliY", "RX", "Rot"])
        m.setattr(QubitDevice, "observables", ["PauliZ"])
        m.setattr(QubitDevice, "short_name", "MockDevice")

        def get_qubit_device(wires=1):
            return QubitDevice(wires=wires)

        yield get_qubit_device


mock_qubit_device_capabilities = {
    "measurements": "everything",
    "returns_state": True,
    "noise_models": ["depolarizing", "bitflip"],
}


@pytest.fixture(scope="function")
def mock_qubit_device_with_paulis_and_methods(monkeypatch):
    """A function to create a mock device that supports Paulis in its capabilities"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, "__abstractmethods__", frozenset())
        m.setattr(QubitDevice, "_capabilities", mock_qubit_device_capabilities)
        m.setattr(QubitDevice, "operations", mock_qubit_device_paulis)
        m.setattr(QubitDevice, "observables", mock_qubit_device_paulis)
        m.setattr(QubitDevice, "short_name", "MockDevice")
        m.setattr(QubitDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "apply", lambda self, x, rotations: None)

        def get_qubit_device(wires=1):
            return QubitDevice(wires=wires)

        yield get_qubit_device


@pytest.fixture(scope="function")
def mock_qubit_device_with_paulis_rotations_and_methods(monkeypatch):
    """A function to create a mock device that supports Paulis in its capabilities"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, "__abstractmethods__", frozenset())
        m.setattr(QubitDevice, "_capabilities", mock_qubit_device_capabilities)
        m.setattr(QubitDevice, "operations", mock_qubit_device_paulis + mock_qubit_device_rotations)
        m.setattr(QubitDevice, "observables", mock_qubit_device_paulis)
        m.setattr(QubitDevice, "short_name", "MockDevice")
        m.setattr(QubitDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QubitDevice, "apply", lambda self, x, **kwargs: None)

        def get_qubit_device(wires=1):
            return QubitDevice(wires=wires)

        yield get_qubit_device


def _working_get_batch_size(tensor, expected_shape, expected_size):
    size = QubitDevice._size(tensor)
    if QubitDevice._ndim(tensor) > len(expected_shape) or size > expected_size:
        return size // expected_size

    return None


class TestOperations:
    """Tests the logic related to operations"""

    def test_op_queue_accessed_outside_execution_context(self, mock_qubit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError, match="Cannot access the operation queue outside of the execution context!"
        ):
            dev = mock_qubit_device()
            dev.op_queue

    def test_op_queue_is_filled_during_execution(
        self, mock_qubit_device_with_paulis_and_methods, monkeypatch
    ):
        """Tests that the op_queue is correctly filled when apply is called and that accessing
        op_queue raises no error"""

        with qml.tape.QuantumTape() as tape:
            queue = [qml.PauliX(wires=0), qml.PauliY(wires=1), qml.PauliZ(wires=2)]
            observables = [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))]

        call_history = []

        with monkeypatch.context() as m:
            m.setattr(
                QubitDevice,
                "apply",
                lambda self, x, **kwargs: call_history.extend(x + kwargs.get("rotations", [])),
            )
            m.setattr(QubitDevice, "analytic_probability", lambda *args: None)
            dev = mock_qubit_device_with_paulis_and_methods()
            dev.execute(tape)

        assert call_history == queue

        assert len(call_history) == 3
        assert isinstance(call_history[0], qml.PauliX)
        assert call_history[0].wires == Wires([0])

        assert isinstance(call_history[1], qml.PauliY)
        assert call_history[1].wires == Wires([1])

        assert isinstance(call_history[2], qml.PauliZ)
        assert call_history[2].wires == Wires([2])

    def test_unsupported_operations_raise_error(self, mock_qubit_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        with qml.tape.QuantumTape() as tape:
            queue = [qml.PauliX(wires=0), qml.PauliY(wires=1), qml.Hadamard(wires=2)]
            observables = [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))]

        with pytest.raises(DeviceError, match="Gate Hadamard not supported on device"):
            dev = mock_qubit_device_with_paulis_and_methods()
            dev.execute(tape)

    numeric_queues = [
        [qml.RX(0.3, wires=[0])],
        [
            qml.RX(0.3, wires=[0]),
            qml.RX(0.4, wires=[1]),
            qml.RX(0.5, wires=[2]),
        ],
    ]

    observables = [[qml.PauliZ(0)], [qml.PauliX(0)], [qml.PauliY(0)]]

    @pytest.mark.parametrize("observables", observables)
    @pytest.mark.parametrize("queue", numeric_queues)
    def test_passing_keyword_arguments_to_execute(
        self, mock_qubit_device_with_paulis_rotations_and_methods, monkeypatch, queue, observables
    ):
        """Tests that passing keyword arguments to execute propagates those kwargs to the apply()
        method"""
        with qml.tape.QuantumTape() as tape:
            for op in queue + observables:
                op.queue()

        call_history = {}

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "apply", lambda self, x, **kwargs: call_history.update(kwargs))
            dev = mock_qubit_device_with_paulis_rotations_and_methods()
            dev.execute(tape, hash=tape.graph.hash)

        len(call_history.items()) == 1
        call_history["hash"] = tape.graph.hash


class TestObservables:
    """Tests the logic related to observables"""

    # pylint: disable=no-self-use, redefined-outer-name

    def test_obs_queue_accessed_outside_execution_context(self, mock_qubit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the observable value queue outside of the execution context!",
        ):
            dev = mock_qubit_device()
            dev.obs_queue

    def test_unsupported_observables_raise_error(self, mock_qubit_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        with qml.tape.QuantumTape() as tape:
            queue = [qml.PauliX(wires=0), qml.PauliY(wires=1), qml.PauliZ(wires=2)]

            observables = [
                qml.expval(qml.Hadamard(0)),
                qml.var(qml.PauliZ(1)),
                qml.sample(qml.PauliZ(2)),
            ]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            dev = mock_qubit_device_with_paulis_and_methods()
            dev.execute(tape)

    def test_unsupported_observable_return_type_raise_error(
        self, mock_qubit_device_with_paulis_and_methods, monkeypatch
    ):
        """Check that an error is raised if the return type of an observable is unsupported"""

        with qml.tape.QuantumTape() as tape:
            qml.PauliX(wires=0)
            qml.measurements.MeasurementProcess(
                return_type="SomeUnsupportedReturnType", obs=qml.PauliZ(0)
            )

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "apply", lambda self, x, **kwargs: None)
            with pytest.raises(
                qml.QuantumFunctionError, match="Unsupported return type specified for observable"
            ):
                dev = mock_qubit_device_with_paulis_and_methods()
                dev.execute(tape)


class TestParameters:
    """Test for checking device parameter mappings"""

    def test_parameters_accessed_outside_execution_context(self, mock_qubit_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            dev = mock_qubit_device()
            dev.parameters


class TestExtractStatistics:
    """Test the statistics method"""

    @pytest.mark.parametrize("returntype", [Expectation, Variance, Sample, Probability, State])
    def test_results_created(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method simply builds a results list without any side-effects"""

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            dev = mock_qubit_device_extract_stats()
            results = dev.statistics([obs])

        assert results == [0]

    def test_results_no_state(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the statistics method raises an AttributeError when a State return type is
        requested when QubitDevice does not have a state attribute"""
        with monkeypatch.context():
            dev = mock_qubit_device_extract_stats()
            delattr(dev.__class__, "state")
            with pytest.raises(
                qml.QuantumFunctionError, match="The state is not available in the current"
            ):
                dev.statistics([state()])

    @pytest.mark.parametrize("returntype", [None])
    def test_results_created_empty(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method returns an empty list if the return type is None"""

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            dev = mock_qubit_device_extract_stats()
            results = dev.statistics([obs])

        assert results == []

    @pytest.mark.parametrize("returntype", ["not None"])
    def test_error_return_type_none(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method raises an error if the return type is not well-defined and is not None"""

        assert returntype not in [Expectation, Variance, Sample, Probability, State, None]

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with pytest.raises(qml.QuantumFunctionError, match="Unsupported return type"):
            dev = mock_qubit_device_extract_stats()
            dev.statistics([obs])


class TestGenerateSamples:
    """Test the generate_samples method"""

    def test_auxiliary_methods_called_correctly(self, mock_qubit_device, monkeypatch):
        """Tests that the generate_samples method calls on its auxiliary methods correctly"""

        dev = mock_qubit_device()
        number_of_states = 2**dev.num_wires

        with monkeypatch.context() as m:
            # Mock the auxiliary methods such that they return the expected values
            m.setattr(QubitDevice, "sample_basis_states", lambda self, wires, b: wires)
            m.setattr(QubitDevice, "states_to_binary", staticmethod(lambda a, b: (a, b)))
            m.setattr(QubitDevice, "analytic_probability", lambda *args: None)
            m.setattr(QubitDevice, "shots", 1000)
            dev._samples = dev.generate_samples()

        assert dev._samples == (number_of_states, dev.num_wires)


class TestSampleBasisStates:
    """Test the sample_basis_states method"""

    def test_sampling_with_correct_arguments(self, mock_qubit_device, monkeypatch):
        """Tests that the sample_basis_states method samples with the correct arguments"""

        shots = 1000

        number_of_states = 4
        dev = mock_qubit_device()
        dev.shots = shots
        state_probs = [0.1, 0.2, 0.3, 0.4]

        with monkeypatch.context() as m:
            # Mock the numpy.random.choice method such that it returns the expected values
            m.setattr("numpy.random.choice", lambda x, y, p: (x, y, p))
            res = dev.sample_basis_states(number_of_states, state_probs)

        assert np.array_equal(res[0], np.array([0, 1, 2, 3]))
        assert res[1] == shots
        assert res[2] == state_probs

    def test_raises_deprecation_warning(self, mock_qubit_device, monkeypatch):
        """Test that sampling basis states on a device with shots=None produces a warning."""

        dev = mock_qubit_device()
        number_of_states = 4
        dev.shots = None
        state_probs = [0.1, 0.2, 0.3, 0.4]

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of shots has to be explicitly set on the device",
        ):
            dev.sample_basis_states(number_of_states, state_probs)

    @pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested")
    def test_sampling_with_broadcasting(self, mock_qubit_device, monkeypatch):
        """Tests that the sample_basis_states method samples with the correct arguments
        when using broadcasted probabilities"""

        shots = 1000

        number_of_states = 4
        dev = mock_qubit_device()
        dev.shots = shots
        state_probs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.1, 0.2]]
        # First run the sampling to see that it is using numpy.random.choice correctly
        res = dev.sample_basis_states(number_of_states, state_probs)
        assert qml.math.shape(res) == (2, shots)
        assert set(res.flat).issubset({0, 1, 2, 3})

        with monkeypatch.context() as m:
            # Mock the numpy.random.choice method such that it returns the expected values
            m.setattr("numpy.random.choice", lambda x, y, p: (x, y, p))
            res = dev.sample_basis_states(number_of_states, state_probs)

        assert len(res) == 2
        for _res, prob in zip(res, state_probs):
            assert np.array_equal(_res[0], np.array([0, 1, 2, 3]))
            assert _res[1] == 1000
            assert _res[2] == prob


class TestStatesToBinary:
    """Test the states_to_binary method"""

    def test_correct_conversion_two_states(self, mock_qubit_device):
        """Tests that the sample_basis_states method converts samples to binary correctly"""
        wires = 4
        shots = 10

        number_of_states = 2**wires
        basis_states = np.arange(number_of_states)
        samples = np.random.choice(basis_states, shots)

        dev = mock_qubit_device()
        res = dev.states_to_binary(samples, wires)

        format_smt = f"{{:0{wires}b}}"
        expected = np.array([[int(x) for x in list(format_smt.format(i))] for i in samples])

        assert np.all(res == expected)

    test_binary_conversion_data = [
        (np.array([2, 3, 2, 0, 0]), np.array([[1, 0], [1, 1], [1, 0], [0, 0], [0, 0]])),
        (np.array([2, 3, 1, 3, 1]), np.array([[1, 0], [1, 1], [0, 1], [1, 1], [0, 1]])),
        (
            np.array([7, 7, 1, 5, 2]),
            np.array([[1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0]]),
        ),
    ]

    @pytest.mark.parametrize("samples, binary_states", test_binary_conversion_data)
    def test_correct_conversion(self, mock_qubit_device, samples, binary_states, tol):
        """Tests that the states_to_binary method converts samples to binary correctly"""
        dev = mock_qubit_device()
        dev.shots = 5
        wires = binary_states.shape[1]
        res = dev.states_to_binary(samples, wires)
        assert np.allclose(res, binary_states, atol=tol, rtol=0)

    test_binary_conversion_data_broadcasted = [
        (
            np.array([[2, 3, 2, 0, 0], [3, 0, 0, 1, 1], [2, 2, 0, 1, 3]]),
            np.array(
                [
                    [[1, 0], [1, 1], [1, 0], [0, 0], [0, 0]],
                    [[1, 1], [0, 0], [0, 0], [0, 1], [0, 1]],
                    [[1, 0], [1, 0], [0, 0], [0, 1], [1, 1]],
                ]
            ),
        ),
        (
            np.array([[7, 7, 1, 5, 2], [3, 3, 2, 4, 6], [0, 0, 7, 2, 1]]),
            np.array(
                [
                    [[1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0]],
                    [[0, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                    [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1]],
                ]
            ),
        ),
    ]

    @pytest.mark.parametrize("samples, binary_states", test_binary_conversion_data_broadcasted)
    def test_correct_conversion_broadcasted(self, mock_qubit_device, samples, binary_states, tol):
        """Tests that the states_to_binary method converts broadcasted
        samples to binary correctly"""
        dev = mock_qubit_device()
        dev.shots = 5
        wires = binary_states.shape[-1]
        res = dev.states_to_binary(samples, wires)
        assert np.allclose(res, binary_states, atol=tol, rtol=0)


class TestExpval:
    """Test the expval method"""

    def test_analytic_expval(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that expval method when the analytic attribute is True

        Additional QubitDevice methods that are mocked:
        -probability
        """
        obs = qml.PauliX(0)
        probs = [0.5, 0.5]
        dev = mock_qubit_device_with_original_statistics()

        assert dev.shots is None

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "probability", lambda self, wires=None: probs)
            res = dev.expval(obs)

        assert res == (obs.eigvals() @ probs).real

    def test_analytic_expval_broadcasted(
        self, mock_qubit_device_with_original_statistics, monkeypatch
    ):
        """Tests expval method when the analytic attribute is True and using broadcasting

        Additional QubitDevice methods that are mocked:
        -probability
        """
        obs = qml.PauliX(0)
        probs = np.array([[0.5, 0.5], [0.2, 0.8], [0.1, 0.9]])
        dev = mock_qubit_device_with_original_statistics()

        assert dev.shots is None

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "probability", lambda self, wires=None: probs)
            res = dev.expval(obs)

        assert np.allclose(res, (probs @ obs.eigvals()).real)

    def test_non_analytic_expval(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that expval method when the analytic attribute is False

        Additional QubitDevice methods that are mocked:
        -sample
        -numpy.mean
        """
        obs = qml.PauliX(0)
        dev = mock_qubit_device_with_original_statistics()

        dev.shots = 1000

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "sample", lambda self, obs, *args, **kwargs: obs)
            m.setattr("numpy.mean", lambda obs, axis=None: obs)
            res = dev.expval(obs)

        assert res == obs

    def test_no_eigval_error(self, mock_qubit_device_with_original_statistics):
        """Tests that an error is thrown if expval is called with an observable that does
        not have eigenvalues defined."""
        dev = mock_qubit_device_with_original_statistics()

        # observable with no eigenvalue representation defined
        class MyObs(qml.operation.Observable):
            num_wires = 1

            def eigvals(self):
                raise qml.operation.EigvalsUndefinedError

        obs = MyObs(wires=0)

        with pytest.raises(
            qml.operation.EigvalsUndefinedError, match="Cannot compute analytic expectations"
        ):
            dev.expval(obs)


class TestVar:
    """Test the var method"""

    def test_analytic_var(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the analytic attribute is True

        Additional QubitDevice methods that are mocked:
        -probability
        """
        obs = qml.PauliX(0)
        probs = [0.5, 0.5]
        dev = mock_qubit_device_with_original_statistics()

        assert dev.shots is None

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "probability", lambda self, wires=None: probs)
            res = dev.var(obs)

        assert res == (obs.eigvals() ** 2) @ probs - (obs.eigvals() @ probs).real ** 2

    def test_analytic_var_broadcasted(
        self, mock_qubit_device_with_original_statistics, monkeypatch
    ):
        """Tests var method when the analytic attribute is True and using broadcasting

        Additional QubitDevice methods that are mocked:
        -probability
        """
        obs = qml.PauliX(0)
        probs = [0.5, 0.5]
        dev = mock_qubit_device_with_original_statistics()

        assert dev.shots is None

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "probability", lambda self, wires=None: probs)
            res = dev.var(obs)

        assert np.allclose(res, probs @ (obs.eigvals() ** 2) - (probs @ obs.eigvals()).real ** 2)

    def test_non_analytic_var(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the analytic attribute is False

        Additional QubitDevice methods that are mocked:
        -sample
        -numpy.var
        """
        obs = qml.PauliX(0)
        dev = mock_qubit_device_with_original_statistics()

        dev.shots = 1000

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "sample", lambda self, obs, *args, **kwargs: obs)
            m.setattr("numpy.var", lambda obs, axis=None: obs)
            res = dev.var(obs)

        assert res == obs

    def test_no_eigval_error(self, mock_qubit_device_with_original_statistics):
        """Tests that an error is thrown if var is called with an observable that does not have eigenvalues defined."""
        dev = mock_qubit_device_with_original_statistics()

        # observable with no eigenvalue representation defined
        class MyObs(qml.operation.Observable):
            num_wires = 1

            def eigvals(self):
                raise qml.operation.EigvalsUndefinedError

        obs = MyObs(wires=0)

        with pytest.raises(
            qml.operation.EigvalsUndefinedError, match="Cannot compute analytic variance"
        ):
            dev.var(obs)


class TestSample:
    """Test the sample method"""

    def test_only_ones_minus_ones(
        self, mock_qubit_device_with_original_statistics, monkeypatch, tol
    ):
        """Test that sample for a single Pauli observable only produces -1 and 1 samples"""
        obs = qml.PauliX(0)
        dev = mock_qubit_device_with_original_statistics()
        dev._samples = np.array([[1, 0], [0, 0]])

        with monkeypatch.context() as m:
            res = dev.sample(obs)

        assert np.shape(res) == (2,)
        assert np.allclose(res**2, 1, atol=tol, rtol=0)

    def test_correct_custom_eigenvalues(
        self, mock_qubit_device_with_original_statistics, monkeypatch, tol
    ):
        """Test that sample for a product of Pauli observables produces samples of eigenvalues"""
        obs = qml.PauliX(0) @ qml.PauliZ(1)
        dev = mock_qubit_device_with_original_statistics(wires=2)
        dev._samples = np.array([[1, 0], [0, 0]])

        with monkeypatch.context() as m:
            res = dev.sample(obs)

        assert np.array_equal(res, np.array([-1, 1]))

    def test_sample_with_no_observable_and_no_wires(
        self, mock_qubit_device_with_original_statistics, tol
    ):
        """Test that when we sample a device without providing an observable or wires then it
        will return the raw samples"""
        obs = qml.measurements.sample(op=None, wires=None)
        dev = mock_qubit_device_with_original_statistics(wires=2)
        generated_samples = np.array([[1, 0], [1, 1]])
        dev._samples = generated_samples

        res = dev.sample(obs)
        assert np.array_equal(res, generated_samples)

    def test_sample_with_no_observable_and_with_wires(
        self, mock_qubit_device_with_original_statistics, tol
    ):
        """Test that when we sample a device without providing an observable but we specify
        wires then it returns the generated samples for only those wires"""
        obs = qml.measurements.sample(op=None, wires=[1])
        dev = mock_qubit_device_with_original_statistics(wires=2)
        generated_samples = np.array([[1, 0], [1, 1]])
        dev._samples = generated_samples

        wire_samples = np.array([[0], [1]])
        res = dev.sample(obs)

        assert np.array_equal(res, wire_samples)

    def test_no_eigval_error(self, mock_qubit_device_with_original_statistics):
        """Tests that an error is thrown if sample is called with an observable
        that does not have eigenvalues defined."""
        dev = mock_qubit_device_with_original_statistics()
        dev._samples = np.array([[1, 0], [0, 0]])
        with pytest.raises(qml.operation.EigvalsUndefinedError, match="Cannot compute samples"):
            dev.sample(qml.Hamiltonian([1.0], [qml.PauliX(0)]))

    def test_no_error_measuring__commuting_ops_in_sample(self):
        """Test that no error is raised when computational basis samples are
        taken with other commuting measurements."""
        dev = qml.device("default.qubit", wires=1, shots=10)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(wires=0)), qml.sample()

        _ = circuit()

    def test_no_error_measuring__commuting_ops_in_counts(self):
        """Test that no error is raised when computational basis samples are
        taken with other commuting measurements."""
        dev = qml.device("default.qubit", wires=1, shots=10)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(wires=0)), qml.counts()

        circuit()

class TestSampleWithBroadcasting:
    """Test the sample method when broadcasting is used"""

    def test_only_ones_minus_ones(
        self, mock_qubit_device_with_original_statistics, monkeypatch, tol
    ):
        """Test that sample for a single Pauli observable only produces -1 and 1 samples
        when using broadcasting"""
        obs = qml.PauliX(0)
        dev = mock_qubit_device_with_original_statistics()
        dev._samples = np.array([[[0, 0], [0, 0]], [[1, 0], [1, 0]], [[0, 0], [1, 0]]])

        with monkeypatch.context() as m:
            res = dev.sample(obs)

        assert np.allclose(res, [[1, 1], [-1, -1], [1, -1]], atol=tol, rtol=0)

    def test_correct_custom_eigenvalues(
        self, mock_qubit_device_with_original_statistics, monkeypatch, tol
    ):
        """Test that sample for a product of Pauli observables produces samples
        of eigenvalues when using broadcasting"""
        obs = qml.PauliX(0) @ qml.PauliZ(1)
        dev = mock_qubit_device_with_original_statistics(wires=2)
        dev._samples = np.array([[1, 0], [0, 0]])
        dev._samples = np.array([[[1, 0], [0, 0]], [[0, 1], [1, 1]], [[1, 0], [0, 1]]])

        with monkeypatch.context() as m:
            res = dev.sample(obs)

        assert np.array_equal(res, np.array([[-1, 1], [-1, 1], [-1, -1]]))

    def test_sample_with_no_observable_and_no_wires(
        self, mock_qubit_device_with_original_statistics, tol
    ):
        """Test that when we sample a device without providing an observable or wires then it
        will return the raw samples when using broadcasting"""
        obs = qml.measurements.sample(op=None, wires=None)
        dev = mock_qubit_device_with_original_statistics(wires=2)
        generated_samples = np.array([[[1, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])
        dev._samples = generated_samples

        res = dev.sample(obs)
        assert np.array_equal(res, generated_samples)

    def test_sample_with_no_observable_and_with_wires(
        self, mock_qubit_device_with_original_statistics, tol
    ):
        """Test that when we sample a device without providing an observable but we specify wires
        then it returns the generated samples for only those wires when using broadcasting"""
        obs = qml.measurements.sample(op=None, wires=[1])
        dev = mock_qubit_device_with_original_statistics(wires=2)
        generated_samples = np.array([[[1, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])
        dev._samples = generated_samples

        wire_samples = np.array([[[0], [1]], [[1], [0]], [[1], [0]]])
        res = dev.sample(obs)

        assert np.array_equal(res, wire_samples)

    def test_no_eigval_error(self, mock_qubit_device_with_original_statistics):
        """Tests that an error is thrown if sample is called with an observable
        that does not have eigenvalues defined when using broadcasting."""
        dev = mock_qubit_device_with_original_statistics()
        dev._samples = np.array([[[1, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])
        with pytest.raises(qml.operation.EigvalsUndefinedError, match="Cannot compute samples"):
            dev.sample(qml.Hamiltonian([1.0], [qml.PauliX(0)]))


class TestEstimateProb:
    """Test the estimate_probability method"""

    @pytest.mark.parametrize(
        "wires, expected", [([0], [0.5, 0.5]), (None, [0.5, 0, 0, 0.5]), ([0, 1], [0.5, 0, 0, 0.5])]
    )
    def test_estimate_probability(
        self, wires, expected, mock_qubit_device_with_original_statistics, monkeypatch
    ):
        """Tests the estimate_probability method"""
        dev = mock_qubit_device_with_original_statistics(wires=2)
        samples = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "wires, expected",
        [
            ([0], [[0.0, 0.5], [1.0, 0.5]]),
            (None, [[0.0, 0.5], [0, 0], [0, 0.5], [1.0, 0]]),
            ([0, 1], [[0.0, 0.5], [0, 0], [0, 0.5], [1.0, 0]]),
        ],
    )
    def test_estimate_probability_with_binsize(
        self, wires, expected, mock_qubit_device_with_original_statistics, monkeypatch
    ):
        """Tests the estimate_probability method with a bin size"""
        dev = mock_qubit_device_with_original_statistics(wires=2)
        samples = np.array([[1, 1], [1, 1], [1, 0], [0, 0]])
        bin_size = 2

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires, bin_size=bin_size)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "wires, expected",
        [
            ([0], [[0.0, 1.0], [0.5, 0.5], [0.25, 0.75]]),
            (None, [[0, 0, 0.25, 0.75], [0.5, 0, 0, 0.5], [0.25, 0, 0.25, 0.5]]),
            ([0, 1], [[0, 0, 0.25, 0.75], [0.5, 0, 0, 0.5], [0.25, 0, 0.25, 0.5]]),
        ],
    )
    def test_estimate_probability_with_broadcasting(
        self, wires, expected, mock_qubit_device_with_original_statistics, monkeypatch
    ):
        """Tests the estimate_probability method with parameter broadcasting"""
        dev = mock_qubit_device_with_original_statistics(wires=2)
        samples = np.array(
            [
                [[1, 0], [1, 1], [1, 1], [1, 1]],
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                [[1, 0], [1, 1], [1, 1], [0, 0]],
            ]
        )

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "wires, expected",
        [
            (
                [0],
                [
                    [[0, 0, 0.5], [1, 1, 0.5]],
                    [[0.5, 0.5, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 1], [1, 0.5, 0]],
                ],
            ),
            (
                None,
                [
                    [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0], [0.5, 1, 0.5]],
                    [[0.5, 0.5, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 0.5], [0, 0, 0.5], [0.5, 0, 0], [0.5, 0.5, 0]],
                ],
            ),
            (
                [0, 1],
                [
                    [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0], [0.5, 1, 0.5]],
                    [[0.5, 0.5, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 0.5], [0, 0, 0.5], [0.5, 0, 0], [0.5, 0.5, 0]],
                ],
            ),
        ],
    )
    def test_estimate_probability_with_binsize_with_broadcasting(
        self, wires, expected, mock_qubit_device_with_original_statistics, monkeypatch
    ):
        """Tests the estimate_probability method with a bin size and parameter broadcasting"""
        dev = mock_qubit_device_with_original_statistics(wires=2)
        bin_size = 2
        samples = np.array(
            [
                [[1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1]],
                [[0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1]],
                [[1, 0], [1, 1], [1, 1], [0, 0], [0, 1], [0, 0]],
            ]
        )

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires, bin_size=bin_size)

        assert np.allclose(res, expected)


class TestMarginalProb:
    """Test the marginal_prob method"""

    @pytest.mark.parametrize(
        "wires, inactive_wires",
        [
            ([0], [1, 2]),
            ([1], [0, 2]),
            ([2], [0, 1]),
            ([0, 1], [2]),
            ([0, 2], [1]),
            ([1, 2], [0]),
            ([0, 1, 2], []),
            (Wires([0]), [1, 2]),
            (Wires([0, 1]), [2]),
            (Wires([0, 1, 2]), []),
        ],
    )
    def test_correct_arguments_for_marginals(
        self, mock_qubit_device_with_original_statistics, mocker, wires, inactive_wires, tol
    ):
        """Test that the correct arguments are passed to the marginal_prob method"""

        # Generate probabilities
        probs = np.array([random() for i in range(2**3)])
        probs /= sum(probs)

        spy = mocker.spy(np, "sum")
        dev = mock_qubit_device_with_original_statistics(wires=3)
        res = dev.marginal_prob(probs, wires=wires)
        array_call = spy.call_args[0][0]
        axis_call = spy.call_args[1]["axis"]

        assert np.allclose(array_call.flatten(), probs, atol=tol, rtol=0)
        assert axis_call == tuple(inactive_wires)

    marginal_test_data = [
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.6]), [1]),
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.3, 0.7]), Wires([0])),
        (
            np.array(
                [
                    0.17794671,
                    0.06184147,
                    0.21909549,
                    0.04932204,
                    0.19595214,
                    0.19176834,
                    0.08495311,
                    0.0191207,
                ]
            ),
            np.array([0.3970422, 0.28090525, 0.11116351, 0.21088904]),
            [2, 0],
        ),
    ]

    @pytest.mark.parametrize("probs, marginals, wires", marginal_test_data)
    def test_correct_marginals_returned(
        self, mock_qubit_device_with_original_statistics, probs, marginals, wires, tol
    ):
        """Test that the correct marginals are returned by the marginal_prob method"""
        num_wires = int(np.log2(len(probs)))
        dev = mock_qubit_device_with_original_statistics(num_wires)
        res = dev.marginal_prob(probs, wires=wires)
        assert np.allclose(res, marginals, atol=tol, rtol=0)

    @pytest.mark.parametrize("probs, marginals, wires", marginal_test_data)
    def test_correct_marginals_returned_wires_none(
        self, mock_qubit_device_with_original_statistics, probs, marginals, wires, tol
    ):
        """Test that passing wires=None simply returns the original probability."""
        num_wires = int(np.log2(len(probs)))
        dev = mock_qubit_device_with_original_statistics(wires=num_wires)
        dev.num_wires = num_wires

        res = dev.marginal_prob(probs, wires=None)
        assert np.allclose(res, probs, atol=tol, rtol=0)

    # Note that the broadcasted probs enter `marginal_probs` as a flattened array
    broadcasted_marginal_test_data = [
        (
            np.array([[0.1, 0.2, 0.3, 0.4], [0.8, 0.02, 0.05, 0.13], [0.6, 0.3, 0.02, 0.08]]),
            np.array([[0.4, 0.6], [0.85, 0.15], [0.62, 0.38]]),
            [1],
            2,
        ),
        (
            np.array(
                [
                    [0.17, 0.06, 0.21, 0.04, 0.19, 0.19, 0.08, 0.06],
                    [0.07, 0.04, 0.11, 0.04, 0.29, 0.04, 0.18, 0.23],
                ]
            ),
            np.array([[0.38, 0.27, 0.1, 0.25], [0.18, 0.47, 0.08, 0.27]]),
            [2, 0],
            3,
        ),
    ]

    @pytest.mark.parametrize("probs, marginals, wires, num_wires", broadcasted_marginal_test_data)
    def test_correct_broadcasted_marginals_returned(
        self,
        monkeypatch,
        mock_qubit_device_with_original_statistics,
        probs,
        marginals,
        wires,
        num_wires,
        tol,
    ):
        """Test that the correct marginals are returned by the marginal_prob method when
        broadcasting is used"""
        dev = mock_qubit_device_with_original_statistics(num_wires)
        with monkeypatch.context() as m:
            m.setattr(dev, "_get_batch_size", _working_get_batch_size)
            res = dev.marginal_prob(probs, wires=wires)

        assert np.allclose(res, marginals, atol=tol, rtol=0)

    @pytest.mark.parametrize("probs, marginals, wires, num_wires", broadcasted_marginal_test_data)
    def test_correct_broadcasted_marginals_returned_wires_none(
        self, mock_qubit_device_with_original_statistics, probs, marginals, wires, num_wires, tol
    ):
        """Test that the correct marginals are returned by the marginal_prob method when
        broadcasting is used"""
        dev = mock_qubit_device_with_original_statistics(num_wires)

        res = dev.marginal_prob(probs, wires=None)
        assert np.allclose(res, probs.reshape((-1, 2**num_wires)), atol=tol, rtol=0)


class TestActiveWires:
    """Test that the active_wires static method works as required."""

    def test_active_wires_from_queue(self, mock_qubit_device):
        queue = [qml.CNOT(wires=[0, 2]), qml.RX(0.2, wires=0), qml.expval(qml.PauliX(wires=5))]

        dev = mock_qubit_device(wires=6)
        res = dev.active_wires(queue)

        assert res == Wires([0, 2, 5])


class TestCapabilities:
    """Test that a default qubit device defines capabilities that all devices inheriting
    from it will automatically have."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""
        capabilities = {
            "model": "qubit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "supports_broadcasting": False,
        }
        assert capabilities == QubitDevice.capabilities()


class TestExecution:
    """Tests for the execute method"""

    def test_device_executions(self):
        """Test the number of times a qubit device is executed over a QNode's
        lifetime is tracked by `num_executions`"""

        dev_1 = qml.device("default.qubit", wires=2)

        def circuit_1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_1 = qml.QNode(circuit_1, dev_1)
        num_evals_1 = 10

        for _ in range(num_evals_1):
            node_1(0.432, 0.12)
        assert dev_1.num_executions == num_evals_1

        # test a second instance of a default qubit device
        dev_2 = qml.device("default.qubit", wires=2)

        def circuit_2(x, y):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_2 = qml.QNode(circuit_2, dev_2)
        num_evals_2 = 5

        for _ in range(num_evals_2):
            node_2(0.432, 0.12)
        assert dev_2.num_executions == num_evals_2

        # test a new circuit on an existing instance of a qubit device
        def circuit_3(x, y):
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_3 = qml.QNode(circuit_3, dev_1)
        num_evals_3 = 7

        for _ in range(num_evals_3):
            node_3(0.432, 0.12)
        assert dev_1.num_executions == num_evals_1 + num_evals_3

    def test_raise_error_measuring_non_commuting_ops_in_sample(self):
        """Test that an error is raised in the execution method when
        computational basis samples are taken and other non-commuting
        measurements are present"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1)), qml.sample()

        with pytest.raises(
            qml.QuantumFunctionError, match="Computational basis measurements do not commute"
        ):
            circuit()

    def test_raise_error_measuring_non_commuting_ops_in_counts(self):
        """Test that an error is raised in the execution method when
        computational basis samples are taken and other non-commuting
        measurements are present"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.counts()

        with pytest.raises(
            qml.QuantumFunctionError, match="Computational basis measurements do not commute"
        ):
            circuit()

class TestExecutionBroadcasted:
    """Tests for the execute method with broadcasted parameters"""

    def test_device_executions(self):
        """Test the number of times a qubit device is executed over a QNode's
        lifetime is tracked by `num_executions`"""

        dev_1 = qml.device("default.qubit", wires=2)

        def circuit_1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_1 = qml.QNode(circuit_1, dev_1)
        num_evals_1 = 10

        for _ in range(num_evals_1):
            node_1(0.432, np.array([0.12, 0.5, 3.2]))
        assert dev_1.num_executions == num_evals_1

        # test a second instance of a default qubit device
        dev_2 = qml.device("default.qubit", wires=2)

        assert dev_2.num_executions == 0

        def circuit_2(x, y):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_2 = qml.QNode(circuit_2, dev_2)
        num_evals_2 = 5

        for _ in range(num_evals_2):
            node_2(np.array([0.432, 0.61, 8.2]), 0.12)
        assert dev_2.num_executions == num_evals_2

        # test a new circuit on an existing instance of a qubit device
        def circuit_3(x, y):
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node_3 = qml.QNode(circuit_3, dev_1)
        num_evals_3 = 7

        for _ in range(num_evals_3):
            node_3(np.array([0.432, 0.2]), np.array([0.12, 1.214]))
        assert dev_1.num_executions == num_evals_1 + num_evals_3


class TestBatchExecution:
    """Tests for the batch_execute method."""

    with qml.tape.QuantumTape() as tape1:
        qml.PauliX(wires=0)
        qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

    with qml.tape.QuantumTape() as tape2:
        qml.PauliX(wires=0)
        qml.expval(qml.PauliZ(wires=0))

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_execute(self, n_tapes, mocker, mock_qubit_device_with_paulis_and_methods):
        """Tests that the device's execute method is called the correct number of times."""

        dev = mock_qubit_device_with_paulis_and_methods(wires=2)
        spy = mocker.spy(QubitDevice, "execute")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_reset(self, n_tapes, mocker, mock_qubit_device_with_paulis_and_methods):
        """Tests that the device's reset method is called the correct number of times."""

        dev = mock_qubit_device_with_paulis_and_methods(wires=2)

        spy = mocker.spy(QubitDevice, "reset")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_result(self, mock_qubit_device_with_paulis_and_methods, r_dtype, tol):
        """Tests that the result has the correct shape and entry types."""

        dev = mock_qubit_device_with_paulis_and_methods(wires=2)
        dev.R_DTYPE = r_dtype

        tapes = [self.tape1, self.tape2]
        res = dev.batch_execute(tapes)

        assert len(res) == 2
        assert np.allclose(res[0], dev.execute(self.tape1), rtol=tol, atol=0)
        assert np.allclose(res[1], dev.execute(self.tape2), rtol=tol, atol=0)
        assert res[0].dtype == r_dtype
        assert res[1].dtype == r_dtype

    def test_result_empty_tape(self, mock_qubit_device_with_paulis_and_methods, tol):
        """Tests that the result has the correct shape and entry types for empty tapes."""

        dev = mock_qubit_device_with_paulis_and_methods(wires=2)

        empty_tape = qml.tape.QuantumTape()
        tapes = [empty_tape] * 3
        res = dev.batch_execute(tapes)

        assert len(res) == 3
        assert np.allclose(res[0], dev.execute(empty_tape), rtol=tol, atol=0)


class TestShotList:
    """Tests for passing shots as a list"""

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4,), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11,),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3,), 30],
        [[(10, 3)], [(10, 3)], (3,), 30],
    ]

    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_single_expval(self, shot_list, shot_vector, expected_shape, total_shots):
        """Test a single expectation value"""
        dev = qml.device("default.qubit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        res = circuit(0.5)

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 2), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 2),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 2), 30],
        [[(10, 3)], [(10, 3)], (3, 2), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_multiple_expval(self, shot_list, shot_vector, expected_shape, total_shots):
        """Test multiple expectation values"""
        dev = qml.device("default.qubit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        res = circuit(0.5, 0.1)

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        res = qml.jacobian(circuit)(*pnp.array([0.5, 0.1], requires_grad=True))
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == expected_shape
        assert res[1].shape == expected_shape

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 4), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 4),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 4), 30],
        [[(10, 3)], [(10, 3)], (3, 4), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_probs(self, shot_list, shot_vector, expected_shape, total_shots):
        """Test a probability return"""
        dev = qml.device("default.qubit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        res = circuit(0.5, 0.1)

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        res = qml.jacobian(circuit, argnum=[0, 1])(0.5, 0.1)

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 2, 2), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 2, 2),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 2, 2), 30],
        [[(10, 3)], [(10, 3)], (3, 2, 2), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_multiple_probs(self, shot_list, shot_vector, expected_shape, total_shots):
        """Test multiple probability returns"""
        dev = qml.device("default.qubit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        res = circuit(0.5, 0.1)

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        res = qml.jacobian(circuit, argnum=[0, 1])(0.5, 0.1)

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], [(), (2,), (3,), (10,)], 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            [(), (2,), (2,), (2,), (10,), (), (), (5,), (), (), ()],
            27,
        ],
        [[10, 10, 10], [(10, 3)], [(10,), (10,), (10,)], 30],
        [[(10, 3)], [(10, 3)], [(10,), (10,), (10,)], 30],
    ]

    @pytest.mark.parametrize("shot_list,shot_vector,expected_shapes,total_shots", shot_data)
    def test_sample(self, shot_list, shot_vector, expected_shapes, total_shots):
        """Test sample returns"""
        dev = qml.device("default.qubit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(wires=0))

        res = circuit(0.5, 0.1)

        for r, shape in zip(res, expected_shapes):
            assert r.shape == shape

        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], [(2,), (2, 2), (3, 2), (10, 2)], 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            [(2,), (2, 2), (2, 2), (2, 2), (10, 2), (2,), (2,), (5, 2), (2,), (2,), (2,)],
            27,
        ],
        [[10, 10, 10], [(10, 3)], [(10, 2), (10, 2), (10, 2)], 30],
        [[(10, 3)], [(10, 3)], [(10, 2), (10, 2), (10, 2)], 30],
    ]

    @pytest.mark.parametrize("shot_list,shot_vector,expected_shapes,total_shots", shot_data)
    def test_multiple_sample(self, shot_list, shot_vector, expected_shapes, total_shots):
        """Test sample returns"""
        dev = qml.device("default.qubit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(wires=0)), qml.sample(qml.PauliZ(wires=1))

        res = circuit(0.5, 0.1)

        for r, shape in zip(res, expected_shapes):
            assert r.shape == shape

        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

    def test_invalid_shot_list(self):
        """Test exception raised if the shot list is the wrong type"""
        with pytest.raises(qml.DeviceError, match="Shots must be"):
            qml.device("default.qubit", wires=2, shots=0.5)

        with pytest.raises(ValueError, match="Unknown shot sequence"):
            qml.device("default.qubit", wires=2, shots=["a", "b", "c"])

    @pytest.mark.parametrize("shot_list", [[1, 2, 3, 10], [1, 2, 2, 10, 1, 1], [10, 10, 10], [1]])
    def test_error_shot_list_with_broadcasting(self, shot_list):
        """Test that an error is raised when using parameter broadcasting with a
        shot vector is attempted."""
        dev = qml.device("default.qubit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit():
            qml.RY(np.zeros(3), wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(wires=0)), qml.sample(qml.PauliZ(wires=1))

        with pytest.raises(NotImplementedError, match="Parameter broadcasting when using"):
            circuit()


class TestGetBatchSize:
    """Tests for the helper method ``_get_batch_size`` of ``QubitDevice``."""

    @pytest.mark.parametrize("shape", [(4, 4), (1, 8), (4,)])
    def test_batch_size_always_None(self, mock_qubit_device, shape):
        """Test that QubitDevice always reports a batch_size of None."""
        dev = mock_qubit_device()
        tensor0 = np.ones(shape, dtype=complex)
        assert dev._get_batch_size(tensor0, shape, qml.math.prod(shape)) is None
        tensor1 = np.arange(np.prod(shape)).reshape(shape)
        assert dev._get_batch_size(tensor1, shape, qml.math.prod(shape)) is None

        broadcasted_shape = (1,) + shape
        tensor0 = np.ones(broadcasted_shape, dtype=complex)
        assert (
            dev._get_batch_size(tensor0, broadcasted_shape, qml.math.prod(broadcasted_shape))
            is None
        )
        tensor1 = np.arange(np.prod(broadcasted_shape)).reshape(broadcasted_shape)
        assert (
            dev._get_batch_size(tensor1, broadcasted_shape, qml.math.prod(broadcasted_shape))
            is None
        )

        broadcasted_shape = (3,) + shape
        tensor0 = np.ones(broadcasted_shape, dtype=complex)
        assert (
            dev._get_batch_size(tensor0, broadcasted_shape, qml.math.prod(broadcasted_shape))
            is None
        )
        tensor1 = np.arange(np.prod(broadcasted_shape)).reshape(broadcasted_shape)
        assert (
            dev._get_batch_size(tensor1, broadcasted_shape, qml.math.prod(broadcasted_shape))
            is None
        )
