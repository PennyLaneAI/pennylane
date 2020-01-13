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

import pennylane as qml
from pennylane import QubitDevice, DeviceError
from pennylane.qnodes import QuantumFunctionError
from pennylane import expval, var, sample
from pennylane.operation import Sample, Variance, Expectation, Probability

mock_qubit_device_paulis = ["PauliX", "PauliY", "PauliZ"]

# pylint: disable=abstract-class-instantiated, no-self-use, redefined-outer-name, invalid-name


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

@pytest.fixture(scope="function")
def mock_qubit_device_extract_stats(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, '_capabilities', mock_qubit_device_capabilities)
        m.setattr(QubitDevice, 'operations', ["PauliY", "RX", "Rot"])
        m.setattr(QubitDevice, 'observables', ["PauliZ"])
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        m.setattr(QubitDevice, 'expval', lambda self, x: 0)
        m.setattr(QubitDevice, 'var', lambda self, x: 0)
        m.setattr(QubitDevice, 'sample', lambda self, x: 0)
        m.setattr(QubitDevice, 'probability', lambda self, wires=None: 0 if wires is None else wires)
        m.setattr(QubitDevice, 'apply', lambda self, x: x)
        yield QubitDevice()

@pytest.fixture(scope="function")
def mock_qubit_device_with_original_statistics(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, '_capabilities', mock_qubit_device_capabilities)
        m.setattr(QubitDevice, 'operations', ["PauliY", "RX", "Rot"])
        m.setattr(QubitDevice, 'observables', ["PauliZ"])
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        yield QubitDevice()

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

class TestExtractStatistics:
    """Test the extract_statistics method"""

    @pytest.mark.parametrize("returntype", [Expectation, Variance, Sample, Probability])
    def test_results_created(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the extract_statistics simply builds a results list without any side-effects"""

        class SomeObservable(qml.operation.Observable):
            num_params = 0
            num_wires = 1
            par_domain = 'F'
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            results = mock_qubit_device_extract_stats.extract_statistics([obs])

        assert results == [0]

    @pytest.mark.parametrize("returntype", [None])
    def test_results_created(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the extract_statistics returns an empyt list if the return type is None"""

        class SomeObservable(qml.operation.Observable):
            num_params = 0
            num_wires = 1
            par_domain = 'F'
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            results = mock_qubit_device_extract_stats.extract_statistics([obs])

        assert results == []

    @pytest.mark.parametrize("returntype", ['not None'])
    def test_error_return_type_none(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the extract_statistics raises an error if the return type is not well-defined and is not None"""

        assert returntype not in [Expectation, Variance, Sample, Probability, None]

        class SomeObservable(qml.operation.Observable):
            num_params = 0
            num_wires = 1
            par_domain = 'F'
            return_type = returntype

        obs = SomeObservable(wires=0)

        with pytest.raises(
                QuantumFunctionError, match="Unsupported return type"
        ):
           results = mock_qubit_device_extract_stats.extract_statistics([obs])

class TestRotateBasis:
    """Test the rotate_basis method"""

    def test_wires_used_correct(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the rotate_basis method correctly stored the wires used"""

        assert mock_qubit_device_extract_stats._wires_used is None

        obs_queue = [qml.PauliX(0), qml.PauliZ(1)]

        with monkeypatch.context() as m:
            results = mock_qubit_device_extract_stats.rotate_basis(obs_queue)

        assert mock_qubit_device_extract_stats._wires_used == [0, 1]

    def test_wires_used_correct_for_empyt_obs_queue(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the rotate_basis method correctly stores an empty list when an empty
        observable queue is specified"""

        assert mock_qubit_device_extract_stats._wires_used is None

        obs_queue = []

        with monkeypatch.context() as m:
            results = mock_qubit_device_extract_stats.rotate_basis(obs_queue)

        assert mock_qubit_device_extract_stats._wires_used == []

    def test_probabilities_set_correctly(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the rotate_basis method correctly sets probabilities correctly"""

        assert mock_qubit_device_extract_stats._prob is None
        assert mock_qubit_device_extract_stats._rotated_prob is None

        obs_queue = [qml.PauliX(0), qml.PauliZ(1)]

        with monkeypatch.context() as m:
            mock_qubit_device_extract_stats.rotate_basis(obs_queue)

        assert mock_qubit_device_extract_stats._prob == 0
        assert mock_qubit_device_extract_stats._rotated_prob == [0, 1]

    def test_diagonalizing_gates_applied(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the rotate_basis method applies the diagonalizing gates"""
        obs_queue = [qml.PauliX(0), qml.PauliZ(1)]

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'apply', lambda self, op: call_history.append(op))
            mock_qubit_device_extract_stats.rotate_basis(obs_queue)

        assert len(call_history) == 1
        assert isinstance(call_history[0], qml.Hadamard)
        assert call_history[0].wires == [0]

    def test_memory_set_if_sample_return_type(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the rotate_basis method sets the _memory attribute correctly"""

        assert mock_qubit_device_extract_stats._memory is None

        with monkeypatch.context() as m:
            mock_qubit_device_extract_stats.rotate_basis([])

        assert not mock_qubit_device_extract_stats._memory

        class SomeObservable(qml.operation.Observable):
            num_params = 0
            num_wires = 1
            par_domain = 'F'
            return_type = Sample
            diagonalizing_gates = lambda self: []

        obs_queue = [SomeObservable(0)]

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'apply', lambda self, op: None)
            mock_qubit_device_extract_stats.rotate_basis(obs_queue)

        assert mock_qubit_device_extract_stats._memory

class TestGenerateSamples:
    """Test the generate_samples method"""

    def test_auxiliary_methods_called_correctly(self, mock_qubit_device, monkeypatch):
        """Tests that the generate_samples method calls on its auxiliary methods correctly"""

        mock_qubit_device._wires_used = [1,2]
        number_of_states = 2 ** len(mock_qubit_device._wires_used)

        with monkeypatch.context() as m:
            # Mock the auxiliary methods such that they return the expected values
            m.setattr(QubitDevice, 'sample_basis_states', lambda self, wires, b: wires)
            m.setattr(QubitDevice, 'states_to_binary', lambda a, b: (a, b))
            mock_qubit_device.generate_samples()

        assert mock_qubit_device._samples == (number_of_states, number_of_states)

class TestSampleBasisStates:
    """Test the sample_basis_states method"""

    def test_sampling_with_correct_arguments(self, mock_qubit_device, monkeypatch):
        """Tests that the sample_basis_states method samples with the correct arguments"""

        shots = 1000

        number_of_states = 4
        mock_qubit_device.shots = shots
        state_probs = [0.1, 0.2, 0.3, 0.4]

        with monkeypatch.context() as m:
            # Mock the numpy.random.choice method such that it returns the expected values
            m.setattr("numpy.random.choice", lambda x, y, p: (x, y, p))
            res = mock_qubit_device.sample_basis_states(number_of_states, state_probs)

        assert np.array_equal(res[0], np.array([0, 1, 2, 3]))
        assert res[1] == shots
        assert res[2] == state_probs

class TestExpval:
    """Test the expval method"""

    def test_rotate_basis(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that expval method when the _rotated_prob attribute is None"""
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        obs = qml.PauliX(0)

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: call_history.extend(op))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            res = mock_qubit_device_with_original_statistics.expval(obs)

        assert len(call_history) == 1
        assert isinstance(call_history[0], qml.PauliX)
        assert call_history[0].wires == [0]

