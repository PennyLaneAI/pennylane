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
from pennylane import QubitDevice, DeviceError
from pennylane.qnodes import QuantumFunctionError
from pennylane import expval, var, sample
from pennylane.operation import Sample, Variance, Expectation, Probability

mock_qubit_device_paulis = ["PauliX", "PauliY", "PauliZ"]

# pylint: disable=abstract-class-instantiated, no-self-use, redefined-outer-name, invalid-name


@pytest.fixture(scope="function")
def mock_qubit_device(monkeypatch):
    """ A mock device that mocks most of the methods except for e.g. probability()"""
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
    """ A mock device that mocks the methods related to statistics (expval, var, sample, probability)"""
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
    """ A mock device that mocks only basis methods and uses the original statistics related methods"""
    with monkeypatch.context() as m:
        m.setattr(QubitDevice, '__abstractmethods__', frozenset())
        m.setattr(QubitDevice, '_capabilities', mock_qubit_device_capabilities)
        m.setattr(QubitDevice, 'operations', ["PauliY", "RX", "Rot"])
        m.setattr(QubitDevice, 'observables', ["PauliZ"])
        m.setattr(QubitDevice, 'short_name', 'MockDevice')
        yield QubitDevice()

mock_qubit_device_capabilities = {
    "measurements": "everything",
    "noise_models": ["depolarizing", "bitflip"],
}


@pytest.fixture(scope="function")
def mock_qubit_device_with_paulis_and_methods(monkeypatch):
    """A mock instance of the abstract QubitDevice class that supports Paulis in its capabilities"""
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
    """Test the statistics method"""

    @pytest.mark.parametrize("returntype", [Expectation, Variance, Sample, Probability])
    def test_results_created(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method simply builds a results list without any side-effects"""

        class SomeObservable(qml.operation.Observable):
            num_params = 0
            num_wires = 1
            par_domain = 'F'
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            results = mock_qubit_device_extract_stats.statistics([obs])

        assert results == [0]

    @pytest.mark.parametrize("returntype", [None])
    def test_results_created(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method returns an empty list if the return type is None"""

        class SomeObservable(qml.operation.Observable):
            num_params = 0
            num_wires = 1
            par_domain = 'F'
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            results = mock_qubit_device_extract_stats.statistics([obs])

        assert results == []

    @pytest.mark.parametrize("returntype", ['not None'])
    def test_error_return_type_none(self, mock_qubit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method raises an error if the return type is not well-defined and is not None"""

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
           results = mock_qubit_device_extract_stats.statistics([obs])

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

        assert mock_qubit_device_extract_stats._rotated_prob is None

        obs_queue = [qml.PauliX(0), qml.PauliZ(1)]

        with monkeypatch.context() as m:
            mock_qubit_device_extract_stats.rotate_basis(obs_queue)

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

class TestStatesToBinary:
    """Test the states_to_binary method"""

    def test_correct_conversion_two_states(self, mock_qubit_device, monkeypatch):
        """Tests that the sample_basis_states method converts samples to binary correctly"""
        number_of_states = 2
        basis_states = np.arange(number_of_states)
        samples = np.random.choice(basis_states, mock_qubit_device.shots)

        with monkeypatch.context() as m:
            res = mock_qubit_device.states_to_binary(samples, number_of_states)

        assert np.array_equal(res[:,0], samples)
        assert np.array_equal(res[:,1], np.zeros(mock_qubit_device.shots))


    # Note: in this visual matrix representation, the first columns stands for the first qubit
    # contrary to the bra-ket notation, so e.g. 2 will be represented as [0, 1, 0, 0] whereas
    # in bra-ket notation it would be |0010>
    @pytest.mark.parametrize("samples, binary_states",
                            [
                            (
                              np.array([2, 3, 2, 0, 0]),
                              np.array([[0, 1, 0, 0],
                                        [1, 1, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]],
                             )
                               ),
                            (
                            np.array([2, 3, 1, 3, 1]),
                            np.array([[0, 1, 0, 0],
                                       [1, 1, 0, 0],
                                       [1, 0, 0, 0],
                                       [1, 1, 0, 0],
                                       [1, 0, 0, 0]])

                            )
                            ])
    def test_correct_conversion_four_states(self, mock_qubit_device, monkeypatch, samples, binary_states, tol):
        """Tests that the states_to_binary method converts samples to binary correctly for four states"""
        mock_qubit_device.shots = 5

        number_of_states = 4

        with monkeypatch.context() as m:
            res = mock_qubit_device.states_to_binary(samples, number_of_states)

        assert np.allclose(res, binary_states, atol=tol, rtol=0)

    @pytest.mark.parametrize("samples, binary_states",
                            [
                            (
                            np.array([7, 7, 1, 5, 2]),
                            np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0, 0, 0]])
                            )
                            ])
    def test_correct_conversion_eight_states(self, mock_qubit_device, monkeypatch, samples, binary_states, tol):
        """Tests that the states_to_binary method converts samples to binary correctly for eight states"""
        mock_qubit_device.shots = 5

        number_of_states = 8

        with monkeypatch.context() as m:
            res = mock_qubit_device.states_to_binary(samples, number_of_states)

        assert np.allclose(res, binary_states, atol=tol, rtol=0)


class TestExpval:
    """Test the expval method"""

    def test_rotated_prob_none_rotate_basis(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that expval method when the _rotated_prob attribute is None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        """
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

    def test_rotated_prob_not_none_no_rotate_basis(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that expval method when the _rotated_prob attribute is another value than None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        mock_qubit_device_with_original_statistics._rotated_prob = []

        obs = qml.PauliX(0)

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: call_history.extend(op))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            res = mock_qubit_device_with_original_statistics.expval(obs)

        assert call_history == []

    def test_analytic_expval(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that expval method when the analytic attribute is True

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        """
        obs = qml.PauliX(0)
        probs = [0.5, 0.5]

        assert mock_qubit_device_with_original_statistics.analytic

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: probs)
            res = mock_qubit_device_with_original_statistics.expval(obs)

        assert res == (obs.eigvals @ probs).real

    def test_non_analytic_expval(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that expval method when the analytic attribute is False

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -sample
        -numpy.mean
        """
        obs = qml.PauliX(0)

        assert mock_qubit_device_with_original_statistics.analytic
        mock_qubit_device_with_original_statistics.analytic = False

        assert not mock_qubit_device_with_original_statistics.analytic

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'sample', lambda self, obs: obs)
            m.setattr("numpy.mean", lambda obs: obs)
            res = mock_qubit_device_with_original_statistics.expval(obs)

        assert res == obs

class TestVar:
    """Test the var method"""

    def test_rotated_prob_none_rotate_basis(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the _rotated_prob attribute is None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        obs = qml.PauliX(0)

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: call_history.extend(op))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            res = mock_qubit_device_with_original_statistics.var(obs)

        assert len(call_history) == 1
        assert isinstance(call_history[0], qml.PauliX)
        assert call_history[0].wires == [0]

    def test_rotated_prob_not_none_no_rotate_basis(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the _rotated_prob attribute is another value than None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        mock_qubit_device_with_original_statistics._rotated_prob = []

        obs = qml.PauliX(0)

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: call_history.extend(op))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            res = mock_qubit_device_with_original_statistics.var(obs)

        assert call_history == []

    def test_analytic_var(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the analytic attribute is True

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        """
        obs = qml.PauliX(0)
        probs = [0.5, 0.5]

        assert mock_qubit_device_with_original_statistics.analytic

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: probs)
            res = mock_qubit_device_with_original_statistics.var(obs)

        assert res == (obs.eigvals ** 2) @ probs - (obs.eigvals @ probs).real ** 2

    def test_non_analytic_var(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the analytic attribute is False

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -sample
        -numpy.var
        """
        obs = qml.PauliX(0)

        assert mock_qubit_device_with_original_statistics.analytic
        mock_qubit_device_with_original_statistics.analytic = False

        assert not mock_qubit_device_with_original_statistics.analytic

        call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'sample', lambda self, obs: obs)
            m.setattr("numpy.var", lambda obs: obs)
            res = mock_qubit_device_with_original_statistics.var(obs)

        assert res == obs

class TestSample:
    """Test the sample method"""

    def test_rotated_prob_none_rotate_with_memory(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that the sample method calls on the following methods the _rotated_prob attribute is None:
        -rotate_basis
        -generate_samples

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        -generate_samples
        -pauli_eigvals_as_samples
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        obs = qml.PauliX(0)

        mock_qubit_device_with_original_statistics._memory = True
        mock_qubit_device_with_original_statistics.analytic = False

        rotate_basis_call_history = []
        generate_samples_called = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: rotate_basis_call_history.extend(op))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            m.setattr(QubitDevice, 'generate_samples', lambda self: generate_samples_called.append(1))
            m.setattr(QubitDevice, 'pauli_eigvals_as_samples', lambda self, wires: None)
            m.setattr(QubitDevice, 'custom_eigvals_as_samples', lambda self, wires, eigvals: eigvals)

            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert len(rotate_basis_call_history) == 1
        assert isinstance(rotate_basis_call_history[0], qml.PauliX)
        assert rotate_basis_call_history[0].wires == [0]

        assert generate_samples_called == [1]

    def test_rotated_prob_none_rotate_basis_without_memory(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the _rotated_prob attribute is None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        -generate_samples
        -pauli_eigvals_as_samples
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        obs = qml.PauliX(0)

        mock_qubit_device_with_original_statistics._memory = False
        mock_qubit_device_with_original_statistics.analytic = True

        rotate_basis_call_history = []
        generate_samples_called = []

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: rotate_basis_call_history.extend(op))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            m.setattr(QubitDevice, 'generate_samples', lambda self: generate_samples_called.append(1))
            m.setattr(QubitDevice, 'pauli_eigvals_as_samples', lambda self, wires: None)
            m.setattr(QubitDevice, 'custom_eigvals_as_samples', lambda self, wires, eigvals: eigvals)

            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert len(rotate_basis_call_history) == 1
        assert isinstance(rotate_basis_call_history[0], qml.PauliX)
        assert rotate_basis_call_history[0].wires == [0]

        assert generate_samples_called == []

    def test_rotated_prob_not_none_no_rotate_basis(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that var method when the _rotated_prob attribute is another value than None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        -pauli_eigvals_as_samples
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        mock_qubit_device_with_original_statistics._rotated_prob = []

        obs = qml.PauliX(0)

        call_history_1 = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: call_history_1.extend(op))
            m.setattr(QubitDevice, 'generate_samples', lambda self: times_called.append(1))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            m.setattr(QubitDevice, 'pauli_eigvals_as_samples', lambda self, wires: None)
            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert call_history_1 == []

    def test_pauli_eigvals_called_rotated_prob_none(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that the sample method calls the pauli_eigvals_as_samples when the _rotated_prob attribute is  None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        -pauli_eigvals_as_samples
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        obs = qml.PauliX(0)

        pauli_eigvals_call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'generate_samples', lambda self: times_called.append(1))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            m.setattr(QubitDevice, 'pauli_eigvals_as_samples', lambda self, wires: pauli_eigvals_call_history.append(1))
            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert pauli_eigvals_call_history == [1]

    def test_pauli_eigvals_called_rotated_prob_not_none(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that the sample method calls the pauli_eigvals_as_samples when the _rotated_prob attribute is  None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        -pauli_eigvals_as_samples
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        # Setting the attribute to not None
        mock_qubit_device_with_original_statistics._rotated_prob = 'NotNone'

        obs = qml.PauliX(0)

        pauli_eigvals_call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'generate_samples', lambda self: times_called.append(1))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            m.setattr(QubitDevice, 'pauli_eigvals_as_samples', lambda self, wires: pauli_eigvals_call_history.append(1))
            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert pauli_eigvals_call_history == [1]

    def test_custom_eigvals_called_rotated_prob_none(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that the sample method calls the custom_eigvals_as_samples when the _rotated_prob attribute is  None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        -pauli_eigvals_as_samples
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        obs = qml.PauliX(0) @ qml.PauliX(1)

        custom_eigvals_call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'generate_samples', lambda self: times_called.append(1))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            m.setattr(QubitDevice, 'custom_eigvals_as_samples', lambda self, wires, eigvals: custom_eigvals_call_history.append(1))
            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert custom_eigvals_call_history == [1]

    def test_custom_eigvals_called_rotated_prob_not_none(self, mock_qubit_device_with_original_statistics, monkeypatch):
        """Tests that the sample method calls the pauli_eigvals_as_samples when the _rotated_prob attribute is  None

        Additional QubitDevice methods that are mocked:
        -rotate_basis
        -probability
        -pauli_eigvals_as_samples
        """
        assert mock_qubit_device_with_original_statistics._rotated_prob is None

        # Setting the attribute to not None
        mock_qubit_device_with_original_statistics._rotated_prob = 'NotNone'

        obs = qml.PauliX(0) @ qml.PauliX(1)

        custom_eigvals_call_history = []
        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'rotate_basis', lambda self, op: None)
            m.setattr(QubitDevice, 'generate_samples', lambda self: times_called.append(1))
            m.setattr(QubitDevice, 'probability', lambda self, wires=None: [1,1])
            m.setattr(QubitDevice, 'custom_eigvals_as_samples', lambda self, wires, eigvals: custom_eigvals_call_history.append(1))
            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert custom_eigvals_call_history == [1]

class TestPauliEigvalsAsSamples:
    """Test the pauli_eigvals_as_samples method"""

    def test_only_ones_minus_ones(self, mock_qubit_device_with_original_statistics, monkeypatch, tol):
        """Test that pauli_eigvals_as_samples method only produces -1 and 1 samples"""

        some_wires = [0]

        mock_qubit_device_with_original_statistics._samples = np.array([[1,0], [0,0]])
        with monkeypatch.context() as m:
            res = mock_qubit_device_with_original_statistics.pauli_eigvals_as_samples(some_wires)

        assert np.allclose(res ** 2, 1, atol=tol, rtol=0)

class TestCustomEigvalsAsSamples:
    """Test the custom_eigvals_as_samples method"""

    def test_correct_custom_eigenvalues(self, mock_qubit_device_with_original_statistics, monkeypatch, tol):
        """Test that pauli_eigvals_as_samples method only produces samples of eigenvalues"""
        some_wires = [0]

        mock_qubit_device_with_original_statistics._samples = np.array([[1,0], [0,0]])

        eigenvalues = np.array([6, 5, 12, -54])

        with monkeypatch.context() as m:
            res = mock_qubit_device_with_original_statistics.custom_eigvals_as_samples(some_wires, eigenvalues)

        assert np.array_equal(res, np.array([5, 6]))

class TestProbability:
    """Test the probability method"""

    def test_state_none(self, mock_qubit_device_with_original_statistics, monkeypatch, tol):
        """Test that None is returned if the state is None"""
        mock_qubit_device_with_original_statistics._state = None
        with monkeypatch.context() as m:
            res = mock_qubit_device_with_original_statistics.probability()

        assert res is None

    @pytest.mark.parametrize("wires", [[0],
                                      [1],
                                      [2],
                                      [0,1],
                                      [0,2],
                                      [1,2],
                                      [0, 1, 2],
                                      ])
    def test_correct_arguments_for_marginals(self, mock_qubit_device_with_original_statistics, wires, monkeypatch, tol):
        """Test that None is returned if the state is None"""
        mock_qubit_device_with_original_statistics._state = None

        # Generate array to be used as a state
        state = np.array([random() for i in range(2 ** 3)])

        mock_qubit_device_with_original_statistics._state = state

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, "marginal_prob", lambda self, state, wires: (state, wires))
            res = mock_qubit_device_with_original_statistics.probability(wires=wires)

        assert np.allclose(res[0], np.abs(state)**2, atol=tol, rtol=0)
        assert res[1] == wires

class TestMarginalProb:
    """Test the marginal_prob method"""

    @pytest.mark.parametrize("wires, inactive_wires", [([0], [1, 2]),
                                                      ([1], [0, 2]),
                                                      ([2], [0, 1]),
                                                      ([0,1], [2]),
                                                      ([0,2], [1]),
                                                      ([1,2], [0]),
                                                      ([0, 1, 2], []),
                                                      ])
    def test_correct_arguments_for_marginals(self, mock_qubit_device_with_original_statistics, monkeypatch, wires, inactive_wires, tol):
        """Test that the correct arguments are passed to the marginal_prob method"""

        mock_qubit_device_with_original_statistics.num_wires = 3

        # Generate probabilities
        probs = np.array([random() for i in range(2 ** 3)])
        probs /= sum(probs)

        def apply_over_axes_mock(x, y, p):
            arguments_apply_over_axes.append((y, p))
            return np.array([0])

        arguments_apply_over_axes = []
        with monkeypatch.context() as m:
            m.setattr("numpy.apply_over_axes", apply_over_axes_mock)
            res = mock_qubit_device_with_original_statistics.marginal_prob(probs, wires=wires)

        assert np.array_equal(arguments_apply_over_axes[0][0].flatten(), probs)
        assert np.array_equal(arguments_apply_over_axes[0][1], inactive_wires)

    def test_correct_arguments_for_marginals_no_wires(self, mock_qubit_device_with_original_statistics, monkeypatch, tol):
        """Test that the correct arguments are passed to the marginal_prob method with no wires specified"""

        mock_qubit_device_with_original_statistics.num_wires = 3

        # Generate probabilities
        probs = np.array([random() for i in range(2 ** 3)])
        probs /= sum(probs)

        def apply_over_axes_mock(x, y, p):
            arguments_apply_over_axes.append((y, p))
            return np.array([0])

        arguments_apply_over_axes = []
        with monkeypatch.context() as m:
            m.setattr("numpy.apply_over_axes", apply_over_axes_mock)
            res = mock_qubit_device_with_original_statistics.marginal_prob(probs)

        assert np.array_equal(arguments_apply_over_axes[0][0].flatten(), probs)
        assert np.array_equal(arguments_apply_over_axes[0][1], [])

    @pytest.mark.parametrize("probs, marginals, wires",
                                    [(np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.4, 0.6]), [1]),
                                    (np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.3, 0.7]), [0])])
    def test_correct_marginals_returned(self, mock_qubit_device_with_original_statistics, monkeypatch, probs, marginals, wires, tol):
        """Test that the correct marginals are returned by the marginal_prob method"""

        mock_qubit_device_with_original_statistics.num_wires = 2

        with monkeypatch.context() as m:
            res = mock_qubit_device_with_original_statistics.marginal_prob(probs, wires=wires)

        assert np.allclose(res, marginals, atol=tol, rtol=0)

