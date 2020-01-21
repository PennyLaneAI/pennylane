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
from pennylane.circuit_graph import CircuitGraph

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

    def test_op_queue_is_filled_during_execution(self, mock_qubit_device_with_paulis_and_methods, monkeypatch):
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

        circuit_graph = CircuitGraph(queue + observables, {})

        call_history = []

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'apply', lambda self, x, y: call_history.extend(x + y))
            mock_qubit_device_with_paulis_and_methods.execute(circuit_graph)

        assert call_history == queue

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

        circuit_graph = CircuitGraph(queue + observables, {})

        with pytest.raises(DeviceError, match="Gate Hadamard not supported on device"):
            mock_qubit_device_with_paulis_and_methods.execute(circuit_graph)


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

        circuit_graph = CircuitGraph(queue + observables, {})

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            mock_qubit_device_with_paulis_and_methods.execute(circuit_graph)

    def test_unsupported_observable_return_type_raise_error(self, mock_qubit_device_with_paulis_and_methods, monkeypatch):
        """Check that an error is raised if the return type of an observable is unsupported"""

        queue = [qml.PauliX(wires=0)]

        # Make a observable without specifying a return operation upon measuring
        obs = qml.PauliZ(0)
        obs.return_type = "SomeUnsupportedReturnType"
        observables = [obs]

        circuit_graph = CircuitGraph(queue + observables, {})

        with monkeypatch.context() as m:
            m.setattr(QubitDevice, 'apply', lambda self, x, y: None)
            with pytest.raises(QuantumFunctionError, match="Unsupported return type specified for observable"):
                mock_qubit_device_with_paulis_and_methods.execute(circuit_graph)


class TestParameters:
    """Test for checking device parameter mappings"""

    def test_parameters_accessed_outside_execution_context(self, mock_qubit_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        with pytest.raises(
                ValueError,
                match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            mock_qubit_device.parameters

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

    def test_wires_measured_correct(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the rotate_basis method correctly stored the wires used"""

        assert not mock_qubit_device_extract_stats._wires_measured

        obs_queue = [qml.PauliX(0), qml.PauliZ(1)]

        assert mock_qubit_device_extract_stats.active_wires(obs_queue) == {0, 1}

    def test_wires_measured_correct_for_empyt_obs_queue(self, mock_qubit_device_extract_stats, monkeypatch):
        """Tests that the rotate_basis method correctly stores an empty list when an empty
        observable queue is specified"""

        assert not mock_qubit_device_extract_stats._wires_measured

        obs_queue = []

        assert mock_qubit_device_extract_stats.active_wires(obs_queue) == set()


class TestGenerateSamples:
    """Test the generate_samples method"""

    def test_auxiliary_methods_called_correctly(self, mock_qubit_device, monkeypatch):
        """Tests that the generate_samples method calls on its auxiliary methods correctly"""

        mock_qubit_device._wires_measured = [1,2]
        number_of_states = 2 ** len(mock_qubit_device._wires_measured)

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
            m.setattr(QubitDevice, 'sample', lambda self, obs: obs)
            m.setattr("numpy.mean", lambda obs: obs)
            res = mock_qubit_device_with_original_statistics.expval(obs)

        assert res == obs

class TestVar:
    """Test the var method"""

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
            m.setattr(QubitDevice, 'sample', lambda self, obs: obs)
            m.setattr("numpy.var", lambda obs: obs)
            res = mock_qubit_device_with_original_statistics.var(obs)

        assert res == obs

class TestSample:
    """Test the sample method"""

    def test_only_ones_minus_ones(self, mock_qubit_device_with_original_statistics, monkeypatch, tol):
        """Test that pauli_eigvals_as_samples method only produces -1 and 1 samples"""
        obs = qml.PauliX(0)

        mock_qubit_device_with_original_statistics._samples = np.array([[1,0], [0,0]])

        with monkeypatch.context() as m:
            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert np.allclose(res ** 2, 1, atol=tol, rtol=0)

    def test_correct_custom_eigenvalues(self, mock_qubit_device_with_original_statistics, monkeypatch, tol):
        """Test that pauli_eigvals_as_samples method only produces samples of eigenvalues"""
        obs = qml.PauliX(0) @ qml.PauliZ(1)

        mock_qubit_device_with_original_statistics._samples = np.array([[1,0], [0,0]])

        with monkeypatch.context() as m:
            res = mock_qubit_device_with_original_statistics.sample(obs)

        assert np.array_equal(res, np.array([-1, 1]))

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

