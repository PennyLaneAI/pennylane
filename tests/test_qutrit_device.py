# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`QutritDevice` class.
"""
import pytest
import numpy as np
from random import random
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import QutritDevice, DeviceError, QuantumFunctionError, QubitDevice
from pennylane.devices import DefaultQubit
from pennylane.measurements import Sample, Variance, Expectation, Probability, State, Counts
from pennylane.circuit_graph import CircuitGraph
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.measurements import state


@pytest.fixture(scope="function")
def mock_qutrit_device(monkeypatch):
    """A function to create a mock qutrit device that mocks most of the methods except for e.g. probability()"""
    with monkeypatch.context() as m:
        m.setattr(QutritDevice, "__abstractmethods__", frozenset())
        m.setattr(QutritDevice, "_capabilities", mock_qutrit_device_capabilities)
        m.setattr(QutritDevice, "short_name", "MockQutritDevice")
        m.setattr(QutritDevice, "operations", ["QutritUnitary", "Identity"])
        m.setattr(QutritDevice, "observables", ["Identity"])
        m.setattr(QutritDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "apply", lambda self, *args, **kwargs: None)

        def get_qutrit_device(wires=1):
            return QutritDevice(wires=wires)

        yield get_qutrit_device


mock_qutrit_device_capabilities = {
    "measurements": "everything",
    "returns_state": True,
}


@pytest.fixture(scope="function")
def mock_qutrit_device_extract_stats(monkeypatch):
    """A function to create a mock device that mocks the methods related to
    statistics (expval, var, sample, probability)"""
    with monkeypatch.context() as m:
        m.setattr(QutritDevice, "__abstractmethods__", frozenset())
        m.setattr(QutritDevice, "_capabilities", mock_qutrit_device_capabilities)
        m.setattr(QutritDevice, "short_name", "MockQutritDevice")
        m.setattr(QutritDevice, "operations", ["QutritUnitary", "Identity"])
        m.setattr(QutritDevice, "observables", ["Identity"])
        m.setattr(QutritDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "state", 0)
        m.setattr(QutritDevice, "density_matrix", lambda self, wires=None: 0)
        m.setattr(QutritDevice, "probability", lambda self, wires=None, *args, **kwargs: 0)
        m.setattr(QutritDevice, "apply", lambda self, x, **kwargs: x)

        def get_qutrit_device(wires=1):
            return QutritDevice(wires=wires)

        yield get_qutrit_device


@pytest.fixture(scope="function")
def mock_qutrit_device_shots(monkeypatch):
    """A function to create a mock device that mocks the methods related to
    statistics (expval, var, sample, probability)"""
    with monkeypatch.context() as m:
        m.setattr(QutritDevice, "__abstractmethods__", frozenset())
        m.setattr(QutritDevice, "_capabilities", mock_qutrit_device_capabilities)
        m.setattr(QutritDevice, "short_name", "MockQutritDevice")
        m.setattr(QutritDevice, "operations", ["QutritUnitary", "Identity"])
        m.setattr(QutritDevice, "observables", ["Identity"])
        m.setattr(QutritDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "state", 0)
        m.setattr(QutritDevice, "density_matrix", lambda self, wires=None: 0)
        m.setattr(QutritDevice, "apply", lambda self, x, **kwargs: x)
        m.setattr(
            QutritDevice,
            "analytic_probability",
            lambda self, wires=None: QutritDevice.marginal_prob(self, np.ones(9) / 9.0, wires),
        )

        def get_qutrit_device(wires=1, shots=None):
            return QutritDevice(wires=wires, shots=shots)

        yield get_qutrit_device


@pytest.fixture(scope="function")
def mock_qutrit_device_with_original_statistics(monkeypatch):
    """A function to create a mock qutrit device that uses the original statistics related
    methods"""
    with monkeypatch.context() as m:
        m.setattr(QutritDevice, "__abstractmethods__", frozenset())
        m.setattr(QutritDevice, "_capabilities", mock_qutrit_device_capabilities)
        m.setattr(QutritDevice, "short_name", "MockQutritDevice")
        m.setattr(QutritDevice, "operations", ["QutritUnitary", "Identity"])
        m.setattr(QutritDevice, "observables", ["Identity"])

        def get_qutrit_device(wires=1):
            return QutritDevice(wires=wires)

        yield get_qutrit_device


# TODO: Add tests for expval, var after observables are added


class TestOperations:
    """Tests the logic related to operations"""

    def test_op_queue_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        dev = mock_qutrit_device()
        with pytest.raises(
            ValueError, match="Cannot access the operation queue outside of the execution context!"
        ):
            dev.op_queue

    def test_op_queue_is_filled_during_execution(self, mock_qutrit_device, monkeypatch):
        """Tests that the op_queue is correctly filled when apply is called and that accessing
        op_queue raises no error"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.QutritUnitary(U, wires=0), qml.QutritUnitary(U, wires=0)]
            observables = [qml.expval(qml.Identity(0))]

        call_history = []

        with monkeypatch.context() as m:
            m.setattr(
                QutritDevice,
                "apply",
                lambda self, x, **kwargs: call_history.extend(x + kwargs.get("rotations", [])),
            )
            m.setattr(QutritDevice, "analytic_probability", lambda *args: None)
            m.setattr(QutritDevice, "statistics", lambda self, *args, **kwargs: 0)
            dev = mock_qutrit_device()
            dev.execute(tape)

        assert call_history == queue

        assert len(call_history) == 2
        assert isinstance(call_history[0], qml.QutritUnitary)
        assert call_history[0].wires == Wires([0])

        assert isinstance(call_history[1], qml.QutritUnitary)
        assert call_history[1].wires == Wires([0])

    def test_unsupported_operations_raise_error(self, mock_qutrit_device):
        """Tests that the operations are properly applied and queued"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            queue = [
                qml.QutritUnitary(U, wires=0),
                qml.Hadamard(wires=1),
                qml.QutritUnitary(U, wires=2),
            ]
            observables = [qml.expval(qml.Identity(0)), qml.var(qml.Identity(1))]

        dev = mock_qutrit_device()
        with pytest.raises(DeviceError, match="Gate Hadamard not supported on device"):
            dev.execute(tape)

    unitaries = [unitary_group.rvs(3, random_state=1967) for _ in range(3)]
    numeric_queues = [
        [qml.QutritUnitary(unitaries[0], wires=[0])],
        [
            qml.QutritUnitary(unitaries[0], wires=[0]),
            qml.QutritUnitary(unitaries[1], wires=[1]),
            qml.QutritUnitary(unitaries[2], wires=[2]),
        ],
    ]

    observables = [[qml.Identity(0)], [qml.Identity(1)]]

    @pytest.mark.parametrize("observables", observables)
    @pytest.mark.parametrize("queue", numeric_queues)
    def test_passing_keyword_arguments_to_execute(
        self, mock_qutrit_device, monkeypatch, queue, observables
    ):
        """Tests that passing keyword arguments to execute propagates those kwargs to the apply()
        method"""
        with qml.tape.QuantumTape() as tape:
            for op in queue + observables:
                op.queue()

        call_history = {}

        with monkeypatch.context() as m:
            m.setattr(QutritDevice, "apply", lambda self, x, **kwargs: call_history.update(kwargs))
            m.setattr(QutritDevice, "statistics", lambda self, *args, **kwargs: 0)
            dev = mock_qutrit_device()
            dev.execute(tape, hash=tape.graph.hash)

        len(call_history.items()) == 1
        call_history["hash"] = tape.graph.hash


class TestObservables:
    """Tests the logic related to observables"""

    U = unitary_group.rvs(3, random_state=10)

    def test_obs_queue_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        dev = mock_qutrit_device()
        with pytest.raises(
            ValueError,
            match="Cannot access the observable value queue outside of the execution context!",
        ):
            dev.obs_queue

    def test_unsupported_observables_raise_error(self, mock_qutrit_device):
        """Tests that the operations are properly applied and queued"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.QutritUnitary(U, wires=0)]
            observables = [qml.expval(qml.Hadamard(0))]

        dev = mock_qutrit_device()
        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            dev.execute(tape)

    def test_unsupported_observable_return_type_raise_error(self, mock_qutrit_device, monkeypatch):
        """Check that an error is raised if the return type of an observable is unsupported"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            qml.QutritUnitary(U, wires=0)
            qml.measurements.MeasurementProcess(
                return_type="SomeUnsupportedReturnType", obs=qml.Identity(0)
            )

        with monkeypatch.context() as m:
            m.setattr(QutritDevice, "apply", lambda self, x, **kwargs: None)
            dev = mock_qutrit_device()
            with pytest.raises(
                qml.QuantumFunctionError, match="Unsupported return type specified for observable"
            ):
                dev.execute(tape)


class TestParameters:
    """Test for checking device parameter mappings"""

    def test_parameters_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        dev = mock_qutrit_device()
        with pytest.raises(
            ValueError,
            match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            dev.parameters


class TestExtractStatistics:
    """Test the statistics method"""

    @pytest.mark.parametrize(
        "returntype", [Expectation, Variance, Sample, Probability, State, Counts]
    )
    def test_results_created(self, mock_qutrit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method simply builds a results list without any side-effects"""

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            dev = mock_qutrit_device_extract_stats()
            results = dev.statistics([obs])

        assert results == [0]

    def test_results_no_state(self, mock_qutrit_device, monkeypatch):
        """Tests that the statistics method raises an AttributeError when a State return type is
        requested when QutritDevice does not have a state attribute"""
        with monkeypatch.context() as m:
            dev = mock_qutrit_device()
            m.delattr(QubitDevice, "state")
            with pytest.raises(
                qml.QuantumFunctionError, match="The state is not available in the current"
            ):
                dev.statistics([qml.state()])

    @pytest.mark.parametrize("returntype", [None])
    def test_results_created_empty(self, mock_qutrit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method returns an empty list if the return type is None"""

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            dev = mock_qutrit_device_extract_stats()
            results = dev.statistics([obs])

        assert results == []

    @pytest.mark.parametrize("returntype", ["not None"])
    def test_error_return_type_not_none(
        self, mock_qutrit_device_extract_stats, monkeypatch, returntype
    ):
        """Tests that the statistics method raises an error if the return type is not well-defined and is not None"""

        assert returntype not in [Expectation, Variance, Sample, Probability, State, Counts, None]

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        dev = mock_qutrit_device_extract_stats()
        with pytest.raises(qml.QuantumFunctionError, match="Unsupported return type"):
            dev.statistics([obs])

    def test_return_state_with_multiple_observables(self, mock_qutrit_device_extract_stats):
        """Checks that an error is raised if multiple observables are being returned
        and one of them is state
        """
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            qml.QutritUnitary(U, wires=0)
            qml.state()
            qml.probs(wires=0)

        dev = mock_qutrit_device_extract_stats()

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The state or density matrix cannot be returned in combination",
        ):
            dev.execute(tape)


class TestSample:
    """Test the sample method"""

    # TODO: Add tests for sampling with observables that have eigenvalues to sample from once
    # such observables are added for qutrits.
    # TODO: Add tests for counts for observables with eigenvalues once such observables are
    # added for qutrits.

    def test_sample_with_no_observable_and_no_wires(
        self, mock_qutrit_device_with_original_statistics, tol
    ):
        """Test that when we sample a device without providing an observable or wires then it
        will return the raw samples"""
        obs = qml.measurements.sample(op=None, wires=None)
        dev = mock_qutrit_device_with_original_statistics(wires=2)
        generated_samples = np.array([[1, 2], [0, 1]])
        dev._samples = generated_samples

        res = dev.sample(obs)
        assert np.array_equal(res, generated_samples)

    def test_sample_with_no_observable_and_with_wires(
        self, mock_qutrit_device_with_original_statistics, tol
    ):
        """Test that when we sample a device without providing an observable but we specify
        wires then it returns the generated samples for only those wires"""
        obs = qml.measurements.sample(op=None, wires=[1])
        dev = mock_qutrit_device_with_original_statistics(wires=2)
        generated_samples = np.array([[1, 0], [2, 1]])
        dev._samples = generated_samples

        wire_samples = np.array([[0], [1]])
        res = dev.sample(obs)

        assert np.array_equal(res, wire_samples)

    def test_no_eigval_error(self, mock_qutrit_device_with_original_statistics):
        """Tests that an error is thrown if sample is called with an observable that does not have eigenvalues defined."""
        dev = mock_qutrit_device_with_original_statistics(wires=2)
        dev._samples = np.array([[1, 0], [0, 2]])

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = Sample

        obs = SomeObservable(wires=0)
        with pytest.raises(qml.operation.EigvalsUndefinedError, match="Cannot compute samples"):
            dev.sample(SomeObservable(wires=0))

    def test_samples_with_bins(self, mock_qutrit_device_with_original_statistics, monkeypatch):
        """Tests that sample works correctly when instantiating device with shot list"""

        dev = mock_qutrit_device_with_original_statistics(wires=2)
        samples = np.array([[0, 1], [2, 0], [2, 1], [1, 1], [2, 2], [1, 2]])
        dev._samples = samples
        obs = qml.measurements.sample(op=None, wires=[0, 1])

        shot_range = [0, 6]
        bin_size = 3

        out = dev.sample(obs, shot_range=shot_range, bin_size=bin_size)
        expected_samples = samples.reshape(-1, 3, 2)

        assert np.array_equal(out, expected_samples)

    def test_counts(self, mock_qutrit_device_with_original_statistics, monkeypatch):
        dev = mock_qutrit_device_with_original_statistics(wires=2)
        samples = np.array([[0, 1], [2, 0], [2, 0], [0, 1], [2, 2], [1, 2]])
        dev._samples = samples
        obs = qml.measurements.sample(op=None, wires=[0, 1])

        out = dev.sample(obs, counts=True)
        expected_counts = {
            "01": 2,
            "20": 2,
            "22": 1,
            "12": 1,
        }

        assert out == expected_counts

    def test_raw_counts_with_bins(self, mock_qutrit_device_with_original_statistics, monkeypatch):
        dev = mock_qutrit_device_with_original_statistics(wires=2)
        samples = np.array(
            [
                [0, 1],
                [2, 0],
                [2, 0],
                [0, 1],
                [2, 2],
                [1, 2],
                [0, 1],
                [2, 0],
                [2, 1],
                [0, 2],
                [2, 1],
                [1, 2],
            ]
        )
        dev._samples = samples
        obs = qml.measurements.sample(op=None, wires=[0, 1])

        shot_range = [0, 12]
        bin_size = 4
        out = dev.sample(obs, shot_range=shot_range, bin_size=bin_size, counts=True)

        expected_counts = [
            {"01": 2, "20": 2},
            {"22": 1, "12": 1, "01": 1, "20": 1},
            {"21": 2, "02": 1, "12": 1},
        ]

        assert out == expected_counts


class TestGenerateSamples:
    """Test the generate_samples method"""

    def test_auxiliary_methods_called_correctly(self, mock_qutrit_device, monkeypatch):
        """Tests that the generate_samples method calls on its auxiliary methods correctly"""

        dev = mock_qutrit_device()
        number_of_states = 3**dev.num_wires

        with monkeypatch.context() as m:
            # Mock the auxiliary methods such that they return the expected values
            m.setattr(QutritDevice, "sample_basis_states", lambda self, wires, b: wires)
            m.setattr(QutritDevice, "states_to_ternary", staticmethod(lambda a, b: (a, b)))
            m.setattr(QutritDevice, "analytic_probability", lambda *args: None)
            m.setattr(QutritDevice, "shots", 1000)
            dev._samples = dev.generate_samples()

        assert dev._samples == (number_of_states, dev.num_wires)


class TestSampleBasisStates:
    """Test the sample_basis_states method"""

    def test_sampling_with_correct_arguments(self, mock_qutrit_device, monkeypatch):
        """Tests that the sample_basis_states method samples with the correct arguments"""

        shots = 1000

        number_of_states = 9
        dev = mock_qutrit_device()
        dev.shots = shots
        state_probs = [0.1] * 9
        state_probs[0] = 0.2

        with monkeypatch.context() as m:
            # Mock the numpy.random.choice method such that it returns the expected values
            m.setattr("numpy.random.choice", lambda x, y, p: (x, y, p))
            res = dev.sample_basis_states(number_of_states, state_probs)

        assert np.array_equal(res[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
        assert res[1] == shots
        assert res[2] == state_probs

    def test_raises_deprecation_error(self, mock_qutrit_device, monkeypatch):
        """Test that sampling basis states on a device with shots=None produces an error."""

        dev = mock_qutrit_device()
        number_of_states = 9
        dev.shots = None
        state_probs = [0.1] * 9
        state_probs[0] = 0.2

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of shots has to be explicitly set on the device",
        ):
            dev.sample_basis_states(number_of_states, state_probs)


class TestStatesToTernary:
    """Test the states_to_ternary method"""

    def test_correct_conversion_three_states(self, mock_qutrit_device):
        """Tests that the sample_basis_states method converts samples to ternary correctly"""
        wires = 4
        samples = [10, 31, 80, 65, 44, 2]

        dev = mock_qutrit_device()
        res = dev.states_to_ternary(samples, wires)

        expected = [
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [2, 2, 2, 2],
            [2, 1, 0, 2],
            [1, 1, 2, 2],
            [0, 0, 0, 2],
        ]

        assert np.array_equal(res, np.array(expected))

    test_ternary_conversion_data = [
        (
            np.array([2, 3, 2, 0, 0, 1, 6, 8, 5, 6]),
            np.array(
                [
                    [0, 2],
                    [1, 0],
                    [0, 2],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [2, 0],
                    [2, 2],
                    [1, 2],
                    [2, 0],
                ]
            ),
        ),
        (
            np.array([2, 7, 6, 8, 4, 1, 5]),
            np.array(
                [
                    [0, 2],
                    [2, 1],
                    [2, 0],
                    [2, 2],
                    [1, 1],
                    [0, 1],
                    [1, 2],
                ]
            ),
        ),
        (
            np.array([10, 7, 2, 15, 26, 20, 18, 24, 11, 6, 1, 0]),
            np.array(
                [
                    [1, 0, 1],
                    [0, 2, 1],
                    [0, 0, 2],
                    [1, 2, 0],
                    [2, 2, 2],
                    [2, 0, 2],
                    [2, 0, 0],
                    [2, 2, 0],
                    [1, 0, 2],
                    [0, 2, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
        ),
    ]

    @pytest.mark.parametrize("samples, ternary_states", test_ternary_conversion_data)
    def test_correct_conversion(self, mock_qutrit_device, samples, ternary_states, tol):
        """Tests that the states_to_ternary method converts samples to ternary correctly"""
        dev = mock_qutrit_device()
        dev.shots = 5
        wires = ternary_states.shape[1]
        res = dev.states_to_ternary(samples, wires)
        assert np.allclose(res, ternary_states, atol=tol, rtol=0)


class TestEstimateProb:
    """Test the estimate_probability method"""

    @pytest.mark.parametrize(
        "wires, bin_size, expected",
        [
            ([0], None, [0.5, 0.25, 0.25]),
            (None, None, [0.25, 0, 0.25, 0, 0.25, 0, 0, 0, 0.25]),
            ([0, 1], None, [0.25, 0, 0.25, 0, 0.25, 0, 0, 0, 0.25]),
            ([1], None, [0.25, 0.25, 0.5]),
            ([0], 4, [[0.5], [0.25], [0.25]]),
        ],
    )
    def test_estimate_probability(
        self, wires, bin_size, expected, mock_qutrit_device_with_original_statistics, monkeypatch
    ):
        """Tests probability method when the analytic attribute is True."""
        dev = mock_qutrit_device_with_original_statistics(wires=2)
        samples = np.array([[0, 0], [2, 2], [1, 1], [0, 2]])

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            m.setattr(dev, "shots", 4)
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
        self, mock_qutrit_device_with_original_statistics, mocker, wires, inactive_wires, tol
    ):
        """Test that the correct arguments are passed to the marginal_prob method"""

        # Generate probabilities
        probs = np.array([random() for i in range(3**3)])
        probs /= sum(probs)

        spy = mocker.spy(np, "sum")
        dev = mock_qutrit_device_with_original_statistics(wires=3)
        res = dev.marginal_prob(probs, wires=wires)
        array_call = spy.call_args[0][0]
        axis_call = spy.call_args[1]["axis"]

        assert np.allclose(array_call.flatten(), probs, atol=tol, rtol=0)
        assert axis_call == tuple(inactive_wires)

    p = np.arange(0.01, 0.28, 0.01) / np.sum(np.arange(0.01, 0.28, 0.01))
    probs = np.reshape(p, [3] * 3)
    s00 = np.sum(probs[0, :, 0])
    s10 = np.sum(probs[1, :, 0])
    s20 = np.sum(probs[2, :, 0])
    s01 = np.sum(probs[0, :, 1])
    s11 = np.sum(probs[1, :, 1])
    s21 = np.sum(probs[2, :, 1])
    s02 = np.sum(probs[0, :, 2])
    s12 = np.sum(probs[1, :, 2])
    s22 = np.sum(probs[2, :, 2])
    m_probs = np.array([s00, s10, s20, s01, s11, s21, s02, s12, s22])

    marginal_test_data = [
        (
            np.array([0.1, 0.2, 0.3, 0.04, 0.03, 0.02, 0.01, 0.18, 0.12]),
            np.array([0.15, 0.41, 0.44]),
            [1],
        ),
        (
            np.array([0.1, 0.2, 0.3, 0.04, 0.03, 0.02, 0.01, 0.18, 0.12]),
            np.array([0.6, 0.09, 0.31]),
            [0],
        ),
        (p, m_probs, [2, 0]),
    ]

    @pytest.mark.parametrize("probs, marginals, wires", marginal_test_data)
    def test_correct_marginals_returned(
        self, mock_qutrit_device_with_original_statistics, probs, marginals, wires, tol
    ):
        """Test that the correct marginals are returned by the marginal_prob method"""
        num_wires = int(np.log(len(probs)) / np.log(3))  # Same as log_3(len(probs))
        dev = mock_qutrit_device_with_original_statistics(num_wires)
        res = dev.marginal_prob(probs, wires=wires)
        assert np.allclose(res, marginals, atol=tol, rtol=0)

    @pytest.mark.parametrize("probs, marginals, wires", marginal_test_data)
    def test_correct_marginals_returned_wires_none(
        self, mock_qutrit_device_with_original_statistics, probs, marginals, wires, tol
    ):
        """Test that passing wires=None simply returns the original probability."""
        num_wires = int(np.log(len(probs)) / np.log(3))  # Same as log_3(len(probs))
        dev = mock_qutrit_device_with_original_statistics(wires=num_wires)
        dev.num_wires = num_wires

        res = dev.marginal_prob(probs, wires=None)
        assert np.allclose(res, probs, atol=tol, rtol=0)


class TestActiveWires:
    """Test that the active_wires static method works as required."""

    def test_active_wires_from_queue(self, mock_qutrit_device):
        queue = [
            qml.QutritUnitary(np.eye(9), wires=[0, 2]),
            qml.QutritUnitary(np.eye(3), wires=0),
            qml.expval(qml.Identity(wires=5)),
        ]

        dev = mock_qutrit_device(wires=6)
        res = dev.active_wires(queue)

        assert res == Wires([0, 2, 5])


class TestCapabilities:
    """Test that a qutrit device defines capabilities that all devices inheriting
    from it will automatically have."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""
        capabilities = {
            "model": "qutrit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "supports_broadcasting": False,
        }
        assert capabilities == QutritDevice.capabilities()


class TestExecution:
    """Tests for the execute method"""

    def test_device_executions(self, mock_qutrit_device_extract_stats):
        """Test the number of times a qutrit device is executed over a QNode's
        lifetime is tracked by `num_executions`"""

        dev_1 = mock_qutrit_device_extract_stats(wires=2)

        def circuit_1(U1, U2, U3):
            qml.QutritUnitary(U1, wires=[0])
            qml.QutritUnitary(U2, wires=[1])
            qml.QutritUnitary(U3, wires=[0, 1])
            return qml.state()

        node_1 = qml.QNode(circuit_1, dev_1)
        num_evals_1 = 10

        for _ in range(num_evals_1):
            node_1(np.eye(3), np.eye(3), np.eye(9))
        assert dev_1.num_executions == num_evals_1

        # test a new circuit on an existing instance of a qutrit device
        def circuit_3(U1, U2):
            qml.QutritUnitary(U1, wires=[0])
            qml.QutritUnitary(U2, wires=[0, 1])
            return qml.state()

        node_3 = qml.QNode(circuit_3, dev_1)
        num_evals_3 = 7

        for _ in range(num_evals_3):
            node_3(np.eye(3), np.eye(9))
        assert dev_1.num_executions == num_evals_1 + num_evals_3


class TestBatchExecution:
    """Tests for the batch_execute method."""

    with qml.tape.QuantumTape() as tape1:
        qml.QutritUnitary(np.eye(3), wires=0)
        qml.expval(qml.Identity(0)), qml.expval(qml.Identity(1))

    with qml.tape.QuantumTape() as tape2:
        qml.QutritUnitary(np.eye(3), wires=0)
        qml.expval(qml.Identity(0))

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_execute(self, n_tapes, mocker, mock_qutrit_device):
        """Tests that the device's execute method is called the correct number of times."""

        dev = mock_qutrit_device(wires=2)
        spy = mocker.spy(QutritDevice, "execute")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_reset(self, n_tapes, mocker, mock_qutrit_device):
        """Tests that the device's reset method is called the correct number of times."""

        dev = mock_qutrit_device(wires=2)

        spy = mocker.spy(QutritDevice, "reset")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_result(self, mock_qutrit_device, r_dtype, tol):
        """Tests that the result has the correct shape and entry types."""

        dev = mock_qutrit_device(wires=2)
        dev.R_DTYPE = r_dtype

        tapes = [self.tape1, self.tape2]
        res = dev.batch_execute(tapes)

        assert len(res) == 2
        assert np.allclose(res[0], dev.execute(self.tape1), rtol=tol, atol=0)
        assert np.allclose(res[1], dev.execute(self.tape2), rtol=tol, atol=0)
        assert res[0].dtype == r_dtype
        assert res[1].dtype == r_dtype

    def test_result_empty_tape(self, mock_qutrit_device, tol):
        """Tests that the result has the correct shape and entry types for empty tapes."""

        dev = mock_qutrit_device(wires=2)

        empty_tape = qml.tape.QuantumTape()
        tapes = [empty_tape] * 3
        res = dev.batch_execute(tapes)

        assert len(res) == 3
        assert np.allclose(res[0], dev.execute(empty_tape), rtol=tol, atol=0)


class TestShotList:
    """Tests for passing shots as a list"""

    # TODO: Add tests for expval and sample with shot lists after observables are added

    def test_invalid_shot_list(self, mock_qutrit_device_shots):
        """Test exception raised if the shot list is the wrong type"""
        with pytest.raises(qml.DeviceError, match="Shots must be"):
            mock_qutrit_device_shots(wires=2, shots=0.5)

        with pytest.raises(ValueError, match="Unknown shot sequence"):
            mock_qutrit_device_shots(wires=2, shots=["a", "b", "c"])

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 9), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 9),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 9), 30],
        [[(10, 3)], [(10, 3)], (3, 9), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_probs(
        self, mock_qutrit_device_shots, shot_list, shot_vector, expected_shape, total_shots
    ):
        """Test a probability return"""
        dev = mock_qutrit_device_shots(wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x, z):
            RZ_01 = pnp.array(
                [
                    [pnp.exp(-1j * z / 2), 0.0, 0.0],
                    [0.0, pnp.exp(1j * z / 2), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            c = pnp.cos(x / 2)
            s = pnp.sin(x / 2) * 1j
            RX_01 = pnp.array([[c, -s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

            qml.QutritUnitary(RZ_01, wires=0)
            qml.QutritUnitary(RX_01, wires=1)
            return qml.probs(wires=[0, 1])

        res = circuit(0.1, 0.6)

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        # TODO: Add after differentiability of qutrit circuits is implemented
        # res = qml.jacobian(circuit, argnum=[0, 1])(0.1, 0.6)

    marginal_shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 3), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 3),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 3), 30],
        [[(10, 3)], [(10, 3)], (3, 3), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", marginal_shot_data)
    def test_marginal_probs(
        self, mock_qutrit_device_shots, shot_list, shot_vector, expected_shape, total_shots
    ):
        dev = mock_qutrit_device_shots(wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(x, z):
            RZ_01 = pnp.array(
                [
                    [pnp.exp(-1j * z / 2), 0.0, 0.0],
                    [0.0, pnp.exp(1j * z / 2), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            c = pnp.cos(x / 2)
            s = pnp.sin(x / 2) * 1j
            RX_01 = pnp.array([[c, -s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

            qml.QutritUnitary(RZ_01, wires=0)
            qml.QutritUnitary(RX_01, wires=1)
            return qml.probs(wires=0)

        res = circuit(0.1, 0.6)

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        # TODO: Uncomment after parametric operations are added for qutrits and decomposition
        # for QutritUnitary exists
        # res = qml.jacobian(circuit, argnum=[0, 1])(0.1, 0.6)

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 3, 2), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 3, 2),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 3, 2), 30],
        [[(10, 3)], [(10, 3)], (3, 3, 2), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_multiple_probs(
        self, mock_qutrit_device_shots, shot_list, shot_vector, expected_shape, total_shots
    ):
        """Test multiple probability returns"""
        dev = mock_qutrit_device_shots(wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(U):
            qml.QutritUnitary(np.eye(3), wires=0)
            qml.QutritUnitary(np.eye(3), wires=0)
            qml.QutritUnitary(U, wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        res = circuit(pnp.eye(9))

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        # TODO: Uncomment after parametric operations are added for qutrits and decomposition
        # for QutritUnitary exists
        # res = qml.jacobian(circuit, argnum=[0])(pnp.eye(9, dtype=np.complex128))


class TestUnimplemented:
    """Tests for class methods that aren't implemented

    These tests are for reaching 100% coverage of :class:`pennylane.QutritDevice`, as the
    methods/properties being tested here have been overriden from :class:`pennylane.QubitDevice`
    to avoid unexpected behaviour, but do not yet have working implementations.
    """

    def test_adjoint_jacobian(self, mock_qutrit_device):
        """Test that adjoint_jacobian is unimplemented"""
        dev = mock_qutrit_device()
        tape = qml.tape.QuantumTape()

        with pytest.raises(NotImplementedError):
            dev.adjoint_jacobian(tape)

    def test_state(self, mock_qutrit_device):
        """Test that state is unimplemented"""
        dev = mock_qutrit_device()

        with pytest.raises(NotImplementedError):
            dev.state()

    def test_density_matrix(self, mock_qutrit_device):
        """Test that density_matrix is unimplemented"""
        dev = mock_qutrit_device()

        with pytest.raises(qml.QuantumFunctionError, match="Unsupported return type"):
            dev.density_matrix(wires=0)

    def test_vn_entropy(self, mock_qutrit_device):
        """Test that vn_entropy is unimplemented"""
        dev = mock_qutrit_device()

        with pytest.raises(qml.QuantumFunctionError, match="Unsupported return type"):
            dev.vn_entropy(wires=0, log_base=3)

    def test_mutual_info(self, mock_qutrit_device):
        """Test that mutual_info is unimplemented"""
        dev = mock_qutrit_device()

        with pytest.raises(qml.QuantumFunctionError, match="Unsupported return type"):
            dev.mutual_info(0, 1, log_base=3)
