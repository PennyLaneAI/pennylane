import pytest
import numpy as np
from random import random
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import QutritDevice, DeviceError, QuantumFunctionError
from pennylane.measurements import Sample, Variance, Expectation, Probability, State
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
        m.setattr(QutritDevice, "apply", lambda self, x: x)

        def get_qutrit_device(wires=1):
            return QutritDevice(wires=wires)

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


class TestOperations:
    """Tests the logic related to operations"""

    def test_op_queue_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError, match="Cannot access the operation queue outside of the execution context!"
        ):
            dev = mock_qutrit_device()
            dev.op_queue

    def test_op_queue_is_filled_during_execution(
        self, mock_qutrit_device, monkeypatch
    ):
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
            queue = [qml.QutritUnitary(U, wires=0), qml.Hadamard(wires=1), qml.QutritUnitary(U, wires=2)]
            observables = [qml.expval(qml.Identity(0)), qml.var(qml.Identity(1))]

        with pytest.raises(DeviceError, match="Gate Hadamard not supported on device"):
            dev = mock_qutrit_device()
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
            dev = mock_qutrit_device()
            dev.execute(tape, hash=tape.graph.hash)

        len(call_history.items()) == 1
        call_history["hash"] = tape.graph.hash


class TestObservables:
    """Tests the logic related to observables"""

    U = unitary_group.rvs(3, random_state=10)

    def test_obs_queue_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the observable value queue outside of the execution context!",
        ):
            dev = mock_qutrit_device()
            dev.obs_queue

    def test_unsupported_observables_raise_error(self, mock_qutrit_device):
        """Tests that the operations are properly applied and queued"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.QutritUnitary(U, wires=0)]
            observables = [qml.expval(qml.Hadamard(0))]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            dev = mock_qutrit_device()
            dev.execute(tape)

    def test_unsupported_observable_return_type_raise_error(
        self, mock_qutrit_device, monkeypatch
    ):
        """Check that an error is raised if the return type of an observable is unsupported"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            qml.QutritUnitary(U, wires=0)
            qml.measurements.MeasurementProcess(
                return_type="SomeUnsupportedReturnType", obs=qml.Identity(0)
            )

        with monkeypatch.context() as m:
            m.setattr(QutritDevice, "apply", lambda self, x, **kwargs: None)
            with pytest.raises(
                qml.QuantumFunctionError, match="Unsupported return type specified for observable"
            ):
                dev = mock_qutrit_device()
                dev.execute(tape)


class TestParameters:
    """Test for checking device parameter mappings"""

    def test_parameters_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            dev = mock_qutrit_device()
            dev.parameters


class TestExtractStatistics:
    @pytest.mark.parametrize("returntype", [Expectation, Variance, Sample, Probability, State])
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

    def test_results_no_state(self, mock_qutrit_device_extract_stats, monkeypatch):
        """Tests that the statistics method raises an AttributeError when a State return type is
        requested when QutritDevice does not have a state attribute"""
        with monkeypatch.context():
            dev = mock_qutrit_device_extract_stats()
            delattr(dev.__class__, "state")
            with pytest.raises(
                qml.QuantumFunctionError, match="The state is not available in the current"
            ):
                dev.statistics([state()])

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
    def test_error_return_type_none(self, mock_qutrit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method raises an error if the return type is not well-defined and is not None"""

        assert returntype not in [Expectation, Variance, Sample, Probability, State, None]

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with pytest.raises(qml.QuantumFunctionError, match="Unsupported return type"):
            dev = mock_qutrit_device_extract_stats()
            dev.statistics([obs])


class TestGenerateSamples:
    """Test the generate_samples method"""

    def test_auxiliary_methods_called_correctly(self, mock_qutrit_device, monkeypatch):
        """Tests that the generate_samples method calls on its auxiliary methods correctly"""

        dev = mock_qutrit_device()
        number_of_states = 3**dev.num_wires

        with monkeypatch.context() as m:
            # Mock the auxiliary methods such that they return the expected values
            m.setattr(QutritDevice, "sample_basis_states", lambda self, wires, b: wires)
            m.setattr(QutritDevice, "states_to_ternary", lambda a, b: (a, b))
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

    def test_raises_deprecation_warning(self, mock_qutrit_device, monkeypatch):
        """Test that sampling basis states on a device with shots=None produces a warning."""

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
    pass


# TODO: Add tests for expval, var
class TestExpval:
    pass


class TestVar:
    pass


class TestSample:
    pass


class TestEstimateProb:
    pass


class TestMarginalProb:
    pass


class TestActiveWires:
    pass


class TestCapabilities:
    pass


class TestExecution:
    pass


class TestBatchExecution:
    pass


class TestShotList:
    pass
