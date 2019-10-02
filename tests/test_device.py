# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`Device` class.
"""
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.operation import Sample, Variance, Expectation
from pennylane.qnode import QuantumFunctionError


@pytest.fixture(scope="function")
def mock_device_with_operations():
    """A mock instance of the abstract Device class with non-empty operations"""

    with patch.multiple(
        Device,
        __abstractmethods__=set(),
        operations=PropertyMock(return_value=["PauliX", "PauliZ", "CNOT"]),
    ):
        yield Device()


@pytest.fixture(scope="function")
def mock_device_with_observables():
    """A mock instance of the abstract Device class with non-empty observables"""

    with patch.multiple(
        Device,
        __abstractmethods__=set(),
        observables=PropertyMock(return_value=["PauliX", "PauliZ"]),
    ):
        yield Device()


class TestDeviceSupportedLogic:
    """Test the logic associated with the supported operations and observables"""

    def test_supports_operation_argument_types(self, mock_device_with_operations):
        """Checks that device.supports_operations returns the correct result
           when passed both string and Operation class arguments"""

        assert mock_device_with_operations.supports_operation("PauliX")
        assert mock_device_with_operations.supports_operation(qml.PauliX)

        assert not mock_device_with_operations.supports_operation("PauliY")
        assert not mock_device_with_operations.supports_operation(qml.PauliY)

    def test_supports_observable_argument_types(self, mock_device_with_observables):
        """Checks that device.supports_observable returns the correct result
           when passed both string and Operation class arguments"""

        assert mock_device_with_observables.supports_observable("PauliX")
        assert mock_device_with_observables.supports_observable(qml.PauliX)

        assert not mock_device_with_observables.supports_observable("PauliY")
        assert not mock_device_with_observables.supports_observable(qml.PauliY)

    def test_supports_operation_exception(self, mock_device):
        """check that device.supports_operation raises proper errors
           if the argument is of the wrong type"""

        with pytest.raises(
            ValueError,
            match="The given operation must either be a pennylane.Operation class or a string.",
        ):
            mock_device.supports_operation(3)

        with pytest.raises(
            ValueError,
            match="The given operation must either be a pennylane.Operation class or a string.",
        ):
            mock_device.supports_operation(Device)

    def test_supports_observable_exception(self, mock_device):
        """check that device.supports_observable raises proper errors
           if the argument is of the wrong type"""

        with pytest.raises(
            ValueError,
            match="The given operation must either be a pennylane.Observable class or a string.",
        ):
            mock_device.supports_observable(3)

        with pytest.raises(
            ValueError,
            match="The given operation must either be a pennylane.Observable class or a string.",
        ):
            mock_device.supports_observable(qml.CNOT)


mock_device_paulis = ["PauliX", "PauliY", "PauliZ"]


@pytest.fixture(scope="function")
def mock_device_supporting_paulis():
    """A mock instance of the abstract Device class with non-empty observables"""

    with patch.multiple(
        Device,
        __abstractmethods__=set(),
        operations=PropertyMock(return_value=mock_device_paulis),
        observables=PropertyMock(return_value=mock_device_paulis),
        short_name=PropertyMock(return_value="MockDevice"),
    ):
        yield Device()


class TestInternalFunctions:
    """Test the internal functions of the abstract Device class"""

    def test_check_validity_on_valid_queue(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with valid queue and observables"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [qml.expval(qml.PauliZ(0, do_queue=False))]

        # Raises an error if queue or observables are invalid
        mock_device_supporting_paulis.check_validity(queue, observables)

    def test_check_validity_on_invalid_queue(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with invalid queue and valid observables"""
        queue = [
            qml.RX(1.0, wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [qml.expval(qml.PauliZ(0, do_queue=False))]

        with pytest.raises(DeviceError, match="Gate RX not supported on device"):
            mock_device_supporting_paulis.check_validity(queue, observables)

    def test_check_validity_on_invalid_observable(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with valid queue and invalid observables"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [qml.expval(qml.Hadamard(0, do_queue=False))]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            mock_device_supporting_paulis.check_validity(queue, observables)


mock_device_capabilities = {
    "measurements": "everything",
    "noise_models": ["depolarizing", "bitflip"],
}


@pytest.fixture(scope="function")
def mock_device_with_capabilities():
    """A mock instance of the abstract Device class with non-empty observables"""

    with patch.multiple(Device, __abstractmethods__=set(), _capabilities=mock_device_capabilities):
        yield Device()


class TestClassmethods:
    """Test the classmethods of Device"""

    def test_capabilities(self, mock_device_with_capabilities):
        """check that device can give a dict of further capabilities"""

        assert mock_device_with_capabilities.capabilities() == mock_device_capabilities


@pytest.fixture(scope="function")
def mock_device_with_paulis_and_methods():
    """A mock instance of the abstract Device class with non-empty observables"""

    with patch.multiple(
        Device,
        __abstractmethods__=set(),
        _capabilities=mock_device_capabilities,
        expval=MagicMock(return_value=0),
        var=MagicMock(return_value=0),
        sample=MagicMock(return_value=[0]),
        apply=MagicMock(),
        operations=PropertyMock(return_value=mock_device_paulis),
        observables=PropertyMock(return_value=mock_device_paulis),
        short_name=PropertyMock(return_value="MockDevice"),
    ):
        yield Device()


class TestOperations:
    """Tests the logic related to operations"""

    def test_shots_setter(self, mock_device):
        """Tests that the property setter of shots changes the number of shots."""
        
        assert mock_device._shots == 1000

        mock_device.shots = 10

        assert mock_device._shots == 10

    @pytest.mark.parametrize("shots", [-10, 0])
    def test_shots_setter_error(self, mock_device, shots):
        """Tests that the property setter of shots raises an error if the requested number of shots
        is erroneous."""

        with pytest.raises(qml.DeviceError, match="The specified number of shots needs to be at least 1"):
            mock_device.shots = shots

    def test_op_queue_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError, match="Cannot access the operation queue outside of the execution context!"
        ):
            mock_device.op_queue

    def test_op_queue_is_filled_at_pre_measure(self, mock_device_with_paulis_and_methods):
        """Tests that the op_queue is correctly filled when pre_measure is called and that accessing
           op_queue raises no error"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [
            qml.expval(qml.PauliZ(0, do_queue=False)),
            qml.var(qml.PauliZ(1, do_queue=False)),
            qml.sample(qml.PauliZ(2, do_queue=False)),
        ]

        queue_at_pre_measure = []

        with patch.object(
            Device, "pre_measure", lambda self: queue_at_pre_measure.extend(self.op_queue)
        ):
            mock_device_with_paulis_and_methods.execute(queue, observables)

        assert queue_at_pre_measure == queue

    def test_op_queue_is_filled_during_execution(self, mock_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [
            qml.expval(qml.PauliZ(0, do_queue=False)),
            qml.var(qml.PauliZ(1, do_queue=False)),
            qml.sample(qml.PauliZ(2, do_queue=False)),
        ]

        call_history = []
        mock_device_with_paulis_and_methods.apply = Mock(
            wraps=lambda op, wires, params: call_history.append([op, wires, params])
        )

        mock_device_with_paulis_and_methods.execute(queue, observables)

        assert call_history[0] == ["PauliX", [0], []]
        assert call_history[1] == ["PauliY", [1], []]
        assert call_history[2] == ["PauliZ", [2], []]

    def test_unsupported_operations_raise_error(self, mock_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.Hadamard(wires=2, do_queue=False),
        ]

        observables = [
            qml.expval(qml.PauliZ(0, do_queue=False)),
            qml.var(qml.PauliZ(1, do_queue=False)),
            qml.sample(qml.PauliZ(2, do_queue=False)),
        ]

        with pytest.raises(DeviceError, match="Gate Hadamard not supported on device"):
            mock_device_with_paulis_and_methods.execute(queue, observables)


class TestObservables:
    """Tests the logic related to observables"""

    def test_obs_queue_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the observable value queue outside of the execution context!",
        ):
            mock_device.obs_queue

    def test_obs_queue_is_filled_at_pre_measure(self, mock_device_with_paulis_and_methods):
        """Tests that the op_queue is correctly filled when pre_measure is called and that accessing
           op_queue raises no error"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [
            qml.expval(qml.PauliZ(0, do_queue=False)),
            qml.var(qml.PauliZ(1, do_queue=False)),
            qml.sample(qml.PauliZ(2, do_queue=False)),
        ]

        queue_at_pre_measure = []

        with patch.object(
            Device, "pre_measure", lambda self: queue_at_pre_measure.extend(self.obs_queue)
        ):
            mock_device_with_paulis_and_methods.execute(queue, observables)

        assert queue_at_pre_measure == observables

    def test_obs_queue_is_filled_during_execution(self, mock_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [
            qml.expval(qml.PauliZ(0, do_queue=False)),
            qml.var(qml.PauliZ(1, do_queue=False)),
            qml.sample(qml.PauliZ(2, do_queue=False)),
        ]

        # The methods expval, var and sample are MagicMock'ed in the fixture

        mock_device_with_paulis_and_methods.execute(queue, observables)

        mock_device_with_paulis_and_methods.expval.assert_called_with("PauliZ", [0], [])
        mock_device_with_paulis_and_methods.var.assert_called_with("PauliZ", [1], [])
        mock_device_with_paulis_and_methods.sample.assert_called_with("PauliZ", [2], [])

    def test_unsupported_observables_raise_error(self, mock_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        queue = [
            qml.PauliX(wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [
            qml.expval(qml.Hadamard(0, do_queue=False)),
            qml.var(qml.PauliZ(1, do_queue=False)),
            qml.sample(qml.PauliZ(2, do_queue=False)),
        ]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            mock_device_with_paulis_and_methods.execute(queue, observables)

    def test_unsupported_observable_return_type_raise_error(self, mock_device_with_paulis_and_methods):
        """Check that an error is raised if the return type of an observable is unsupported"""

        queue = [qml.PauliX(wires=0, do_queue=False)]

        # Make a observable without specifying a return operation upon measuring
        obs = qml.PauliZ(0, do_queue=False)
        obs.return_type = "SomeUnsupportedReturnType"
        observables = [obs]

        with pytest.raises(QuantumFunctionError, match="Unsupported return type specified for observable"):
            mock_device_with_paulis_and_methods.execute(queue, observables)

    def test_supported_observable_return_types(self, mock_device_with_paulis_and_methods):
        """Check that no error is raised if the return types of observables are supported"""

        queue = [qml.PauliX(wires=0, do_queue=False)]

        # Make observables with specifying supported return types

        obs1 = qml.PauliZ(0, do_queue=False)
        obs2 = qml.PauliZ(1, do_queue=False)
        obs3 = qml.PauliZ(2, do_queue=False)

        obs1.return_type = Expectation
        obs2.return_type = Variance
        obs3.return_type = Sample

        observables = [obs1,
                       obs2,
                       obs3,
        ]

        # The methods expval, var and sample are MagicMock'ed in the fixture
        mock_device_with_paulis_and_methods.execute(queue, observables)

        mock_device_with_paulis_and_methods.expval.assert_called_with("PauliZ", [0], [])
        mock_device_with_paulis_and_methods.var.assert_called_with("PauliZ", [1], [])
        mock_device_with_paulis_and_methods.sample.assert_called_with("PauliZ", [2], [])


class TestParameters:
    """Test for checking device parameter mappings"""

    @pytest.fixture(scope="function")
    def mock_device(self):
        with patch.multiple(
            Device,
            __abstractmethods__=set(),
            operations=PropertyMock(return_value=["PauliY", "RX", "Rot"]),
            observables=PropertyMock(return_value=["PauliZ"]),
            short_name=PropertyMock(return_value="MockDevice"),
            expval=MagicMock(return_value=0),
            var=MagicMock(return_value=0),
            sample=MagicMock(return_value=[0]),
            apply=MagicMock()
        ):
            yield Device()

    def test_parameters_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            mock_device.parameters

    def test_parameters_available_at_pre_measure(self, mock_device):
        """Tests that the parameter mapping is available when pre_measure is called and that accessing
           Device.parameters raises no error"""


        p0 = 0.54
        p1 = -0.32

        queue = [
            qml.RX(p0, wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.Rot(0.432, 0.123, p1, wires=2, do_queue=False),
        ]

        parameters = {0: (0, 0), 1: (2, 3)}

        observables = [
            qml.expval(qml.PauliZ(0, do_queue=False)),
            qml.var(qml.PauliZ(1, do_queue=False)),
            qml.sample(qml.PauliZ(2, do_queue=False)),
        ]

        p_mapping = {}

        with patch.object(Device, "pre_measure", lambda self: p_mapping.update(self.parameters)):
            mock_device.execute(queue, observables, parameters=parameters)

        assert p_mapping == parameters


class TestDeviceInit:
    """Tests for device loader in __init__.py"""

    def test_no_device(self):
        """Test that an exception is raised for a device that doesn't exist"""

        with pytest.raises(DeviceError, match="Device does not exist"):
            qml.device("None", wires=0)

    def test_outdated_API(self):
        """Test that an exception is raised if plugin that targets an old API is loaded"""

        with patch.object(qml, "version", return_value="0.0.1"):
            with pytest.raises(DeviceError, match="plugin requires PennyLane versions"):
                qml.device("default.qubit", wires=0)
