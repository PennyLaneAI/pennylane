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
Unit tests for the :mod:`pennylane` :class:`Device` class.
"""

import pytest
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.qnodes import QuantumFunctionError

mock_device_paulis = ["PauliX", "PauliY", "PauliZ"]

# pylint: disable=abstract-class-instantiated, no-self-use, redefined-outer-name, invalid-name

@pytest.fixture(scope="function")
def mock_device_with_operations(monkeypatch):
    """A mock instance of the abstract Device class with non-empty operations"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, 'operations', mock_device_paulis)
        m.setattr(Device, 'observables', mock_device_paulis)
        m.setattr(Device, 'short_name', 'MockDevice')
        yield Device()


@pytest.fixture(scope="function")
def mock_device_with_observables(monkeypatch):
    """A mock instance of the abstract Device class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, 'operations', mock_device_paulis)
        m.setattr(Device, 'observables', mock_device_paulis)
        m.setattr(Device, 'short_name', 'MockDevice')
        yield Device()


@pytest.fixture(scope="function")
def mock_device_supporting_paulis(monkeypatch):
    """A mock instance of the abstract Device class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, 'operations', mock_device_paulis)
        m.setattr(Device, 'observables', mock_device_paulis)
        m.setattr(Device, 'short_name', 'MockDevice')
        yield Device()


@pytest.fixture(scope="function")
def mock_device_supporting_paulis_and_inverse(monkeypatch):
    """A mock instance of the abstract Device class with non-empty operations
    and supporting inverses"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, 'operations', mock_device_paulis)
        m.setattr(Device, 'observables', mock_device_paulis)
        m.setattr(Device, 'short_name', 'MockDevice')
        m.setattr(Device, '_capabilities', {"inverse_operations": True})
        yield Device()

@pytest.fixture(scope="function")
def mock_device_supporting_observables_and_inverse(monkeypatch):
    """A mock instance of the abstract Device class with non-empty operations
    and supporting inverses"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, 'operations', mock_device_paulis)
        m.setattr(Device, 'observables', mock_device_paulis + ['Hermitian'])
        m.setattr(Device, 'short_name', 'MockDevice')
        m.setattr(Device, '_capabilities', {"inverse_operations": True})
        yield Device()

mock_device_capabilities = {
    "measurements": "everything",
    "noise_models": ["depolarizing", "bitflip"],
}


@pytest.fixture(scope="function")
def mock_device_with_capabilities(monkeypatch):
    """A mock instance of the abstract Device class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, '_capabilities', mock_device_capabilities)
        yield Device()


@pytest.fixture(scope="function")
def mock_device_with_paulis_and_methods(monkeypatch):
    """A mock instance of the abstract Device class with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, '_capabilities', mock_device_capabilities)
        m.setattr(Device, 'operations', mock_device_paulis)
        m.setattr(Device, 'observables', mock_device_paulis)
        m.setattr(Device, 'short_name', 'MockDevice')
        m.setattr(Device, 'expval', lambda self, x, y, z: 0)
        m.setattr(Device, 'var', lambda self, x, y, z: 0)
        m.setattr(Device, 'sample', lambda self, x, y, z: 0)
        m.setattr(Device, 'apply', lambda self, x, y, z: None)
        yield Device()


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, '_capabilities', mock_device_capabilities)
        m.setattr(Device, 'operations', ["PauliY", "RX", "Rot"])
        m.setattr(Device, 'observables', ["PauliZ"])
        m.setattr(Device, 'short_name', 'MockDevice')
        m.setattr(Device, 'expval', lambda self, x, y, z: 0)
        m.setattr(Device, 'var', lambda self, x, y, z: 0)
        m.setattr(Device, 'sample', lambda self, x, y, z: 0)
        m.setattr(Device, 'apply', lambda self, x, y, z: None)
        yield Device()


class TestDeviceSupportedLogic:
    """Test the logic associated with the supported operations and observables"""

    # pylint: disable=no-self-use, redefined-outer-name

    def test_supports_operation_argument_types(self, mock_device_with_operations):
        """Checks that device.supports_operations returns the correct result
           when passed both string and Operation class arguments"""

        assert mock_device_with_operations.supports_operation("PauliX")
        assert mock_device_with_operations.supports_operation(qml.PauliX)

        assert not mock_device_with_operations.supports_operation("S")
        assert not mock_device_with_operations.supports_operation(qml.CNOT)

    def test_supports_observable_argument_types(self, mock_device_with_observables):
        """Checks that device.supports_observable returns the correct result
           when passed both string and Operation class arguments"""

        assert mock_device_with_observables.supports_observable("PauliX")
        assert mock_device_with_observables.supports_observable(qml.PauliX)

        assert not mock_device_with_observables.supports_observable("Identity")
        assert not mock_device_with_observables.supports_observable(qml.Identity)

    def test_supports_obeservable_inverse(self, mock_device_supporting_paulis_and_inverse):

        assert mock_device_supporting_paulis_and_inverse.supports_observable("PauliX.inv")
        assert not mock_device_supporting_paulis_and_inverse.supports_observable("Identity.inv")

    def test_supports_obeservable_raise_error_hermitian_inverse(self, mock_device_supporting_observables_and_inverse):

        assert mock_device_supporting_observables_and_inverse.supports_observable("PauliX")
        assert mock_device_supporting_observables_and_inverse.supports_observable("PauliX.inv")
        assert mock_device_supporting_observables_and_inverse.supports_observable("Hermitian")

        assert not mock_device_supporting_observables_and_inverse.supports_observable("Hermitian.inv")

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
                match="The given observable must either be a pennylane.Observable class or a string.",
        ):
            mock_device.supports_observable(3)

        operation = qml.CNOT

        with pytest.raises(
                ValueError,
                match="The given observable must either be a pennylane.Observable class or a string.",
        ):
            mock_device.supports_observable(operation)


class TestInternalFunctions:
    """Test the internal functions of the abstract Device class"""

    def test_check_validity_on_valid_queue(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with valid queue and observables"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.PauliZ(0))]

        # Raises an error if queue or observables are invalid
        mock_device_supporting_paulis.check_validity(queue, observables)

    def test_check_validity_on_valid_queue_with_inverses(self,
                                                         mock_device_supporting_paulis_and_inverse):
        """Tests the function Device.check_validity with valid queue
        and the inverse of operations"""
        queue = [
            qml.PauliX(wires=0).inv(),
            qml.PauliY(wires=1).inv(),
            qml.PauliZ(wires=2).inv(),

            qml.PauliX(wires=0).inv().inv(),
            qml.PauliY(wires=1).inv().inv(),
            qml.PauliZ(wires=2).inv().inv(),
        ]

        observables = [qml.expval(qml.PauliZ(0))]

        # Raises an error if queue or observables are invalid
        mock_device_supporting_paulis_and_inverse.check_validity(queue, observables)

    def test_check_validity_with_not_supported_operation_inverse(self, mock_device_supporting_paulis_and_inverse):
        """Tests the function Device.check_validity with an valid queue
        and the inverse of not supported operations"""
        queue = [
            qml.CNOT(wires=[0, 1]).inv(),
        ]

        observables = [qml.expval(qml.PauliZ(0))]

        with pytest.raises(
                DeviceError,
                match="Gate {} not supported on device {}".format("CNOT", 'MockDevice'),
        ):
            mock_device_supporting_paulis_and_inverse.check_validity(queue, observables)

    def test_check_validity_on_tensor_support(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with tensor support capability"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.PauliZ(0) @ qml.PauliX(1))]

        # mock device does not support Tensor product
        with pytest.raises(DeviceError, match="Tensor observables not supported"):
            mock_device_supporting_paulis.check_validity(queue, observables)

    def test_check_validity_on_invalid_observable_with_tensor_support(self, monkeypatch):
        """Tests the function Device.check_validity with tensor support capability
        but with an invalid observable"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.PauliZ(0) @ qml.Hadamard(1))]

        D = Device
        with monkeypatch.context() as m:
            m.setattr(D, '__abstractmethods__', frozenset())
            m.setattr(D, 'operations', ["PauliX", "PauliY", "PauliZ"])
            m.setattr(D, 'observables', ["PauliX", "PauliY", "PauliZ"])
            m.setattr(D, 'capabilities', lambda self: {"tensor_observables": True})
            m.setattr(D, 'short_name', "Dummy")

            dev = D()

            # mock device supports Tensor products but not hadamard
            with pytest.raises(DeviceError, match="Observable Hadamard not supported"):
                dev.check_validity(queue, observables)

    def test_check_validity_on_invalid_queue(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with invalid queue and valid observables"""
        queue = [
            qml.RX(1.0, wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.PauliZ(0))]

        with pytest.raises(DeviceError, match="Gate RX not supported on device"):
            mock_device_supporting_paulis.check_validity(queue, observables)

    def test_check_validity_on_invalid_observable(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with valid queue and invalid observables"""
        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.Hadamard(0))]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            mock_device_supporting_paulis.check_validity(queue, observables)

    def test_check_validity_on_invalid_queue_of_inverses(self, mock_device_supporting_paulis_and_inverse):
        """Tests the function Device.check_validity with invalid queue and valid inverses of operations"""
        queue = [
            qml.PauliY(wires=1).inv(),
            qml.PauliZ(wires=2).inv(),
            qml.RX(1.0, wires=0).inv(),
        ]

        observables = [qml.expval(qml.PauliZ(0))]

        with pytest.raises(DeviceError, match="Gate RX not supported on device"):
            mock_device_supporting_paulis_and_inverse.check_validity(queue, observables)

    def test_supports_inverse(self, mock_device_supporting_paulis_and_inverse):
        """Tests the function Device.supports_inverse on device which supports inverses"""
        assert mock_device_supporting_paulis_and_inverse.check_validity([qml.PauliZ(0).inv()], []) is None
        assert mock_device_supporting_paulis_and_inverse.check_validity([], [qml.PauliZ(0).inv()]) is None

    def test_supports_inverse_device_does_not_support_inverses(self, mock_device_supporting_paulis):
        """Tests the function Device.supports_inverse on device which does not support inverses"""
        with pytest.raises(DeviceError, match="The inverse of gates are not supported on device {}".
                format(mock_device_supporting_paulis.short_name)):
            mock_device_supporting_paulis.check_validity([qml.PauliZ(0).inv()], [])

        with pytest.raises(DeviceError, match="The inverse of gates are not supported on device {}".
                format(mock_device_supporting_paulis.short_name)):
            mock_device_supporting_paulis.check_validity([], [qml.PauliZ(0).inv()])

    def test_args(self, mock_device):
        """Test that the device requires correct arguments"""
        with pytest.raises(qml.DeviceError, match="specified number of shots needs to be at least 1"):
            Device(mock_device, shots=0)

class TestClassmethods:
    """Test the classmethods of Device"""

    def test_capabilities(self, mock_device_with_capabilities):
        """check that device can give a dict of further capabilities"""

        assert mock_device_with_capabilities.capabilities() == mock_device_capabilities


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

    def test_op_queue_is_filled_at_pre_measure(self, mock_device_with_paulis_and_methods, monkeypatch):
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
            m.setattr(Device, 'pre_measure', lambda self: queue_at_pre_measure.extend(self.op_queue))
            mock_device_with_paulis_and_methods.execute(queue, observables)

        assert queue_at_pre_measure == queue

    def test_op_queue_is_filled_during_execution(self, mock_device_with_paulis_and_methods, monkeypatch):
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
            m.setattr(Device, 'apply', lambda self, op, wires, params: call_history.append([op, wires, params]))
            mock_device_with_paulis_and_methods.execute(queue, observables)

        assert call_history[0] == ["PauliX", [0], []]
        assert call_history[1] == ["PauliY", [1], []]
        assert call_history[2] == ["PauliZ", [2], []]

    def test_unsupported_operations_raise_error(self, mock_device_with_paulis_and_methods):
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
            mock_device_with_paulis_and_methods.execute(queue, observables)


class TestObservables:
    """Tests the logic related to observables"""

    # pylint: disable=no-self-use, redefined-outer-name

    def test_obs_queue_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
                ValueError,
                match="Cannot access the observable value queue outside of the execution context!",
        ):
            mock_device.obs_queue

    def test_obs_queue_is_filled_at_pre_measure(self, mock_device_with_paulis_and_methods, monkeypatch):
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
            m.setattr(Device, 'pre_measure', lambda self: queue_at_pre_measure.extend(self.obs_queue))
            mock_device_with_paulis_and_methods.execute(queue, observables)

        assert queue_at_pre_measure == observables

    def test_obs_queue_is_filled_during_execution(self, monkeypatch, mock_device_with_paulis_and_methods):
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
            m.setattr(Device, 'expval', lambda self, *args: expval_args.extend(args))
            m.setattr(Device, 'var', lambda self, *args: var_args.extend(args))
            m.setattr(Device, 'sample', lambda self, *args: sample_args.extend(args))
            mock_device_with_paulis_and_methods.execute([], observables)

        assert expval_args == ["PauliX", [0], []]
        assert var_args == ["PauliY", [1], []]
        assert sample_args == ["PauliZ", [2], []]

    def test_unsupported_observables_raise_error(self, mock_device_with_paulis_and_methods):
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
            mock_device_with_paulis_and_methods.execute(queue, observables)

    def test_unsupported_observable_return_type_raise_error(self, mock_device_with_paulis_and_methods):
        """Check that an error is raised if the return type of an observable is unsupported"""

        queue = [qml.PauliX(wires=0)]

        # Make a observable without specifying a return operation upon measuring
        obs = qml.PauliZ(0)
        obs.return_type = "SomeUnsupportedReturnType"
        observables = [obs]

        with pytest.raises(QuantumFunctionError, match="Unsupported return type specified for observable"):
            mock_device_with_paulis_and_methods.execute(queue, observables)


class TestParameters:
    """Test for checking device parameter mappings"""

    def test_parameters_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        with pytest.raises(
                ValueError,
                match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            mock_device.parameters

    def test_parameters_available_at_pre_measure(self, mock_device, monkeypatch):
        """Tests that the parameter mapping is available when pre_measure is called and that accessing
           Device.parameters raises no error"""

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
            m.setattr(Device, "pre_measure", lambda self: p_mapping.update(self.parameters))
            mock_device.execute(queue, observables, parameters=parameters)

        assert p_mapping == parameters


class TestDeviceInit:
    """Tests for device loader in __init__.py"""

    def test_no_device(self):
        """Test that an exception is raised for a device that doesn't exist"""

        with pytest.raises(DeviceError, match="Device does not exist"):
            qml.device("None", wires=0)

    def test_outdated_API(self, monkeypatch):
        """Test that an exception is raised if plugin that targets an old API is loaded"""

        with monkeypatch.context() as m:
            m.setattr(qml, "version", lambda: "0.0.1")
            with pytest.raises(DeviceError, match="plugin requires PennyLane versions"):
                qml.device("default.qubit", wires=0)
