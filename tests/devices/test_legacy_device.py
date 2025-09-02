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
from collections import OrderedDict
from importlib import metadata, reload
from sys import version_info

import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane.devices import LegacyDevice as Device
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.wires import Wires

mock_device_paulis = ["PauliX", "PauliY", "PauliZ"]
mock_device_paulis_and_hamiltonian = ["Hamiltonian", "PauliX", "PauliY", "PauliZ"]

# pylint: disable=abstract-class-instantiated, no-self-use, redefined-outer-name, invalid-name, missing-function-docstring


@pytest.fixture(scope="function")
def mock_device_with_identity(monkeypatch):
    """A function to create a mock device with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "operations", mock_device_paulis + ["Identity"])
        m.setattr(Device, "observables", mock_device_paulis + ["Identity"])
        m.setattr(Device, "short_name", "MockDevice")

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


@pytest.fixture(scope="function")
def mock_device_supporting_paulis(monkeypatch):
    """A function to create a mock device with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "operations", mock_device_paulis)
        m.setattr(Device, "observables", mock_device_paulis)
        m.setattr(Device, "short_name", "MockDevice")

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


@pytest.fixture(scope="function")
def mock_device_supporting_paulis_and_inverse(monkeypatch):
    """A function to create a mock device with non-empty operations
    and supporting inverses"""
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "operations", mock_device_paulis)
        m.setattr(Device, "observables", mock_device_paulis)
        m.setattr(Device, "short_name", "MockDevice")
        m.setattr(Device, "_capabilities", {"supports_inverse_operations": True})

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


@pytest.fixture(scope="function")
def mock_device_supporting_observables_and_inverse(monkeypatch):
    """A function to create a mock device with non-empty operations
    and supporting inverses"""
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "operations", mock_device_paulis)
        m.setattr(Device, "observables", mock_device_paulis + ["Hermitian"])
        m.setattr(Device, "short_name", "MockDevice")
        m.setattr(Device, "_capabilities", {"supports_inverse_operations": True})

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


mock_device_capabilities = {
    "measurements": "everything",
    "noise_models": ["depolarizing", "bitflip"],
}


@pytest.fixture(scope="function")
def mock_device_with_capabilities(monkeypatch):
    """A function to create a mock device with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "_capabilities", mock_device_capabilities)

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


@pytest.fixture(scope="function")
def mock_device_with_paulis_and_methods(monkeypatch):
    """A function to create a mock device with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "_capabilities", mock_device_capabilities)
        m.setattr(Device, "operations", mock_device_paulis)
        m.setattr(Device, "observables", mock_device_paulis)
        m.setattr(Device, "short_name", "MockDevice")
        m.setattr(Device, "expval", lambda self, x, y, z: 0)
        m.setattr(Device, "var", lambda self, x, y, z: 0)
        m.setattr(Device, "sample", lambda self, x, y, z: 0)
        m.setattr(Device, "apply", lambda self, x, y, z: None)

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


@pytest.fixture(scope="function")
def mock_device_with_paulis_hamiltonian_and_methods(monkeypatch):
    """A function to create a mock device with non-empty observables"""
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "_capabilities", mock_device_capabilities)
        m.setattr(Device, "operations", mock_device_paulis)
        m.setattr(Device, "observables", mock_device_paulis_and_hamiltonian)
        m.setattr(Device, "short_name", "MockDevice")
        m.setattr(Device, "expval", lambda self, x, y, z: 0)
        m.setattr(Device, "var", lambda self, x, y, z: 0)
        m.setattr(Device, "sample", lambda self, x, y, z: 0)
        m.setattr(Device, "apply", lambda self, x, y, z: None)

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "_capabilities", mock_device_capabilities)
        m.setattr(Device, "operations", ["PauliY", "RX", "Rot"])
        m.setattr(Device, "observables", ["PauliZ"])
        m.setattr(Device, "short_name", "MockDevice")
        m.setattr(Device, "expval", lambda self, x, y, z: 0)
        m.setattr(Device, "var", lambda self, x, y, z: 0)
        m.setattr(Device, "sample", lambda self, x, y, z: 0)
        m.setattr(Device, "apply", lambda self, x, y, z: None)

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


@pytest.fixture(scope="function")
def mock_device_supporting_prod(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(Device, "__abstractmethods__", frozenset())
        m.setattr(Device, "_capabilities", mock_device_capabilities)
        m.setattr(Device, "operations", ["PauliX", "PauliZ"])
        m.setattr(Device, "observables", ["PauliX", "PauliZ", "Prod"])
        m.setattr(Device, "short_name", "MockDevice")

        def get_device(wires=1):
            return Device(wires=wires)

        yield get_device


# pylint: disable=pointless-statement
def test_invalid_attribute_in_devices_raises_error():
    with pytest.raises(AttributeError, match="'pennylane.devices' has no attribute 'blabla'"):
        qml.devices.blabla


def test_gradients_record():
    """Test that execute_and_gradients and gradient both track the number of gradients requested."""

    dev = DefaultQubitLegacy(wires=1)

    tape = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])

    with dev.tracker:
        dev.execute_and_gradients((tape, tape), method="adjoint_jacobian", use_device_state=True)

    assert dev.tracker.totals["execute_and_derivative_batches"] == 1
    assert dev.tracker.totals["derivatives"] == 2

    with dev.tracker:
        dev.gradients((tape, tape), method="adjoint_jacobian", use_device_state=True)

    assert dev.tracker.totals["derivatives"] == 2


class TestDeviceSupportedLogic:
    """Test the logic associated with the supported operations and observables"""

    # pylint: disable=no-self-use, redefined-outer-name

    def test_supports_operation_argument_types(self, mock_device_supporting_paulis):
        """Checks that device.supports_operations returns the correct result
        when passed both string and Operation class arguments"""

        dev = mock_device_supporting_paulis()

        assert dev.supports_operation("PauliX")
        assert dev.supports_operation(qml.PauliX)

        assert not dev.supports_operation("S")
        assert not dev.supports_operation(qml.CNOT)

    def test_supports_observable_argument_types(self, mock_device_supporting_paulis):
        """Checks that device.supports_observable returns the correct result
        when passed both string and Operation class arguments"""
        dev = mock_device_supporting_paulis()

        assert dev.supports_observable("PauliX")
        assert dev.supports_observable(qml.PauliX)

        assert not dev.supports_observable("Identity")
        assert not dev.supports_observable(qml.Identity)

    def test_supports_operation_exception(self, mock_device):
        """check that device.supports_operation raises proper errors
        if the argument is of the wrong type"""
        dev = mock_device()

        with pytest.raises(
            ValueError,
            match="The given operation must either be a pennylane.Operation class or a string.",
        ):
            dev.supports_operation(3)

        with pytest.raises(
            ValueError,
            match="The given operation must either be a pennylane.Operation class or a string.",
        ):
            dev.supports_operation(Device)

    def test_supports_observable_exception(self, mock_device):
        """check that device.supports_observable raises proper errors
        if the argument is of the wrong type"""
        dev = mock_device()

        with pytest.raises(
            ValueError,
            match="The given observable must either be a pennylane.operation.Operator class or a string.",
        ):
            dev.supports_observable(3)

    @pytest.mark.parametrize("supported_multi_term_obs", ["Hamiltonian", "LinearCombination"])
    @pytest.mark.parametrize("obs_type", [qml.ops.LinearCombination, qml.Hamiltonian])
    def test_all_multi_term_obs_supported_linear_combination(
        self, mock_device, supported_multi_term_obs, obs_type
    ):
        """Test that LinearCombination is supported when the device supports either
        LinearCombination or Hamiltonian."""
        dev = mock_device()
        dev.observables = dev.observables + [supported_multi_term_obs]

        obs = obs_type([1.0, 2.0], [qml.Z(0), qml.Z(0)])
        circuit = qml.tape.QuantumScript([], [qml.expval(obs)])
        assert dev._all_multi_term_obs_supported(circuit)  # pylint: disable=protected-access


class TestInternalFunctions:  # pylint:disable=too-many-public-methods
    """Test the internal functions of the abstract Device class"""

    # pylint: disable=unnecessary-dunder-call
    def test_repr(self, mock_device_supporting_paulis):
        """Tests the __repr__ function"""
        dev = mock_device_supporting_paulis()
        assert "<Device device (wires=1, shots=1000) at " in dev.__repr__()

    def test_str(self, mock_device_supporting_paulis):
        """Tests the __str__ function"""
        dev = mock_device_supporting_paulis()
        string = str(dev)
        assert "Short name: MockDevice" in string
        assert "Package: pennylane" in string
        assert "Plugin version: None" in string
        assert "Author: None" in string
        assert "Wires: 1" in string
        assert "Shots: 1000" in string

    def test_check_validity_on_valid_queue(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with valid queue and observables"""
        dev = mock_device_supporting_paulis()

        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.PauliZ(0))]

        # Raises an error if queue or observables are invalid
        dev.check_validity(queue, observables)

    def test_check_validity_containing_prod(self, mock_device_supporting_prod):
        """Tests that the function Device.check_validity works with Prod"""

        dev = mock_device_supporting_prod()

        queue = [
            qml.PauliX(wires=0),
            qml.PauliZ(wires=1),
        ]

        observables = [
            qml.expval(qml.PauliX(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliZ(0) @ (qml.PauliX(1) @ qml.PauliZ(2))),
        ]

        dev.check_validity(queue, observables)

    def test_prod_containing_unsupported_nested_observables(self, mock_device_supporting_prod):
        """Tests that the observables nested within Prod are checked for validity"""

        dev = mock_device_supporting_prod()

        queue = [
            qml.PauliX(wires=0),
            qml.PauliZ(wires=1),
        ]

        unsupported_nested_observables = [
            qml.expval(qml.PauliZ(0) @ (qml.PauliX(1) @ qml.PauliY(2)))
        ]

        with pytest.raises(DeviceError, match="Observable PauliY not supported"):
            dev.check_validity(queue, unsupported_nested_observables)

    def test_check_validity_on_prod_support(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with prod support capability"""
        dev = mock_device_supporting_paulis()

        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.PauliZ(0) @ qml.PauliX(1))]

        # mock device does not support Tensor product
        with pytest.raises(DeviceError, match="Observable Prod not supported"):
            dev.check_validity(queue, observables)

    def test_check_validity_on_invalid_queue(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with invalid queue and valid observables"""
        dev = mock_device_supporting_paulis()

        queue = [
            qml.RX(1.0, wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.PauliZ(0))]

        with pytest.raises(DeviceError, match="Gate RX not supported on device"):
            dev.check_validity(queue, observables)

    def test_check_validity_on_invalid_observable(self, mock_device_supporting_paulis):
        """Tests the function Device.check_validity with valid queue and invalid observables"""
        dev = mock_device_supporting_paulis()

        queue = [
            qml.PauliX(wires=0),
            qml.PauliY(wires=1),
            qml.PauliZ(wires=2),
        ]

        observables = [qml.expval(qml.Hadamard(0))]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            dev.check_validity(queue, observables)

    def test_check_validity_on_projector_as_operation(self, mock_device_supporting_paulis):
        """Test that an error is raised if the operation queue contains qml.Projector"""
        dev = mock_device_supporting_paulis(wires=1)

        queue = [qml.PauliX(0), qml.Projector([0], wires=0), qml.PauliZ(0)]
        observables = []

        with pytest.raises(ValueError, match="Postselection is not supported"):
            dev.check_validity(queue, observables)

    def test_check_validity_on_non_observable_measurement(self, mock_device_with_identity, recwarn):
        """Test that using non-observable measurements like state() works."""
        dev = mock_device_with_identity(wires=1)
        queue = []
        observables = [qml.state()]

        dev.check_validity(queue, observables)
        assert len(recwarn) == 0

    def test_args(self, mock_device):
        """Test that the device requires correct arguments"""
        with pytest.raises(DeviceError, match="specified number of shots needs to be at least 1"):
            Device(mock_device, shots=0)

    @pytest.mark.parametrize(
        "wires, expected",
        [(["a1", "q", -1, 3], Wires(["a1", "q", -1, 3])), (3, Wires([0, 1, 2])), ([3], Wires([3]))],
    )
    def test_wires_property(self, mock_device, wires, expected):
        """Tests that the wires attribute is set correctly."""
        dev = mock_device(wires=wires)
        assert dev.wires == expected

    def test_wire_map_property(self, mock_device):
        """Tests that the wire_map is constructed correctly."""
        dev = mock_device(wires=["a1", "q", -1, 3])
        expected = OrderedDict([("a1", 0), ("q", 1), (-1, 2), (3, 3)])
        assert dev.wire_map == expected

    def test_execution_property(self, mock_device):
        """Tests that the number of executions is initialised correctly"""
        dev = mock_device()
        expected = 0
        assert dev.num_executions == expected

    def test_device_executions(self):
        """Test the number of times a device is executed over a QNode's
        lifetime is tracked by `num_executions`"""

        # test default Gaussian device
        dev_gauss = qml.device("default.gaussian", wires=1)

        def circuit_gauss(mag_alpha, phase_alpha, phi):
            qml.Displacement(mag_alpha, phase_alpha, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.expval(qml.NumberOperator(0))

        node_gauss = qml.QNode(circuit_gauss, dev_gauss)
        num_evals_gauss = 12

        for _ in range(num_evals_gauss):
            node_gauss(0.015, 0.02, 0.005)
        assert dev_gauss.num_executions == num_evals_gauss

    @pytest.mark.parametrize(
        "depth, expanded_ops",
        [
            (0, [qml.PauliX(0), qml.BasisEmbedding([1, 0], wires=[1, 2])]),
            (1, [qml.PauliX(wires=0), qml.PauliX(wires=1)]),
        ],
    )
    def test_device_default_expand_ops(
        self, depth, expanded_ops, mock_device_with_paulis_hamiltonian_and_methods
    ):
        """Test that the default expand method can selectively expand operations
        without expanding measurements."""

        ops = [qml.PauliX(0), qml.BasisEmbedding([1, 0], wires=[1, 2])]
        measurements = [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.Hamiltonian([2], [qml.PauliX(0) @ qml.PauliY(1)])),
        ]
        circuit = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        dev = mock_device_with_paulis_hamiltonian_and_methods(wires=3)
        expanded_tape = dev.default_expand_fn(circuit, max_expansion=depth)

        for op, expected_op in zip(
            expanded_tape.operations[expanded_tape.num_preps :],
            expanded_ops,
        ):
            qml.assert_equal(op, expected_op)

        for mp, expected_mp in zip(expanded_tape.measurements, measurements):
            qml.assert_equal(mp, expected_mp)

    wires_to_try = [
        (1, Wires([0])),
        (4, Wires([1, 3])),
        (["a", 2], Wires([2])),
        (["a", 2], Wires([2, "a"])),
    ]

    @pytest.mark.parametrize("dev_wires, wires_to_map", wires_to_try)
    def test_map_wires_caches(self, dev_wires, wires_to_map, mock_device):
        """Test that multiple calls to map_wires will use caching."""
        dev = mock_device(dev_wires)
        original_hits = dev.map_wires.cache_info().hits
        original_misses = dev.map_wires.cache_info().misses

        # The first call is computed: it's a miss as it didn't come from the cache
        dev.map_wires(wires_to_map)

        # The number of misses increased
        assert dev.map_wires.cache_info().misses > original_misses

        # The second call comes from the cache: it's a hit
        dev.map_wires(wires_to_map)

        # The number of hits increased
        assert dev.map_wires.cache_info().hits > original_hits

    def test_mcm_unsupported_error(self, mock_device_with_paulis_and_methods):
        """Test that an error is raised if mid-circuit measurements are not
        supported natively"""
        dev = mock_device_with_paulis_and_methods(wires=2)

        # mid-circuit measurements are part of the queue (for now)
        with qml.queuing.AnnotatedQueue() as q:
            qml.measure(1)
            qml.PauliZ(0)

        tape = qml.tape.QuantumScript.from_queue(q)
        # Raises an error for device that doesn't support mid-circuit measurements natively
        with pytest.raises(DeviceError, match="Mid-circuit measurements are not natively"):
            dev.check_validity(tape.operations, tape.observables)

    @pytest.mark.parametrize(
        "wires, subset, expected_subset",
        [
            (Wires(["a", "b", "c"]), Wires(["c", "b"]), Wires(["b", "c"])),
            (Wires([0, 1, 2]), Wires([1, 0, 2]), Wires([0, 1, 2])),
            (Wires([3, "beta", "a"]), Wires(["a", "beta", 3]), Wires([3, "beta", "a"])),
            (Wires([0]), Wires([0]), Wires([0])),
        ],
    )
    def test_order_wires(self, wires, subset, expected_subset, mock_device):
        dev = mock_device(wires=wires)
        ordered_subset = dev.order_wires(subset_wires=subset)
        assert ordered_subset == expected_subset

    @pytest.mark.parametrize(
        "wires, subset",
        [
            (Wires(["a", "b", "c"]), Wires(["c", "d"])),
            (Wires([0, 1, 2]), Wires([3, 4, 5])),
            (Wires([3, "beta", "a"]), Wires(["alpha", "beta", "gamma"])),
            (Wires([0]), Wires([2])),
        ],
    )
    def test_order_wires_raises_value_error(self, wires, subset, mock_device):
        dev = mock_device(wires=wires)
        with pytest.raises(ValueError, match="Could not find some or all subset wires"):
            _ = dev.order_wires(subset_wires=subset)

    def test_default_expand_fn_with_invalid_op(self, mock_device_supporting_paulis, recwarn):
        """Test that default_expand_fn works with an invalid op and some measurement."""
        invalid_tape = qml.tape.QuantumScript([qml.S(0)], [qml.expval(qml.PauliZ(0))])
        expected_tape = qml.tape.QuantumScript([qml.RZ(np.pi / 2, 0)], [qml.expval(qml.PauliZ(0))])
        dev = mock_device_supporting_paulis(wires=1)
        expanded_tape = dev.expand_fn(invalid_tape, max_expansion=3)
        qml.assert_equal(expanded_tape, expected_tape)
        assert len(recwarn) == 0

    def test_stopping_condition_passes_with_non_obs_mp(self, mock_device_with_identity, recwarn):
        """Test that Device.stopping_condition passes with non-observable measurements"""
        dev = mock_device_with_identity(wires=1)
        assert dev.stopping_condition(qml.state())
        assert len(recwarn) == 0


# pylint: disable=too-few-public-methods
class TestClassmethods:
    """Test the classmethods of Device"""

    def test_capabilities(self, mock_device_with_capabilities):
        """check that device can give a dict of further capabilities"""
        dev = mock_device_with_capabilities()

        assert dev.capabilities() == mock_device_capabilities


class TestOperations:
    """Tests the logic related to operations"""

    # pylint: disable=protected-access
    def test_shots_setter(self, mock_device):
        """Tests that the property setter of shots changes the number of shots."""
        dev = mock_device()

        assert dev._shots == 1000

        dev.shots = 10

        assert dev._shots == 10

    @pytest.mark.parametrize("shots", [-10, 0])
    def test_shots_setter_error(self, mock_device, shots):
        """Tests that the property setter of shots raises an error if the requested number of shots
        is erroneous."""
        dev = mock_device()

        with pytest.raises(
            DeviceError,
            match="The specified number of shots needs to be at least 1",
        ):
            dev.shots = shots

    # pylint: disable=pointless-statement
    def test_op_queue_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""
        dev = mock_device()

        with pytest.raises(
            ValueError, match="Cannot access the operation queue outside of the execution context!"
        ):
            dev.op_queue

    def test_op_queue_is_filled_at_pre_measure(
        self, mock_device_with_paulis_and_methods, monkeypatch
    ):
        """Tests that the op_queue is correctly filled when pre_measure is called and that accessing
        op_queue raises no error"""
        dev = mock_device_with_paulis_and_methods(wires=3)

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
            m.setattr(
                Device, "pre_measure", lambda self: queue_at_pre_measure.extend(self.op_queue)
            )
            dev.execute(queue, observables)

        assert queue_at_pre_measure == queue

    def test_op_queue_is_filled_during_execution(
        self, mock_device_with_paulis_and_methods, monkeypatch
    ):
        """Tests that the operations are properly applied and queued"""
        dev = mock_device_with_paulis_and_methods(wires=3)

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
            m.setattr(
                Device,
                "apply",
                lambda self, op, wires, params: call_history.append([op, wires, params]),
            )
            dev.execute(queue, observables)

        assert call_history[0] == ["PauliX", Wires([0]), []]
        assert call_history[1] == ["PauliY", Wires([1]), []]
        assert call_history[2] == ["PauliZ", Wires([2]), []]

    def test_unsupported_operations_raise_error(self, mock_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        dev = mock_device_with_paulis_and_methods()

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
            dev.execute(queue, observables)

    def test_execute_obs_probs(self, mock_device_supporting_paulis):
        """Tests that the execute function raises an error if probabilities are
        not supported by the device"""
        dev = mock_device_supporting_paulis()
        obs = qml.probs(op=qml.PauliZ(0))
        with pytest.raises(NotImplementedError):
            dev.execute([], [obs])

    def test_var(self, mock_device_supporting_paulis):
        """Tests that the variance method are not implemented by the device by
        default"""
        dev = mock_device_supporting_paulis()
        with pytest.raises(NotImplementedError):
            dev.var(qml.PauliZ, 0, [])

    def test_sample(self, mock_device_supporting_paulis):
        """Tests that the sample method are not implemented by the device by
        default"""
        dev = mock_device_supporting_paulis()
        with pytest.raises(NotImplementedError):
            dev.sample(qml.PauliZ, 0, [])

    @pytest.mark.parametrize("wires", [None, []])
    def test_probability(self, mock_device_supporting_paulis, wires):
        """Tests that the probability method are not implemented by the device
        by default"""
        dev = mock_device_supporting_paulis()
        with pytest.raises(NotImplementedError):
            dev.probability(wires=wires)


class TestObservables:
    """Tests the logic related to observables"""

    # pylint: disable=no-self-use, redefined-outer-name, pointless-statement
    def test_obs_queue_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""
        dev = mock_device()

        with pytest.raises(
            ValueError,
            match="Cannot access the observable value queue outside of the execution context!",
        ):
            dev.obs_queue

    def test_obs_queue_is_filled_at_pre_measure(
        self, mock_device_with_paulis_and_methods, monkeypatch
    ):
        """Tests that the op_queue is correctly filled when pre_measure is called and that accessing
        op_queue raises no error"""
        dev = mock_device_with_paulis_and_methods(wires=3)

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
            m.setattr(
                Device, "pre_measure", lambda self: queue_at_pre_measure.extend(self.obs_queue)
            )
            dev.execute(queue, observables)

        assert queue_at_pre_measure == observables

    def test_obs_queue_is_filled_during_execution(
        self, monkeypatch, mock_device_with_paulis_and_methods
    ):
        """Tests that the operations are properly applied and queued"""
        dev = mock_device_with_paulis_and_methods(wires=3)

        observables = [qml.expval(qml.PauliX(0)), qml.var(qml.PauliY(1)), qml.sample(qml.PauliZ(2))]

        # capture the arguments passed to dev methods
        expval_args = []
        var_args = []
        sample_args = []
        with monkeypatch.context() as m:
            m.setattr(Device, "expval", lambda self, *args: expval_args.extend(args))
            m.setattr(Device, "var", lambda self, *args: var_args.extend(args))
            m.setattr(Device, "sample", lambda self, *args: sample_args.extend(args))
            dev.execute([], observables)

        assert expval_args == ["PauliX", Wires([0]), []]
        assert var_args == ["PauliY", Wires([1]), []]
        assert sample_args == ["PauliZ", Wires([2]), []]

    def test_unsupported_observables_raise_error(self, mock_device_with_paulis_and_methods):
        """Tests that the operations are properly applied and queued"""
        dev = mock_device_with_paulis_and_methods()

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
            dev.execute(queue, observables)

    def test_unsupported_observable_return_type_raise_error(
        self, mock_device_with_paulis_and_methods
    ):
        """Check that an error is raised if the return type of an observable is unsupported"""
        dev = mock_device_with_paulis_and_methods()

        queue = [qml.PauliX(wires=0)]

        # Make a observable without specifying a return operation upon measuring
        observables = [qml.counts(op=qml.PauliZ(0))]

        with pytest.raises(
            QuantumFunctionError,
            match="Unsupported return type specified for observable",
        ):
            dev.execute(queue, observables)


class TestParameters:
    """Test for checking device parameter mappings"""

    # pylint: disable=pointless-statement
    def test_parameters_accessed_outside_execution_context(self, mock_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""
        dev = mock_device()

        with pytest.raises(
            ValueError,
            match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            dev.parameters

    def test_parameters_available_at_pre_measure(self, mock_device, monkeypatch):
        """Tests that the parameter mapping is available when pre_measure is called and that accessing
        Device.parameters raises no error"""
        dev = mock_device(wires=3)

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
            dev.execute(queue, observables, parameters=parameters)

        assert p_mapping == parameters


class TestDeviceInit:
    """Tests for device loader in __init__.py"""

    def test_no_device(self):
        """Test that an exception is raised for a device that doesn't exist"""

        with pytest.raises(DeviceError, match="Device None does not exist"):
            qml.device("None", wires=0)

    def test_outdated_API(self, monkeypatch):
        """Test that an exception is raised if plugin that targets an old API is loaded"""

        with monkeypatch.context() as m:
            m.setattr(qml, "version", lambda: "0.0.1")
            with pytest.raises(DeviceError, match="plugin requires PennyLane versions"):
                qml.device("default.qutrit", wires=0)

    def test_plugin_devices_from_devices_triggers_getattr(self, mocker):
        spied = mocker.spy(qml.devices, "__getattr__")

        qml.devices.plugin_devices

        spied.assert_called_once()

    def test_refresh_entrypoints(self, monkeypatch):
        """Test that new entrypoints are found by the refresh_devices function"""
        assert qml.plugin_devices

        with monkeypatch.context() as m:
            # remove all entry points
            retval = {"pennylane.plugins": []} if version_info[:2] == (3, 9) else []
            m.setattr(metadata, "entry_points", lambda **kwargs: retval)

            # reimporting PennyLane within the context sets qml.plugin_devices to {}
            reload(qml)
            reload(qml.devices.device_constructor)

            # since there are no entry points, there will be no plugin devices
            assert not qml.plugin_devices

        # outside of the context, entrypoints will now be found
        assert not qml.plugin_devices
        qml.refresh_devices()
        assert qml.plugin_devices

        # Test teardown: re-import PennyLane to revert all changes and
        # restore the plugin_device dictionary
        reload(qml)
        reload(qml.devices.device_constructor)

    def test_hot_refresh_entrypoints(self, monkeypatch):
        """Test that new entrypoints are found by the device loader if not currently present"""
        assert qml.plugin_devices

        with monkeypatch.context() as m:
            # remove all entry points
            retval = {"pennylane.plugins": []} if version_info[:2] == (3, 9) else []
            m.setattr(metadata, "entry_points", lambda **kwargs: retval)

            # reimporting PennyLane within the context sets qml.plugin_devices to {}
            reload(qml.devices)
            reload(qml.devices.device_constructor)

            m.setattr(qml.devices.device_constructor, "refresh_devices", lambda: None)
            assert not qml.plugin_devices

            # since there are no entry points, there will be no plugin devices
            with pytest.raises(DeviceError, match="Device default.qubit does not exist"):
                qml.device("default.qubit", wires=0)

        # outside of the context, entrypoints will now be found automatically
        assert not qml.plugin_devices
        dev = qml.device("default.qubit", wires=0)
        assert qml.plugin_devices
        assert dev.name == "default.qubit"

        # Test teardown: re-import PennyLane to revert all changes and
        # restore the plugin_device dictionary
        reload(qml)
        reload(qml.devices.device_constructor)

    def test_shot_vector_property(self):
        """Tests shot vector initialization."""
        with pytest.warns(
            qml.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            dev = qml.device("default.qutrit", wires=1, shots=[1, 3, 3, 4, 4, 4, 3])
        shot_vector = dev.shot_vector
        assert len(shot_vector) == 4
        assert shot_vector[0].shots == 1
        assert shot_vector[0].copies == 1
        assert shot_vector[1].shots == 3
        assert shot_vector[1].copies == 2
        assert shot_vector[2].shots == 4
        assert shot_vector[2].copies == 3
        assert shot_vector[3].shots == 3
        assert shot_vector[3].copies == 1

        assert dev.shots.total_shots == 22

    def test_has_partitioned_shots(self):
        """Tests _has_partitioned_shots returns correct values"""
        dev = DefaultQubitLegacy(wires=1, shots=100)
        assert not dev._has_partitioned_shots()  # pylint:disable=protected-access

        dev.shots = [10, 20]
        assert dev._has_partitioned_shots()  # pylint:disable=protected-access

        dev.shots = 10
        assert not dev._has_partitioned_shots()  # pylint:disable=protected-access

        dev.shots = None
        assert not dev._has_partitioned_shots()  # pylint:disable=protected-access


class TestBatchExecution:
    """Tests for the batch_execute method."""

    with qml.queuing.AnnotatedQueue() as q1:
        qml.PauliX(wires=0)
        qml.expval(qml.PauliZ(wires=0))
        qml.expval(qml.PauliZ(wires=1))

    tape1 = qml.tape.QuantumScript.from_queue(q1)
    with qml.queuing.AnnotatedQueue() as q2:
        qml.PauliX(wires=0)
        qml.expval(qml.PauliZ(wires=0))

    tape2 = qml.tape.QuantumScript.from_queue(q2)

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_execute(self, n_tapes, mocker, mock_device_with_paulis_and_methods):
        """Tests that the device's execute method is called the correct number of times."""

        dev = mock_device_with_paulis_and_methods(wires=2)
        spy = mocker.spy(Device, "execute")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_reset(self, n_tapes, mocker, mock_device_with_paulis_and_methods):
        """Tests that the device's reset method is called the correct number of times."""

        dev = mock_device_with_paulis_and_methods(wires=2)
        spy = mocker.spy(Device, "reset")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    def test_result(self, mock_device_with_paulis_and_methods, tol):
        """Tests that the result has the correct shape and entry types."""

        dev = mock_device_with_paulis_and_methods(wires=2)
        tapes = [self.tape1, self.tape2]
        res = dev.batch_execute(tapes)

        assert len(res) == 2
        assert np.allclose(
            res[0], dev.execute(self.tape1.operations, self.tape1.measurements), rtol=tol, atol=0
        )
        assert np.allclose(
            res[1], dev.execute(self.tape2.operations, self.tape2.measurements), rtol=tol, atol=0
        )

    def test_result_empty_tape(self, mock_device_with_paulis_and_methods, tol):
        """Tests that the result has the correct shape and entry types for empty tapes."""

        dev = mock_device_with_paulis_and_methods(wires=2)

        empty_tape = qml.tape.QuantumScript()
        tapes = [empty_tape] * 3
        res = dev.batch_execute(tapes)

        assert len(res) == 3
        assert np.allclose(
            res[0], dev.execute(empty_tape.operations, empty_tape.measurements), rtol=tol, atol=0
        )


class TestGrouping:
    """Tests for the use_grouping option for devices."""

    # pylint: disable=too-few-public-methods, unused-argument, missing-function-docstring, missing-class-docstring
    class SomeDevice(qml.devices.LegacyDevice):
        name = ""
        short_name = ""
        pennylane_requires = ""
        version = ""
        author = ""
        operations = ""
        observables = ""

        def apply(self, *args, **kwargs):
            return 0

        def expval(self, *args, **kwargs):
            return 0

        def reset(self, *args, **kwargs):
            return 0

        def supports_observable(self, *args, **kwargs):
            return True

    # pylint: disable=attribute-defined-outside-init
    @pytest.mark.parametrize("use_grouping", (True, False))
    def test_batch_transform_checks_use_grouping_property(self, use_grouping, mocker):
        """If the device specifies `use_grouping=False`, the batch transform
        method won't expand the hamiltonian when the measured hamiltonian has
        grouping indices.
        """

        H = qml.Hamiltonian([1.0, 1.0], [qml.PauliX(0), qml.PauliY(0)], grouping_type="qwc")
        qs = qml.tape.QuantumScript(measurements=[qml.expval(H)])
        spy = mocker.spy(qml.transforms, "split_non_commuting")

        dev = self.SomeDevice()
        dev.use_grouping = use_grouping
        new_qscripts, _ = dev.batch_transform(qs)

        if use_grouping:
            assert len(new_qscripts) == 2
            spy.assert_called_once()
        else:
            assert len(new_qscripts) == 1
            spy.assert_not_called()

    def test_batch_transform_does_not_expand_supported_sum(self, mocker):
        """Tests that batch_transform does not expand Sums if they are supported."""
        H = qml.sum(qml.PauliX(0), qml.PauliY(0))
        qs = qml.tape.QuantumScript(measurements=[qml.expval(H)])
        spy = mocker.spy(qml.transforms, "split_non_commuting")

        dev = self.SomeDevice(shots=None)
        new_qscripts, _ = dev.batch_transform(qs)

        assert len(new_qscripts) == 1
        spy.assert_not_called()

    def test_batch_transform_expands_not_supported_sums(self, mocker):
        """Tests that batch_transform expand Sums if they are not supported."""
        H = qml.sum(qml.PauliX(0), qml.PauliY(0))
        qs = qml.tape.QuantumScript(measurements=[qml.expval(H)])
        spy = mocker.spy(qml.transforms, "split_non_commuting")

        dev = self.SomeDevice()
        dev.supports_observable = lambda *args, **kwargs: False
        new_qscripts, _ = dev.batch_transform(qs)

        assert len(new_qscripts) == 2
        spy.assert_called()

    def test_batch_transform_expands_prod_containing_sums(self, mocker):
        """Tests that batch_transform expands a Prod with a nested Sum"""

        H = qml.prod(qml.PauliX(0), qml.sum(qml.PauliY(0), qml.PauliZ(0)))
        qs = qml.tape.QuantumScript(measurements=[qml.expval(H)])
        spy = mocker.spy(qml.transforms, "split_non_commuting")

        dev = self.SomeDevice()
        dev.supports_observable = lambda *args, **kwargs: False
        new_qscripts, _ = dev.batch_transform(qs)

        assert len(new_qscripts) == 2
        spy.assert_called()
