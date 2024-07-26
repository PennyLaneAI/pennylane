# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains unit tests for the LegacyDeviceFacade class.
"""
import numpy as np

# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane.devices.default_qubit_autograd import DefaultQubitAutograd
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.devices.legacy_facade import (
    LegacyDeviceFacade,
    legacy_device_batch_transform,
    legacy_device_expand_fn,
)


class DummyDevice(qml.devices.LegacyDevice):
    """A minimal device that does not do anything."""

    author = "some string"
    name = "my legacy device"
    short_name = "something"
    version = 0.0

    observables = {"PauliX", "PauliY", "PauliZ"}
    operations = {"Rot", "RX", "RY", "RZ", "PauliX", "PauliY", "PauliZ", "CNOT"}
    pennylane_requires = 0.38

    def reset(self):
        pass

    # pylint: disable=unused-argument
    def apply(self, operation, wires, par):
        return 0.0

    # pylint: disable=unused-argument
    def expval(self, observable, wires, par):
        return 0.0


def test_error_if_not_legacy_device():
    """Test that a ValueError is raiuised if the target is not a legacy device."""

    target = qml.devices.DefaultQubit()
    with pytest.raises(ValueError, match="The LegacyDeviceFacade only accepts"):
        LegacyDeviceFacade(target)


def test_shots():
    """Test the shots behavior of a dummy legacy device."""
    legacy_dev = DummyDevice(shots=(100, 100))
    dev = LegacyDeviceFacade(legacy_dev)

    assert dev.shots == qml.measurements.Shots((100, 100))

    assert legacy_dev._shot_vector == list(qml.measurements.Shots((100, 100)).shot_vector)
    assert legacy_dev.shots == 200
    assert dev.shots == qml.measurements.Shots((100, 100))


def test_tracker():
    """Test that tracking works with the Facade."""

    dev = LegacyDeviceFacade(DummyDevice())

    with qml.Tracker(dev) as tracker:
        _ = dev.execute(qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=50))

    assert tracker.totals == {"executions": 1, "shots": 50, "batches": 1, "batch_len": 1}


def test_debugger():
    """Test that snapshots still work with legacy devices."""

    dev = LegacyDeviceFacade(qml.devices.DefaultQubitLegacy(wires=1))

    @qml.qnode(dev)
    def circuit():
        qml.Snapshot()
        qml.Hadamard(0)
        qml.Snapshot()
        return qml.expval(qml.Z(0))

    res = qml.snapshots(circuit)()
    assert qml.math.allclose(res[0], np.array([1, 0]))
    assert qml.math.allclose(res[1], 1 / np.sqrt(2) * np.array([1, 1]))
    assert qml.math.allclose(res["execution_results"], 0)


def test_shot_distribution():
    """Test that different numbers of shots in a batch all get executed."""

    class DummyJacobianDevice(DummyDevice):

        _capabilities = {"provides_jacobian": True}

        def jacobian(self, circuit):  # pylint: disable=unused-argument
            return 0

    dev = LegacyDeviceFacade(DummyJacobianDevice())

    tape1 = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=5)
    tape2 = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=100)

    with dev.tracker:
        dev.execute((tape1, tape2))
    assert dev.tracker.history["shots"] == [5, 100]

    with dev.tracker:
        dev.compute_derivatives((tape1, tape2))

    assert dev.tracker.history["derivatives"] == [1, 1]  # two calls

    with dev.tracker:
        dev.execute_and_compute_derivatives((tape1, tape2))

    assert dev.tracker.history["batches"] == [1, 1]  # broken up into multiple calls
    assert dev.tracker.history["shots"] == [5, 100]
    assert dev.tracker.history["derivatives"] == [1, 1]


def test_legacy_device_expand_fn():
    """Test that legacy_device_expand_fn expands operations to the target gateset."""

    tape = qml.tape.QuantumScript([qml.X(0), qml.IsingXX(0.5, wires=(0, 1))], [qml.state()])

    expected = qml.tape.QuantumScript(
        [qml.X(0), qml.CNOT((0, 1)), qml.RX(0.5, 0), qml.CNOT((0, 1))], [qml.state()]
    )
    (new_tape,), fn = legacy_device_expand_fn(tape, device=DummyDevice())
    assert qml.equal(new_tape, expected)
    assert fn(("A",)) == "A"


def test_legacy_device_batch_transform():
    """Test that legacy_device_batch_transform still performs the operations that the legacy batch transform did."""

    tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0) + qml.Y(0))])
    (tape1, tape2), fn = legacy_device_batch_transform(tape, device=DummyDevice())

    assert qml.equal(tape1, qml.tape.QuantumScript([], [qml.expval(qml.X(0))]))
    assert qml.equal(tape2, qml.tape.QuantumScript([], [qml.expval(qml.Y(0))]))
    assert fn((1.0, 2.0)) == np.array(3.0)


def test_batch_transform_supports_hamiltonian():
    """Assert that the legacy device batch transform absorbs the shot sensitive behavior of the
    hamiltonian expansion."""

    class HamiltonianDev(DummyDevice):
        """A device that supports a hamiltonian."""

        observables = {"Hamiltonian"}

    H = qml.Hamiltonian([1, 1], [qml.X(0), qml.Y(0)])

    tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=None)
    batch, _ = legacy_device_batch_transform(tape, device=HamiltonianDev())
    assert len(batch) == 1

    tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=50)
    batch, _ = legacy_device_batch_transform(tape, device=HamiltonianDev())
    assert len(batch) == 2


def test_basic_properties():
    """Test the basic properties of the device."""

    ld = DummyDevice(wires=(0, 1))
    d = LegacyDeviceFacade(ld)
    assert d.target_device is ld
    assert d.name == "something"
    assert d.author == "some string"
    assert d.version == 0.0
    assert repr(d) == f"<LegacyDeviceFacade: {repr(ld)}>"
    assert d.wires == qml.wires.Wires((0, 1))


def test_preprocessing_program():
    """Test the population of the preprocessing program."""

    dev = DummyDevice(wires=(0, 1))
    program, _ = LegacyDeviceFacade(dev).preprocess()

    assert (
        program[0].transform == legacy_device_batch_transform.transform
    )  # pylint: disable=no-member
    assert program[1].transform == legacy_device_expand_fn.transform  # pylint: disable=no-member
    assert program[2].transform == qml.defer_measurements.transform  # pylint: disable=no-member

    m0 = qml.measure(0)
    tape = qml.tape.QuantumScript(
        [qml.X(0), qml.IsingXX(0.5, (0, 1)), *m0.measurements],
        [qml.expval(qml.Hamiltonian([1, 1], [qml.X(0), qml.Y(0)]))],
        shots=50,
    )

    (tape1, tape2), fn = program((tape,))
    expected_ops = [qml.X(0), qml.CNOT((0, 1)), qml.RX(0.5, 0), qml.CNOT((0, 1)), qml.CNOT((0, 2))]
    assert tape1.operations == expected_ops
    assert tape2.operations == expected_ops

    assert tape1.measurements == [qml.expval(qml.X(0))]
    assert tape2.measurements == [qml.expval(qml.Y(0))]
    assert tape1.shots == qml.measurements.Shots(50)
    assert tape2.shots == qml.measurements.Shots(50)

    assert qml.math.allclose(fn((1.0, 2.0)), 3.0)


def test_preprocessing_program_supports_mid_measure():
    """Test that if the device natively supports mid measure, defer_measurements wont be applied."""

    class MidMeasureDev(DummyDevice):
        """A dummy device that supports mid circuit measurements."""

        _capabilities = {"supports_mid_measure": True}

    dev = MidMeasureDev()
    program, _ = LegacyDeviceFacade(dev).preprocess()
    assert qml.defer_measurements not in program


@pytest.mark.parametrize("t_postselect_mode", ("hw-like", "fill-shots"))
def test_pass_postselect_mode_to_dev(t_postselect_mode):
    """test that postselect mode is passed to the target if it supports mid measure."""

    class MidMeasureDev(DummyDevice):
        """A dummy device that supports mid circuit measurements."""

        _capabilities = {"supports_mid_measure": True}

        def batch_execute(self, circuits, postselect_mode):
            assert postselect_mode == t_postselect_mode
            return tuple(0 for _ in circuits)

    target = MidMeasureDev()
    dev = LegacyDeviceFacade(target)

    mcm_config = qml.devices.MCMConfig(postselect_mode=t_postselect_mode)
    config = qml.devices.ExecutionConfig(mcm_config=mcm_config)

    dev.execute(qml.tape.QuantumScript(), config)


class TestGradientSupport:
    """Test integration with various kinds of device derivatives."""

    def test_no_derivatives_case(self):
        """Test that the relevant errors are raised when the device does not support derivatives."""

        dev = LegacyDeviceFacade(DummyDevice())

        assert not dev.supports_derivatives()
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="adjoint"))
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="device"))

        with pytest.raises(qml.DeviceError):
            dev.preprocess(ExecutionConfig(gradient_method="device"))

        with pytest.raises(qml.DeviceError):
            dev.preprocess(ExecutionConfig(gradient_method="adjoint"))

        with pytest.raises(qml.DeviceError):
            dev.preprocess(ExecutionConfig(gradient_method="backprop"))

    @pytest.mark.parametrize("gradient_method", ("best", "adjoint"))
    def test_adjoint_support(self, gradient_method):
        """Test that the facade can handle devices that support adjoint."""

        # pylint: disable=unnecessary-lambda-assignment
        class AdjointDev(DummyDevice):
            """A dummy device that supports adjoint diff"""

            _capabilities = {"returns_state": True}

            _apply_operation = lambda *args, **kwargs: 0
            _apply_unitary = lambda *args, **kwargs: 0
            adjoint_jacobian = lambda *args, **kwargs: "a"

        adj_dev = AdjointDev()
        dev = LegacyDeviceFacade(adj_dev)

        tape = qml.tape.QuantumScript([], [], shots=None)
        assert dev._validate_adjoint_method(tape)
        tape_shots = qml.tape.QuantumScript([], [], shots=50)
        assert not dev._validate_adjoint_method(tape_shots)

        config = qml.devices.ExecutionConfig(gradient_method=gradient_method)
        assert dev.supports_derivatives(config, tape)
        assert not dev.supports_derivatives(config, tape_shots)

        program, processed_config = dev.preprocess(config)
        assert processed_config.use_device_gradient is True
        assert processed_config.gradient_keyword_arguments == {
            "use_device_state": True,
            "method": "adjoint_jacobian",
        }
        assert processed_config.grad_on_execution is True

        tape = qml.tape.QuantumScript(
            [qml.Rot(qml.numpy.array(1.2), 2.3, 3.4, 0)], [qml.expval(qml.Z(0))]
        )
        (new_tape,), _ = program((tape,))
        expected = qml.tape.QuantumScript(
            [qml.RZ(qml.numpy.array(1.2), 0), qml.RY(2.3, 0), qml.RZ(3.4, 0)],
            [qml.expval(qml.Z(0))],
        )
        qml.assert_equal(new_tape, expected)

        out = dev.compute_derivatives(tape, processed_config)
        assert out == "a"  # the output of adjoint_jacobian

        res, jac = dev.execute_and_compute_derivatives(tape, processed_config)
        assert qml.math.allclose(res, 0)
        assert jac == "a"

    def test_device_derivatives(self):
        """Test that a device that provides a derivative processed the config correctly."""

        class DeviceDerivatives(DummyDevice):
            """A dummy device with a jacobian."""

            _capabilities = {"provides_jacobian": True}

        ddev = DeviceDerivatives()
        dev = LegacyDeviceFacade(ddev)

        assert dev.supports_derivatives()
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="device"))
        assert dev._validate_device_method(qml.tape.QuantumScript())

        config = qml.devices.ExecutionConfig(gradient_method="best")
        _, processed_config = dev.preprocess(config)
        assert processed_config.use_device_gradient is True
        assert processed_config.grad_on_execution is True

    def test_no_backprop_with_sparse_hamiltonian(self):
        """Test that backpropagation is not supported with SparseHamiltonian."""

        class BackpropDevice(DummyDevice):

            _capabilities = {"passthru_interface": "autograd"}

        H = qml.SparseHamiltonian(qml.X.compute_sparse_matrix(), wires=0)
        x = qml.numpy.array(0.1)
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(H)])
        dev = LegacyDeviceFacade(BackpropDevice())
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"), tape)

        tape2 = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"), tape2)

    def test_passthru_interface_no_substitution(self):
        """Test that if the passthru interface is set, no substitution occurs."""

        class BackpropDevice(DummyDevice):

            _capabilities = {"passthru_interface": "autograd"}

        dev = LegacyDeviceFacade(BackpropDevice(wires=2, shots=None))

        x = qml.numpy.array(0.1)
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])

        assert dev.supports_derivatives(qml.devices.ExecutionConfig(gradient_method="backprop"))
        assert dev._create_temp_device((tape,)) is dev.target_device

    def test_backprop_has_passthru_devices(self):
        """Test that backprop is supported if the device has passthru devices."""

        class BackpropDevice(DummyDevice):

            _capabilities = {"passthru_devices": {"autograd": "default.qubit.autograd"}}

        dev = LegacyDeviceFacade(BackpropDevice(shots=None))

        x = qml.numpy.array(0.1)
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
        assert dev.supports_derivatives()
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"), tape)

        tmp_device = dev._create_temp_device((tape,))
        assert tmp_device.short_name == "default.qubit.autograd"

    def test_backprop_passthru_device_self(self):
        """Test that the temporary device is the original device if the passthru device is itself."""

        class BackpropSelfDevice(DummyDevice):

            short_name = "BackpropSelfDevice"

            _capabilities = {"passthru_devices": {"autograd": "BackpropSelfDevice"}}

        dev = LegacyDeviceFacade(BackpropSelfDevice(wires=2))

        x = qml.numpy.array(0.1)
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
        tmp_dev = dev._create_temp_device((tape,))
        assert tmp_dev is dev.target_device

    def test_passthru_device_does_not_exist(self):
        """Test that if backprop is requested for a device that does not support it, a device error is raised."""

        x = qml.numpy.array(0.1)
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])

        dev = LegacyDeviceFacade(DummyDevice(wires=2))
        config = qml.devices.ExecutionConfig(gradient_method="backprop")
        with pytest.raises(qml.DeviceError, match=r"does not support backpropagation"):
            dev.execute(tape, config)

    @pytest.mark.parametrize("dev_class", (qml.devices.DefaultQubitLegacy, DefaultQubitAutograd))
    def test_backprop_device_substitution(self, dev_class):
        """Test that default.qubit.legacy is substituted for a backprop device during backprop execution."""

        dq_legacy = dev_class(wires=2)
        dev = LegacyDeviceFacade(dq_legacy)

        def f(x):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
            return dev.execute(tape, qml.devices.ExecutionConfig(gradient_method="backprop"))

        assert qml.math.allclose(dq_legacy.state, np.array([1, 0, 0, 0]))

        with dev.tracker:
            g = qml.grad(f)(qml.numpy.array(0.5))
        assert qml.math.allclose(g, -np.sin(0.5))
        assert dev.tracker.totals["executions"] == 1
        assert not qml.math.allclose(dq_legacy.state, np.array([1, 0, 0, 0]))
