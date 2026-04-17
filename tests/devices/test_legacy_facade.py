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

# pylint: disable=protected-access
import copy

import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qp
from pennylane.devices.execution_config import ExecutionConfig, MCMConfig
from pennylane.devices.legacy_facade import (
    LegacyDeviceFacade,
    legacy_device_batch_transform,
    legacy_device_expand_fn,
)
from pennylane.exceptions import DeviceError, PennyLaneDeprecationWarning, QuantumFunctionError


class DummyDevice(qp.devices.LegacyDevice):
    """A minimal device that does not do anything."""

    author = "some string"
    name = "my legacy device"
    short_name = "something"
    version = 0.0

    observables = {"PauliX", "PauliY", "PauliZ"}
    operations = {"Rot", "RX", "RY", "RZ", "PauliX", "PauliY", "PauliZ", "CNOT"}
    pennylane_requires = 0.38

    def __init__(self, wires=1, shots=None, *, analytic=None):
        super().__init__(wires=wires, shots=shots, analytic=analytic)

    def reset(self):
        pass

    # pylint: disable=unused-argument
    def apply(self, operation, wires, par):
        return 0.0

    # pylint: disable=unused-argument
    def expval(self, observable, wires, par):
        return 0.0


def test_double_facade_raises_error():
    """Test that a RuntimeError is raised if a facaded device is passed to constructor"""
    dev = qp.device("default.qutrit", wires=1)

    with pytest.raises(RuntimeError, match="already-facaded device can not be wrapped"):
        qp.devices.LegacyDeviceFacade(dev)


def test_error_if_not_legacy_device():
    """Test that a ValueError is raised if the target is not a legacy device."""

    target = qp.devices.DefaultQubit()
    with pytest.raises(ValueError, match="The LegacyDeviceFacade only accepts"):
        LegacyDeviceFacade(target)


def test_copy():
    """Test that copy works correctly"""
    dev = qp.device("default.qutrit", wires=1)

    for copied_devs in (copy.copy(dev), copy.deepcopy(dev)):
        assert copied_devs is not dev
        assert copied_devs.target_device is not dev.target_device
        assert isinstance(copied_devs.target_device, type(dev.target_device))


def test_shots():
    """Test the shots behavior of a dummy legacy device."""
    legacy_dev = DummyDevice(shots=(100, 100))
    # Expect a deprecation warning when wrapping a legacy device with shots
    with pytest.warns(
        PennyLaneDeprecationWarning,
        match="Setting shots on device is deprecated",
    ):
        dev = LegacyDeviceFacade(legacy_dev)

    assert dev.shots == qp.measurements.Shots((100, 100))

    assert legacy_dev._shot_vector == list(qp.measurements.Shots((100, 100)).shot_vector)
    assert legacy_dev.shots == 200
    assert dev.shots == qp.measurements.Shots((100, 100))


def test_tracker():
    """Test that tracking works with the Facade."""

    dev = LegacyDeviceFacade(DummyDevice())

    with qp.Tracker(dev) as tracker:
        _ = dev.execute(qp.tape.QuantumScript([], [qp.expval(qp.Z(0))], shots=50))

    assert tracker.totals == {"executions": 1, "shots": 50, "batches": 1, "batch_len": 1}


def test_debugger():
    """Test that snapshots still work with legacy devices."""

    dev = LegacyDeviceFacade(DefaultQubitLegacy(wires=1))

    @qp.qnode(dev)
    def circuit():
        qp.Snapshot()
        qp.Hadamard(0)
        qp.Snapshot()
        return qp.expval(qp.Z(0))

    res = qp.snapshots(circuit)()
    assert qp.math.allclose(res[0], np.array([1, 0]))
    assert qp.math.allclose(res[1], 1 / np.sqrt(2) * np.array([1, 1]))
    assert qp.math.allclose(res["execution_results"], 0)


@pytest.mark.parametrize(
    "execution_config",
    (None, ExecutionConfig(gradient_keyword_arguments={"method": "new_gradient"})),
)
def test_shot_distribution(execution_config):
    """Test that different numbers of shots in a batch all get executed."""

    class DummyJacobianDevice(DummyDevice):
        _capabilities = {"provides_jacobian": True}

        def new_gradient(self, circuit):  # pylint: disable=unused-argument
            return 0

        def jacobian(self, circuit):  # pylint: disable=unused-argument
            return 1

    dev = LegacyDeviceFacade(DummyJacobianDevice())

    tape1 = qp.tape.QuantumScript([], [qp.expval(qp.Z(0))], shots=5)
    tape2 = qp.tape.QuantumScript([], [qp.expval(qp.Z(0))], shots=100)

    with dev.tracker:
        dev.execute((tape1, tape2))
    assert dev.tracker.history["shots"] == [5, 100]

    with dev.tracker:
        dev.compute_derivatives((tape1, tape2), execution_config)

    assert dev.tracker.history["derivatives"] == [1, 1]  # two calls

    with dev.tracker:
        dev.execute_and_compute_derivatives((tape1, tape2), execution_config)

    assert dev.tracker.history["batches"] == [1, 1]  # broken up into multiple calls
    assert dev.tracker.history["shots"] == [5, 100]
    assert dev.tracker.history["derivatives"] == [1, 1]


def test_legacy_device_expand_fn():
    """Test that legacy_device_expand_fn expands operations to the target gateset."""

    tape = qp.tape.QuantumScript([qp.X(0), qp.IsingXX(0.5, wires=(0, 1))], [qp.state()])

    expected = qp.tape.QuantumScript(
        [qp.X(0), qp.CNOT((0, 1)), qp.RX(0.5, 0), qp.CNOT((0, 1))], [qp.state()]
    )
    (new_tape,), fn = legacy_device_expand_fn(tape, device=DummyDevice())
    qp.assert_equal(new_tape, expected)
    assert fn(("A",)) == "A"


def test_legacy_device_batch_transform():
    """Test that legacy_device_batch_transform still performs the operations that the legacy batch transform did."""

    tape = qp.tape.QuantumScript([], [qp.expval(qp.X(0) + qp.Y(0))])
    (tape1, tape2), fn = legacy_device_batch_transform(tape, device=DummyDevice())

    qp.assert_equal(tape1, qp.tape.QuantumScript([], [qp.expval(qp.X(0))]))
    qp.assert_equal(tape2, qp.tape.QuantumScript([], [qp.expval(qp.Y(0))]))
    assert fn((1.0, 2.0)) == np.array(3.0)


def test_batch_transform_supports_hamiltonian():
    """Assert that the legacy device batch transform absorbs the shot sensitive behavior of the
    hamiltonian expansion."""

    class HamiltonianDev(DummyDevice):
        """A device that supports a hamiltonian."""

        observables = {"Hamiltonian"}

    H = qp.Hamiltonian([1, 1], [qp.X(0), qp.Y(0)])

    tape = qp.tape.QuantumScript([], [qp.expval(H)], shots=None)
    batch, _ = legacy_device_batch_transform(tape, device=HamiltonianDev())
    assert len(batch) == 1

    tape = qp.tape.QuantumScript([], [qp.expval(H)], shots=50)
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
    assert d.wires == qp.wires.Wires((0, 1))


def test_preprocessing_program():
    """Test the population of the preprocessing program."""

    m0 = qp.measure(0)
    tape = qp.tape.QuantumScript(
        [qp.X(0), qp.IsingXX(0.5, (0, 1)), *m0.measurements],
        [qp.expval(qp.Hamiltonian([1, 1], [qp.X(0), qp.Y(0)]))],
    )

    dev = DummyDevice(wires=(0, 1))
    facade = LegacyDeviceFacade(dev)
    config = facade.setup_execution_config(circuit=tape)
    program = facade.preprocess_transforms(config)

    assert (
        program[0].tape_transform == qp.defer_measurements.tape_transform
    )  # pylint: disable=no-member
    assert (
        program[1].tape_transform == legacy_device_batch_transform.tape_transform
    )  # pylint: disable=no-member
    assert (
        program[2].tape_transform == legacy_device_expand_fn.tape_transform
    )  # pylint: disable=no-member

    m0 = qp.measure(0)
    tape = qp.tape.QuantumScript(
        [qp.X(0), qp.IsingXX(0.5, (0, 1)), *m0.measurements],
        [qp.expval(qp.Hamiltonian([1, 1], [qp.X(0), qp.Y(0)]))],
        shots=50,
    )

    (tape1, tape2), fn = program((tape,))
    expected_ops = [qp.X(0), qp.CNOT((0, 1)), qp.RX(0.5, 0), qp.CNOT((0, 1)), qp.CNOT((0, 2))]
    assert tape1.operations == expected_ops
    assert tape2.operations == expected_ops

    assert tape1.measurements == [qp.expval(qp.X(0))]
    assert tape2.measurements == [qp.expval(qp.Y(0))]
    assert tape1.shots == qp.measurements.Shots(50)
    assert tape2.shots == qp.measurements.Shots(50)

    assert qp.math.allclose(fn((1.0, 2.0)), 3.0)


def test_mcm_validation():
    """Tests that the setup_execution_config correctly validates mcm methods."""

    dev = DummyDevice(wires=[0, 1])
    facade = LegacyDeviceFacade(dev)

    m0 = qp.measure(0)
    tape = qp.tape.QuantumScript([qp.X, *m0.measurements], [qp.expval(qp.Z(0))])
    with pytest.raises(QuantumFunctionError, match="only supported with finite shots"):
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="one-shot"))
        facade.setup_execution_config(config, tape)

    with pytest.raises(QuantumFunctionError, match="unsupported by the device"):
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="tree-traversal"))
        facade.setup_execution_config(config, tape)


def test_mcm_resolution_when_supported():
    """Tests resolution of the mcm method when mcm is supported."""

    class MidMeasureDev(DummyDevice):
        """A dummy device that supports mid circuit measurements."""

        _capabilities = {"supports_mid_measure": True}

    dev = MidMeasureDev(wires=[0, 1])
    facade = LegacyDeviceFacade(dev)

    m0 = qp.measure(0)
    tape = qp.tape.QuantumScript([qp.X, *m0.measurements], [qp.expval(qp.Z(0))], shots=100)
    config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="one-shot"))
    config = facade.setup_execution_config(config, tape)
    assert config.mcm_config.mcm_method == "one-shot"

    config = ExecutionConfig()
    config = facade.setup_execution_config(config, tape)
    assert config.mcm_config.mcm_method == "one-shot"


@pytest.mark.usefixtures("create_temporary_toml_file")
@pytest.mark.parametrize(
    "create_temporary_toml_file",
    [
        """
        schema = 3

        [compilation]

        supported_mcm_methods = ["one-shot"]
        """
    ],
    indirect=True,
)
def test_mcm_resolution_toml_present(request):
    """Tests resolution of the mcm methods when a toml file is provided."""

    dev = DummyDevice(wires=[0, 1])
    dev.config_filepath = request.node.toml_file  # pylint: disable=attribute-defined-outside-init
    facade = LegacyDeviceFacade(dev)

    m0 = qp.measure(0)
    tape = qp.tape.QuantumScript([qp.X, *m0.measurements], [qp.expval(qp.Z(0))], shots=100)
    config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="one-shot"))
    config = facade.setup_execution_config(config, tape)
    assert config.mcm_config.mcm_method == "one-shot"

    config = ExecutionConfig()
    config = facade.setup_execution_config(config, tape)
    assert config.mcm_config.mcm_method == "one-shot"


def test_preprocessing_program_supports_mid_measure():
    """Test that if the device natively supports mid measure, defer_measurements wont be applied."""

    class MidMeasureDev(DummyDevice):
        """A dummy device that supports mid circuit measurements."""

        _capabilities = {"supports_mid_measure": True}

    dev = MidMeasureDev()
    program = LegacyDeviceFacade(dev).preprocess_transforms()
    assert qp.defer_measurements not in program


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

    mcm_config = qp.devices.MCMConfig(postselect_mode=t_postselect_mode)
    config = qp.devices.ExecutionConfig(mcm_config=mcm_config)

    dev.execute(qp.tape.QuantumScript(), config)


class TestGradientSupport:
    """Test integration with various kinds of device derivatives."""

    def test_no_derivatives_case(self):
        """Test that the relevant errors are raised when the device does not support derivatives."""

        dev = LegacyDeviceFacade(DummyDevice())

        assert not dev.supports_derivatives()
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="adjoint"))
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="device"))
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="param_shift"))

        with pytest.raises(DeviceError):
            config = dev.setup_execution_config(ExecutionConfig(gradient_method="device"))
            dev.preprocess(config)

        with pytest.raises(DeviceError):
            config = dev.setup_execution_config(ExecutionConfig(gradient_method="adjoint"))
            dev.preprocess(config)

        with pytest.raises(DeviceError):
            config = dev.setup_execution_config(ExecutionConfig(gradient_method="backprop"))
            dev.preprocess(config)

    def test_adjoint_support(self):
        """Test that the facade can handle devices that support adjoint."""

        gradient_method = "adjoint"

        # pylint: disable=unnecessary-lambda-assignment
        class AdjointDev(DummyDevice):
            """A dummy device that supports adjoint diff"""

            _capabilities = {"returns_state": True}

            _apply_operation = lambda *args, **kwargs: 0
            _apply_unitary = lambda *args, **kwargs: 0
            adjoint_jacobian = lambda *args, **kwargs: "a"

        adj_dev = AdjointDev()
        dev = LegacyDeviceFacade(adj_dev)

        tape = qp.tape.QuantumScript([], [], shots=None)
        assert dev._validate_adjoint_method(tape)
        tape_shots = qp.tape.QuantumScript([], [], shots=50)
        assert not dev._validate_adjoint_method(tape_shots)

        config = qp.devices.ExecutionConfig(gradient_method=gradient_method)
        assert dev.supports_derivatives(config, tape)
        assert not dev.supports_derivatives(config, tape_shots)

        unsupported_tape = qp.tape.QuantumScript([], [qp.state()])
        assert not dev.supports_derivatives(config, unsupported_tape)

        program, processed_config = dev.preprocess(config)
        assert processed_config.use_device_gradient is True
        assert processed_config.gradient_keyword_arguments == {
            "use_device_state": True,
            "method": "adjoint_jacobian",
        }
        assert processed_config.grad_on_execution is True

        tape = qp.tape.QuantumScript(
            [qp.Rot(qp.numpy.array(1.2), 2.3, 3.4, 0)], [qp.expval(qp.Z(0))]
        )
        (new_tape,), _ = program((tape,))
        expected = qp.tape.QuantumScript(
            [qp.RZ(qp.numpy.array(1.2), 0), qp.RY(2.3, 0), qp.RZ(3.4, 0)],
            [qp.expval(qp.Z(0))],
        )
        qp.assert_equal(new_tape, expected)

        out = dev.compute_derivatives(tape, processed_config)
        assert out == "a"  # the output of adjoint_jacobian

        res, jac = dev.execute_and_compute_derivatives(tape, processed_config)
        assert qp.math.allclose(res, 0)
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
        assert dev._validate_device_method(qp.tape.QuantumScript())

        config = qp.devices.ExecutionConfig(gradient_method="best")
        processed_config = dev.setup_execution_config(config)
        assert processed_config.use_device_gradient is True
        assert processed_config.grad_on_execution is True

    def test_no_backprop_with_sparse_hamiltonian(self):
        """Test that backpropagation is not supported with SparseHamiltonian."""

        class BackpropDevice(DummyDevice):
            _capabilities = {"passthru_interface": "autograd"}

        H = qp.SparseHamiltonian(qp.X.compute_sparse_matrix(), wires=0)
        x = qp.numpy.array(0.1)
        tape = qp.tape.QuantumScript([qp.RX(x, 0)], [qp.expval(H)])
        dev = LegacyDeviceFacade(BackpropDevice())
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"), tape)

        tape2 = qp.tape.QuantumScript([qp.RX(x, 0)], [qp.expval(qp.Z(0))])
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"), tape2)

    def test_passthru_interface_no_substitution(self):
        """Test that if the passthru interface is set, no substitution occurs."""

        class BackpropDevice(DummyDevice):
            _capabilities = {"passthru_interface": "autograd"}

        dev = LegacyDeviceFacade(BackpropDevice(wires=2, shots=None))

        assert dev.supports_derivatives(qp.devices.ExecutionConfig(gradient_method="backprop"))

        config = qp.devices.ExecutionConfig(gradient_method="backprop", use_device_gradient=True)
        assert dev.setup_execution_config(config) is config  # unchanged
