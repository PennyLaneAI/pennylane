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
Unit tests for the debugging module.
"""
from unittest.mock import patch

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.debugging import PLDB, pldb_device_manager


class TestSnapshot:
    """Test the Snapshot instruction for simulators."""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "backprop", "parameter-shift", "adjoint"])
    def test_default_qubit_legacy_opmath(self, method):
        """Test that multiple snapshots are returned correctly on the state-vector simulator."""
        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None
        if method is not None:
            assert circuit.interface == "auto"

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "backprop", "parameter-shift", "adjoint"])
    def test_default_qubit2(self, method):
        """Test that multiple snapshots are returned correctly on the new
        state-vector simulator."""
        dev = qml.device("default.qubit")

        # TODO: add additional QNode test once the new device supports it

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None
        if method is not None:
            assert circuit.interface == "auto"

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_mixed(self, method):
        """Test that multiple snapshots are returned correctly on the density-matrix simulator."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            "very_important_state": np.array(
                [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
            ),
            2: np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_gaussian(self, method):
        """Test that multiple snapshots are returned correctly on the CV simulator."""
        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Displacement(0.5, 0, wires=0)
            qml.Snapshot("very_important_state")
            qml.Beamsplitter(0.5, 0.7, wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.QuadX(0))

        circuit()
        assert dev._debugger is None

        result = qml.snapshots(circuit)()
        expected = {
            0: {
                "cov_matrix": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "means": np.array([0, 0, 0, 0]),
            },
            1: {
                "cov_matrix": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "means": np.array([1, 0, 0, 0]),
            },
            2: {
                "cov_matrix": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "means": np.array([0.87758256, 0.36668488, 0, 0.30885441]),
            },
            "execution_results": np.array(0.87758256),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert np.allclose(result["execution_results"], expected["execution_results"])
        del result["execution_results"]
        del expected["execution_results"]
        assert all(
            np.allclose(v1["cov_matrix"], v2["cov_matrix"])
            for v1, v2 in zip(result.values(), expected.values())
        )
        assert all(
            np.allclose(v1["means"], v2["means"])
            for v1, v2 in zip(result.values(), expected.values())
        )

    @pytest.mark.parametrize("method", [None, "parameter-shift", "adjoint"])
    def test_lightning_qubit(self, method):
        """Test that an error is (currently) raised on the lightning simulator."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        # can run the circuit
        result = circuit()
        assert result == 0

        with pytest.raises(qml.DeviceError, match="Device does not support snapshots."):
            qml.snapshots(circuit)()

    def test_unsupported_device(self):
        """Test that an error is raised on unsupported devices."""
        dev = qml.device("default.qubit.legacy", wires=2)
        # remove attributes to simulate unsupported device
        delattr(dev, "_debugger")
        dev.operations.remove("Snapshot")

        @qml.qnode(dev, interface=None)  # iterface=None prevents new device creation internally
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        # can run the circuit
        result = circuit()
        assert result == 0

        with pytest.raises(qml.DeviceError, match="Device does not support snapshots."):
            qml.snapshots(circuit)()

        # need to revert change to not affect other tests (since operations a static attribute)
        dev.operations.add("Snapshot")

    def test_unsupported_device_new(self):
        """Test that an error is raised on unsupported devices."""

        class DummyDevice(qml.devices.Device):  # pylint: disable=too-few-public-methods
            def execute(self, *args, **kwargs):
                return args, kwargs

        dev = DummyDevice()

        with pytest.raises(qml.DeviceError, match="Device does not support snapshots."):
            with qml.debugging.snapshot._Debugger(dev):
                dev.execute([])

    def test_empty_snapshots(self):
        """Test that snapshots function in the absence of any Snapshot operations."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        result = qml.snapshots(circuit)()
        expected = {"execution_results": np.array(0)}

        assert result == expected

    @pytest.mark.parametrize("shots", [None, 1, 1000000])
    def test_different_shots(self, shots):
        """Test that snapshots are returned correctly with different QNode shot values."""
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        for v1, v2 in zip(result.values(), expected.values()):
            if v1.shape == ():
                # Expectation value comparison
                assert v2.shape == ()
                if shots != 1:
                    # If many shots or analytic mode, we can compare the result
                    assert np.allclose(v1, v2, atol=0.006)
                else:
                    # If a single shot, it must be in the eigvals of qml.Z
                    assert v1.item() in {-1.0, 1.0}
            else:
                assert np.allclose(v1, v2)

    @pytest.mark.parametrize(
        "m,expected_result",
        [
            ("expval", np.array(0)),
            ("var", np.array(1)),
            ("probs", np.array([0.5, 0, 0, 0.5])),
            ("state", np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])),
        ],
    )
    def test_different_measurements(self, m, expected_result):
        """Test that snapshots are returned correctly with different QNode measurements."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            if m == "expval":
                return qml.expval(qml.PauliZ(0))
            if m == "var":
                return qml.var(qml.PauliY(1))
            if m == "probs":
                return qml.probs([0, 1])
            return qml.state()

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": expected_result,
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

        if m == "state":
            assert np.allclose(result[2], result["execution_results"])

    def test_controlled_circuit(self):
        """Test that snapshots are returned correctly even with a controlled circuit."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(params, wire):
            qml.Hadamard(wire)
            qml.Snapshot()
            qml.Rot(*params, wire)

        @qml.qnode(dev)
        def qnode(params):
            qml.Hadamard(0)
            qml.ctrl(circuit, 0)(params, wire=1)
            return qml.expval(qml.PauliZ(1))

        params = np.array([1.3, 1.4, 0.2])
        result = qml.snapshots(qnode)(params)
        expected = {
            0: np.array([1 / np.sqrt(2), 0, 0.5, 0.5]),
            "execution_results": np.array(0.36819668),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    def test_all_measurement_snapshot(self):
        """Test that the correct measurement snapshots are returned for different measurement types."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(x):
            qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)))
            qml.RX(x, wires=0)
            qml.Snapshot(measurement=qml.var(qml.PauliZ(0)))
            qml.Snapshot(measurement=qml.probs(0))
            qml.Snapshot(measurement=qml.state())
            return qml.expval(qml.PauliZ(0))

        phi = 0.1
        result = qml.snapshots(circuit)(phi)
        expected = {
            0: np.array(1),
            1: np.array(1 - np.cos(phi) ** 2),
            2: np.array([np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2]),
            3: np.array([np.cos(phi / 2), -1j * np.sin(phi / 2)]),
            "execution_results": np.array(np.cos(phi)),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    def test_unsupported_snapshot_measurement(self):
        """Test that an exception is raised when an unsupported measurement is provided to the snapshot."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            with pytest.raises(
                ValueError,
                match="The measurement PauliZ is not supported as it is not an instance "
                "of <class 'pennylane.measurements.measurements.StateMeasurement'>",
            ):
                qml.Snapshot(measurement=qml.PauliZ(0))
            return qml.expval(qml.PauliZ(0))

        qml.snapshots(circuit)()


# pylint: disable=protected-access
class TestPLDB:
    """Test the interactive debugging integration"""

    def test_pldb_init(self):
        """Test that PLDB initializes correctly"""
        debugger = PLDB()
        assert debugger.prompt == "[pldb]: "
        assert getattr(debugger, "_PLDB__active_dev") is None

    def test_valid_context_outside_qnode(self):
        """Test that valid_context raises an error when breakpoint
        is called outside of a qnode execution."""

        with pytest.raises(
            RuntimeError, match="Can't call breakpoint outside of a qnode execution"
        ):
            with qml.queuing.AnnotatedQueue() as _:
                qml.X(0)
                qml.breakpoint()
                qml.Hadamard(0)

        def my_qfunc():
            qml.X(0)
            qml.breakpoint()
            qml.Hadamard(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            RuntimeError, match="Can't call breakpoint outside of a qnode execution"
        ):
            _ = my_qfunc()

    def test_valid_context_not_compatible_device(self):
        """Test that valid_context raises an error when breakpoint
        is called with an incompatible device."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def my_circ():
            qml.X(0)
            qml.breakpoint()
            qml.Hadamard(0)
            return qml.expva(qml.Z(0))

        with pytest.raises(TypeError, match="Breakpoints not supported on this device"):
            _ = my_circ()

        PLDB.reset_active_dev()

    def test_add_device(self):
        """Test that we can add a device to the global active device list."""
        assert not PLDB.has_active_dev()

        dev1, dev2, dev3 = (
            qml.device("default.qubit", wires=3),
            qml.device("default.qubit"),
            qml.device("lightning.qubit", wires=1),
        )

        PLDB.add_device(dev1)
        assert PLDB.get_active_device() == dev1

        PLDB.add_device(dev2)  # overwrites dev1
        PLDB.add_device(dev3)  # overwrites dev2

        assert PLDB.get_active_device() == dev3

        PLDB.reset_active_dev()  # clean up the debugger active devices

    dev_names = (
        "default.qubit",
        "lightning.qubit",
    )

    @pytest.mark.parametrize("device_name", dev_names)
    def test_get_active_device(self, device_name):
        """Test that we can access the active device."""
        dev = qml.device(device_name, wires=2)
        with pldb_device_manager(dev) as _:
            assert PLDB.get_active_device() is dev

    def test_get_active_device_error_when_no_active_device(self):
        """Test that an error is raised if we try to get
        the active device when there are no active devices."""
        assert not PLDB.has_active_dev()

        with pytest.raises(RuntimeError, match="No active device to get"):
            _ = PLDB.get_active_device()

    @pytest.mark.parametrize("device_name", dev_names)
    def test_reset_active_device(self, device_name):
        """Test that we can rest the global active device list."""
        dev = qml.device(device_name, wires=2)
        PLDB.add_device(dev)
        assert PLDB.get_active_device() == dev

        PLDB.reset_active_dev()
        assert not PLDB.has_active_dev()

    def test_has_active_device(self):
        """Test that we can determine if there is an active device."""
        assert getattr(PLDB, "_PLDB__active_dev") is None

        dev = qml.device("default.qubit")
        PLDB.add_device(dev)
        assert PLDB.has_active_dev()

        PLDB.reset_active_dev()
        assert not PLDB.has_active_dev()

    tapes = (
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.CNOT([0, 1])],
            measurements=[qml.state()],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.X(1)],
            measurements=[qml.expval(qml.Z(1))],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.CNOT([0, 1])],
            measurements=[qml.probs()],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.CNOT([0, 1])],
            measurements=[qml.probs(wires=[0])],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.state()],
        ),  # Test that state expands to number of device wires
    )

    results = (
        qnp.array([1 / qnp.sqrt(2), 0, 0, 1 / qnp.sqrt(2)], dtype=complex),
        qnp.array(-1),
        qnp.array([1 / 2, 0, 0, 1 / 2]),
        qnp.array([1 / 2, 1 / 2]),
        qnp.array([1 / qnp.sqrt(2), 0, 1 / qnp.sqrt(2), 0], dtype=complex),
    )

    @pytest.mark.parametrize("tape, expected_result", zip(tapes, results))
    @pytest.mark.parametrize(
        "dev", (qml.device("default.qubit", wires=2), qml.device("lightning.qubit", wires=2))
    )
    def test_execute(self, dev, tape, expected_result):
        """Test that the _execute method works as expected."""
        PLDB.add_device(dev)
        executed_results = PLDB._execute((tape,))
        assert qnp.allclose(expected_result, executed_results)
        PLDB.reset_active_dev()


def test_tape():
    """Test that we can access the tape from the active queue."""
    with qml.queuing.AnnotatedQueue() as queue:
        qml.X(0)

        for i in range(3):
            qml.Hadamard(i)

        qml.Y(1)
        qml.Z(0)
        qml.expval(qml.Z(0))

        executed_tape = qml.debugging.tape()

    expected_tape = qml.tape.QuantumScript.from_queue(queue)
    assert qml.equal(expected_tape, executed_tape)


@pytest.mark.parametrize("measurement_process", (qml.expval(qml.Z(0)), qml.state(), qml.probs()))
@patch.object(PLDB, "_execute")
def test_measure(mock_method, measurement_process):
    """Test that the private measure function doesn't modify the active queue"""
    with qml.queuing.AnnotatedQueue() as queue:
        ops = [qml.X(0), qml.Y(1), qml.Z(0)] + [qml.Hadamard(i) for i in range(3)]
        measurements = [qml.expval(qml.X(2)), qml.state(), qml.probs(), qml.var(qml.Z(3))]
        qml.debugging.debugger._measure(measurement_process)

    executed_tape = qml.tape.QuantumScript.from_queue(queue)
    expected_tape = qml.tape.QuantumScript(ops, measurements)

    assert qml.equal(expected_tape, executed_tape)  # no unexpected queuing

    expected_debugging_tape = qml.tape.QuantumScript(ops, measurements + [measurement_process])
    executed_debugging_tape = mock_method.call_args.args[0][0]

    assert qml.equal(
        expected_debugging_tape, executed_debugging_tape
    )  # _execute was called with new measurements


@patch.object(PLDB, "_execute")
def test_state(_mock_method):
    """Test that the state function works as expected."""
    with qml.queuing.AnnotatedQueue() as queue:
        qml.RX(1.23, 0)
        qml.RY(0.45, 2)
        qml.sample()

        qml.debugging.state()

    assert qml.state() not in queue


@patch.object(PLDB, "_execute")
def test_expval(_mock_method):
    """Test that the expval function works as expected."""
    for op in [qml.X(0), qml.Y(1), qml.Z(2), qml.Hadamard(0)]:
        with qml.queuing.AnnotatedQueue() as queue:
            qml.RX(1.23, 0)
            qml.RY(0.45, 2)
            qml.sample()

            qml.debugging.expval(op)

        assert op not in queue
        assert qml.expval(op) not in queue


@patch.object(PLDB, "_execute")
def test_probs_with_op(_mock_method):
    """Test that the probs function works as expected."""

    for op in [None, qml.X(0), qml.Y(1), qml.Z(2)]:
        with qml.queuing.AnnotatedQueue() as queue:
            qml.RX(1.23, 0)
            qml.RY(0.45, 2)
            qml.sample()

            qml.debugging.probs(op=op)

        assert op not in queue
        assert qml.probs(op=op) not in queue


@patch.object(PLDB, "_execute")
def test_probs_with_wires(_mock_method):
    """Test that the probs function works as expected."""

    for wires in [None, [0, 1], [2]]:
        with qml.queuing.AnnotatedQueue() as queue:
            qml.RX(1.23, 0)
            qml.RY(0.45, 2)
            qml.sample()

            qml.debugging.probs(wires=wires)

        assert qml.probs(wires=wires) not in queue


@pytest.mark.parametrize("device_name", ("default.qubit", "lightning.qubit"))
def test_pldb_device_manager(device_name):
    """Test that the context manager works as expected."""
    assert not PLDB.has_active_dev()
    dev = qml.device(device_name, wires=2)

    with pldb_device_manager(dev) as _:
        assert PLDB.get_active_device() == dev

    assert not PLDB.has_active_dev()


@patch.object(PLDB, "set_trace")
def test_breakpoint_integration(mock_method):
    """Test that qml.breakpoint behaves as expected"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def my_circ():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.breakpoint()
        return qml.expval(qml.Z(1))

    mock_method.assert_not_called()  # Did not hit breakpoint
    my_circ()
    mock_method.assert_called_once()  # Hit breakpoint once.


@patch.object(PLDB, "set_trace")
def test_breakpoint_integration_with_valid_context_error(mock_method):
    """Test that the PLDB.valid_context() integrates well with qml.breakpoint"""
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def my_circ():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.breakpoint()
        return qml.expval(qml.Z(1))

    with pytest.raises(TypeError, match="Breakpoints not supported on this device"):
        _ = my_circ()

    mock_method.assert_not_called()  # Error was raised before we triggered breakpoint
