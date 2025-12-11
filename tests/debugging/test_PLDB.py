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
from pennylane.debugging import PLDB, pldb_device_manager


# pylint: disable=protected-access
class TestPLDB:
    """Test the interactive debugging integration"""

    def test_pldb_init(self):
        """Test that PLDB initializes correctly"""
        debugger = PLDB()
        assert debugger.prompt == "[pldb] "
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
            return qml.expval(qml.Z(0))

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
        """Test that we can reset the global active device list."""
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
        np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex),
        np.array(-1),
        np.array([1 / 2, 0, 0, 1 / 2]),
        np.array([1 / 2, 1 / 2]),
        np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0], dtype=complex),
    )

    @pytest.mark.parametrize("tape, expected_result", zip(tapes, results))
    @pytest.mark.parametrize(
        "dev", (qml.device("default.qubit", wires=2), qml.device("lightning.qubit", wires=2))
    )
    def test_execute(self, dev, tape, expected_result):
        """Test that the _execute method works as expected."""
        PLDB.add_device(dev)
        executed_results = PLDB._execute((tape,))
        assert np.allclose(expected_result, executed_results)
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

        executed_tape = qml.debug_tape()

    expected_tape = qml.tape.QuantumScript.from_queue(queue)
    qml.assert_equal(expected_tape, executed_tape)


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

    qml.assert_equal(expected_tape, executed_tape)  # no unexpected queuing

    expected_debugging_tape = qml.tape.QuantumScript(ops, measurements + [measurement_process])
    executed_debugging_tape = mock_method.call_args.args[0][0]

    qml.assert_equal(
        expected_debugging_tape, executed_debugging_tape
    )  # _execute was called with new measurements


@patch.object(PLDB, "_execute")
def test_state(_mock_method):
    """Test that the state function works as expected."""
    with qml.queuing.AnnotatedQueue() as queue:
        qml.RX(1.23, 0)
        qml.RY(0.45, 2)
        qml.sample()

        qml.debug_state()

    assert qml.state() not in queue


@patch.object(PLDB, "_execute")
def test_expval(_mock_method):
    """Test that the expval function works as expected."""
    for op in [qml.X(0), qml.Y(1), qml.Z(2), qml.Hadamard(0)]:
        with qml.queuing.AnnotatedQueue() as queue:
            qml.RX(1.23, 0)
            qml.RY(0.45, 2)
            qml.sample()

            qml.debug_expval(op)

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

            qml.debug_probs(op=op)

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

            qml.debug_probs(wires=wires)

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
