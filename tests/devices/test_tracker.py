# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the Tracker and constructor
"""
# pylint: disable=use-implicit-booleaness-not-comparison
import pytest

import pennylane as qml
from pennylane import Tracker


class TestTrackerCoreBehavior:
    """Unittests for the tracker class"""

    def test_default_initialization(self):
        """Tests default initializalition"""

        tracker = Tracker()

        assert tracker.persistent is False
        assert tracker.callback is None

        assert tracker.history == {}
        assert tracker.totals == {}
        assert tracker.latest == {}

        assert tracker.active is False

    def test_device_assignment(self):
        """Assert gets assigned to device"""

        # pylint: disable=too-few-public-methods
        class TempDevice:
            short_name = "temp"

            def __init__(self):
                self.tracker = Tracker()

        temp = TempDevice()

        tracker = Tracker(dev=temp)

        assert id(temp.tracker) == id(tracker)

    def test_incompatible_device_assignment_no_tracker(self):
        """Assert exception raised when `supports_tracker` not True"""

        # pylint: disable=too-few-public-methods
        class TempDevice:
            short_name = "temp"

            def capabilities(self):
                return {}

        temp = TempDevice()

        with pytest.raises(Exception, match=r"Device 'temp' does not support device tracking"):
            Tracker(dev=temp)

    def test_incompatible_device_assignment_explicit_false(self):
        """Assert exception raised when `supports_tracker` is False"""

        # pylint: disable=too-few-public-methods
        class TempDevice:
            short_name = "temp"

        temp = TempDevice()

        with pytest.raises(Exception, match=r"Device 'temp' does not support device tracking"):
            Tracker(dev=temp)

    def test_reset(self):
        """Assert reset empties totals, history and latest"""

        tracker = Tracker()

        tracker.totals = {"a": 1}
        tracker.history = {"a": [1]}
        tracker.latest = {"a": 1}

        tracker.reset()

        assert tracker.totals == {}
        assert tracker.history == {}
        assert tracker.latest == {}

    def test_enter_and_exit(self):
        """Assert entering and exit work as expected"""

        tracker = Tracker()
        tracker.totals = {"a": 1}
        tracker.history = {"a": [1]}
        tracker.latest = {"a": 1}

        returned = tracker.__enter__()  # pylint: disable=unnecessary-dunder-call

        assert id(tracker) == id(returned)
        assert tracker.active is True

        assert tracker.totals == {}
        assert tracker.history == {}
        assert tracker.latest == {}

        tracker.__exit__(None, None, None)

        assert tracker.active is False

    def test_context(self):
        """Assert works with runtime context"""

        with Tracker() as tracker:
            assert isinstance(tracker, Tracker)
            assert tracker.active is True

        assert tracker.active is False

    def test_update(self):
        """Checks update stores to history and totals"""

        tracker = Tracker()

        tracker.update(a=1, b="b", c=None)
        tracker.update(a=2, b="b2", c=1)

        assert tracker.history == {"a": [1, 2], "b": ["b", "b2"], "c": [None, 1]}

        assert tracker.totals == {"a": 3, "c": 1}

        assert tracker.latest == {"a": 2, "b": "b2", "c": 1}

    def test_record_callback(self, mocker):
        # pylint: disable=too-few-public-methods
        class callback_wrapper:
            @staticmethod
            def callback(totals, history, latest):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        tracker = Tracker(callback=wrapper.callback)

        tracker.totals = {"a": 1, "b": 2}
        tracker.history = {"a": [1], "b": [1, 1]}
        tracker.latest = {"a": 1, "b": 1}

        tracker.record()

        _, kwargs_called = spy.call_args_list[-1]

        assert kwargs_called["totals"] == tracker.totals
        assert kwargs_called["history"] == tracker.history
        assert kwargs_called["latest"] == tracker.latest


# Integration test definitions

dev_qubit = qml.device("default.qubit", wires=1)


@qml.qnode(dev_qubit)
def circuit_qubit():
    return qml.expval(qml.PauliZ(0))


dev_gaussian = qml.device("default.gaussian", wires=1)


@qml.qnode(dev_gaussian)
def circuit_gaussian():
    return qml.expval(qml.QuadX(0))


@pytest.mark.xfail
@pytest.mark.parametrize("circuit", (circuit_qubit, circuit_gaussian))
class TestDefaultTrackerIntegration:
    """Tests integration behavior with 'default.gaussian'.

    Integration with several `QubitDevice`-inherited devices are tested in the
    device suite. Using `default.gaussian`, we test one that inherits from `Device`.
    """

    def test_single_execution_default(self, circuit, mocker):
        """Test correct behavior with single circuit execution"""

        # pylint: disable=too-few-public-methods
        class callback_wrapper:
            @staticmethod
            def callback(totals=None, history=None, latest=None):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        with Tracker(circuit.device, callback=wrapper.callback) as tracker:
            circuit()
            circuit()

        assert tracker.totals == {"executions": 2}
        assert tracker.history == {"executions": [1, 1], "shots": [None, None]}
        assert tracker.latest == {"executions": 1, "shots": None}

        _, kwargs_called = spy.call_args_list[-1]

        assert kwargs_called["totals"] == {"executions": 2}
        assert kwargs_called["history"] == {"executions": [1, 1], "shots": [None, None]}
        assert kwargs_called["latest"] == {"executions": 1, "shots": None}

    def test_shots_execution_default(self, circuit, mocker):
        """Test correct tracks shots as well."""

        # pylint: disable=too-few-public-methods
        class callback_wrapper:
            @staticmethod
            def callback(totals=None, history=None, latest=None):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        with Tracker(circuit.device, callback=wrapper.callback) as tracker:
            circuit(shots=10)
            circuit(shots=20)

        assert tracker.totals == {"executions": 2, "shots": 30}
        assert tracker.history == {"executions": [1, 1], "shots": [10, 20]}
        assert tracker.latest == {"executions": 1, "shots": 20}

        assert spy.call_count == 2

        _, kwargs_called = spy.call_args_list[-1]
        assert kwargs_called["totals"] == {"executions": 2, "shots": 30}
        assert kwargs_called["history"] == {"executions": [1, 1], "shots": [10, 20]}
        assert kwargs_called["latest"] == {"executions": 1, "shots": 20}

    def test_batch_execution(self, circuit, mocker):
        """Tests that batch execute also updates information stored."""

        # pylint: disable=too-few-public-methods
        class callback_wrapper:
            @staticmethod
            def callback(totals=None, history=None, latest=None):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        tape = qml.workflow.construct_tape(circuit)()

        with Tracker(circuit.device, callback=wrapper.callback) as tracker:
            circuit.device.batch_execute([tape, tape])

        assert tracker.totals == {"executions": 2, "batches": 1, "batch_len": 2}
        assert tracker.history == {
            "executions": [1, 1],
            "shots": [None, None],
            "batches": [1],
            "batch_len": [2],
        }
        assert tracker.latest == {"batches": 1, "batch_len": 2}

        _, kwargs_called = spy.call_args_list[-1]
        assert kwargs_called["totals"] == {"executions": 2, "batches": 1, "batch_len": 2}
        assert kwargs_called["history"] == {
            "executions": [1, 1],
            "shots": [None, None],
            "batches": [1],
            "batch_len": [2],
        }
        assert kwargs_called["latest"] == {"batches": 1, "batch_len": 2}
