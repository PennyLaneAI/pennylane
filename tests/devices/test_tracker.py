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

import numpy as np

# pylint: disable=use-implicit-booleaness-not-comparison
import pytest

import pennylane as qp
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

    def test_repr(self):
        """Tests the string representation contains all relevant fields with correct values."""
        tracker = Tracker()
        r = repr(tracker)

        assert r.startswith("Tracker(")
        assert f"active={tracker.active}" in r
        assert f"totals={tracker.totals}" in r
        assert f"history={tracker.history}" in r
        assert f"latest={tracker.latest}" in r
        assert f"persistent={tracker.persistent}" in r
        assert f"callback={tracker.callback!r}" in r

        tracker.update(a=2, b="b2", c=1)
        r2 = repr(tracker)

        assert f"active={tracker.active}" in r2
        assert f"totals={tracker.totals}" in r2
        assert f"history={tracker.history}" in r2
        assert f"latest={tracker.latest}" in r2
        assert f"persistent={tracker.persistent}" in r2
        assert f"callback={tracker.callback!r}" in r2

    def test_repr_with_callback(self):
        """Tests that the callback is shown correctly in the repr."""

        def my_callback(_totals, _history, _latest):
            pass  # pragma: no cover

        tracker = Tracker(callback=my_callback)
        r = repr(tracker)
        assert f"callback={my_callback!r}" in r

    def test_repr_active(self):
        """Tests that active state is reflected in the repr."""
        tracker = Tracker()
        with tracker:
            r = repr(tracker)
            assert "active=True" in r
        r = repr(tracker)
        assert "active=False" in r

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

dev_qubit = qp.device("default.qubit", wires=1)


@qp.qnode(dev_qubit)
def circuit():
    return qp.expval(qp.PauliZ(0))


class TestDefaultTrackerIntegration:
    """Tests integration behavior with 'default.gaussian'.

    Integration with several `QubitDevice`-inherited devices are tested in the
    device suite. Using `default.gaussian`, we test one that inherits from `Device`.
    """

    def test_single_execution_default(self, mocker):
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

        assert tracker.totals == {"executions": 2, "batches": 2, "results": 2, "simulations": 2}
        # lots of other things get tracked now
        assert tracker.history["executions"] == [1, 1]
        assert tracker.latest["executions"] == 1

        _, kwargs_called = spy.call_args_list[-1]

        assert kwargs_called["totals"] == {
            "executions": 2,
            "batches": 2,
            "results": 2,
            "simulations": 2,
        }
        assert kwargs_called["history"]["executions"] == [1, 1]
        assert kwargs_called["latest"]["executions"] == 1

    def test_shots_execution_default(self, mocker):
        """Test correct tracks shots as well."""

        # pylint: disable=too-few-public-methods
        class callback_wrapper:
            @staticmethod
            def callback(totals=None, history=None, latest=None):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        with Tracker(circuit.device, callback=wrapper.callback) as tracker:
            qp.set_shots(circuit, 10)()
            qp.set_shots(circuit, 20)()

        assert tracker.totals == {
            "executions": 2,
            "shots": 30,
            "batches": 2,
            "results": 2.0,
            "simulations": 2,
        }
        assert tracker.history["shots"] == [10, 20]
        assert tracker.latest["shots"] == 20

        assert spy.call_count == 4

        _, kwargs_called = spy.call_args_list[-1]
        assert kwargs_called["totals"] == {
            "executions": 2,
            "shots": 30,
            "batches": 2,
            "results": 2.0,
            "simulations": 2,
        }
        assert kwargs_called["history"]["shots"] == [10, 20]
        assert kwargs_called["latest"]["shots"] == 20

    def test_batch_execution(self, mocker):
        """Tests that batch execute also updates information stored."""

        # pylint: disable=too-few-public-methods
        class callback_wrapper:
            @staticmethod
            def callback(totals=None, history=None, latest=None):
                pass

        wrapper = callback_wrapper()
        spy = mocker.spy(wrapper, "callback")

        tape = qp.workflow.construct_tape(circuit)()

        with Tracker(circuit.device, callback=wrapper.callback) as tracker:
            circuit.device.execute([tape, tape])

        assert tracker.totals == {
            "executions": 2,
            "batches": 1,
            "results": np.float64(2.0),
            "simulations": 2,
        }
        assert tracker.history["executions"] == [1, 1]
        assert tracker.history["batches"] == [1]
        assert tracker.latest["simulations"] == 1

        _, kwargs_called = spy.call_args_list[-1]
        assert kwargs_called["totals"] == {
            "executions": 2,
            "batches": 1,
            "results": np.float64(2.0),
            "simulations": 2,
        }
        assert kwargs_called["history"]["executions"] == [1, 1]
        assert kwargs_called["latest"]["simulations"] == 1
