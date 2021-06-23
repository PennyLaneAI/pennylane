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
Unit tests for the DeviceTracker and constructor
"""
import pytest
from collections import defaultdict
from itertools import count
import time

import pennylane as qml
from pennylane import track
from pennylane.device_tracker import DefaultTracker


def custom_recorder(totals={}, history={}, current={}):
    print("Hello world")

@pytest.mark.parametrize("record", (None, "totals", "current", custom_recorder))
@pytest.mark.parametrize("update", (None, "timings"))
class TestTrackerCoreBehaviour:
    def test_default_initialization(self, record, update):
        """Tests default initializalition"""

        tracker = track(record=record, update=update)

        assert tracker.reset_on_enter == True
        assert tracker.tracking == False
        assert tracker.history == dict()
        assert tracker.totals == dict()
        assert tracker.current == dict()

    def test_device_assignment(self, record, update):
        """Assert gets assigned to device"""
        dev = qml.device("default.qubit", wires=2)

        tracker = track(dev=dev, record=record, update=update)

        assert id(dev.tracker) == id(tracker)

    def test_reset(self, record, update):
        """Assert reset empties totals and history"""

        tracker = track(record=record, update=update)

        tracker.totals = {"a": 1}
        tracker.history = {"a": [1]}
        tracker.current = {"a": 1}

        tracker.reset()

        assert tracker.totals == dict()
        assert tracker.history == dict()
        assert tracker.current == dict()

    def test_enter_and_exit(self, record=record, update=update):
        """Assert entering and exit work as expected"""

        tracker = track(record=record, update=update)
        tracker.totals = {"a": 1}
        tracker.history = {"a": [1]}
        tracker.current = {"a": 1}

        returned = tracker.__enter__()

        assert id(tracker) == id(returned)
        assert tracker.tracking == True

        assert tracker.totals == dict()
        assert tracker.history == dict()
        assert tracker.current == dict()

        tracker.__exit__(1, 1, 1)

        assert tracker.tracking == False

    def test_context(self, record, update):
        """Assert works with runtime context"""

        with track(record=record, update=update) as tracker:
            assert isinstance(tracker, DefaultTracker)
            assert tracker.tracking == True

        assert tracker.tracking == False

    def test_update(self, record, update):
        """Checks update stores to history and totals"""

        tracker = track(record=record, update=update)

        tracker.update(a=1, b=2, c=None)

        assert tracker.history["a"] == [1]
        assert tracker.history["b"] == [2]
        assert tracker.history["c"] == [None]

        assert tracker.totals["a"] == 1
        assert tracker.totals["b"] == 2
        assert tracker.totals["c"] == 0


class TestRecordOptions:

    def test_None(self, capsys):
        """Check record prints information properly"""

        tracker = track()
        tracker.totals = {"a": 1, "b": 2}
        tracker.record()

        captured = capsys.readouterr()

        predicted = ""

        assert captured.out == predicted

    def test_totals(self, capsys):
        """Check record totals prints correctly"""

        tracker = track(record="totals")
        tracker.totals = {"a": 1, "b":2}

        tracker.record()

        captured = capsys.readouterr()

        predicted = "Total: a = 1\tb = 2\t\n"

        assert captured.out == predicted
    
    def test_current(self, capsys):
        """Check record current print correctly"""

        tracker = track(record="current")
        tracker.current = {"a": 1, "b": 2}

        tracker.record()

        captured = capsys.readouterr()

        predicted = "Current: a = 1\tb = 2\t\n"

        assert caputred.out == predicted

    def test_custom(self, capsys):
        """Test custom record works"""

        def custom(totals = {}, history={}, current={}):
            global gtotals = totals
            global ghistory = history
            global gcurrent = current
            print("hello world")

        tracker = track(record=custom)
        tracker.totals = {"a":1}
        tracker.history = {"a": [1]}
        tracker.current = {"a": 1}

        tracker.record()
        captured = capsys.readouterr()

        assert captured.out == "hello world"
        assert gtotals == tracker.totals
        assert ghistory == tracker.history
        assert gcurrent == tracker.current
        

class TestTimeTracker:
    def test_time(self, monkeypatch):

        # each time mocker called, increments return value by one
        counter = count()

        def mocktime():
            return float(next(counter))

        monkeypatch.setattr(time, "time", mocktime)

        tracker = TimingTracker()

        # reset upon initialization
        assert tracker._time_last == 0.0

        with tracker:
            # got reset upon enter
            assert tracker._time_last == 1.0

            tracker.update(a=1)

            # called again at update
            assert tracker._time_last == 2.0
            assert tracker.totals == {"a": 1, "time": 1.0}
            assert tracker.history == {"a": [1], "time": [1.0]}

            tracker.update(b=1)

            # called again
            assert tracker._time_last == 3.0
            assert tracker.totals == {"a": 1, "b": 1, "time": 2.0}
            assert tracker.history == {"a": [1], "b": [1], "time": [1.0, 1.0]}

        tracker.reset()
        # incremented once again
        assert tracker._time_last == 4.0


class TestDefaultTrackerIntegration:
    def test_single_execution_default(self, capsys):
        """Test correct behaviour with single circuit execution"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        with DefaultTracker(circuit.device) as tracker:
            circuit()

        captured = capsys.readouterr()

        predicted_text = "Total: executions = 1\t\n"
        assert captured.out == predicted_text
        assert tracker.totals == {"executions": 1}
        assert tracker.history == {"executions": [1], "shots": [None]}

    def test_shots_execution_default(self, capsys):
        """Test correct tracks shots as well."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        with DefaultTracker(circuit.device) as tracker:
            circuit(shots=10)
            circuit(shots=20)

        captured = capsys.readouterr()

        predicted_text = "Total: executions = 1\tshots = 10\t\nTotal: executions = 2\tshots = 30\t\n"
        assert captured.out == predicted_text
        assert tracker.totals == {"executions": 2, "shots": 30}
        assert tracker.history == {"executions": [1,1], "shots": [10, 20]}
