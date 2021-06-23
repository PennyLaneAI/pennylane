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
This module contains the device tracker stuff.
"""

import time

class DefaultTracker:
    """Default base class for device trackers.

    Args:
        dev (Device): a PennyLane compatible device

    Keyword Args:
        reset_on_enter=True : whether to reset stored information upon entering 
            a runtime context.
    """

    def __init__(self, dev=None, reset_on_enter=True):
        self.reset_on_enter = reset_on_enter

        self.reset()
        self.tracking = False

        if dev is not None:
            dev.tracker = self

    def __enter__(self):
        if self.reset_on_enter:
            self.reset()

        self.tracking = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.tracking = False

    def update(self, **current):
        """Store passed keyword-value pairs into ``totals``,``history``, and ``current`` attributes.

        There is no restriction on information passed to this function, other than python
        must be able to add the value to `0`.

        >>> tracker.update(a=1, b=2)
        >>> tracker.current
        {"a":1, "b":2}
        >>> tracker.history
        {"a": [1], "b": [2]}
        >>> tracker.totals
        {"a": 1, "b": 2}

        """
        self.current = current

        for key, value in current.items():
            # update history
            if key in self.history.keys():
                self.history[key].append(value)
            else:
                self.history[key] = [value]

            # updating totals
            if value is not None:
                self.totals[key] = value + self.totals.get(key, 0)

    def reset(self):
        """Resets stored information."""
        self.totals = dict()
        self.history = dict()
        self.current = dict()

    def record(self):
        """Move stored information to some other location.

        While blank for the default base class, inheriting classes can print or log the data.
        """
        pass

class UpdateTimings(DefaultTracker):

    def update(self, **current):
        current_time = time.time()
        current["time"] = current_time - self._time_last
        self._time_last = current_time

        super(UpdateTimings, self).update(**current)

    def reset(self):
        super(UpdateTimings, self).reset()
        self._time_last = time.time()

class PrintTotals(DefaultTracker):

    def record(self):
        """Print all key-value pairs stored in ``totals``.
        """
        print("Total: ", end="")
        for key, value in self.totals.items():
            print(f"{key} = {value}", end="\t")
        print()

class PrintCurrent(DefaultTracker):

    def record(self):
        """Print all key-value pairs stored in ``current``.
        """
        print("Current: ", end="")
        for key, value in self.current.items():
            print(f"{key} = {value}", end="\t")
        print()

class PrintCustom(DefaultTracker):
    def __init__(self, dev=None, reset_on_enter=True, custom_recorder=None):
        self.custom_recorder=custom_recorder
        super(PrintCustom, self).__init__(dev=dev, reset_on_enter=reset_on_enter)

    def record(self):
        """Executes user-provided record function."""
        self.custom_recorder(totals=self.totals, history=self.history, current=self.current,)

record_mapping = {"totals": PrintTotals, "current": PrintCurrent, "custom": PrintCustom}
update_mapping = {"timings": UpdateTimings}

def track(dev, record=None, update=None, **kwargs):
    r"""Creates a tracking context and applies it to a device.

    Args:
        dev (Device): a PennyLane-compatible device
        record (callable or str or None): If callable, this function is used to record information. Must be a
            function of ``current``, ``totals`` and ``history`` keywords. If string, selects an built record method.
            Current available options are ``"totals"`` and ``"current"``. If ``None``, no recording happens.
        update (str or None): if ``"timings"``, the update method will also store the length of system time between
            subsequent `update` calls.

    Keyword Args:
        reset_on_enter=True (bool): whether or not to reset information
            entering the context

    **Example**

    With the default settings on most devices, the tracker will store execution and shot information without 
    printing or logging of the information.  This information can be accessed through the `totals`, `history`,
    and `current` attributes of the tracker.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1, shots=100)
        
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        x = np.array(0.1)

        with qml.track(dev) as tracker:
            qml.grad(circuit)(0.1)

        You can then access the tabulated information through ``totals``, ``history``, and ``current``:

        >>> tracker.totals
        {'executions': 3, 'shots': 300}
        >>> tracker.history
        {'executions': [1, 1, 1], 'shots': [100, 100, 100]}
        >>> tracker.current
        {'executions': 1, 'shots': 100}

    .. UsageDetails::

    .. note::
        With backpropagation, this functions should take ``qnode.device``
        instead of the device used to create the QNode.

    For a print out of information on each execution, use one of the ``"record"``
    keywords. You can use the `"totals"` keyword:

    >>> with qml.track(circuit.device, record="totals") as tracker:
    ...    qml.grad(circuit)(0.1)
    Total: executions = 1	shots = 100	
    Total: executions = 2	shots = 200	
    Total: executions = 3	shots = 300

    or the ``"current"`` keyword:

    >>> with qml.track(circuit.device, record="current") as tracker:
    ...    qml.grad(circuit)(0.1)
    Current: executions = 1	shots = 100	
    Current: executions = 1	shots = 100	
    Current: executions = 1	shots = 100	

    Users can also pass a custom record function to the record keyword.  The 
    function passed must accept ``totals``, ``history``, and ``current`` as 
    keyword arguments:

    >>> def just_shot_info(totals=dict(), history=dict(), current=dict()):
    ...     print("Totals shots: ", totals['shots'])
    >>> with qml.track(circuit.device, record=just_shot_info) as tracker:
    ...     qml.grad(circuit)(0.1)
    Totals shots:  100
    Totals shots:  200
    Totals shots:  300

    By passing ``update="timings"``, the tracker also stores the time difference between
    `update` calls.

    >>> with qml.track(circuit.device, update="timings") as timing_tracker:
    ...    circuit(0.1)
    ...    circuit(0.2)
    >>> timing_tracker.history['time']
    [0.0010597705841064453, 0.0011420249938964844]

    By specifying ``reset_on_enter=False``, you can reuse the same tracker accross
    multiple runtime contexts.

    >>> with qml.track(circuit.device, reset_on_enter=False) as tracker:
    ...     circuit(0.1)
    >>> with tracker:
    ...     circuit(0.2)
    >>> tracker.totals['executions']
    2

    """
    mixin_list = []

    if update is not None:
        update_class = update_mapping.get(update)
        mixin_list.append(update_class)

    if callable(record):
        mixin_list.append(record_mapping.get("custom"))
    elif record is not None:
        record_class = record_mapping.get(record)
        mixin_list.append(record_class)
    
    mixin_list.append(DefaultTracker)

    class Tracker(*mixin_list):
        pass

    if callable(record):
        tracker = Tracker(dev, custom_recorder=record, **kwargs)
        
    else:
        tracker = Tracker(dev, **kwargs)

    return tracker

