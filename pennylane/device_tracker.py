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

import time
from collections import defaultdict

class DefaultTracker:
    """
    Class docstring
    """

    def __init__(self, dev=None, reset_on_enter=True):
        """
        docstring
        """
        self.reset_on_enter = reset_on_enter

        self.reset()
        self.tracking = False

        if dev is not None:
            dev.tracker = self

    def __enter__(self):
        """
        docstring for enter
        """
        if self.reset_on_enter:
            self.reset()

        self.tracking = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        docstring for exit
        """
        self.tracking = False

    def update(self, **current):
        """updating data"""
        self.current = current

        for key, value in current.items():
            # update history
            self.history[key].append(value)

            # updating totals
            if value is not None:
                self.totals[key] += value

    def reset(self):
        """reseting data"""
        self.totals = defaultdict(int)
        self.history = defaultdict(list)
        self.current = dict()

    def record(self):
        """
        record data somehow
        """
        pass

class TimingsMixin(DefaultTracker):

    def update(self, **current):

        current_time = time.time()
        current["time"] = current_time - self._time_last
        self._time_last = current_time

        super(TimingsMixin, self).update(**current)

    def reset(self):
        super(TimingsMixin, self).reset()
        self._time_last = time.time()

class PrintTotals(DefaultTracker):

    def record(self):
        """
        Print out totals 
        """
        print("Total: ", end="")
        for key, value in self.totals.items():
            print(f"{key} = {value}", end="\t")
        print()

class PrintCurrent(DefaultTracker):

    def record(self):
        """
        Print out current values
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
        """
        Use user provided logging function
        """
        self.custom_recorder(current=self.current, totals=self.totals, history=self.history)

record_mapping = {"totals": PrintTotals, "current": PrintCurrent, "custom": PrintCustom}
update_mapping = {"timings": TimingsMixin}

def track(dev, record=None, update=None, **kwargs):
    r"""Creates a tracking context and applies it to a device.

    Args:
        dev (Device): a PennyLane-compatible device
        version (str): name of tracker to use.  The current options are
            `default` and `timing`.

    Keyword Args:
        reset_on_enter=True (bool): whether or not to reset information
            entering the context

    **Usage Information**

    Note that with backpropagation, this functions should take ``qnode.device``
    instead of the device used to create the QNode.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

    With the default version, total execution information is printed on
    each device execution.  The printed data depends on the device and tracker version,
    but for standard PennyLane devices, the object will track executions and shots.

    >>> with qml.track(circuit.device) as tracker:
    ...    qml.grad(circuit)(0.1, shots=10)
    Total: executions = 1	shots = 10
    Total: executions = 2	shots = 20
    Total: executions = 3	shots = 30

    In with the ``'timing'`` implementation, the instance also tracks the time
    between entering the context and the completion of an execution.

    >>> with qml.track(circuit.device, version='timing') as timing_tracker:
    ...    circuit(0.1)
    ...    circuit(0.2)
    Total: executions = 1	time = 0.0011134147644042969
    Total: executions = 2	time = 0.0027322769165039062

    After completion, one can also access the recorded information:

    >>> timing_tracker.totals
    defaultdict(int, {'executions': 2, 'shots': 30, 'time': 0.00311279296875})
    >>> timing_tracker.history
    defaultdict(list,
            {'executions': [1, 1],
             'shots': [None, None],
             'time': [0.0012764930725097656, 0.0018362998962402344]})

    By specifying ``reset_on_enter=False``, you can reuse the same tracker accross
    multiple runtime contexts.

    >>> with qml.track(circuit.device, reset_on_enter=False) as tracker:
    ...     circuit(0.1)
    Total: executions = 1
    >>> with tracker:
    ...     circuit(0.2)
    Total: executions = 2

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

    class temp_class(*mixin_list):
        pass

    if callable(record):
        tracker = temp_class(dev, custom_recorder=record, **kwargs)
    else:
        tracker = temp_class(dev, **kwargs)

    return tracker

