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
r"""
This module contains a constructor and classes for tracking device execution
information.
"""


class DefaultTracker:
    """Default base class for device trackers.

    Args:
        dev (Device): a PennyLane compatible device
        callback=None (callable or None): a function of the keywords ``totals``,
            ``history`` and ``latest``.  Run on each ``record`` call with current values of
            the corresponding attributes.

    Keyword Args:
        persistent=False (bool): whether to reset stored information upon entering
            a runtime context.

    **Example**

    ..code-block :: python

        dev = qml.device('default.qubit', wires=1. shots=10)

        @qml.qnode(dev, wires=1, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        with qml.device_tracker.DefaultTracker(circuit.device) as tracker:
            qml.grad(circuit)(0.1)
            circuit(0.1, shots=100)

    >>> tracker.totals
    {'executions': 4, 'shots': 130}
    >>> tracker.history
    {'executions': [1, 1, 1, 1], 'shots': [10, 10, 10, 100]}
    >>> tracker.latest
    {'executions': 1, 'shots': 100}

    """

    def __init__(self, dev=None, callback=None, persistent=False):
        self.persistent = persistent

        self.callback = callback

        # same code as self.reset
        self.totals = dict()
        self.history = dict()
        self.latest = dict()

        self.tracking = False

        if dev is not None:
            if not dev.capabilities().get("supports_tracker", False):
                raise Exception(f"Device '{dev.short_name}' does not support device tracking")
            dev.tracker = self

    def __enter__(self):
        if not self.persistent:
            self.reset()

        self.tracking = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.tracking = False

    def update(self, **kwargs):
        """Store passed keyword-value pairs into ``totals``,``history``, and ``latest`` attributes.

        There is no restriction on the key-value pairs passed.

        >>> tracker.update(a=1, b=2)
        >>> tracker.latest
        {"a":1, "b":2}
        >>> tracker.history
        {"a": [1], "b": [2]}
        >>> tracker.totals
        {"a": 1, "b": 2}

        """
        self.latest = kwargs

        for key, value in kwargs.items():
            # update history
            if key in self.history.keys():
                self.history[key].append(value)
            else:
                self.history[key] = [value]

            # updating totals
            if value is not None:
                # Only total numeric values
                try:
                    self.totals[key] = value + self.totals.get(key, 0)
                except TypeError:
                    pass

    def reset(self):
        """Resets stored information."""
        self.totals = dict()
        self.history = dict()
        self.latest = dict()

    def record(self):
        """If a ``callback`` is passed to the class upon initialization, it is called."""
        if self.callback is not None:
            self.callback(totals=self.totals, history=self.history, latest=self.latest)


def track(dev=None, **kwargs):
    r"""Creates a tracking context and applies it to a device.

    Args:
        dev (Device): a PennyLane-compatible device.

    Keyword Args:
        callback=None (callable or None): This function is used to record information. Must be a
            function of ``totals``, ``history``, and ``latest`` keywords.
        persistent=False (bool): whether or not to reset information
            entering the context

    **Example**

    With the default settings on most devices, the tracker will store execution and shot information without
    printing or logging of the information.  This information can be accessed through the ``totals``, ``history``,
    and ``latest`` attributes of the tracker.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1, shots=100)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1)

        with qml.track(dev) as tracker:
            qml.grad(circuit)(0.1)

    You can then access the tabulated information through ``totals``, ``history``, and ``latest``:

    >>> tracker.totals
    {'executions': 3, 'shots': 300}
    >>> tracker.history
    {'executions': [1, 1, 1], 'shots': [100, 100, 100]}
    >>> tracker.latest
    {'executions': 1, 'shots': 100}

    .. UsageDetails::

    .. note::
        With backpropagation, this function should take ``qnode.device``
        instead of the device used to create the QNode.

    Users can pass a custom callback function to the ``callback`` keyword. This
    function is run each time ``record()`` is called. The
    function passed must accept ``totals``, ``history``, and ``latest`` as
    keyword arguments:

    >>> def shots_info(totals=dict(), history=dict(), latest=dict()):
    ...     print("Total shots: ", totals['shots'])
    >>> with qml.track(circuit.device, callback=shots_info) as tracker:
    ...     qml.grad(circuit)(0.1)
    Total shots:  100
    Total shots:  200
    Total shots:  300

    By specifying ``persistent=False``, you can reuse the same tracker accross
    multiple runtime contexts.

    >>> with qml.track(circuit.device, persistent=False) as tracker:
    ...     circuit(0.1)
    >>> with tracker:
    ...     circuit(0.2)
    >>> tracker.totals['executions']
    2

    """
    # New features may be added later
    # Hence why this function exists

    return DefaultTracker(dev, **kwargs)
