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
This module contains a class for updating and recording information about device executions.
"""


class Tracker:
    """This class stores information about device executions and allows users to interact with that
    data upon each executions, even within parameter-shift gradients and optimization steps.

    The information is stored in three class attributes: ``totals``, ``history``, and ``latest``.
    Standard devices will track the number of executions, number of shots, and number of batch'
    executions, but plugins may store additional information with no changes to this class.

    Information is only stored when the class attribute ``tracking`` is set to ``True``. This
    attribute can be toggled via a context manager and Python's ``with`` statement. Upon entering a
    context, the stored information is reset, unless ``persistent=True``.

    Args:
        dev (Device): a PennyLane compatible device
        callback=None (callable or None): a function of the keywords ``totals``,
            ``history`` and ``latest``.  Run on each ``record`` call with current values of
            the corresponding attributes.
        persistent=False (bool): whether to reset stored information upon
            entering a runtime context.


    **Example**

    Using a ``with`` statement to toggle the tracking mode, we can see the number of executions
    and shots used to calculate a parameter-shift derivative.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1, shots=100)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1)

        with qml.Tracker(dev) as tracker:
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
    function is run each time the ``record()`` method is called. The
    function passed must accept ``totals``, ``history``, and ``latest`` as
    keyword arguments:

    >>> def shots_info(totals=dict(), history=dict(), latest=dict()):
    ...     print("Total shots: ", totals['shots'])
    >>> with qml.Tracker(circuit.device, callback=shots_info) as tracker:
    ...     qml.grad(circuit)(0.1)
    Total shots:  100
    Total shots:  200
    Total shots:  300

    By specifying ``persistent=False``, you can reuse the same tracker accross
    multiple contexts.

    >>> with qml.Tracker(circuit.device, persistent=False) as tracker:
    ...     circuit(0.1)
    >>> with tracker:
    ...     circuit(0.2)
    >>> tracker.totals['executions']
    2

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
