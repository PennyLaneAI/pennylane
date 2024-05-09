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

# pylint: disable=attribute-defined-outside-init

from numbers import Number


class Tracker:
    """This class stores information about device executions and allows users to interact with that
    data upon individual executions and batches, even within parameter-shift gradients and
    optimization steps.

    The information is stored in three class attribute dictionaries: ``totals``, ``history``,
    and ``latest``:

    * ``latest`` tracks the last set of information passed to the tracker.
    * ``history`` stores a list of values passed for each keyword.
    * ``totals`` keeps a running sum per keyword when the values are numeric.

    Standard devices will track the number of executions, number of shots, number of batch
    executions, batch execution length, and results of circuit executions, but plugins may store
    additional information with no changes to this class.

    Information is only stored when the class attribute ``active`` is set to ``True``. This
    attribute can be toggled via a context manager and Python's ``with`` statement. Upon entering a
    context, the stored information is reset, unless ``persistent=True``. Tracking mode can also be
    manually triggered by setting ``tracker.active = True`` without the use of a context manager.

    Args:
        dev (Device): A PennyLane compatible device
        callback=None (callable or None): A function of the keywords ``totals``,
            ``history`` and ``latest``.  Run on each ``record`` call with current values of
            the corresponding attributes.
        persistent=False (bool): Whether to reset stored information upon
            entering a runtime context.


    **Example**

    Using a ``with`` statement to toggle the active mode, we can see the number of executions
    and shots used to calculate a parameter-shift derivative.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1, shots=100)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        x = np.array(0.1, requires_grad=True)

        with qml.Tracker(dev) as tracker:
            qml.grad(circuit)(x)

    You can then access the tabulated information through ``totals``, ``history``, and ``latest``:

    >>> tracker.totals
    {'batches': 2, 'simulations': 3, 'executions': 3, 'shots': 300}
    >>> tracker.latest
    {'simulations': 1,
     'executions': 1,
     'results': array(-0.08),
     'shots': 100,
     'resources': Resources(num_wires=1, num_gates=1,
                            gate_types=defaultdict(<class 'int'>, {'RX': 1}),
                            gate_sizes=defaultdict(<class 'int'>, {1: 1}),
                            depth=1,
                            shots=Shots(total_shots=100, shot_vector=(ShotCopies(100 shots x 1),))),
     'errors': {}
    }
    >>> tracker.history.keys()
    dict_keys(['batches', 'simulations', 'executions', 'results', 'shots', 'resources'])
    >>> tracker.history['results']
    [array(1.), array(0.02), array(-0.08)]
    >>> print(tracker.history['resources'][0])
    wires: 1
    gates: 1
    depth: 1
    shots: Shots(total=100)
    gate_types:
    {'RX': 1}
    gate_sizes:
    {1: 1}

    We can see that calculating the gradient of ``circuit`` takes three total evaluations: one
    forward pass and one batch of length two for the derivative of ``qml.RX``.

    .. details::
        :title: Usage Details

        .. note::
            With backpropagation, this function should take ``qnode.device``
            instead of the device used to create the QNode.

        Users can pass a custom callback function to the ``callback`` keyword. This
        function is run each time the ``record()`` method is called, which occurs near
        the end of a device's ``execute`` and ``batch_execute`` methods. Using ``print``
        or logging, users can monitor completion during a long set of jobs.

        The function passed must accept ``totals``, ``history``, and ``latest`` as
        keyword arguments. The dictionary ``latest`` will contain different keywords based on whether
        whether ``execute`` or ``batch_execute`` last performed an update.

        >>> def shots_info(totals, history, latest):
        ...     if 'shots' in latest:
        ...         print("Total shots: ", totals['shots'])
        >>> x = np.array(0.1, requires_grad=True)
        >>> with qml.Tracker(circuit.device, callback=shots_info) as tracker:
        ...     qml.grad(circuit)(x)
        Total shots:  100
        Total shots:  200
        Total shots:  300

        By specifying ``persistent=False``, you can reuse the same tracker across
        multiple contexts.

        >>> with qml.Tracker(circuit.device, persistent=False) as tracker:
        ...     circuit(0.1)
        >>> with tracker:
        ...     circuit(0.2)
        >>> tracker.totals['executions']
        2

        When used with the null qubit device (eg. ``dev = qml.device("null.qubit")``), we can track the resources
        used in the circuit without execution!

        >>> dev = qml.device("null.qubit", wires=[0], shots=10)
        >>> @qml.qnode(dev)
        ... def circuit(x):
        ...     qml.RX(x, wires=0)
        ...     return qml.expval(qml.Z(0))
        ...
        >>> with qml.Tracker(dev) as tracker:
        ...     circuit(0.1)
        ...
        >>> resources_lst = tracker.history['resources']
        >>> print(resources_lst[0])
        wires: 1
        gates: 1
        depth: 1
        shots: Shots(total=10)
        gate_types:
        {"RX": 1}
        gate_sizes:
        {1: 1}
    """

    def __init__(self, dev=None, callback=None, persistent=False):
        self.persistent = persistent

        self.callback = callback

        self.reset()

        self.active = False

        if dev is not None:
            if not hasattr(dev, "tracker"):
                raise ValueError(f"Device '{dev.short_name}' does not support device tracking")
            dev.tracker = self

    def __enter__(self):
        if not self.persistent:
            self.reset()

        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.active = False

    def update(self, **kwargs):
        """Store passed keyword-value pairs into ``totals``,``history``, and ``latest`` attributes.

        There is no restriction on the key-value pairs passed, but in the standard devices, the
        device ``execute`` method will pass ``executions`` and ``shots``, and the ``batch_execute``
        method will pass ``batches`` and ``batch_len``.

        Only numeric values will be added to ``totals``.

        >>> tracker.update(a=1, b=2, c="c")
        >>> tracker.latest
        {"a":1, "b":2, "c":"c"}
        >>> tracker.history
        {"a": [1], "b": [2], "c": ["c"]}
        >>> tracker.totals
        {"a": 1, "b": 2}

        """
        self.latest = kwargs

        for key, value in kwargs.items():
            # update history
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

            # updating totals
            if value is not None:
                # Only total numeric values
                if isinstance(value, Number):
                    self.totals[key] = value + self.totals.get(key, 0)

    def reset(self):
        """Resets stored information."""
        self.totals = {}
        self.history = {}
        self.latest = {}

    def record(self):
        """This method allows users to interact with the stored data.  While it's intended purpose
        is monitoring large jobs through ``print`` statements or logging, the function is
        completely flexible and customizable.

        If a user provided a ``callback`` function during initialization, that function is called
        with the current ``totals``, ``history``, and ``latest`` data variables as keyword arguments.
        """
        if self.callback is not None:
            self.callback(totals=self.totals, history=self.history, latest=self.latest)
