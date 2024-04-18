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
This module contains the qml.measure measurement.
"""
import uuid
from typing import Generic, TypeVar, Optional
import numpy as np

import pennylane as qml
from pennylane.wires import Wires

from .measurements import MeasurementProcess, MidMeasure


def measure(wires: Wires, reset: Optional[bool] = False, postselect: Optional[int] = None):
    r"""Perform a mid-circuit measurement in the computational basis on the
    supplied qubit.

    Computational basis measurements are performed using the 0, 1 convention
    rather than the ±1 convention.
    Measurement outcomes can be used to conditionally apply operations, and measurement
    statistics can be gathered and returned by a quantum function.

    If a device doesn't support mid-circuit measurements natively, then the
    QNode will apply the :func:`defer_measurements` transform.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = qml.measure(1)

            qml.cond(m_0, qml.RY)(y, wires=0)
            return qml.probs(wires=[0])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.90165331, 0.09834669], requires_grad=True)

    Wires can be reused after measurement. Moreover, measured wires can be reset
    to the :math:`|0 \rangle` state by setting ``reset=True``.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def func():
            qml.X(1)
            m_0 = qml.measure(1, reset=True)
            return qml.probs(wires=[1])

    Executing this QNode:

    >>> func()
    tensor([1., 0.], requires_grad=True)

    Mid-circuit measurements can be manipulated using the following arithmetic operators:
    ``+``, ``-``, ``*``, ``/``, ``~`` (not), ``&`` (and), ``|`` (or), ``==``, ``<=``,
    ``>=``, ``<``, ``>`` with other mid-circuit measurements or scalars.

    .. Note ::

        Python ``not``, ``and``, ``or``, do not work since these do not have dunder methods.
        Instead use ``~``, ``&``, ``|``.

    Mid-circuit measurement results can be processed with the usual measurement functions such as
    :func:`~.expval`. For QNodes with finite shots, :func:`~.sample` applied to a mid-circuit measurement
    result will return a binary sequence of samples.
    See :ref:`here <mid_circuit_measurements_statistics>` for more details.

    .. Note ::

        Computational basis measurements are performed using the 0, 1 convention rather than the ±1 convention.
        So, for example, ``expval(qml.measure(0))`` and ``expval(qml.Z(0))`` will give different answers.

    .. code-block:: python3

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            m0 = qml.measure(1)
            return (
                qml.sample(m0), qml.expval(m0), qml.var(m0), qml.probs(op=m0), qml.counts(op=m0),
            )

    >>> circuit(1.0, 2.0, shots=1000)
    (array([0, 1, 1, ..., 1, 1, 1])), 0.702, 0.20919600000000002, array([0.298, 0.702]), {0: 298, 1: 702})

    Args:
        wires (Wires): The wire to measure.
        reset (Optional[bool]): Whether to reset the wire to the :math:`|0 \rangle`
            state after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.

    Returns:
        MidMeasureMP: measurement process instance

    Raises:
        QuantumFunctionError: if multiple wires were specified

    .. details::
        :title: Postselection

        Postselection discards outcomes that do not meet the criteria provided by the ``postselect``
        argument. For example, specifying ``postselect=1`` on wire 0 would be equivalent to projecting
        the state vector onto the :math:`|1\rangle` state on wire 0:

        .. code-block:: python3

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def func(x):
                qml.RX(x, wires=0)
                m0 = qml.measure(0, postselect=1)
                qml.cond(m0, qml.X)(wires=1)
                return qml.sample(wires=1)

        By postselecting on ``1``, we only consider the ``1`` measurement outcome on wire 0. So, the probability of
        measuring ``1`` on wire 1 after postselection should also be 1. Executing this QNode with 10 shots:

        >>> func(np.pi / 2, shots=10)
        array([1, 1, 1, 1, 1, 1, 1])

        Note that only 7 samples are returned. This is because samples that do not meet the postselection criteria are
        thrown away.

        If postselection is requested on a state with zero probability of being measured, the result may contain ``NaN``
        or ``Inf`` values:

        .. code-block:: python3

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def func(x):
                qml.RX(x, wires=0)
                m0 = qml.measure(0, postselect=1)
                qml.cond(m0, qml.X)(wires=1)
                return qml.probs(wires=1)

        >>> func(0.0)
        tensor([nan, nan], requires_grad=True)

        In the case of ``qml.sample``, an empty array will be returned:

        .. code-block:: python3

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def func(x):
                qml.RX(x, wires=0)
                m0 = qml.measure(0, postselect=1)
                qml.cond(m0, qml.X)(wires=1)
                return qml.sample(wires=[0, 1])

        >>> func(0.0, shots=[10, 10])
        (array([], shape=(0, 2), dtype=int64), array([], shape=(0, 2), dtype=int64))

        .. note::

            Currently, postselection support is only available on ``default.qubit``. Using postselection
            on other devices will raise an error.

        .. warning::

            All measurements are supported when using postselection. However, postselection on a zero probability
            state can cause some measurements to break:

            * With finite shots, one must be careful when measuring ``qml.probs`` or ``qml.counts``, as these
              measurements will raise errors if there are no valid samples after postselection. This will occur
              with postselection states that have zero or close to zero probability.

            * With analytic execution, ``qml.mutual_info`` will raise errors when using any interfaces except
              ``jax``, and ``qml.vn_entropy`` will raise an error with the ``tensorflow`` interface when the
              postselection state has zero probability.

            * When using JIT, ``QNode``'s may have unexpected behaviour when postselection on a zero
              probability state is performed. Due to floating point precision, the zero probability may not be
              detected, thus letting execution continue as normal without ``NaN`` or ``Inf`` values or empty
              samples, leading to unexpected or incorrect results.

    """

    wire = Wires(wires)
    if len(wire) > 1:
        raise qml.QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())[:8]
    mp = MidMeasureMP(wires=wire, reset=reset, postselect=postselect, id=measurement_id)
    return MeasurementValue([mp], processing_fn=lambda v: v)


T = TypeVar("T")


class MidMeasureMP(MeasurementProcess):
    """Mid-circuit measurement.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit in the computational basis are assumed.

    Please refer to :func:`measure` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("reset", self.reset), ("id", self.id))
        return (None, None), metadata

    def __init__(
        self,
        wires: Optional[Wires] = None,
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        self.batch_size = None
        super().__init__(wires=Wires(wires), id=id)
        self.reset = reset
        self.postselect = postselect

    def label(self, decimals=None, base_label=None, cache=None):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Must be same length as ``obs`` attribute.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings
        """
        _label = "┤↗"
        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        _label += "├" if not self.reset else "│  │0⟩"

        return _label

    @property
    def return_type(self):
        return MidMeasure

    @property
    def samples_computational_basis(self):
        return False

    @property
    def _queue_category(self):
        return "_ops"

    @property
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            tuple(self.wires.tolist()),
            self.id,
        )

        return hash(fingerprint)

    @property
    def data(self):
        """The data of the measurement. Needed to match the Operator API."""
        return []

    @property
    def name(self):
        """The name of the measurement. Needed to match the Operator API."""
        return "MidMeasureMP"


class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurements (list[.MidMeasureMP]): The measurement(s) that this object depends on.
        processing_fn (callable): A lazily transformation applied to the measurement values.
    """

    name = "MeasurementValue"

    def __init__(self, measurements, processing_fn):
        self.measurements = measurements
        self.processing_fn = processing_fn

    def _items(self):
        """A generator representing all the possible outcomes of the MeasurementValue."""
        for i in range(2 ** len(self.measurements)):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurements)))
            yield branch, self.processing_fn(*branch)

    @property
    def wires(self):
        """Returns a list of wires corresponding to the mid-circuit measurements."""
        return Wires.all_wires([m.wires for m in self.measurements])

    @property
    def branches(self):
        """A dictionary representing all possible outcomes of the MeasurementValue."""
        ret_dict = {}
        for i in range(2 ** len(self.measurements)):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurements)))
            ret_dict[branch] = self.processing_fn(*branch)
        return ret_dict

    def map_wires(self, wire_map):
        """Returns a copy of the current ``MeasurementValue`` with the wires of each measurement changed
        according to the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            MeasurementValue: new ``MeasurementValue`` instance with measurement wires mapped
        """
        mapped_measurements = [m.map_wires(wire_map) for m in self.measurements]
        return MeasurementValue(mapped_measurements, self.processing_fn)

    def _transform_bin_op(self, base_bin, other):
        """Helper function for defining dunder binary operations."""
        if isinstance(other, MeasurementValue):
            # pylint: disable=protected-access
            return self._merge(other)._apply(lambda t: base_bin(t[0], t[1]))
        # if `other` is not a MeasurementValue then apply it to each branch
        return self._apply(lambda v: base_bin(v, other))

    def __invert__(self):
        """Return a copy of the measurement value with an inverted control
        value."""
        return self._apply(lambda v: not v)

    def __eq__(self, other):
        return self._transform_bin_op(lambda a, b: a == b, other)

    def __ne__(self, other):
        return self._transform_bin_op(lambda a, b: a != b, other)

    def __add__(self, other):
        return self._transform_bin_op(lambda a, b: a + b, other)

    def __radd__(self, other):
        return self._apply(lambda v: other + v)

    def __sub__(self, other):
        return self._transform_bin_op(lambda a, b: a - b, other)

    def __rsub__(self, other):
        return self._apply(lambda v: other - v)

    def __mul__(self, other):
        return self._transform_bin_op(lambda a, b: a * b, other)

    def __rmul__(self, other):
        return self._apply(lambda v: other * v)

    def __truediv__(self, other):
        return self._transform_bin_op(lambda a, b: a / b, other)

    def __rtruediv__(self, other):
        return self._apply(lambda v: other / v)

    def __lt__(self, other):
        return self._transform_bin_op(lambda a, b: a < b, other)

    def __le__(self, other):
        return self._transform_bin_op(lambda a, b: a <= b, other)

    def __gt__(self, other):
        return self._transform_bin_op(lambda a, b: a > b, other)

    def __ge__(self, other):
        return self._transform_bin_op(lambda a, b: a >= b, other)

    def __and__(self, other):
        return self._transform_bin_op(lambda a, b: a and b, other)

    def __or__(self, other):
        return self._transform_bin_op(lambda a, b: a or b, other)

    def _apply(self, fn):
        """Apply a post computation to this measurement"""
        return MeasurementValue(self.measurements, lambda *x: fn(self.processing_fn(*x)))

    def concretize(self, measurements: dict):
        """Returns a concrete value from a dictionary of hashes with concrete values."""
        values = tuple(measurements[meas] for meas in self.measurements)
        return self.processing_fn(*values)

    def _merge(self, other: "MeasurementValue"):
        """Merge two measurement values"""

        # create a new merged list with no duplicates and in lexical ordering
        merged_measurements = list(set(self.measurements).union(set(other.measurements)))
        merged_measurements.sort(key=lambda m: m.id)

        # create a new function that selects the correct indices for each sub function
        def merged_fn(*x):
            sub_args_1 = (x[i] for i in [merged_measurements.index(m) for m in self.measurements])
            sub_args_2 = (x[i] for i in [merged_measurements.index(m) for m in other.measurements])

            out_1 = self.processing_fn(*sub_args_1)
            out_2 = other.processing_fn(*sub_args_2)

            return out_1, out_2

        return MeasurementValue(merged_measurements, merged_fn)

    def __getitem__(self, i):
        branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurements)))
        return self.processing_fn(*branch)

    def __str__(self):
        lines = []
        for i in range(2 ** (len(self.measurements))):
            branch = tuple(int(b) for b in np.binary_repr(i, width=len(self.measurements)))
            id_branch_mapping = [
                f"{self.measurements[j].id}={branch[j]}" for j in range(len(branch))
            ]
            lines.append(
                "if " + ",".join(id_branch_mapping) + " => " + str(self.processing_fn(*branch))
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"MeasurementValue(wires={self.wires.tolist()})"
