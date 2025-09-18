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
from collections.abc import Hashable
from functools import lru_cache

from pennylane.capture import enabled as capture_enabled
from pennylane.exceptions import QuantumFunctionError
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurement_value import MeasurementValue


def _measure_impl(
    wires: Hashable | Wires, reset: bool | None = False, postselect: int | None = None
):
    """Concrete implementation of qml.measure"""
    wires = Wires(wires)
    if len(wires) > 1:
        raise QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())
    mp = MidMeasureMP(wires=wires, reset=reset, postselect=postselect, id=measurement_id)
    return MeasurementValue([mp])


@lru_cache
def _create_mid_measure_primitive():
    """Create a primitive corresponding to an mid-circuit measurement type.

    Called when using :func:`~pennylane.measure`.

    Returns:
        jax.extend.core.Primitive: A new jax primitive corresponding to a mid-circuit
        measurement.

    """
    # pylint: disable=import-outside-toplevel
    import jax

    from pennylane.capture.custom_primitives import QmlPrimitive

    mid_measure_p = QmlPrimitive("measure")

    @mid_measure_p.def_impl
    def _(wires, reset=False, postselect=None):
        return _measure_impl(wires, reset=reset, postselect=postselect)

    @mid_measure_p.def_abstract_eval
    def _(*_, **__):
        dtype = jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32
        return jax.core.ShapedArray((), dtype)

    return mid_measure_p


def get_mcm_predicates(conditions: tuple[MeasurementValue]) -> list[MeasurementValue]:
    r"""Function to make mid-circuit measurement predicates mutually exclusive.

    The ``conditions`` are predicates to the ``if`` and ``elif`` branches of ``qml.cond``.
    This function updates all the ``MeasurementValue``\ s in ``conditions`` such that
    reconciling the correct branch is never ambiguous.

    Args:
        conditions (Sequence[MeasurementValue]): Sequence containing predicates for ``if``
            and all ``elif`` branches of a function decorated with :func:`~pennylane.cond`.

    Returns:
        Sequence[MeasurementValue]: Updated sequence of mutually exclusive predicates.
    """
    new_conds = [conditions[0]]
    false_cond = ~conditions[0]

    for c in conditions[1:]:
        new_conds.append(false_cond & c)
        false_cond = false_cond & ~c

    new_conds.append(false_cond)
    return new_conds


def find_post_processed_mcms(circuit):
    """Return the subset of mid-circuit measurements which are required for post-processing.

    This includes any mid-circuit measurement that is post-selected or the object of a terminal
    measurement.
    """
    post_processed_mcms = {
        op
        for op in circuit.operations
        if isinstance(op, MidMeasureMP) and op.postselect is not None
    }
    for m in circuit.measurements:
        if isinstance(m.mv, list):
            for mv in m.mv:
                post_processed_mcms = post_processed_mcms | set(mv.measurements)
        elif m.mv is not None:
            post_processed_mcms = post_processed_mcms | set(m.mv.measurements)
    return post_processed_mcms


class MidMeasureMP(Operator):
    """Mid-circuit measurement.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit in the computational basis are assumed.

    Please refer to :func:`pennylane.measure` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    num_wires = 1
    num_params = 0
    batch_size = None

    def __init__(
        self,
        wires: Wires | None = None,
        reset: bool | None = False,
        postselect: int | None = None,
        id: str | None = None,
    ):
        super().__init__(wires=Wires(wires), id=id)
        self._hyperparameters = {"reset": reset, "postselect": postselect, "id": id}

    @property
    def reset(self) -> bool | None:
        """Whether to reset the wire into the zero state after the measurement."""
        return self.hyperparameters["reset"]

    @property
    def postselect(self) -> int | None:
        """Which basis state to postselect after a mid-circuit measurement."""
        return self.hyperparameters["postselect"]

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def compute_diagonalizing_gates(*params, wires, **hyperparams) -> list[Operator]:
        return []

    def label(self, decimals=None, base_label=None, cache=None):
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
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            tuple(self.wires.tolist()),
            self.id,
        )

        return hash(fingerprint)


def measure(
    wires: Hashable | Wires, reset: bool = False, postselect: int | None = None
) -> MeasurementValue:
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

        >>> qml.set_shots(func, 10)(np.pi / 2)
        array([[1],
        [1],
        [1],
        [1]])

        Note that less than 10 samples are returned. This is because samples that do not meet the postselection criteria are
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
    if capture_enabled():
        primitive = _create_mid_measure_primitive()
        return primitive.bind(wires, reset=reset, postselect=postselect)

    return _measure_impl(wires, reset=reset, postselect=postselect)
