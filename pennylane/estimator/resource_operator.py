# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This submodule contains base classes for resource operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np

from pennylane.exceptions import ResourcesUndefinedError
from pennylane.operation import classproperty
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

from .resources_base import Resources


class CompressedResourceOp:
    r"""This class is a minimal representation of a :class:`~.pennylane.estimator.ResourceOperator`,
    containing only the operator type and the necessary parameters to estimate its resources.

    The ``CompressedResourceOp`` object is returned by the ``.resource_rep()`` method of resource
    operators. The object is used by resource operators to efficiently compute the resource counts.

    .. code-block:: pycon

        >>> import pennylane.estimator as qre
        >>> cmpr_op = qre.PauliRot.resource_rep(pauli_string="XYZ")
        >>> print(cmpr_op)
        CompressedResourceOp(PauliRot, num_wires=3, params={'pauli_string':'XYZ', 'precision':None})

    Args:
        op_type (type[ResourceOperator]): the class object of an operation which inherits from :class:`~.pennylane.estimator.ResourceOperator`
        num_wires (int): The number of wires that the operation acts upon,
            excluding any auxiliary wires that are allocated on decomposition.
        params (dict): A dictionary containing the minimal pairs of parameter names and values
            required to compute the resources for the given operator.
        name (str | None): A custom name for the compressed operator. If not
            provided, a name will be generated using ``op_type.make_tracking_name``
            with the given parameters.

    """

    def __init__(
        self,
        op_type: type[ResourceOperator],
        num_wires: int,
        params: dict | None = None,
        name: str | None = None,
    ) -> None:

        if not issubclass(op_type, ResourceOperator):
            raise TypeError(f"op_type must be a subclass of ResourceOperator. Got {op_type}.")
        self.op_type = op_type
        self.num_wires = num_wires
        self.params = params or {}
        self._hashable_params = _make_hashable(params) if params else ()
        self._name = name or op_type.tracking_name(**self.params)

    def __hash__(self) -> int:
        return hash((self.op_type, self.num_wires, self._hashable_params))

    def __eq__(self, other: CompressedResourceOp) -> bool:
        return (
            isinstance(other, CompressedResourceOp)
            and self.op_type == other.op_type
            and self.num_wires == other.num_wires
            and self.params == other.params
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__qualname__
        op_type_name = self.op_type.__name__

        num_wires_str = f"num_wires={self.num_wires}"

        params_arg_str = ""
        if self.params:
            params = sorted(self.params.items())
            params_str = ", ".join(f"{k!r}:{v!r}" for k, v in params)
            params_arg_str = f", params={{{params_str}}}"

        return f"{class_name}({op_type_name}, {num_wires_str}{params_arg_str})"

    @property
    def name(self) -> str:
        r"""Returns the name of operator."""
        return self._name


def _make_hashable(d: Any) -> tuple:
    r"""Converts a potentially non-hashable object into a hashable tuple.

    Args:
        d (Any): The object to potentially convert to a hashable tuple.
           This can be a dictionary, list, set, or an array.

    Returns:
        A hashable tuple representation of the input.

    """

    if isinstance(d, Hashable):
        return d

    if isinstance(d, dict):
        return tuple(sorted((_make_hashable(k), _make_hashable(v)) for k, v in d.items()))
    if isinstance(d, (list, tuple)):
        return tuple(_make_hashable(elem) for elem in d)
    if isinstance(d, set):
        return tuple(sorted(_make_hashable(elem) for elem in d))
    if isinstance(d, np.ndarray):
        return _make_hashable(d.tolist())

    raise TypeError(f"Object of type {type(d)} is not hashable and cannot be converted.")


class ResourceOperator(ABC):
    r"""Base class to represent quantum operators according to the fundamental set of information
    required for resource estimation.

    A :class:`~.pennylane.estimator.ResourceOperator` is uniquely defined by its
    name (the class type) and its resource parameters (:code:`op.resource_params`).

    .. details::
        :title: Usage Details

        This example shows how to create a custom :class:`~.pennylane.estimator.ResourceOperator`
        class for resource estimation. We use :class:`~.pennylane.QFT` as a well known gate for
        simplicity.

        .. code-block:: python

            import pennylane.estimator as qre

            class QFT(qre.ResourceOperator):

                resource_keys = {"num_wires"}  # the only parameter that its resources depend upon.

                def __init__(self, num_wires, wires=None):  # wire labels are optional
                    self.num_wires = num_wires
                    super().__init__(wires=wires)

                @property
                def resource_params(self) -> dict:        # The keys must match the `resource_keys`
                    return {"num_wires": self.num_wires}  # and values obtained from the operator.

                @classmethod
                def resource_rep(cls, num_wires):   # Takes the same input as `resource_keys` and
                    params = {"num_wires": num_wires}  #  produces a compressed representation
                    return qre.CompressedResourceOp(cls, num_wires, params)

                @classmethod
                def resource_decomp(cls, num_wires):  # `resource_keys` are input

                    # Get compressed reps for each gate in the decomposition:

                    swap = qre.resource_rep(qre.SWAP)
                    hadamard = qre.resource_rep(qre.Hadamard)
                    ctrl_phase_shift = qre.resource_rep(qre.ControlledPhaseShift)

                    # Figure out the associated counts for each type of gate:

                    swap_counts = num_wires // 2
                    hadamard_counts = num_wires
                    ctrl_phase_shift_counts = num_wires*(num_wires - 1) // 2

                    return [    # Return the decomposition
                        qre.GateCount(swap, swap_counts),
                        qre.GateCount(hadamard, hadamard_counts),
                        qre.GateCount(ctrl_phase_shift, ctrl_phase_shift_counts),
                    ]

        Which can be instantiated as a normal operation, but now contains the resources:

        .. code-block:: pycon

            >>> op = QFT(num_wires=3)
            >>> print(qre.estimate(op, gate_set={'Hadamard', 'SWAP', 'ControlledPhaseShift'}))
            --- Resources: ---
             Total wires: 3
                algorithmic wires: 3
                allocated wires: 0
                     zero state: 0
                     any state: 0
             Total gates : 7
              'SWAP': 1,
              'ControlledPhaseShift': 3,
              'Hadamard': 3

    """

    num_wires: int | None = None

    # pylint: disable=unused-argument
    def __init__(self, *args, wires=None, **kwargs) -> None:
        self.wires = None
        if wires is not None:
            wires = Wires(wires)
            self.wires = wires

        self.queue()
        super().__init__()

    def __eq__(self, other):
        """Return True if the operators are equal."""
        if not isinstance(other, ResourceOperator):
            return False

        return (
            self.__class__ is other.__class__
            and self.resource_params == other.resource_params
            and self.num_wires == other.num_wires
        )

    def queue(self, context: QueuingManager = QueuingManager) -> "ResourceOperator":
        """Append the operator to the Operator queue."""
        context.append(self)
        return self

    @classproperty
    @classmethod
    def resource_keys(cls) -> set:  # pylint: disable=no-self-use
        """The set of parameters that affects the resource requirement of the operator.

        All resource decomposition functions for this operator class are expected to accept the
        keyword arguments that match these keys exactly. The :func:`~pennylane.estimator.resource_rep`
        function will also expect keyword arguments that match these keys when called with this
        operator type.

        The default implementation is an empty set, which is suitable for most operators.
        """
        return set()

    @property
    @abstractmethod
    def resource_params(self) -> dict:
        """A dictionary containing the minimal information needed to compute a resource estimate
        of the operator's decomposition. The keys of this dictionary should match the
        ``resource_keys`` attribute of the operator class.
        """

    @classmethod
    @abstractmethod
    def resource_rep(cls, *args, **kwargs) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to estimate the resources."""

    def resource_rep_from_op(self) -> CompressedResourceOp:
        r"""Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params)

    @classmethod
    @abstractmethod
    def resource_decomp(cls, *args, **kwargs) -> list[GateCount]:
        r"""Returns a list of actions that define the resources of the operator."""

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.
        """
        raise ResourcesUndefinedError

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.
        """
        raise ResourcesUndefinedError

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator
        raised to a power.

        Args:
            pow_z (int): exponent that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.
        """
        raise ResourcesUndefinedError

    def __repr__(self) -> str:
        str_rep = self.__class__.__name__ + "(" + str(self.resource_params) + ")"
        return str_rep

    def __mul__(self, scalar: int):
        if not isinstance(scalar, int):
            raise TypeError(f"Cannot multiply resource operator {self} with type {type(scalar)}.")
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})

        return Resources(zeroed_wires=0, algo_wires=self.num_wires, gate_types=gate_types)

    def __matmul__(self, scalar: int):
        if not isinstance(scalar, int):
            raise TypeError(f"Cannot multiply resource operator {self} with type {type(scalar)}.")
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})

        return Resources(zeroed_wires=0, algo_wires=self.num_wires * scalar, gate_types=gate_types)

    def add_series(self, other):
        """Adds a :class:`~.pennylane.estimator.ResourceOperator` or :class:`~.pennylane.estimator.Resources` in series.

        Args:
            other (:class:`~.pennylane.estimator.Resources`|:class:`~.pennylane.estimator.ResourceOperator`): The other object to combine with, it can be
                another ``ResourceOperator`` or a ``Resources`` object.

        Returns:
            :class:`~.pennylane.estimator.Resources`: added ``Resources``
        """
        if isinstance(other, ResourceOperator):
            return (1 * self).add_series(1 * other)
        if isinstance(other, Resources):
            return (1 * self).add_series(other)

        raise TypeError(f"Cannot add resource operator {self} with type {type(other)}.")

    def add_parallel(self, other):
        """Adds a :class:`~.pennylane.estimator.ResourceOperator` or :class:`~.pennylane.estimator.Resources` in parallel.

        Args:
            other (:class:`~.pennylane.estimator.Resources`|:class:`~.pennylane.estimator.ResourceOperator`): The other object to combine with, it can be
                another ``ResourceOperator`` or a ``Resources`` object.

        Returns:
            :class:`~.pennylane.estimator.Resources`: added ``Resources``
        """
        if isinstance(other, ResourceOperator):
            return (1 * self).add_parallel(1 * other)
        if isinstance(other, Resources):
            return (1 * self).add_parallel(other)

        raise TypeError(f"Cannot add resource operator {self} with type {type(other)}.")

    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @classmethod
    def tracking_name(cls, *args, **kwargs) -> str:
        r"""Returns a name used to track the operator during resource estimation."""
        return cls.__name__


def _dequeue(
    op_to_remove: "ResourceOperator" | Iterable,
    context: QueuingManager = QueuingManager,
):
    """Remove the given resource operator(s) from the Operator queue."""
    if not isinstance(op_to_remove, Iterable):
        op_to_remove = [op_to_remove]

    for op in op_to_remove:
        context.remove(op)


class GateCount:
    r"""Stores a lightweight representation of a gate and its number of occurrences in a decomposition.

    The decomposition of a resource operator is tracked as a sequence of gates and the corresponding
    number of times those gates occur in the decomposition. For a given resource operator, this
    decomposition can be accessed with the operator's ``.resource_decomp()`` method. The method
    returns a sequence of ``GateCount`` objects where each object groups the two pieces of
    information, gate and counts, for the decomposition.

    For example, the decomposition of the Quantum Fourier Transform (QFT)
    contains 3 ``Hadamard`` gates, 1 ``SWAP`` gate and 3 ``ControlledPhaseShift`` gates.

    .. code-block:: pycon

        >>> import pennylane.estimator as qre
        >>> lst_of_gate_counts = qre.QFT.resource_decomp(num_wires=3)
        >>> lst_of_gate_counts
        [(3 x Hadamard), (1 x SWAP), (3 x ControlledPhaseShift)]

    **Example**

    This example creates an object to count ``5`` instances of :code:`QFT` acting
    on three wires:

    >>> import pennylane.estimator as qre
    >>> qft = qre.resource_rep(qre.QFT, {"num_wires": 3})
    >>> counts = qre.GateCount(qft, 5)
    >>> counts
    (5 x QFT(3))

    Args:
        gate (CompressedResourceOp): The compressed resource representation of the gate being counted.
        counts (int | None): The number of occurrences of the quantum gate in the circuit or
            decomposition. Defaults to ``1``.

    Returns:
        GateCount: The container object holding both pieces of information.

    """

    def __init__(self, gate: CompressedResourceOp, count: int | None = 1) -> None:
        self.gate = gate
        self.count = count

    def __mul__(self, other):
        if isinstance(other, int):
            return self.__class__(self.gate, self.count * other)
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, self.__class__) and (self.gate == other.gate):
            return self.__class__(self.gate, self.count + other.count)
        raise NotImplementedError

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        if not isinstance(other, GateCount):
            return False
        return self.gate == other.gate and self.count == other.count

    def __repr__(self) -> str:
        return f"({self.count} x {self.gate._name})"


def resource_rep(
    resource_op: type[ResourceOperator],
    resource_params: dict | None = None,
) -> CompressedResourceOp:
    r"""Produce a compressed representation of the resource operator to be used when
    tracking resources.

    This function produces the expected compressed representation of a resource operator class.
    The compressed representation
    (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`) is used instead of
    the resource operator to enable faster performance of the resource estimation functionality.

    This function is used when defining the resource decompositions of a resource operator.
    Specifically, all resource decompositions are represented as a list of operators
    (``CompressedResourceOp``) and the number of times they occur in the decomposition (``int``).
    Those two pieces of information are tracked inside the
    :class:`~.pennylane.estimator.resource_operator.GateCount` class.

    .. note::

        The :code:`resource_params` dictionary should specify the required resource
        parameters of the operator. The required resource parameters are listed in the
        :code:`resource_keys` class property of every :class:`~.pennylane.estimator.ResourceOperator`.

    Args:
        resource_op (type[ResourceOperator]]): The type of operator for which to retrieve the compact representation.
        resource_params (dict | None): The required set of parameters to specify the operator. Defaults to ``None``.

    Returns:
        :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of a resource operator

    **Example**

    This example shows how to obtain the compressed resource representation for the quantum Fourier
    transform (:code:`QFT`) operation. We begin by checking what parameters are required for
    resource estimation and then provide them accordingly:

    >>> import pennylane.estimator as qre
    >>> qre.QFT.resource_keys
    {'num_wires'}
    >>> cmpr_qft = qre.resource_rep(
    ...     qre.QFT,
    ...     {"num_wires": 3},
    ... )
    >>> cmpr_qft
    CompressedResourceOp(QFT, num_wires=3, params={'num_wires':3})

    .. details::
        :title: Usage Details

        In this example we create a custom resource decomposition function which returns the
        decomposition using the ``GateCount`` class. We use the ``resource_rep`` function to
        obtain the compressed representations of each gate in the decomposition.

        .. code-block:: python

            import pennylane.estimator as qre

            def custom_RX_decomp(precision):  # RX = H @ RZ @ H
                h = qre.resource_rep(qre.Hadamard)
                rz = qre.resource_rep(qre.RZ, resource_params={"precision": None})
                return [qre.GateCount(h, 2), qre.GateCount(rz, 1)]

        .. code-block:: pycon

            >>> print(qre.estimate(qre.RX(), gate_set={"Hadamard", "RZ", "T"}))
            --- Resources: ---
             Total wires: 1
               algorithmic wires: 1
               allocated wires: 0
                 zero state: 0
                 any state: 0
             Total gates : 44
               'T': 44

        We override the default decomposition using the
        :class:`~.pennylane.estimator.resource_config.ResourceConfig` class.

        .. code-block:: pycon

            >>> config = qre.ResourceConfig()
            >>> config.set_decomp(qre.RX, custom_RX_decomp)
            >>> print(qre.estimate(qre.RX(), gate_set={"Hadamard", "RZ", "T"}, config=config))
            --- Resources: ---
             Total wires: 1
               algorithmic wires: 1
               allocated wires: 0
                 zero state: 0
                 any state: 0
             Total gates : 3
               'RZ': 1,
               'Hadamard': 2

    """

    resource_params = resource_params or {}
    return resource_op.resource_rep(**resource_params)
