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
r"""Abstract base class for resource operators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import Union

import numpy as np

from pennylane.labs.resource_estimation.qubit_manager import QubitManager
from pennylane.labs.resource_estimation.resources_base import Resources
from pennylane.operation import classproperty
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=unused-argument


class CompressedResourceOp:
    r"""Instantiate a light weight class corresponding to the operator type and parameters.

    This class provides a minimal representation of an operation, containing
    only the operator type and the necessary parameters to estimate its resources.
    It's designed for efficient hashing and comparison, allowing it to be used
    effectively in collections where uniqueness and quick lookups are important.

    Args:
        op_type (Type): the class object of an operation which inherits from :class:'~.pennylane.labs.resource_estimation.ResourceOperator'
        num_wires (int): The number of wires that the operation acts upon,
            excluding any auxiliary wires that are allocated on decomposition.
        params (dict): a dictionary containing the minimal pairs of parameter names and values
            required to compute the resources for the given operator
        name (str, optional): A custom name for the compressed operator. If not
            provided, a name will be generated using `op_type.tracking_name`
            with the given parameters.

    .. details::

        This representation is the minimal amount of information required to estimate resources for the operator.

    **Example**

    >>> from pennylane.labs import resource_estimation as plre
    >>> cmpr_op = plre.CompressedResourceOp(plre.ResourceHadamard, num_wires=1)
    >>> print(cmpr_op)
    CompressedResourceOp(ResourceHadamard, num_wires=1)
    """

    def __init__(
        self,
        op_type: type[ResourceOperator],
        num_wires: int,
        params: dict | None = None,
        name: str = None,
    ):

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


def _make_hashable(d) -> tuple:
    r"""Converts a potentially non-hashable object into a hashable tuple.

    Args:
        d : The object to potentially convert to a hashable tuple.
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
    r"""Base class to represent quantum operators according to the set of information
    required for resource estimation.

    A :class:`~.pennylane.labs.resource_estimation.ResourceOperator` is uniquely defined by its
    name (the class type) and its resource parameters (:code:`op.resource_params`).

    **Example**

    This example shows how to create a custom :class:`~.pennylane.labs.resource_estimation.ResourceOperator`
    class for resource estimation. We use :class:`~.pennylane.QFT` as a well known gate for
    simplicity.

    .. code-block:: python

        from pennylane.labs import resource_estimation as plre

        class ResourceQFT(plre.ResourceOperator):

            resource_keys = {"num_wires"}  # the only parameter that its resources depend upon.

            def __init__(self, num_wires, wires=None):  # wire labels are optional
                self.num_wires = num_wires
                super().__init__(wires=wires)

            @property
            def resource_params(self) -> dict:        # The keys must match the `resource_keys`
                return {"num_wires": self.num_wires}  # and values obtained from the operator.

            @classmethod
            def resource_rep(cls, num_wires):             # Takes the `resource_keys` as input
                params = {"num_wires": num_wires}         #   and produces a compressed
                return plre.CompressedResourceOp(cls, params)  # representation of the operator

            @classmethod
            def resource_decomp(cls, num_wires, **kwargs):  # `resource_keys` are input

                # Get compressed reps for each gate in the decomposition:

                swap = plre.resource_rep(plre.ResourceSWAP)
                hadamard = plre.resource_rep(plre.ResourceHadamard)
                ctrl_phase_shift = plre.resource_rep(plre.ResourceControlledPhaseShift)

                # Figure out the associated counts for each type of gate:

                swap_counts = num_wires // 2
                hadamard_counts = num_wires
                ctrl_phase_shift_counts = num_wires*(num_wires - 1) // 2

                return [                                  # Return the decomposition
                    plre.GateCount(swap, swap_counts),
                    plre.GateCount(hadamard, hadamard_counts),
                    plre.GateCount(ctrl_phase_shift, ctrl_phase_shift_counts),
                ]

    Which can be instantiated as a normal operation, but now contains the resources:

    .. code-block:: pycon

        >>> op = ResourceQFT(num_wires=3)
        >>> print(plre.estimate(op, gate_set={'Hadamard', 'SWAP', 'ControlledPhaseShift'}))
        --- Resources: ---
        Total qubits: 3
        Total gates : 7
        Qubit breakdown:
         clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
        Gate breakdown:
         {'SWAP': 1, 'Hadamard': 3, 'ControlledPhaseShift': 3}

    """

    num_wires = 1

    def __init__(self, *args, wires=None, **kwargs) -> None:
        self.wires = None
        if wires is not None:
            wires = Wires(wires)
            if len(wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {wires}.")
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

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self

    @staticmethod
    def dequeue(
        op_to_remove: Union["ResourceOperator", Iterable],
        context: QueuingManager = QueuingManager,
    ):
        """Remove the given resource operator(s) from the Operator queue."""
        if not isinstance(op_to_remove, Iterable):
            op_to_remove = [op_to_remove]

        for op in op_to_remove:
            context.remove(op)

    @classproperty
    @classmethod
    def resource_keys(cls) -> set:  # pylint: disable=no-self-use
        """The set of parameters that affects the resource requirement of the operator.

        All resource decomposition functions for this operator class are expected to accept the
        keyword arguments that match these keys exactly. The :func:`~pennylane.resource_rep`
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
        the Operator that are needed to estimate the resources."""

    def resource_rep_from_op(self) -> CompressedResourceOp:
        r"""Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params)

    @classmethod
    @abstractmethod
    def resource_decomp(cls, *args, **kwargs) -> list:
        r"""Returns a list of actions that define the resources of the operator."""

    @classmethod
    def adjoint_resource_decomp(cls, *args, **kwargs) -> list:
        r"""Returns a list representing the resources for the adjoint of the operator."""
        raise ResourcesNotDefined

    @classmethod
    def controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires: int, ctrl_num_ctrl_values: int, *args, **kwargs
    ) -> list:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, *args, **kwargs) -> list:
        r"""Returns a list representing the resources for an operator
        raised to a power.

        Args:
            pow_z (int): exponent that the operator is being raised to
        """
        raise ResourcesNotDefined

    def __repr__(self) -> str:
        str_rep = self.__class__.__name__ + "(" + str(self.resource_params) + ")"
        return str_rep

    def __mul__(self, scalar: int):
        assert isinstance(scalar, int)
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
        qubit_manager = QubitManager(0, algo_wires=self.num_wires)

        return Resources(qubit_manager, gate_types)

    def __matmul__(self, scalar: int):
        assert isinstance(scalar, int)
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
        qubit_manager = QubitManager(0, algo_wires=scalar * self.num_wires)

        return Resources(qubit_manager, gate_types)

    def __add__(self, other):
        if isinstance(other, ResourceOperator):
            return (1 * self) + (1 * other)
        if isinstance(other, Resources):
            return (1 * self) + other

        raise TypeError(f"Cannot add resource operator {self} with type {type(other)}.")

    def __and__(self, other):
        if isinstance(other, ResourceOperator):
            return (1 * self) & (1 * other)
        if isinstance(other, Resources):
            return (1 * self) & other

        raise TypeError(f"Cannot add resource operator {self} with type {type(other)}.")

    __radd__ = __add__
    __rand__ = __and__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @classmethod
    def tracking_name(cls, *args, **kwargs) -> str:
        r"""Returns a name used to track the operator during resource estimation."""
        return cls.__name__.replace("Resource", "")

    def tracking_name_from_op(self) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return self.__class__.tracking_name(**self.resource_params)


class ResourcesNotDefined(Exception):
    r"""Exception to be raised when a ``ResourceOperator`` does not implement _resource_decomp"""


class GateCount:
    r"""A class to represent a gate and its number of occurrences in a circuit or decomposition.

    Args:
        gate (CompressedResourceOp): a compressed resource representation of the gate being counted
        counts (int, optional): The number of occurances of the quantum gate in the circuit or
            decomposition. Defaults to 1.

    Returns:
        GateCount: the container object holding both pieces of information

    **Example**

    In this example we create an object to count 5 instances of :code:`plre.ResourceQFT` acting
    on three wires:

    >>> qft = plre.resource_rep(plre.ResourceQFT, {"num_wires": 3})
    >>> counts = plre.GateCount(qft, 5)
    >>> counts
    (5 x QFT(3))

    """

    def __init__(self, gate: CompressedResourceOp, count: int = 1) -> None:
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

    Note, the :code:`resource_params` dictionary should specify the required resource
    parameters of the operator. The required resource parameters are listed in the
    :code:`resource_keys` class property of every :class:`~.pennylane.labs.resource_estimation.ResourceOperator`.

    Args:
        resource_op (Type[ResourceOperator]]): The type of operator we wish to compactify
        resource_params (Dict): The required set of parameters to specify the operator

    Returns:
        CompressedResourceOp: A compressed representation of a resource operator

    **Example**

    In this example we obtain the compressed resource representation for :code:`ResourceQFT`.
    We begin by checking what parameters are required for resource estimation, and then providing
    them accordingly:

    >>> plre.ResourceQFT.resource_keys
    {'num_wires'}
    >>> cmpr_qft = plre.resource_rep(
    ...     plre.ResourceQFT,
    ...     {"num_wires": 3},
    ... )
    >>> cmpr_qft
    QFT(3)

    """

    if resource_params:
        return resource_op.resource_rep(**resource_params)

    return resource_op.resource_rep()
