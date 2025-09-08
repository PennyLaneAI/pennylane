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
from collections.abc import Callable, Iterable
from inspect import signature

from pennylane.decomposition.resources import CompressedResourceOp, Resources
from pennylane.estimator.wires_manager import WireResourceManager
from pennylane.operation import classproperty
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=no-member, unused-argument, no-self-use


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
            def default_resource_decomp(cls, num_wires, **kwargs):  # `resource_keys` are input

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
        >>> print(plre.estimate_resources(op, gate_set={'Hadamard', 'SWAP', 'ControlledPhaseShift'}))
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

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self

    @staticmethod
    def dequeue(
        op_to_remove: ResourceOperator | Iterable,
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
    def default_resource_decomp(cls, *args, **kwargs) -> list:
        r"""Returns a list of actions that define the resources of the operator."""

    @classmethod
    def resource_decomp(cls, *args, **kwargs) -> list:
        r"""Returns a list of actions that define the resources of the operator."""
        return cls.default_resource_decomp(*args, **kwargs)

    @classmethod
    def default_adjoint_resource_decomp(cls, *args, **kwargs) -> list:
        r"""Returns a list representing the resources for the adjoint of the operator."""
        raise ResourcesNotDefined

    @classmethod
    def adjoint_resource_decomp(cls, *args, **kwargs) -> list:
        r"""Returns a list of actions that define the resources of the operator."""
        return cls.default_adjoint_resource_decomp(*args, **kwargs)

    @classmethod
    def default_controlled_resource_decomp(
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
        return cls.default_controlled_resource_decomp(
            ctrl_num_ctrl_wires, ctrl_num_ctrl_values, *args, **kwargs
        )

    @classmethod
    def default_pow_resource_decomp(cls, pow_z: int, *args, **kwargs) -> list:
        r"""Returns a list representing the resources for an operator
        raised to a power.

        Args:
            pow_z (int): exponent that the operator is being raised to
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, *args, **kwargs) -> list:
        r"""Returns a list representing the resources for an operator
        raised to a power.

        Args:
            pow_z (int): exponent that the operator is being raised to
        """
        return cls.default_pow_resource_decomp(pow_z, *args, **kwargs)

    # pylint: disable=no-member
    @classmethod
    def set_resources(cls, new_func: Callable, override_type: str = "base"):
        """Set a custom function to override the default resource decomposition.

        This method allows users to replace any of the `resource_decomp`, `adjoint_resource_decomp`,
        `ctrl_resource_decomp`, or `pow_resource_decomp` methods globally for every instance of
        the class.

        """
        if override_type == "base":
            keys = cls.resource_keys.union({"kwargs"})
            _validate_signature(new_func, keys)
            cls.resource_decomp = new_func
        if override_type == "pow":
            keys = cls.resource_keys.union({"pow_z", "kwargs"})
            _validate_signature(new_func, keys)
            cls.pow_resource_decomp = new_func
        if override_type == "adj":
            keys = cls.resource_keys.union({"kwargs"})
            _validate_signature(new_func, keys)
            cls.adjoint_resource_decomp = new_func
        if override_type == "ctrl":
            keys = cls.resource_keys.union(
                {"ctrl_num_ctrl_wires", "ctrl_num_ctrl_values", "kwargs"}
            )
            _validate_signature(new_func, keys)
            cls.controlled_resource_decomp = new_func

    def __repr__(self) -> str:
        str_rep = self.__class__.__name__ + "(" + str(self.resource_params) + ")"
        return str_rep

    def __mul__(self, scalar: int):
        assert isinstance(scalar, int)
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
        qubit_manager = WireResourceManager(0, algo_wires=self.num_wires)

        return Resources(qubit_manager, gate_types)

    def __matmul__(self, scalar: int):
        assert isinstance(scalar, int)
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
        qubit_manager = WireResourceManager(0, algo_wires=scalar * self.num_wires)

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


def _validate_signature(func: Callable, expected_args: set):
    """Raise an error if the provided function doesn't match expected signature

    Args:
        func (Callable): function to match signature with
        expected_args (set): expected signature
    """

    sig = signature(func)
    actual_args = set(sig.parameters)

    if extra_args := actual_args - expected_args:
        raise ValueError(
            f"The function provided specifies additional arguments ({extra_args}) from"
            + f" the expected arguments ({expected_args}). Please update the function signature or"
            + " modify the base class' `resource_keys` argument."
        )

    if missing_args := expected_args - actual_args:
        raise ValueError(
            f"The function is missing arguments ({missing_args}) which are expected. Please"
            + " update the function signature or modify the base class' `resource_keys` argument."
        )


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
