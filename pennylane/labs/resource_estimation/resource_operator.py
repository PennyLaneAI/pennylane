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
from inspect import signature
from typing import Callable, Hashable, List, Optional, Type, Dict, Union

import numpy as np

from pennylane.labs.resource_estimation.qubit_manager import QubitManager
from pennylane.labs.resource_estimation.resources_base import Resources
from pennylane.operation import classproperty
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=unused-argument


class CompressedResourceOp:  # pylint: disable=too-few-public-methods
    r"""Instantiate the light weight class corresponding to the operator type and parameters.

    Args:
        op_type (Type): the class object of an operation which inherits from '~.ResourceOperator'
        params (dict): a dictionary containing the minimal pairs of parameter names and values
                    required to compute the resources for the given operator

    .. details::

        This representation is the minimal amount of information required to estimate resources for the operator.

        **Example**

        >>> op_tp = CompressedResourceOp(ResourceHadamard, {"num_wires":1})
        >>> print(op_tp)
        Hadamard(num_wires=1)
    """

    def __init__(
        self, op_type: Type[ResourceOperator], params: Optional[dict] = None, name: str = None
    ):

        if not issubclass(op_type, ResourceOperator):
            raise TypeError(f"op_type must be a subclass of ResourceOperator. Got {op_type}.")
        self.op_type = op_type
        self.params = params or {}
        self._hashable_params = _make_hashable(params) if params else ()
        self._name = name or op_type.tracking_name(**self.params)

    def __hash__(self) -> int:
        return hash((self.op_type, self._hashable_params))

    def __eq__(self, other: CompressedResourceOp) -> bool:
        return (
            isinstance(other, CompressedResourceOp)
            and self.op_type == other.op_type
            and self.params == other.params
        )

    def __repr__(self) -> str:

        class_name = self.__class__.__qualname__
        op_type_name = self.op_type.__name__

        params_arg_str = ""
        if self.params:
            params = sorted(self.params.items())
            params_str = ", ".join(f"{k!r}:{v!r}" for k, v in params)
            params_arg_str = f", params={{{params_str}}}"

        return f"{class_name}({op_type_name}{params_arg_str})"

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

    Operators defined for the purpose of resource estimation require less detailed information.
    This is because the cost of a quantum gate can be well approximated without a full description
    of its parameters. For example two :class:`~.RX` gates have the same cost regardless
    of their rotation angle parameters.

    A :class:`~.pennylane.labs.resource_estimations.ResourceOperator` is uniquely defined by its
    name (the class type) and its resource parameters ():code:`op.resource_params`). Additionally,

    .. details::

        **Example**

        A PennyLane Operator can be extended for resource estimation by creating a new class that
        inherits from both the ``Operator`` and ``ResourceOperator``. Here is an example showing how to
        extend ``qml.QFT`` for resource estimation.

        .. code-block:: python

            import pennylane as qml
            from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator

            class ResourceQFT(qml.QFT, ResourceOperator):

                @staticmethod
                def _resource_decomp(num_wires) -> Dict[CompressedResourceOp, int]:
                    gate_types = {}
                    hadamard = ResourceHadamard.resource_rep()
                    swap = ResourceSWAP.resource_rep()
                    ctrl_phase_shift = ResourceControlledPhaseShift.resource_rep()

                    gate_types[hadamard] = num_wires
                    gate_types[swap] = num_wires // 2
                    gate_types[ctrl_phase_shift] = num_wires*(num_wires - 1) // 2

                    return gate_types

                @property
                def resource_params(self, num_wires) -> dict:
                    return {"num_wires": num_wires}

                @classmethod
                def resource_rep(cls, num_wires) -> CompressedResourceOp:
                    params = {"num_wires": num_wires}
                    return CompressedResourceOp(cls, params)

        Which can be instantiated as a normal operation, but now contains the resources:

        .. code-block:: bash

            >>> op = ResourceQFT(range(3))
            >>> op.resources(**op.resource_params)
            {Hadamard: 3, SWAP: 1, ControlledPhaseShift: 3}
    """

    num_wires = 0
    _queue_category = "_resource_op"

    def __init__(self, *args, wires=None, **kwargs) -> None:
        self.wires = None
        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)

        self.queue()
        super().__init__()

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self

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
        """A dictionary containing the minimal information needed to compute a
        resource estimate of the operator's decomposition.

        The keys of this dictionary should match the ``resource_keys`` attribute of the operator
        class. Two instances of the same operator type should have identical ``resource_params`` if
        their decompositions exhibit the same counts for each gate type, even if the individual
        gate parameters differ.

        **Examples**

        The ``MultiRZ`` operator has non-empty ``resource_keys``:

        >>> re.ResourceMultiRZ.resource_keys
        {"num_wires"}

        The ``resource_params`` of an instance of ``MultiRZ`` will contain the number of wires:

        >>> op = re.ResourceMultiRZ(0.5, wires=[0, 1])
        >>> op.resource_params
        {"num_wires": 2}

        Note that another ``MultiRZ`` may have different parameters but the same ``resource_params``:

        >>> op2 = qml.ResourceMultiRZ(0.7, wires=[1, 2])
        >>> op2.resource_params
        {"num_wires": 2}

        """

    @classmethod
    @abstractmethod
    def resource_rep(cls, *args, **kwargs):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

    def resource_rep_from_op(self):
        r"""Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params)

    @classmethod
    @abstractmethod
    def default_resource_decomp(cls, *args, **kwargs) -> List:
        r"""Returns a list of actions that define the resources of the operator."""

    @classmethod
    def resource_decomp(cls, *args, **kwargs) -> List:
        r"""Returns a list of actions that define the resources of the operator."""
        return cls.default_resource_decomp(*args, **kwargs)

    @classmethod
    def default_adjoint_resource_decomp(cls, *args, **kwargs) -> List:
        r"""Returns a list representing the resources for the adjoint of the operator."""
        raise ResourcesNotDefined

    @classmethod
    def adjoint_resource_decomp(cls, *args, **kwargs) -> List:
        r"""Returns a list of actions that define the resources of the operator."""
        return cls.default_adjoint_resource_decomp(*args, **kwargs)

    @classmethod
    def default_controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires: int, ctrl_num_ctrl_values: int, *args, **kwargs
    ) -> List:
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
    ) -> List:
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
    def default_pow_resource_decomp(cls, pow_z: int, *args, **kwargs) -> List:
        r"""Returns a list representing the resources for an operator
        raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, *args, **kwargs) -> List:
        r"""Returns a list representing the resources for an operator
        raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
        """
        return cls.default_pow_resource_decomp(pow_z, *args, **kwargs)

    @classmethod
    def set_resources(cls, new_func: Callable, override_type: str = "base"):
        """Set a custom resource method."""
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
        return

    def __repr__(self) -> str:
        str_rep = self.__class__.__name__ + "(" + str(self.resource_params) + ")"
        return str_rep

    def __mul__(self, scalar: int):
        assert isinstance(scalar, int)
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
        qubit_manager = QubitManager(0)
        qubit_manager._logic_qubit_counts = self.num_wires

        return Resources(qubit_manager, gate_types)

    def __matmul__(self, scalar: int):
        assert isinstance(scalar, int)
        gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
        qubit_manager = QubitManager(0)
        qubit_manager._logic_qubit_counts = scalar * self.num_wires

        return Resources(qubit_manager, gate_types)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return (1 * self) + (1 * other)
        if isinstance(other, Resources):
            return (1 * self) + other

        raise TypeError(f"Cannot add resource operator {self} with type {type(other)}.")

    def __and__(self, other):
        if isinstance(other, self.__class__):
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
            f"The function provided specifies addtional arguments ({extra_args}) from"
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


def set_decomp(cls: Type[ResourceOperator], decomp_func: Callable) -> None:
    cls.set_resources(decomp_func, override_type="base")


def set_ctrl_decomp(cls: Type[ResourceOperator], decomp_func: Callable) -> None:
    cls.set_resources(decomp_func, override_type="ctrl")


def set_adj_decomp(cls: Type[ResourceOperator], decomp_func: Callable) -> None:
    cls.set_resources(decomp_func, override_type="adj")


def set_pow_decomp(cls: Type[ResourceOperator], decomp_func: Callable) -> None:
    cls.set_resources(decomp_func, override_type="pow")


class GateCount:
    """A class to represent a gate and the amount of times it was repeated."""
    
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
    resource_op: Type[ResourceOperator], resource_params: Dict=None,
) -> CompressedResourceOp:
    r"""Produce a compressed representation of the resource operator to be used when
    tracking resources.

    Args:
        resource_op (Type[ResourceOperator]]): The type of operator we wish to compactify 
        resource_params (Dict): The required set of parameters to specify the operator

    Returns:
        CompressedResourceOp: A compressed representation of a resource operator
    """

    if resource_params:
        return resource_op.resource_rep(**resource_params)

    return resource_op.resource_rep()

