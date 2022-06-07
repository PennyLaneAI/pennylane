# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule defines the symbolic operation that indicates the control of an operator.
"""

import warnings
from copy import copy

from pennylane import math as qmlmath
from pennylane import numpy as np

from pennylane import Projector
from pennylane.operation import (
    Operator,
    Operation,
    Observable,
    Tensor,
    expand_matrix,
)
from pennylane.queuing import QueuingContext
from pennylane.wires import Wires


# pylint: disable=no-member
class ControlledOperation(Operation):
    """Operation-specific methods and properties for the ``Controlled`` class.

    Dynamically mixed in based on the provided base operator.  If the base operator is an
    Operation, this class will be mixed in.

    When we no longer rely on certain functionality through `Operation`, we can get rid of this
    class.
    """

    @property
    def grad_method(self):
        return self.base.grad_method

    # pylint: disable=missing-function-docstring
    @property
    def basis(self):
        return self.base.basis

    # TODO: parameter-frequencies


# pylint: disable=too-many-arguments, too-many-public-methods
class Controlled(Operator):
    """Symbolic operator denoting a controlled operator.

    Args:
        base (~.operation.Operator): the operator that is controlled
        control_wires (Any): The wires to control on.

    Keyword Args:
        control_values (Iterable[Bool]): The values to control on. Must be the same
            length as ``control_wires``. Defaults to ``True`` for all control wires.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    **Example:**

    >>> base = qml.RX(1.234, 2)
    >>> op = Controlled(base, (0,1))
    >>> op
    CRX(1.234, wires=[0, 1, 2])
    >>> op.base
    RX(1.234, wires=[2])
    >>> op.data
    [1.234]
    >>> op.wires
    <Wires = [0, 1, 2]>
    >>> op.control_wires
    <Wires = [0, 1]>
    >>> op.target_wires
    <Wires = [2]>
    >>> op.control_values
    [True, True]

    >>> op2 = Controlled(qml.PauliX(1), 0)
    >>> qml.matrix(op2)
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
    >>> qml.eigvals(op2)
    tensor([ 1.,  1.,  1., -1.], requires_grad=True)
    >>> qml.generator(op)
    (Projector([1, 1], wires=[0, 1]) @ PauliX(wires=[2]), -0.5)
    >>> op.pow(-1.2)
    [CRX(-1.4808, wires=[0, 1, 2])]


    """

    _operation_type = None  # type if base inherits from operation and not observable
    _operation_observable_type = None  # type if base inherits from both operation and observable
    _observable_type = None  # type if base inherits from observable and not oepration

    # pylint: disable=unused-argument
    def __new__(
        cls, base, control_wires, control_values=None, work_wires=None, do_queue=True, id=None
    ):
        """Mixes in parents based on inheritance structure of base.

        Though all the types will be named "Pow", their *identity* and location in memory will be different
        based on ``base``'s inheritance.  We cache the different types in private class variables so that:

        """

        if isinstance(base, Operation):
            if isinstance(base, Observable):
                if cls._operation_observable_type is None:
                    base_classes = (ControlledOperation, Controlled, Observable, Operation)
                    cls._operation_observable_type = type(
                        "Controlled", base_classes, dict(cls.__dict__)
                    )
                return object.__new__(cls._operation_observable_type)

            # not an observable
            if cls._operation_type is None:
                base_classes = (ControlledOperation, Controlled, Operation)
                cls._operation_type = type("Controlled", base_classes, dict(cls.__dict__))
            return object.__new__(cls._operation_type)

        if isinstance(base, Observable):
            if cls._observable_type is None:
                base_classes = (Controlled, Observable)
                cls._observable_type = type("Controlled", base_classes, dict(cls.__dict__))
            return object.__new__(cls._observable_type)

        return object.__new__(Controlled)

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten becuase the base must be copied too.
        copied_op = object.__new__(type(self))
        # copied_op must maintain inheritance structure of self
        # For example, it must keep AdjointOperation if self has it
        # this way preserves inheritance structure

        for attr, value in vars(self).items():
            if attr != "_hyperparameters":
                setattr(copied_op, attr, value)
        copied_op._hyperparameters = copy(self._hyperparameters)
        copied_op._hyperparameters["base"] = copy(self.base)

        return copied_op

    # pylint: disable=super-init-not-called
    def __init__(
        self, base, control_wires, control_values=None, work_wires=None, do_queue=True, id=None
    ):
        control_wires = Wires(control_wires)
        if control_values is None:
            control_values = [True] * len(control_wires)
        else:
            if isinstance(control_values, str):
                warnings.warn(
                    "Specifying control values as a string is deprecated. Please use Sequence[Bool]",
                    UserWarning,
                )
                control_values = [(x == "1") for x in control_values]

            assert len(control_values) == len(
                control_wires
            ), "control_values should be the same length as control_wires"
            assert set(control_values).issubset(
                {False, True}
            ), "control_values can only take on True or False"

        assert (
            len(Wires.shared_wires([base.wires, control_wires])) == 0
        ), "The control wires must be different from the base operation wires."

        self.hyperparameters["base"] = base
        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["control_values"] = control_values
        self.hyperparameters["work_wires"] = Wires([]) if work_wires is None else Wires(work_wires)

        self._name = f"C{base.name}"

        self._id = id
        self.queue_idx = None
        self._inverse = False

        if do_queue:
            self.queue()

    @property
    def base(self):
        """The Operator being controlled."""
        return self.hyperparameters["base"]

    @property
    def control_wires(self):
        """The control wires."""
        return self.hyperparameters["control_wires"]

    @property
    def target_wires(self):
        """The wires of the base operation."""
        return self.base.wires

    @property
    def control_values(self):
        """Iterable[Bool]. For each control wire, denotes whether to control on ``True`` or ``False``."""
        return self.hyperparameters["control_values"]

    @property
    def work_wires(self):
        """Additional wires that can be used in the decomposition. Not modified by the operation."""
        return self.hyperparameters["work_wires"]

    @property
    def data(self):
        """Trainable parameters that the operator depends on."""
        return self.base.data

    @data.setter
    def data(self, new_data):
        self.base.data = new_data

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def wires(self):
        return self.control_wires + self.base.wires + self.work_wires

    # pylint: disable=protected-access
    @property
    def _wires(self):
        return self.wires

    # pylint: disable=protected-access
    @_wires.setter
    def _wires(self, new_wires):
        new_wires = new_wires if isinstance(new_wires, Wires) else Wires(new_wires)

        num_control = len(self.control_wires)
        num_base = len(self.base.wires)
        num_control_and_base = num_control + num_base

        assert num_control_and_base <= len(new_wires), (
            f"{self.name} needs at least {num_control_and_base} wires."
            f" {len(new_wires)} provided."
        )

        self.hyperparameters["control_wires"] = new_wires[0:num_control]

        self.base._wires = new_wires[num_control:num_control_and_base]

        if len(new_wires) > num_control_and_base:
            self.hyperparameters["work_wires"] = new_wires[num_control_and_base:]
        else:
            self.hyperparameters["work_wires"] = Wires([])

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def ndim_params(self):
        return self.base.ndim_params

    @property
    def is_hermitian(self):
        return self.base.is_hermitian

    def queue(self, context=QueuingContext):
        context.safe_update_info(self.base, owner=self)
        context.append(self, owns=self.base)
        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)

    # pylint: disable=invalid-overridden-method
    @property
    def has_matrix(self):
        return self.base.has_matrix

    def matrix(self, wire_order=None):

        base_matrix = self.base.matrix()
        interface = qmlmath.get_interface(base_matrix)

        base_matrix_size = qmlmath.shape(base_matrix)[0]
        num_control_states = 2 ** len(self.control_wires)
        total_matrix_size = num_control_states * base_matrix_size

        control_int = sum(2**i for i, val in enumerate(reversed(self.control_values)) if val)
        padding_left = control_int * base_matrix_size
        padding_right = total_matrix_size - padding_left - base_matrix_size

        left_pad = qmlmath.cast_like(qmlmath.eye(padding_left, like=interface), 1j)
        right_pad = qmlmath.cast_like(qmlmath.eye(padding_right, like=interface), 1j)

        canonical_matrix = qmlmath.block_diag([left_pad, base_matrix, right_pad])

        if wire_order is None or self.wires == Wires(wire_order):
            return canonical_matrix

        active_wires = self.control_wires + self.target_wires
        return expand_matrix(canonical_matrix, wires=active_wires, wire_order=wire_order)

    def eigvals(self):
        base_eigvals = self.base.eigvals()
        ones = np.ones(2 ** len(self.control_wires))
        return qmlmath.concatenate([ones, base_eigvals])

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    # TODO: decomposition
    # TODO: sparse_matrix (requires performance team help for optimization)

    def generator(self):
        sub_gen = self.base.generator()
        proj_tensor = Tensor(*(Projector([1], wires=w) for w in self.control_wires))
        return 1.0 * proj_tensor @ sub_gen

    def adjoint(self):
        return Controlled(
            self.base.adjoint(), self.control_wires, self.control_values, self.work_wires
        )

    def pow(self, z):
        base_pow = self.base.pow(z)
        return [
            Controlled(op, self.control_wires, self.control_values, self.work_wires)
            for op in base_pow
        ]

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns ``_queue_cateogory`` for base operator.

        Options are:
            * `"_prep"`
            * `"_ops"`
            * `"_measurements"`
            * `None`
        """
        return self.base._queue_category  # pylint: disable=protected-access
