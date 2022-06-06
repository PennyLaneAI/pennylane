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

from pennylane import math as qmlmath
from pennylane import numpy as np

import pennylane as qml
from pennylane.operation import (
    Operation,
    expand_matrix,
)
from pennylane.queuing import QueuingContext
from pennylane.wires import Wires


# pylint: disable=too-many-arguments
class Controlled(Operation):
    """
    Class docstring

    """

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten becuase the base must be copied too.
        copied_op = object.__new__(type(self))
        # copied_op must maintain inheritance structure of self
        # For example, it must keep AdjointOperation if self has it
        # this way preserves inheritance structure

        for attr, value in vars(self).items():
            if attr not in {"data"}:
                setattr(copied_op, attr, value)
        copied_op._hyperparameters["base"] = self.base.__copy__()

        return copied_op

    # pylint: disable=super-init-not-called
    def __init__(
        self, base, control_wires, control_values=None, work_wires=None, do_queue=True, id=None
    ):
        control_wires = Wires(control_wires)
        if isinstance(control_values, str):
            warnings.warn(
                "Specifying control values as a string is deprecated. Please use Sequence[Bool]",
                UserWarning,
            )
            control_values = [(x == "1") for x in control_values]
        if control_values is None:
            control_values = [True] * len(control_wires)
        if not set(control_values).issubset({False, True}):
            raise ValueError(
                "control_values can only take on True or False.  Provided control"
                f" values are {control_values}."
            )

        if Wires.shared_wires([base.wires, control_wires]):
            raise ValueError("The control wires must be different from the base operation wires.")

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
        num_control = len(self.control_wires)
        num_base = len(self.base.wires)
        num_control_and_base = num_control + num_base

        if len(new_wires) < num_control_and_base:
            raise ValueError(
                f"{self} needs at least {num_control_and_base} wires." f"{len(new_wires)} provided."
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
    def basis(self):
        """Used in compilation."""
        return self.base.basis

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
        if all(self.control_values):
            base_eigvals = self.base.eigvals()
            ones = np.ones(2 ** len(self.control_wires))
            return qml.math.concatenate([ones, base_eigvals])
        return super().eigvals()

    def generator(self):
        sub_gen = self.base.generator()
        proj_ones = np.ones(len(self.control_wires), dtype=int, requires_grad=False)
        proj = qml.Projector(proj_ones, wires=self.control_wires)
        return 1.0 * proj @ sub_gen

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
    def grad_method(self):
        return self.base.grad_method
