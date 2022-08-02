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
This submodule defines a base class for symbolic operations representing operator math.
"""
from copy import copy

from pennylane.operation import Operator
from pennylane.queuing import QueuingContext


class SymbolicOp(Operator):
    """Developer-facing base class for single-operator symbolic operators.

    Args:
        base (~.operation.Operator): The base operation that is modified symbolicly
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified

    This *developer-facing* class can serve as a parent to single base symbolic operators, such as
    :class:`~.ops.op_math.Adjoint` and :class:`~.ops.op_math.Pow`.

    New symbolic operators can inherit from this class to recieve some common default behavior, such as
    deferring properties to the the base class, copying the base class during a shallow copy, and updating
    the metadata of the base operator during queueing.

    The child symbolic operator should define the `_name` property during initialization and define any
    relevant representations, such as :meth:`~.operation.Operator.matrix`, :meth:`~.operation.Operator.diagonalizing_gates`,
    :meth:`~.operation.Operator.eigvals`, and :meth:`~.operation.Operator.decomposition`.
    """

    _name = "Symbolic"

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten because the base must be copied too.
        copied_op = object.__new__(type(self))
        # copied_op must maintain inheritance structure of self
        # Relevant for symbolic ops that mix in operation-specific components.

        for attr, value in vars(self).items():
            if attr not in {"_hyperparameters"}:
                setattr(copied_op, attr, value)

        copied_op._hyperparameters = copy(self._hyperparameters)
        copied_op._hyperparameters["base"] = copy(self.base)

        return copied_op

    # pylint: disable=super-init-not-called
    def __init__(self, base=None, do_queue=True, id=None):
        self.hyperparameters["base"] = base
        self._id = id
        self.queue_idx = None

        if do_queue:
            self.queue()

    @property
    def base(self) -> Operator:
        """The base operator."""
        return self.hyperparameters["base"]

    @property
    def data(self):
        """The trainable parameters"""
        return self.base.data

    @data.setter
    def data(self, new_data):
        self.base.data = new_data

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def wires(self):
        return self.base.wires

    # pylint: disable=protected-access
    @property
    def _wires(self):
        return self.base._wires

    # pylint: disable=protected-access
    @_wires.setter
    def _wires(self, new_wires):
        self.base._wires = new_wires

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def ndim_params(self):
        return self.base.ndim_params

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return self.base.has_matrix

    @property
    def is_hermitian(self):
        return self.base.is_hermitian

    @property
    def _queue_category(self):
        return self.base._queue_category  # pylint: disable=protected-access

    def queue(self, context=QueuingContext):
        context.safe_update_info(self.base, owner=self)
        context.append(self, owns=self.base)
        return self
