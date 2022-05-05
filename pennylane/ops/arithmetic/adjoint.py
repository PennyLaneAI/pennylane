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
This submodule defines an operation modifier that indicates the adjoint of an operator.
"""
from pennylane.operation import Operator, Operation, AnyWires, AdjointUndefinedError
from pennylane.queuing import QueuingContext, QueuingError
from pennylane.math import transpose, conj


class Adjoint(Operator):
    """
    The Adjoint of an operator.

    Args:
        base (~.operation.Operator): The operator that is adjointed.

    .. seealso:: :func:`~.adjoint`, :meth:`~.operation.Operator.adjoint`

    **Example:**

    >>> op = Adjoint(qml.S(0))
    >>> op.name
    'Adjoint(S)'
    >>> qml.matrix(op)
    array([[1.-0.j, 0.-0.j],
       [0.-0.j, 0.-1.j]])
    >>> qml.generator(Adjoint(qml.RX(1.0, wires=0)))
    (PauliX(wires=[0]), 0.5)
    >>> Adjoint(qml.RX(1.234, wires=0)).data
    [1.234]

    """

    num_wires = AnyWires

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_base = self.base.__copy__()
        copied_op.base = copied_base
        copied_op._hyperparameters = {"base": copied_base}
        for attr, value in vars(self).items():
            if attr not in {"data", "base", "_hyperparameters"}:
                setattr(copied_op, attr, value)

        return copied_op

    def __init__(self, base=None, do_queue=True, id=None):
        self.base = base
        self.hyperparameters["base"] = base
        super().__init__(*base.parameters, wires=base.wires, do_queue=do_queue, id=id)
        self._name = f"Adjoint({self.base.name})"

    @property
    def data(self):
        return self.base.data

    @data.setter
    def data(self, new_data):
        """Allows us to set base operation parameters."""
        self.base.data = new_data

    def queue(self, context=QueuingContext):
        try:
            context.update_info(self.base, owner=self)
        except QueuingError:
            self.base.queue(context=context)
            context.update_info(self.base, owner=self)

        context.append(self, owns=self.base)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals, base_label, cache=cache) + "†"

    @staticmethod
    def compute_matrix(*params, base=None):
        base_matrix = base.compute_matrix(*params, **base.hyperparameters)
        return transpose(conj(base_matrix))

    @staticmethod
    def compute_decomposition(*params, wires, base=None):
        try:
            return [base.adjoint()]
        except AdjointUndefinedError:
            base_decomp = base.compute_decomposition(*params, wires, **base.hyperparameters)
            return [Adjoint(op) for op in reversed(base_decomp)]

    def sparse_matrix(self, wires=None):
        base_matrix = self.base.sparse_matrix(wires=wires)
        return transpose(conj(base_matrix))

    def get_eigvals(self):
        # Cannot define ``compute_eigvals`` because Hermitian only defines ``get_eigvals``
        return [conj(x) for x in self.base.get_eigvals()]

    @staticmethod
    def compute_diagonalizing_gates(*params, wires, base=None):
        return base.compute_diagonalizing_gates(*params, wires, **base.hyperparameters)

    @property
    def has_matrix(self):
        return self.base.has_matrix

    def adjoint(self):
        return self.base

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns ``_queue_cateogory`` for base operator.
        """
        return self.base._queue_category

    ## Operation specific properties ##########################################

    @property
    def grad_method(self):
        return self.base.grad_method

    @property
    def grad_recipe(self):
        return self.base.grad_recipe

    @property
    def basis(self):
        return self.base.basis

    @property
    def control_wires(self):
        return self.base.control_wires

    @property
    def single_qubit_rot_angles(self):
        omega, theta, phi = self.base.single_qubit_rot_angles
        return -phi, -theta, -omega

    # get_parameter_shift ?
    @property
    def parameter_frequencies(self):
        return self.base.parameter_frequencies

    def generator(self):
        if isinstance(self.base, Operation):  # stand in for being unitary and inverse=adjoint
            return -1.0 * self.base.generator()
        return super().generator()
