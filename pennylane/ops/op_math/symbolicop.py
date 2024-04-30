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
from abc import abstractmethod
from copy import copy

import numpy as np

import pennylane as qml
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.queuing import QueuingManager


class SymbolicOp(Operator):
    """Developer-facing base class for single-operator symbolic operators.

    Args:
        base (~.operation.Operator): the base operation that is modified symbolicly
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified

    This *developer-facing* class can serve as a parent to single base symbolic operators, such as
    :class:`~.ops.op_math.Adjoint`.

    New symbolic operators can inherit from this class to receive some common default behaviour, such
    as deferring properties to the base class, copying the base class during a shallow copy, and
    updating the metadata of the base operator during queueing.

    The child symbolic operator should define the `_name` property during initialization and define
    any relevant representations, such as :meth:`~.operation.Operator.matrix`,
    :meth:`~.operation.Operator.diagonalizing_gates`, :meth:`~.operation.Operator.eigvals`, and
    :meth:`~.operation.Operator.decomposition`.
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

        copied_op._hyperparameters = copy(self.hyperparameters)
        copied_op.hyperparameters["base"] = copy(self.base)

        return copied_op

    # pylint: disable=super-init-not-called
    def __init__(self, base, id=None):
        self.hyperparameters["base"] = base
        self._id = id
        self.queue_idx = None
        self._pauli_rep = None
        self.queue()

    @property
    def batch_size(self):
        return self.base.batch_size

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

    # pylint:disable = missing-function-docstring
    @property
    def basis(self):
        return self.base.basis

    @property
    def num_wires(self):
        """Number of wires the operator acts on."""
        return len(self.wires)

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

    def queue(self, context=QueuingManager):
        context.remove(self.base)
        context.append(self)
        return self

    @property
    def arithmetic_depth(self) -> int:
        return 1 + self.base.arithmetic_depth

    @property
    def hash(self):
        return hash(
            (
                str(self.name),
                self.base.hash,
            )
        )

    def map_wires(self, wire_map: dict):
        new_op = copy(self)
        new_op.hyperparameters["base"] = self.base.map_wires(wire_map=wire_map)
        if (p_rep := new_op.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)  # pylint:disable=protected-access
        return new_op


class ScalarSymbolicOp(SymbolicOp):
    """Developer-facing base class for single-operator symbolic operators that contain a
    scalar coefficient.

    Args:
        base (~.operation.Operator): the base operation that is modified symbolicly
        scalar (float): the scalar coefficient
        id (str): custom label given to an operator instance, can be useful for some applications
            where the instance has to be identified

    This *developer-facing* class can serve as a parent to single base symbolic operators, such as
    :class:`~.ops.op_math.SProd` and :class:`~.ops.op_math.Pow`.
    """

    _name = "ScalarSymbolicOp"

    def __init__(self, base, scalar: float, id=None):
        self.scalar = np.array(scalar) if isinstance(scalar, list) else scalar
        super().__init__(base, id=id)
        self._batch_size = _UNSET_BATCH_SIZE

    @property
    def batch_size(self):
        if self._batch_size is _UNSET_BATCH_SIZE:
            base_batch_size = self.base.batch_size
            if qml.math.ndim(self.scalar) == 0:
                # coeff is not batched
                self._batch_size = base_batch_size
            else:
                # coeff is batched
                scalar_size = qml.math.size(self.scalar)
                if base_batch_size is not None and base_batch_size != scalar_size:
                    raise ValueError(
                        "Broadcasting was attempted but the broadcasted dimensions "
                        f"do not match: {scalar_size}, {base_batch_size}."
                    )
                self._batch_size = scalar_size
        return self._batch_size

    @property
    def data(self):
        return (self.scalar, *self.base.data)

    @data.setter
    def data(self, new_data):
        self.scalar = new_data[0]
        self.base.data = new_data[1:]

    @property
    def has_matrix(self):
        return self.base.has_matrix or isinstance(self.base, qml.ops.Hamiltonian)

    @property
    def hash(self):
        return hash(
            (
                str(self.name),
                str(self.scalar),
                self.base.hash,
            )
        )

    @staticmethod
    @abstractmethod
    def _matrix(scalar, mat):
        """Scalar-matrix operation that doesn't take into account batching.

        ``ScalarSymbolicOp.matrix`` will call this method to compute the matrix for a single scalar
        and base matrix.

        Args:
            scalar (Union[int, float]): non-broadcasted scalar
            mat (ndarray): non-broadcasted matrix
        """

    def matrix(self, wire_order=None):
        r"""Representation of the operator as a matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        If the matrix depends on trainable parameters, the result
        will be cast in the same autodifferentiation framework as the parameters.

        A ``MatrixUndefinedError`` is raised if the base matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator.compute_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the
            operator's wires

        Returns:
            tensor_like: matrix representation
        """
        # compute base matrix
        if isinstance(self.base, qml.ops.Hamiltonian):
            base_matrix = qml.matrix(self.base)
        else:
            base_matrix = self.base.matrix()

        scalar_interface = qml.math.get_interface(self.scalar)
        scalar = self.scalar
        if scalar_interface == "torch":
            # otherwise get `RuntimeError: Can't call numpy() on Tensor that requires grad.`
            base_matrix = qml.math.convert_like(base_matrix, self.scalar)
        elif scalar_interface == "tensorflow":
            # just cast everything to complex128. Otherwise we may have casting problems
            # where things get truncated like in SProd(tf.Variable(0.1), qml.X(0))
            scalar = qml.math.cast(scalar, "complex128")
            base_matrix = qml.math.cast(base_matrix, "complex128")

        # compute scalar operation on base matrix taking batching into account
        scalar_size = qml.math.size(scalar)
        if scalar_size != 1:
            if scalar_size == self.base.batch_size:
                # both base and scalar are broadcasted
                mat = qml.math.stack([self._matrix(s, m) for s, m in zip(scalar, base_matrix)])
            else:
                # only scalar is broadcasted
                mat = qml.math.stack([self._matrix(s, base_matrix) for s in scalar])
        elif self.base.batch_size is not None:
            # only base is broadcasted
            mat = qml.math.stack([self._matrix(scalar, ar2) for ar2 in base_matrix])
        else:
            # none are broadcasted
            mat = self._matrix(scalar, base_matrix)

        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)
