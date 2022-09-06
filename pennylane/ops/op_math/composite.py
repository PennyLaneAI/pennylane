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
This submodule defines a base class for composite operations.
"""
import abc
import itertools
from copy import copy
from functools import reduce
from itertools import combinations
from typing import List, Tuple, Union

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator


class CompositeOp(Operator, abc.ABC):
    """A base class for operators that are composed of other operators.

    Args:
        operands: (tuple[~.operation.Operator]): a tuple of operators which will be combined.

    Keyword Args:
        do_queue (bool): determines if the operator will be queued. Default is True.
        id (str or None): id for the operator. Default is None.

    The child composite operator should define the `_name` and `_op_symbol` properties
    during initialization and define any relevant representations, such as
    :meth:`~.operation.Operator.matrix` and :meth:`~.operation.Operator.decomposition`.
    """

    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(self, *operands: Operator, do_queue=True, id=None):
        self._id = id
        self.queue_idx = None

        if not hasattr(self, "_name"):
            raise NotImplementedError("Child class must specify _name")
        if not hasattr(self, "_op_symbol"):
            raise NotImplementedError("Child class must specify _op_symbol")
        if len(operands) < 2:
            raise ValueError(f"Require at least two operators to combine; got {len(operands)}")

        self.operands = operands
        self._wires = qml.wires.Wires.all_wires([op.wires for op in self.operands])
        self._hash = None

        if do_queue:
            self.queue()

    def __repr__(self):
        return f" {self.op_symbol} ".join(
            [f"({op})" if op.arithmetic_depth > 0 else f"{op}" for op in self.operands]
        )

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.operands = tuple(s.__copy__() for s in self.operands)

        for attr, value in vars(self).items():
            if attr not in {"operands"}:
                setattr(copied_op, attr, value)  # TODO: exclude data?

        return copied_op

    def __iter__(self):
        """Return the iterator over the underlying operands."""
        return iter(self.operands)

    def __getitem__(self, idx):
        """Return the operand at position ``idx`` of the composition."""
        return self.operands[idx]

    def __len__(self):
        """Return the number of operators in this composite operator"""
        return len(self.operands)

    @property
    def op_symbol(self) -> str:
        """The symbol used when visualizing the composite operator"""
        return self._op_symbol

    @property
    def data(self):
        """Create data property"""
        return [op.parameters for op in self.operands]

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for new_entry, op in zip(new_data, self.operands):
            op.data = new_entry

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def num_params(self):
        return sum(op.num_params for op in self.operands)

    @property
    @abc.abstractmethod
    def is_hermitian(self):
        """
        TODO: should we do `all(op.is_hermitian for op in self.operands)` as default?
        """

    @abc.abstractmethod
    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""

    @abc.abstractmethod
    def sparse_matrix(self, wire_order=None):
        """Compute the sparse matrix representation of the composite operator."""

    @property
    def eigendecomposition(self):
        r"""Return the eigendecomposition of the matrix specified by the operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the
                eigenvectors of the operator.
        """
        Hmat = self.matrix()
        Hmat = math.to_numpy(Hmat)
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in self._eigs:
            w, U = np.linalg.eigh(Hmat)
            self._eigs[Hkey] = {"eigvec": U, "eigval": w}

        return self._eigs[Hkey]

    def diagonalizing_gates(self):
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Returns:
            list[.Operator] or None: a list of operators
        """

        eigen_vectors = self.eigendecomposition["eigvec"]
        return [qml.QubitUnitary(eigen_vectors.conj().T, wires=self.wires)]

    def eigvals(self):
        r"""Return the eigenvalues of the specified operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the operator
        """
        return self.eigendecomposition["eigval"]

    def label(self, decimals=None, base_label=None, cache=None):
        r"""How the composite operator is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Must be same length as ``operands`` attribute.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings
        """

        def _label(op, decimals, base_label, cache):
            sub_label = op.label(decimals, base_label, cache)
            return f"({sub_label})" if op.arithmetic_depth > 0 else sub_label

        if base_label is not None:
            if isinstance(base_label, str) or len(base_label) != len(self.operands):
                raise ValueError(
                    "Composite operator labels require ``base_label`` keyword to be same length as operands."
                )
            return self.op_symbol.join(
                _label(op, decimals, lbl, cache) for op, lbl in zip(self.operands, base_label)
            )

        return self.op_symbol.join(_label(op, decimals, None, cache) for op in self.operands)

    def queue(self, context=qml.QueuingContext):
        """Updates each operator's owner to self, this ensures
        that the operators are not applied to the circuit repeatedly."""
        for op in self.operands:
            context.safe_update_info(op, owner=self)
        context.append(self, owns=self.operands)
        return self

    @property
    def arithmetic_depth(self) -> int:
        return 1 + max(op.arithmetic_depth for op in self.operands)
