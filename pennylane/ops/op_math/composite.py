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
from typing import List

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator
from pennylane.wires import Wires

# pylint: disable=too-many-instance-attributes


class CompositeOp(Operator):
    """A base class for operators that are composed of other operators.

    Args:
        operands: (tuple[~.operation.Operator]): a tuple of operators which will be combined.

    Keyword Args:
        do_queue (bool): determines if the operator will be queued. Default is True.
        id (str or None): id for the operator. Default is None.

    The child composite operator should define the `_op_symbol` property
    during initialization and define any relevant representations, such as
    :meth:`~.operation.Operator.matrix` and :meth:`~.operation.Operator.decomposition`.
    """

    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(
        self, *operands: Operator, do_queue=True, id=None
    ):  # pylint: disable=super-init-not-called
        self._id = id
        self.queue_idx = None
        self._name = self.__class__.__name__

        if len(operands) < 2:
            raise ValueError(f"Require at least two operators to combine; got {len(operands)}")

        self.operands = operands
        self._wires = qml.wires.Wires.all_wires([op.wires for op in operands])
        self._hash = None
        self._has_overlapping_wires = None

        if do_queue:
            self.queue()

    def __repr__(self):
        return f" {self._op_symbol} ".join(
            [f"({op})" if op.arithmetic_depth > 0 else f"{op}" for op in self]
        )

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.operands = tuple(s.__copy__() for s in self)

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
    @abc.abstractmethod
    def _op_symbol(self) -> str:
        """The symbol used when visualizing the composite operator"""

    @property
    def data(self):
        """Create data property"""
        return [op.data for op in self]

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for new_entry, op in zip(new_data, self):
            op.data = new_entry

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def num_params(self):
        return sum(op.num_params for op in self)

    @property
    def has_overlapping_wires(self) -> bool:
        """Boolean expression that indicates if the factors have overlapping wires."""
        if self._has_overlapping_wires is None:
            wires = []
            for op in self:
                wires.extend(list(op.wires))
            self._has_overlapping_wires = len(wires) != len(set(wires))
        return self._has_overlapping_wires

    @property
    @abc.abstractmethod
    def is_hermitian(self):
        """This property determines if the composite operator is hermitian."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return all(op.has_matrix or isinstance(op, qml.Hamiltonian) for op in self)

    @abc.abstractmethod
    def eigvals(self):
        """Return the eigenvalues of the specified operator."""

    @abc.abstractmethod
    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""

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
        if self.hash not in self._eigs:
            mat = (
                self.matrix()
            )  # we no longer check for hermiticity and allow "anything" on simulator
            mat = math.to_numpy(mat)
            w, U = np.linalg.eig(mat)
            self._eigs[self.hash] = {"eigvec": U, "eigval": w}

        return self._eigs[self.hash]

    @property
    def has_diagonalizing_gates(self):
        if self.has_overlapping_wires:
            return self.has_matrix

        return all(op.has_diagonalizing_gates for op in self)

    def diagonalizing_gates(self):
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Returns:
            list[.Operator] or None: a list of operators
        """
        if self.has_overlapping_wires:
            eigen_vectors = self.eigendecomposition["eigvec"]
            return [qml.QubitUnitary(eigen_vectors.conj().T, wires=self.wires, unitary_check=False)]
        diag_gates = []
        for op in self:
            diag_gates.extend(op.diagonalizing_gates())
        return diag_gates

    def label(self, decimals=None, base_label=None, cache=None):
        r"""How the composite operator is represented in diagrams and drawings.

        Args:
            decimals (int): If ``None``, no parameters are included. Else,
                how to round the parameters. Defaults to ``None``.
            base_label (Iterable[str]): Overwrite the non-parameter component of the label.
                Must be same length as ``operands`` attribute. Defaults to ``None``.
            cache (dict): Dictionary that carries information between label calls
                in the same drawing. Defaults to ``None``.

        Returns:
            str: label to use in drawings

        **Example (using the Sum composite operator)**

        >>> op = qml.S(0) + qml.PauliX(0) + qml.Rot(1,2,3, wires=[1])
        >>> op.label()
        '(S+X)+Rot'
        >>> op.label(decimals=2, base_label=[["my_s", "my_x"], "inc_rot"])
        '(my_s+my_x)+inc_rot\n(1.00,\n2.00,\n3.00)'

        """

        def _label(op, decimals, base_label, cache):
            sub_label = op.label(decimals, base_label, cache)
            return f"({sub_label})" if op.arithmetic_depth > 0 else sub_label

        if base_label is not None:
            if isinstance(base_label, str) or len(base_label) != len(self):
                raise ValueError(
                    "Composite operator labels require ``base_label`` keyword to be same length as operands."
                )
            return self._op_symbol.join(
                _label(op, decimals, lbl, cache) for op, lbl in zip(self, base_label)
            )

        return self._op_symbol.join(_label(op, decimals, None, cache) for op in self)

    def queue(self, context=qml.QueuingManager):
        """Updates each operator's owner to self, this ensures
        that the operators are not applied to the circuit repeatedly."""
        for op in self:
            context.update_info(op, owner=self)
        context.append(self, owns=self.operands)
        return self

    @classmethod
    @abc.abstractmethod
    def _sort(cls, op_list, wire_map: dict = None) -> List[Operator]:
        """Sort composite operands by their wire indices."""

    @property
    def hash(self):
        if self._hash is None:
            self._hash = hash(
                (str(self.name), str([factor.hash for factor in self._sort(self.operands)]))
            )
        return self._hash

    @property
    def arithmetic_depth(self) -> int:
        return 1 + max(op.arithmetic_depth for op in self)

    def map_wires(self, wire_map: dict):
        cls = self.__class__
        new_op = cls.__new__(cls)
        new_op.operands = tuple(op.map_wires(wire_map=wire_map) for op in self)
        new_op._wires = Wires(  # pylint: disable=protected-access
            [wire_map.get(wire, wire) for wire in self.wires]
        )
        new_op.data = self.data.copy()
        for attr, value in vars(self).items():
            if attr not in {"data", "operands", "_wires"}:
                setattr(new_op, attr, value)

        return new_op
