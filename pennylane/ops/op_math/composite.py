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
# pylint: disable=too-many-instance-attributes
import abc
from typing import Callable, List
import copy

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires

# pylint: disable=too-many-instance-attributes


class CompositeOp(Operator):
    """A base class for operators that are composed of other operators.

    Args:
        operands: (tuple[~.operation.Operator]): a tuple of operators which will be combined.

    Keyword Args:
        id (str or None): id for the operator. Default is None.

    The child composite operator should define the `_op_symbol` property
    during initialization and define any relevant representations, such as
    :meth:`~.operation.Operator.matrix` and :meth:`~.operation.Operator.decomposition`.
    """

    def _flatten(self):
        return tuple(self.operands), tuple()

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data)

    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(
        self, *operands: Operator, id=None, _pauli_rep=None
    ):  # pylint: disable=super-init-not-called
        self._id = id
        self.queue_idx = None
        self._name = self.__class__.__name__

        self.operands = operands
        self._wires = qml.wires.Wires.all_wires([op.wires for op in operands])
        self._hash = None
        self._has_overlapping_wires = None
        self._overlapping_ops = None
        self._pauli_rep = self._build_pauli_rep() if _pauli_rep is None else _pauli_rep
        self.queue()
        self._batch_size = _UNSET_BATCH_SIZE

    def _check_batching(self):
        batch_sizes = {op.batch_size for op in self if op.batch_size is not None}
        if len(batch_sizes) > 1:
            raise ValueError(
                "Broadcasting was attempted but the broadcasted dimensions "
                f"do not match: {batch_sizes}."
            )
        self._batch_size = batch_sizes.pop() if batch_sizes else None

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
        return tuple(d for op in self for d in op.data)

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for op in self:
            op_num_params = op.num_params
            if op_num_params > 0:
                op.data = new_data[:op_num_params]
                new_data = new_data[op_num_params:]

    @property
    def num_wires(self):
        """Number of wires the operator acts on."""
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
        return all(op.has_matrix or isinstance(op, qml.ops.Hamiltonian) for op in self)

    def eigvals(self):
        """Return the eigenvalues of the specified operator.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the operator
        """
        eigvals = []
        for ops in self.overlapping_ops:
            if len(ops) == 1:
                eigvals.append(
                    qml.utils.expand_vector(ops[0].eigvals(), list(ops[0].wires), list(self.wires))
                )
            else:
                tmp_composite = self.__class__(*ops)
                eigvals.append(
                    qml.utils.expand_vector(
                        tmp_composite.eigendecomposition["eigval"],
                        list(tmp_composite.wires),
                        list(self.wires),
                    )
                )

        return self._math_op(math.asarray(eigvals, like=math.get_deep_interface(eigvals)), axis=0)

    @abc.abstractmethod
    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""

    @property
    def overlapping_ops(self) -> List[List[Operator]]:
        """Groups all operands of the composite operator that act on overlapping wires.

        Returns:
            List[List[Operator]]: List of lists of operators that act on overlapping wires. All the
            inner lists commute with each other.
        """
        if self._overlapping_ops is None:
            overlapping_ops = []  # [(wires, [ops])]
            for op in self:
                ops = [op]
                wires = op.wires
                op_added = False
                for idx, (old_wires, old_ops) in enumerate(overlapping_ops):
                    if any(wire in old_wires for wire in wires):
                        overlapping_ops[idx] = (old_wires + wires, old_ops + ops)
                        op_added = True
                        break
                if not op_added:
                    overlapping_ops.append((op.wires, [op]))

            self._overlapping_ops = [overlapping_op[1] for overlapping_op in overlapping_ops]

        return self._overlapping_ops

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
        eigen_func = math.linalg.eigh if self.is_hermitian else math.linalg.eig

        if self.hash not in self._eigs:
            mat = self.matrix()
            w, U = eigen_func(mat)
            self._eigs[self.hash] = {"eigvec": U, "eigval": w}

        return self._eigs[self.hash]

    @property
    def has_diagonalizing_gates(self):
        if self.has_overlapping_wires:
            for ops in self.overlapping_ops:
                # if any of the single ops doesn't have diagonalizing gates, the overall operator doesn't either
                if len(ops) == 1 and not ops[0].has_diagonalizing_gates:
                    return False
            # the lists of ops with multiple operators can be handled if there is a matrix
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
        diag_gates = []
        for ops in self.overlapping_ops:
            if len(ops) == 1:
                diag_gates.extend(ops[0].diagonalizing_gates())
            else:
                tmp_sum = self.__class__(*ops)
                eigvecs = tmp_sum.eigendecomposition["eigvec"]
                diag_gates.append(
                    qml.QubitUnitary(math.transpose(math.conj(eigvecs)), wires=tmp_sum.wires)
                )
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

        >>> op = qml.S(0) + qml.X(0) + qml.Rot(1,2,3, wires=[1])
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
        if qml.QueuingManager.recording():
            for op in self:
                context.remove(op)
            context.append(self)
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

    # pylint:disable = missing-function-docstring
    @property
    def basis(self):
        return None

    @property
    def arithmetic_depth(self) -> int:
        return 1 + max(op.arithmetic_depth for op in self)

    @property
    @abc.abstractmethod
    def _math_op(self) -> Callable:
        """The function used when combining the operands of the composite operator"""

    def map_wires(self, wire_map: dict):
        # pylint:disable=protected-access
        cls = self.__class__
        new_op = cls.__new__(cls)
        new_op.operands = tuple(op.map_wires(wire_map=wire_map) for op in self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op.data = copy.copy(self.data)
        if self._overlapping_ops is not None:
            new_op._overlapping_ops = [
                [o.map_wires(wire_map) for o in _ops] for _ops in self._overlapping_ops
            ]
        else:
            new_op._overlapping_ops = None

        for attr, value in vars(self).items():
            if attr not in {"data", "operands", "_wires", "_overlapping_ops"}:
                setattr(new_op, attr, value)
        if (p_rep := new_op.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)

        return new_op

    @abc.abstractmethod
    def _build_pauli_rep(self):
        """The function to generate the pauli representation for the composite operator."""
