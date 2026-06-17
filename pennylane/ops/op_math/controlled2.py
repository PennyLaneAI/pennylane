# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the base class for controlled operators."""

from collections.abc import Sequence
from inspect import signature
from typing import Literal

from scipy import sparse
from typing_extensions import override

import pennylane as qp
from pennylane import math
from pennylane.core.operator import Operator
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.exceptions import SparseMatrixUndefinedError
from pennylane.wires import Wires, WiresLike

from .controlled import Controlled
from .symbolicop2 import SymbolicOp2


class Controlled2(SymbolicOp2, is_baseclass=True):  # pylint: disable=too-many-public-methods
    """The base class for controlled operators.

    This class acts as a common interface for all operators that can be considered controlled
    operators. The main purpose of this class is to provide common properties such as ``base``,
    ``control_values``, ``control_wires``, etc., and implement default implementations for
    methods such as ``compute_matrix``, ``compute_eigvals``, etc.

    .. note::

        This class is an interface that is not meant to be instantiated. To properly create a
        controlled version of an individual operator, use the :class:`ControlledOp2` instead.

    **Example**

    .. code-block:: python

        from pennylane.ops import Controlled2

        class CRot(Controlled2):

            dynamic_argnames = ("phi", "theta", "omega")

            wires_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, phi, theta, omega, wires):
                super().__init__(qp.Rot(phi, theta, omega, wires=wires[1]), control_wires=wires[0])

    >>> op = CRot(0.1, 0.2, 0.3, wires=[0, 1])
    >>> isinstance(op, Controlled2)
    True
    >>> op.control_wires
    Wires([0])
    >>> op.target_wires
    Wires([1])
    >>> op.matrix()
    array([[ 1.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  1.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.j        ,
             0.97517...-0.19767...j, -0.09933...+0.00996...j],
           [ 0.        +0.j        ,  0.        +0.j        ,
             0.09933...+0.00996...j,  0.97517...+0.19767...j]])

    """

    _init_args: dict  # initialized in __new__, declared here for type checking purposes.
    """Arguments that the operator is initialized with."""

    def __new__(cls, *args, **kwargs):
        # The purpose of this function here is to intercept the argument passed to the
        # constructor of the subclass and store it on the operator, so that in __init__,
        # we can pass that along to the base Operator2.__init__, which expects the
        # arguments to match the pre-defined signature of the subclass.
        obj = super().__new__(cls)
        sig = signature(cls)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        obj._init_args = bound_args.arguments
        return obj

    def __init__(  # pylint: disable=too-many-arguments
        self,
        base: Operator,
        control_wires: WiresLike,
        control_values: Sequence[int | bool] | None = None,
        work_wires: WiresLike | None = None,
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
    ):

        control_wires = Wires(control_wires)
        work_wires = Wires([] if work_wires is None else work_wires)

        if Wires.shared_wires([base.wires, control_wires]):
            raise ValueError("control_wires must not overlap with the base operator.")

        if Wires.shared_wires([work_wires, base.wires + control_wires]):
            raise ValueError("work_wires must not overlap with the operator or control_wires.")

        accepted = {"zeroed", "borrowed"}
        if work_wire_type not in accepted:
            raise ValueError(f"work_wire_type must be one of {accepted}. Got '{work_wire_type}'.")

        if control_values is None:
            control_values = [True] * len(control_wires)

        if isinstance(control_values, (int, bool)):
            control_values = [bool(control_values)]

        if len(control_values) != len(control_wires):
            raise ValueError("control_values should be the same length as control_wires")

        control_values = [bool(v) for v in control_values]

        self._base = base
        self._control_wires = control_wires
        self._control_values = control_values
        self._work_wires = work_wires
        self._work_wire_type = work_wire_type

        if "control_wires" in self._init_args:
            self._init_args["control_wires"] = control_wires

        if "control_values" in self._init_args:
            self._init_args["control_values"] = control_values

        if "work_wires" in self._init_args:
            self._init_args["work_wires"] = work_wires

        super().__init__(**self._init_args)

    def __init_subclass__(cls, is_baseclass=False) -> None:

        super().__init_subclass__(is_baseclass)

        base_argnames = {"base", "control_wires", "control_values", "work_wires", "work_wire_type"}
        if set(signature(cls).parameters.keys()) == base_argnames:
            return

        if cls.compute_matrix is Controlled2.compute_matrix:

            @staticmethod
            def _compute_matrix(*args, **kwargs):
                op = cls(*args, **kwargs)
                return Controlled2.compute_matrix(op.base, op.control_wires, op.control_values)

            cls.compute_matrix = _compute_matrix

        if cls.compute_sparse_matrix is Controlled2.compute_sparse_matrix:

            @staticmethod
            def _compute_sparse_matrix(*args, format="csr", **kwargs):
                op = cls(*args, **kwargs)
                return Controlled2.compute_sparse_matrix(
                    op.base,
                    op.control_wires,
                    op.control_values,
                    format=format,
                )

            cls.compute_sparse_matrix = _compute_sparse_matrix

        if cls.compute_eigvals is Controlled2.compute_eigvals:

            @staticmethod
            def _compute_eigvals(*args, **kwargs):
                op = cls(*args, **kwargs)
                return Controlled2.compute_eigvals(op.base, op.control_wires)

            cls.compute_eigvals = _compute_eigvals

        if cls.compute_diagonalizing_gates is Controlled2.compute_diagonalizing_gates:

            @staticmethod
            def _compute_diagonalizing_gates(*args, **kwargs):
                op = cls(*args, **kwargs)
                return Controlled2.compute_diagonalizing_gates(op.base)

            cls.compute_diagonalizing_gates = _compute_diagonalizing_gates

    @property
    @override
    def base(self) -> Operator:
        """The target operator."""
        return self._base

    @property
    def target_wires(self) -> Wires:
        """The wires of the target operator."""
        return self.base.wires

    @property
    def control_wires(self) -> Wires:
        """The control wires."""
        return self._control_wires

    @property
    def control_values(self) -> Sequence[bool]:
        """For each control wire, denote whether to control on ``True`` or ``False``"""
        return self._control_values

    @property
    def work_wires(self) -> Wires:
        """Auxiliary wires that can be used in the decomposition."""
        return self._work_wires

    @property
    def work_wire_type(self):
        """The type of work wires, can be ``"zeroed"`` or ``"borrowed"``"""
        return self._work_wire_type

    @property
    @override
    def wires(self):
        return self.control_wires + self.target_wires

    @staticmethod
    @override
    # pylint: disable=arguments-differ
    def compute_matrix(base, control_wires, control_values, **_):

        base_matrix = base.matrix()
        interface = math.get_interface(base_matrix)

        num_target_states = 2 ** len(base.wires)
        num_control_states = 2 ** len(control_wires)
        total_matrix_size = num_control_states * num_target_states

        padding_left = _bool_array_to_int(control_values) * num_target_states
        padding_right = total_matrix_size - padding_left - num_target_states

        padding_left = math.cast_like(math.eye(padding_left, like=interface), 1j)
        padding_right = math.cast_like(math.eye(padding_right, like=interface), 1j)

        left_pad = math.convert_like(padding_left, base_matrix)
        right_pad = math.convert_like(padding_right, base_matrix)

        shape = math.shape(base_matrix)
        if len(shape) == 3:  # stack if batching
            return math.stack([math.block_diag([left_pad, _U, right_pad]) for _U in base_matrix])

        return math.block_diag([left_pad, base_matrix, right_pad])

    @property
    @override
    def has_sparse_matrix(self) -> bool:
        return self.base.has_sparse_matrix or self.base.has_matrix

    @staticmethod
    @override
    # pylint: disable=arguments-differ
    def compute_sparse_matrix(base, control_wires, control_values, format="csr", **_):

        target_matrix = _get_sparse_matrix(base)

        num_target_states = 2 ** len(base.wires)
        num_control_states = 2 ** len(control_wires)
        total_states = num_target_states * num_control_states

        start_idx = _bool_array_to_int(control_values) * num_target_states
        end_idx = start_idx + num_target_states

        m = sparse.eye(total_states, format="lil", dtype=target_matrix.dtype)
        m[start_idx:end_idx, start_idx:end_idx] = target_matrix
        return m.asformat(format=format)

    @staticmethod
    @override
    # pylint: disable=arguments-differ
    def compute_eigvals(base, control_wires, **_):
        base_eigvals = base.eigvals()
        num_target_wires = len(base.wires)
        num_control_wires = len(control_wires)
        total = 2 ** (num_target_wires + num_control_wires)
        ones = math.ones(total - len(base_eigvals))
        return math.concatenate([ones, base_eigvals])

    @property
    @override
    # pylint: disable=invalid-overridden-method,arguments-differ
    def has_adjoint(self):
        return self.base.has_adjoint

    @override
    def adjoint(self):
        return qp.ctrl(
            self.base.adjoint(),
            self.control_wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
            work_wire_type=self.work_wire_type,
        )

    @property
    @override
    # pylint: disable=invalid-overridden-method,arguments-differ
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    @staticmethod
    @override
    # pylint: disable=arguments-differ
    def compute_diagonalizing_gates(base, **_):
        return base.diagonalizing_gates()

    @property
    @override
    # pylint: disable=invalid-overridden-method,arguments-differ
    def has_generator(self):
        return self.base.has_generator

    @override
    def generator(self):
        return qp.prod(
            *(
                qp.Projector([v], wires=w)
                for v, w in zip(self.control_values, self.control_wires, strict=True)
            ),
            self.base.generator(),
        )

    @override
    def simplify(self):
        if isinstance(self.base, (Controlled, Controlled2)):

            simplified_base = self.base.base.simplify()
            if isinstance(simplified_base, qp.Identity):
                return simplified_base

            return qp.ctrl(
                simplified_base,
                control=self.control_wires + self.base.control_wires,
                control_values=self.control_values + self.base.control_values,
                work_wires=self.work_wires + self.base.work_wires,
                work_wire_type=resolve_work_wire_type(
                    self.base.work_wires,
                    self.base.work_wire_type,
                    self.work_wires,
                    self.work_wire_type,
                ),
            )

        simplified_base = self.base.simplify()
        if isinstance(simplified_base, qp.Identity):
            return simplified_base

        return qp.ctrl(
            op=simplified_base,
            control=self.control_wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
            work_wire_type=self.work_wire_type,
        )

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)


def _get_sparse_matrix(base):
    if base.has_sparse_matrix:
        return base.sparse_matrix()
    if base.has_matrix:
        return sparse.lil_matrix(base.matrix())
    raise SparseMatrixUndefinedError()  # pragma: no cover


def _bool_array_to_int(arr: list[bool]):
    return sum(2**i for i, val in enumerate(reversed(arr)) if val)


class ControlledOp2(Controlled2):  # pylint: disable=too-few-public-methods
    """Represents a controlled version of an arbitrary base operator.

    Args:
        base (~.core.Operator): the operator that is controlled
        control_wires (WiresLike): The wires to control on.

    Keyword Args:
        control_values (Iterable[Bool]): The values to control on for each control wire.
            Defaults to ``True`` for all control wires.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition
        work_wire_type: The type of work wire(s), can be ``"zeroed"`` or ``"borrowed"``.
            ``"zeroed"`` indicates that the work wires are in the :math:`|0\rangle` state,
            whereas ``"borrowed"`` work wires can be in any arbitrary state. In both cases,
            it is expected that the work wires are restored to their original states by
            the end of the decomposition. Defaults to ``"borrowed"``.

    """

    dynamic_argnames = ("control_values",)

    wire_argnames = ("control_wires", "work_wires")

    hybrid_argnames = ("base",)

    compilable_argnames = ("work_wire_type",)

    def __init__(  # pylint: disable=too-many-arguments,useless-parent-delegation
        self,
        base: Operator,
        control_wires: WiresLike,
        control_values: Sequence[int | bool] | None = None,
        work_wires: WiresLike | None = None,
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
    ):
        super().__init__(base, control_wires, control_values, work_wires, work_wire_type)

    @property
    @override
    def name(self):
        return f"C({self.base.name})"

    def __repr__(self):
        params = [f"control_wires={self.control_wires.tolist()}"]
        if self.work_wires:
            params.append(f"work_wires={self.work_wires.tolist()}")
        if self.control_values and not all(self.control_values):
            params.append(f"control_values={self.control_values}")
        return f"Controlled({self.base}, {', '.join(params)})"
