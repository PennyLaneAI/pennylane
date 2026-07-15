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
from textwrap import dedent
from typing import Literal

from scipy import sparse
from typing_extensions import override

import pennylane as qp
from pennylane import allocation, math
from pennylane.core.operator import Operator, abstractify
from pennylane.core.operator.operator2 import operator_p, pop_op_eqns  # tach-ignore
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    _decomp_contains_mcm,
    get_fixed_decomp,
    list_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import (
    AbstractOperatorLike,
    CompressedResourceOp,
    controlled_resource_rep,
    resolve_work_wire_type,
    resource_rep,
)
from pennylane.exceptions import SparseMatrixUndefinedError
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.typing import AbstractWires, Bool, Wire
from pennylane.wires import Wires, WiresLike

from .symbolicop2 import SymbolicOp2

# pylint: disable=unused-argument,protected-access,no-value-for-parameter


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

            wire_argnames = ("wires",)

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

        obj = super().__new__(cls)

        # NOTE: If called without arguments (during a __copy__)
        # skip signature binding as attributes will be restored
        # during the copy.
        if not args and not kwargs:
            return obj

        # The purpose of this function here is to intercept the argument passed to the
        # constructor of the subclass and store it on the operator, so that in __init__,
        # we can pass that along to the base Operator2.__init__, which expects the
        # arguments to match the pre-defined signature of the subclass.
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

    @override
    def __abstract_init__(  # pylint: disable=too-many-arguments,arguments-differ
        self,
        base: Operator,
        control_wires: WiresLike,
        control_values: Sequence[int | bool] | None = None,
        work_wires: WiresLike | None = None,
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
    ):

        # abstractify the wires
        if work_wires is None:
            work_wires = Wire[0]
        if not isinstance(work_wires, AbstractWires):
            work_wires = abstractify(Wires(work_wires))
        if not isinstance(control_wires, AbstractWires):
            control_wires = abstractify(Wires(control_wires))

        # abstractify control values
        if control_values is None:
            control_values = Bool[len(control_wires)]
        elif isinstance(control_values, (int, bool)):
            control_values = Bool[1]
        elif isinstance(control_values, (list, tuple, Wires)):
            control_values = Bool[len(control_values)]

        # abstractify the base
        base = abstractify(base)

        # initialize the interface properties
        self._base = base
        self._control_wires = control_wires
        self._control_values = control_values
        self._work_wires = work_wires
        self._work_wire_type = work_wire_type

        if "base" in self._init_args:
            self._init_args["base"] = base

        if "control_wires" in self._init_args:
            self._init_args["control_wires"] = control_wires

        if "control_values" in self._init_args:
            self._init_args["control_values"] = control_values

        if "work_wires" in self._init_args:
            self._init_args["work_wires"] = work_wires

        super().__abstract_init__(**self._init_args)

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
        if isinstance(self.base, (qp.ops.Controlled, Controlled2)):
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


# There are a couple of reasons for defining this as a distinct subclass of Controlled2 instead
# of having just a single Controlled2 class that serve both as a base class interface and a
# concrete operator type. See the PR description of #9647 for more details.
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

    static_argnames = ("work_wire_type",)

    arg_specs = {"control_values": Bool[-1], "control_wires": Wire[-1], "work_wires": Wire[-1]}

    # We cannot remove this __init__, otherwise signature(ControlledOp2) will return the
    # signature of Controlled2.__new__, which is just (*args, **kwargs). When __new__ is
    # overridden with a different signature, we must override __init__ so that the signature
    # of the __init__ is correctly retrieved as the signature of the operator subclass.
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
        ctrl_wires = (
            self.control_wires.tolist()
            if isinstance(self.control_wires, Wires)
            else self.control_wires
        )
        work_wires = (
            self.work_wires.tolist() if isinstance(self.work_wires, Wires) else self.work_wires
        )
        params = [f"control_wires={ctrl_wires}"]
        if self.work_wires:
            params.append(f"work_wires={work_wires}")
        if self.control_values and not all(self.control_values):
            params.append(f"control_values={self.control_values}")
        return f"Controlled({self.base}, {', '.join(params)})"

    @property
    def has_decomposition(self):  # pylint: disable=arguments-differ,invalid-overridden-method
        return any(rule.is_applicable(**self.arguments) for rule in list_decomps(self))

    @override
    def _bind_primitive(self):
        """Bind the operator primitive. ``ControlledOp2`` has to override the method of
        the base ``Operator2`` class so that we can "edit" the original primitive."""
        if not qp.capture.enabled():
            return

        if self.base.tracer is None:
            # pylint: disable=protected-access
            self.base._bind_primitive()
            # NOTE: `self.base.tracer` can still be `None` if we're not in a tracing context.
            # In that case, there is nothing to do, so return early.
            if self.base.tracer is None:
                return

        eqns = pop_op_eqns((self.base,))
        assert len(eqns) == 1, f"Expected exactly one plxpr equation for {self.base}."
        params = eqns[0].params
        n_ctrls = params["n_ctrls"]

        # `eqns` contains `TracingEqns`, not `JaxprEqns`, so invars during tracing will just
        # be tracers, not `Var`s wrapping abstract values.
        if n_ctrls == 0:
            invars = eqns[0].invars + self.control_wires.tolist() + self.control_values
        else:
            # invars are ordered as (*other_args, *control_wires, *control_values), so we
            # need to insert the new control wires before the old ones, and do the same
            # for control values too.
            control_wires = self.control_wires.tolist() + eqns[0].invars[-2 * n_ctrls : -n_ctrls]
            control_values = self.control_values + eqns[0].invars[-n_ctrls:]
            invars = eqns[0].invars[: -2 * n_ctrls] + control_wires + control_values

        params["n_ctrls"] += len(self.control_wires)
        res = operator_p.bind(*invars, **params)

        self.base.tracer = None
        # If we bind the primitive outside a tracing context but with program capture enabled,
        # `res`` will be a concrete operator, not an abstract tracer, so we don't save it.
        if math.is_abstract(res):
            self.tracer = res


@list_decomps.register
def _list_controlled_decomps(op: ControlledOp2) -> DecompCollection:
    """Get all the decomposition rules applicable to this operator."""

    op = abstractify(op)

    # fixed_decomps should override everything
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])

    # Special case for flipping the order of control and adjoint. We prefer to have adjoint
    # wrapping a controlled operator instead of the other way around because there is more
    # likely custom decomposition rules registered for the controlled version.
    if isinstance(op.base, Adjoint2):
        return DecompCollection([flip_control_adjoint])

    # Get custom rules registered for this controlled operator.
    custom_rules = list_decomps.dispatch(object)(op)

    # Get general fallback rules.
    general_rules = DecompCollection([])
    if op.base.has_matrix and len(op.base.wires) == 1:
        general_rules.append(to_controlled_unitary)
    if len(op.control_wires) > 2:
        general_rules.append(ctrl_single_work_wire)

    # Populate controlled versions of the base decomposition rules.
    wrapped_rules = DecompCollection(
        [
            _make_controlled_decomp(rule)
            for rule in list_decomps(op.base)
            if not _decomp_contains_mcm(rule, op.base.arguments)
        ]
    )

    return custom_rules + wrapped_rules + general_rules


def _make_controlled_decomp(base_rule: DecompositionRule):

    def _condition_fn(base, **_):
        return base_rule.is_applicable(**base.arguments)

    def _resource_fn(base, control_wires, control_values, work_wires, work_wire_type):
        base_counts = base_rule.compute_resources(**base.arguments).gate_counts
        # TODO: we need a better startegy for control values, but for now
        #       we're assuming that half the control values are 0s
        gate_counts = {
            _ctrl_abstract(op, control_wires, work_wires, work_wire_type): count
            for op, count in base_counts.items()
        }
        gate_counts[qp.X] = len(control_values)
        return gate_counts

    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_rule._work_wire_spec,
        exact=False,  # TODO:: no reliable way to tell whether control values has 0s.
        name=f"controlled({base_rule.name})",
    )
    def _impl(base, control_wires, control_values, work_wires, work_wire_type):

        @qp.for_loop(0, len(control_values))
        def _x_flips(i):
            qp.cond(qp.math.logical_not(control_values[i]), qp.X)(control_wires[i])

        _x_flips()
        qp.ctrl(
            base_rule._impl,  # pylint: disable=protected-access
            control=control_wires,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )(**base.arguments)
        _x_flips()

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_rule._source).strip()
    )
    return _impl


def _flip_control_adjoint_resource(base, control_wires, control_values, work_wires, work_wire_type):
    return {
        Adjoint2(
            qp.ctrl(
                base.base,
                control=control_wires,
                control_values=control_values,
                work_wires=work_wires,
                work_wire_type=work_wire_type,
            )
        ): 1
    }


@register_resources(_flip_control_adjoint_resource)
def flip_control_adjoint(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose the control of an adjoint by applying control to the base of the adjoint
    and taking the adjoint of the control."""
    qp.adjoint(
        qp.ctrl(
            base.base,
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    )


def _to_controlled_qu_resource(base, control_wires, control_values, work_wires, work_wire_type):
    return {
        resource_rep(
            qp.ControlledQubitUnitary,
            num_target_wires=1,
            num_control_wires=len(control_wires),
            # TODO: again assuming that half the control values are 0s, fix
            #       when we have a better solution here.
            num_zero_control_values=len(control_wires) // 2,
            num_work_wires=len(work_wires),
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_to_controlled_qu_resource)
def to_controlled_unitary(base, control_wires, control_values, work_wires, work_wire_type):
    """Convert a controlled operator to a controlled qubit unitary."""
    qp.ControlledQubitUnitary(
        base.matrix(),
        wires=control_wires + base.wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


def flip_zero_control(rule: DecompositionRule, name: str = "") -> DecompositionRule:
    """Wraps a decomposition for a controlled operator with X gates to flip zero control wires."""

    def _condition_fn(*args, **kwargs):
        return rule.is_applicable(*args, **kwargs)

    def _resource_fn(base, control_wires, control_values, work_wires, work_wire_type):
        gate_counts = rule.compute_resources(
            base=base,
            control_wires=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ).gate_counts
        # TODO: in the eye of the decomposition graph, we're essentially just adding PauliX
        #       gates for no reason. It'll be like this until we have a better solution.
        base_x_count = gate_counts.get(qp.X, 0)
        gate_counts[qp.X] = base_x_count + len(control_values)
        return gate_counts

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=rule._work_wire_spec,
        exact=False,
        name=name or f"flip_zero_ctrl_values({rule.name})",
    )
    def _impl(base, control_wires, control_values, work_wires, work_wire_type):

        @qp.for_loop(0, len(control_values))
        def _x_flips(i):
            qp.cond(qp.math.logical_not(control_values[i]), qp.X)(control_wires[i])

        _x_flips()
        rule(
            base,
            control_wires=control_wires,
            control_values=None,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        _x_flips()

    base_source = rule._source
    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere inner_decomp is defined as:\n\n"
        + dedent(base_source).strip()
    )
    return _impl


def _ctrl_single_work_wire_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return {
        _ctrl_abstract(
            base,
            control_wires=Wire[1],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1,
        _ctrl_abstract(qp.X, Wire[len(control_wires)], Wire[len(work_wires)], work_wire_type): 2,
    }


# pylint: disable=protected-access,unused-argument
@register_resources(_ctrl_single_work_wire_resource, work_wires={"zeroed": 1})
def _ctrl_single_work_wire(base, control_wires, control_values, work_wires, work_wire_type):
    """Implements Lemma 7.11 from https://arxiv.org/abs/quant-ph/9503016."""
    with allocation.allocate(1, state="zero", restored=True) as aux:
        qp.ctrl(qp.X(aux[0]), control=control_wires)
        qp.ctrl(base, control=aux[0])
        qp.ctrl(qp.X(aux[0]), control=control_wires)


ctrl_single_work_wire = flip_zero_control(_ctrl_single_work_wire, name="ctrl_single_work_wire")


def _ctrl_abstract(
    op: AbstractOperatorLike | type[Operator],
    control_wires: AbstractWires,
    work_wires: AbstractWires = Wire[0],
    work_wire_type: str = "borrowed",
    num_zero_control_values: int = 0,
):
    op = abstractify(op)

    if isinstance(op, CompressedResourceOp):
        return controlled_resource_rep(
            op.op_type,
            op.params,
            num_control_wires=len(control_wires),
            num_zero_control_values=num_zero_control_values,
            num_work_wires=len(work_wires),
            work_wire_type=work_wire_type,
        )

    if not num_zero_control_values:
        return qp.ctrl(
            op,
            control=control_wires,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    return qp.ctrl(
        op,
        control=control_wires,
        control_values=Bool[len(control_wires)],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
