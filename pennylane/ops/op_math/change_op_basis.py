# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This submodule defines a class for compute-uncompute patterns.
"""

import copy
import inspect
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import reduce
from typing import override

from pennylane import capture, math
from pennylane.core import queuing
from pennylane.core.operator import Operator, Operator2
from pennylane.core.operator.operator2 import pop_op_eqns  # tach-ignore
from pennylane.core.qscript import make_qscript
from pennylane.decomposition import CompressedResourceOp, add_decomps, register_resources
from pennylane.ops.op_math import adjoint, ctrl, prod
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.controlled2 import _ctrl_abstract, flip_zero_control
from pennylane.typing import Wire
from pennylane.wires import Wires

from .composite import handle_recursion_error


def _validate_callable(func: Callable) -> None:
    """Validates that a callable has no unbound mandatory parameters."""
    sig = inspect.signature(func)

    for param in sig.parameters.values():
        # The function,
        #
        # def f(*args, **kwargs):
        #     pass
        #
        # technically doesn't have any required parameters.
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # If param has no default we can early exit
        if param.default is inspect.Parameter.empty:
            raise TypeError(
                "change_op_basis requires that Callable inputs have no unbound mandatory parameters. Please use functools.partial to bind them."
            )


def _is_abstract_operator(op) -> bool:
    """Return whether ``op`` is an operator-valued JAX tracer."""
    return math.is_abstract(op) and isinstance(op.aval, capture.AbstractOperator)


def _region_ops(region):
    """Return the operators in a region in execution order."""
    return region if isinstance(region, tuple) else (region,)


def _map_region(fn, region, *, reverse=False):
    """Apply ``fn`` to a region while retaining single-operator regions as operators."""
    ops = _region_ops(region)
    if reverse:
        ops = reversed(ops)
    mapped = tuple(fn(op) for op in ops)
    return mapped[0] if not isinstance(region, tuple) and len(mapped) == 1 else mapped


def _adjoint_region(region, *, abstract=False):
    """Return the adjoint of an ordered operator region."""
    adjoint_fn = _adjoint_abstract if abstract else adjoint
    return _map_region(adjoint_fn, region, reverse=True)


def _apply_op_or_func(op_or_func):
    if callable(op_or_func):
        _validate_callable(op_or_func)
        op_or_func()
    elif isinstance(op_or_func, tuple):
        for op in op_or_func:
            _apply_op_or_func(op)
    elif isinstance(op_or_func, Operator2):
        # NOTE: An Operator2 built outside the trace context has no equation
        # so we need to emit one.
        if op_or_func.tracer is None:
            # pylint: disable-next=protected-access
            op_or_func._bind_primitive()
    elif isinstance(op_or_func, Operator):
        queuing.apply(op_or_func)
    elif _is_abstract_operator(op_or_func):
        pass
    else:
        raise TypeError(
            f"The parameters to change_op_basis must be Operator or Callable, not {type(op_or_func)}"
        )


def _convert_to_region(op_or_func):
    if callable(op_or_func):
        _validate_callable(op_or_func)
        operations = tuple(make_qscript(op_or_func)().operations)
        if len(operations) == 1:
            return operations[0]

        # Legacy operations are not supported as leaves of an Operator2 hybrid pytree. Retain
        # the old Prod compatibility shell only for those regions until the contained operations
        # are migrated.
        if not all(isinstance(op, Operator2) for op in operations):
            return prod(*reversed(operations))
        return operations
    if isinstance(op_or_func, tuple):
        if not all(isinstance(op, Operator2) for op in op_or_func):
            return prod(*reversed(op_or_func))
        return op_or_func
    if isinstance(op_or_func, Operator):
        return op_or_func
    raise TypeError(
        f"The parameters to change_op_basis must be Operator or Callable, not {type(op_or_func)}"
    )


def _validate_operands(*operands):
    """Validate the operator operands accepted across concrete and abstract construction."""
    # pylint: disable=import-outside-toplevel
    from pennylane.ops.mid_measure import MidMeasure, PauliMeasure

    def _is_mid_measure(op):
        if isinstance(op, (MidMeasure, PauliMeasure)):
            return True
        return isinstance(op, CompressedResourceOp) and issubclass(
            op.op_type, (MidMeasure, PauliMeasure)
        )

    operators = tuple(op for region in operands for op in _region_ops(region))

    if any(
        isinstance(region, tuple)
        and not all(isinstance(op, Operator2) or _is_abstract_operator(op) for op in region)
        for region in operands
    ):
        raise TypeError("ChangeOpBasis regions can only contain Operator2 operators.")

    if any(_is_mid_measure(op) for op in operators):
        raise ValueError("Composite operators of mid-circuit measurements are not supported.")

    valid_operands = (Operator, CompressedResourceOp)
    if not all(isinstance(op, valid_operands) or _is_abstract_operator(op) for op in operators):
        raise TypeError("ChangeOpBasis operands must be operators.")


# pylint: disable=inconsistent-return-statements
def change_op_basis(
    compute_op: Operator | tuple[Operator, ...] | Callable,
    target_op: Operator | tuple[Operator, ...] | Callable,
    uncompute_op: Operator | tuple[Operator, ...] | Callable | None = None,
):
    """Construct an operator representing a compute-target-uncompute pattern.

    Args:
        compute_op (:class:`~.Operator` | tuple[Operator2, ...] | Callable): An operator, an
            ordered tuple of Operator2 operators, or a no-input callable that applies quantum
            operations.
        target_op (:class:`~.Operator` | tuple[Operator2, ...] | Callable): An operator, an ordered
            tuple of Operator2 operators, or a no-input callable that applies quantum operations.
        uncompute_op (None | :class:`~.Operator` | tuple[Operator2, ...] | Callable): An optional
            operator region. ``None`` applies the adjoint of ``compute_op``. Callable regions that
            still contain legacy operators use a temporary :class:`~.Prod` compatibility shell.

    Returns:
        ~ops.op_math.ChangeOpBasis: the operator representing the compute-uncompute pattern.

    Raises:
        TypeError: if any arguments are not ``Callable`` s or :class:`~.Operator` s, or a ``Callable`` argument has input parameters.

    **Example**

    Consider the following example involving a ``change_op_basis``. The compute, uncompute pattern
    is composed of a Quantum Fourier Transform (``QFT``), followed by a ``PhaseAdder``, and finally
    an inverse ``QFT``.

    .. code-block:: python

        import pennylane as qp
        from functools import partial

        qp.decomposition.enable_graph()

        dev = qp.device("default.qubit")
        @qp.qnode(dev)
        def circuit():
            qp.H(0)
            qp.CNOT([1,2])
            qp.ctrl(
                qp.change_op_basis(qp.QFT([1,2]), qp.PhaseAdder(1, x_wires=[1,2])),
                control=0
            )
            return qp.state()

        circuit2 = qp.decompose(circuit, max_expansion=1)

    When this circuit is decomposed, the ``compute_op`` and ``uncompute_op`` are not controlled,
    resulting in a much more resource-efficient decomposition:

    >>> print(qp.draw(circuit2)())
    0: ──H──────╭●────────────────┤ ╭State
    1: ─╭●─╭QFT─├PhaseAdder─╭QFT†─┤ ├State
    2: ─╰X─╰QFT─╰PhaseAdder─╰QFT†─┤ ╰State

    A ``Callable`` can also be provided as an argument to ``change_op_basis``. This can be a
    function that applies a series of ``Operation`` s. Since ``change_op_basis`` requires this
    ``Callable`` to have no input arguments, ``functools.partial`` can be used to absorb any
    necessary parameters.

    .. code-block:: python

        def my_compute_op(a, reg1, reg2):
            qp.BasisState(np.zeros(len(reg2)), reg2)
            qp.QFT(reg1)
            qp.RX(a, reg1[0])

        def my_target_op(wires):
            qp.PauliX(wires[0])

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            # Use partial to absorb any input parameters
            compute = partial(my_compute_op, 0.1, [0], [1])
            target = partial(my_target_op, [0])
            qp.change_op_basis(compute, target)
            return qp.state()

        circuit3 = qp.decompose(circuit, max_expansion=1)

    >>> print(qp.draw(circuit3)())
    0: ─╭RX(0.10)@QFT@|Ψ⟩──X─╭(RX(0.10)@QFT@|Ψ⟩)†─┤ ╭State
    1: ─╰RX(0.10)@QFT@|Ψ⟩────╰(RX(0.10)@QFT@|Ψ⟩)†─┤ ╰State

    .. warning::

        There is limited support for passing callables to ``change_op_basis`` when program capture
        is enabled. Specifically, passing callables to ``qp.adjoint(qp.change_op_basis)(...)`` and
        ``qp.ctrl(qp.change_op_basis, control=...)(...)`` are not supported with ``@qp.qjit(capture=True)``

    .. seealso:: :class:`~.ops.op_math.ChangeOpBasis`

    """

    if capture.enabled():
        # NOTE: Need to pop any eagerly constructed operators present in the traced function
        # out of the jaxpr. This ensures that the order is kept consistent if any operators
        # were built outside of the traced function. '_apply_op_or_func' will bind the primitives
        # and insert them in the correct order.
        operands = (compute_op, target_op, uncompute_op)
        # Operator1 constructors return AbstractOperator tracers during capture, while
        # Operator2 constructors retain Python wrappers whose ``tracer`` attributes point to
        # their equations. If any operand is already an AbstractOperator tracer, preserve the
        # constructor order instead of moving only the Operator2 equations.
        if not any(
            _is_abstract_operator(op)
            for region in operands
            if region is not None
            for op in _region_ops(region)
        ):
            for _op in (
                op for region in operands if region is not None for op in _region_ops(region)
            ):
                if isinstance(_op, Operator2) and _op.tracer is not None:
                    pop_op_eqns((_op,))
        _apply_op_or_func(compute_op)
        _apply_op_or_func(target_op)
        if uncompute_op is not None:
            _apply_op_or_func(uncompute_op)
        elif isinstance(compute_op, (Operator2, tuple)):
            for op in reversed(_region_ops(compute_op)):
                if isinstance(op, Operator2):
                    # NOTE: The new Adjoint2 will consume the operator as a hybrid pytree
                    # argument. Feed it a detached copy because its equation has already been
                    # moved into execution order above.
                    op = copy.copy(op)
                    op.tracer = None
                _apply_op_or_func(adjoint(op))
        else:
            _apply_op_or_func(adjoint(compute_op))
    else:
        return ChangeOpBasis(
            _convert_to_region(compute_op),
            _convert_to_region(target_op),
            _convert_to_region(uncompute_op) if uncompute_op is not None else None,
        )


class ChangeOpBasis(Operator2):
    """
    Composite operator representing a compute-uncompute pattern of operators, which constitutes changing the basis in
    which an operator is applied.

    Args:
        compute_op (:class:`~.Operator` | tuple[Operator2, ...]): The compute region, in execution
            order.
        target_op (:class:`~.Operator` | tuple[Operator2, ...]): The target region, in execution
            order.
        uncompute_op (:class:`~.Operator` | tuple[Operator2, ...]): The uncompute region, in
            execution order. Defaults to the adjoint of ``compute_op``.

    Returns:
        (Operator): Returns an Operator which is the change_op_basis of the provided Operators: compute_op, target_op, uncompute_op.

    .. note::
        Iterating over a ``ChangeOpBasis`` yields its three regions in matrix-product order:
        uncompute, target, then compute. Operators inside a tuple region are stored and applied in
        execution order.

    .. seealso:: :func:`~.change_op_basis`
    """

    wire_argnames = ()
    hybrid_argnames = ("compute_op", "target_op", "uncompute_op")
    arg_specs = {}

    def __init__(
        self,
        compute_op: Operator | tuple[Operator, ...],
        target_op: Operator | tuple[Operator, ...],
        uncompute_op: Operator | tuple[Operator, ...] | None = None,
    ):
        if uncompute_op is None:
            uncompute_op = _map_region(
                lambda op: (
                    _adjoint_abstract(op) if isinstance(op, CompressedResourceOp) else adjoint(op)
                ),
                compute_op,
                reverse=True,
            )

        _validate_operands(compute_op, target_op, uncompute_op)

        super().__init__(compute_op, target_op, uncompute_op)

        # Operator2 automatically collects wires from Operator2-valued hybrid arguments. Retain
        # support for legacy Operator operands until their own migrations are complete.
        flat_operands = tuple(op for region in self.operands for op in _region_ops(region))
        if all(isinstance(op, Operator) for op in flat_operands):
            self._wires = Wires.all_wires([op.wires for op in flat_operands])
            self._pauli_rep = self._build_pauli_rep()
        else:
            self._wires = Wire[0]
            self._is_abstract = True

    @override
    def __abstract_init__(self, *args, **kwargs):
        bound_args = self._sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        if bound_args.arguments["uncompute_op"] is None:
            bound_args.arguments["uncompute_op"] = _adjoint_region(
                bound_args.arguments["compute_op"], abstract=True
            )
        _validate_operands(*bound_args.arguments.values())
        super().__abstract_init__(*bound_args.args, **bound_args.kwargs)
        if isinstance(self._wires, Wires):
            self._wires = Wire[0]

    @property
    def operands(self):
        """The factors in matrix-product order."""
        return self.uncompute_op, self.target_op, self.compute_op

    def __iter__(self):
        return iter(self.operands)

    def __getitem__(self, idx):
        return self.operands[idx]

    def __len__(self):
        return len(self.operands)

    @property
    def num_wires(self):
        """Number of wires the operator acts on."""
        return len(self.wires)

    def __repr__(self):
        def _repr_region(region):
            return " @ ".join(
                f"({op})" if getattr(op, "arithmetic_depth", 0) > 0 else f"{op}"
                for op in reversed(_region_ops(region))
            )

        return " @ ".join(_repr_region(region) for region in self.operands)

    @handle_recursion_error
    def __hash__(self):
        return hash(
            (
                self.name,
                tuple(tuple(hash(op) for op in _region_ops(region)) for region in self.operands),
            )
        )

    @handle_recursion_error
    def label(self, decimals=None, base_label=None, cache=None):
        def _label_op(op, operand_label):
            sub_label = op.label(decimals, operand_label, cache)
            return f"({sub_label})" if op.arithmetic_depth > 0 else sub_label

        def _label_region(region, operand_label):
            if not isinstance(region, tuple):
                return _label_op(region, operand_label)
            if isinstance(region, tuple) and isinstance(operand_label, str):
                return operand_label

            labels = reversed(operand_label) if isinstance(operand_label, tuple) else None
            return "@".join(
                _label_op(op, label)
                for op, label in zip(
                    reversed(_region_ops(region)),
                    labels or (None for _ in _region_ops(region)),
                    strict=True,
                )
            )

        if base_label is not None:
            if isinstance(base_label, str) or len(base_label) != len(self):
                raise ValueError(
                    "Composite operator labels require ``base_label`` keyword to be same length "
                    "as operands."
                )
            return "@".join(
                _label_region(region, operand_label)
                for region, operand_label in zip(self, base_label, strict=True)
            )

        return "@".join(_label_region(region, None) for region in self)

    @property
    @handle_recursion_error
    def data(self):
        return tuple(
            data for region in self for op in reversed(_region_ops(region)) for data in op.data
        )

    @property
    @handle_recursion_error
    def num_params(self):
        return len(self.data)

    grad_method = None

    @property
    @override
    def arithmetic_depth(self):
        return 1 + max(
            (getattr(op, "arithmetic_depth", 0) for region in self for op in _region_ops(region)),
            default=0,
        )

    @property
    @override
    def is_verified_hermitian(self):
        """Check if the product operator is hermitian.

        Note, this check is not exhaustive. There can be hermitian operators for which this check
        yields false, which ARE hermitian. So a false result only implies that a more explicit check
        must be performed.
        """
        target_ops = _region_ops(self.target_op)
        return len(target_ops) == 1 and target_ops[0].is_verified_hermitian

    @override
    def adjoint(self):
        return ChangeOpBasis(
            _map_region(lambda op: adjoint(op, lazy=False), self.uncompute_op, reverse=True),
            _map_region(lambda op: adjoint(op, lazy=False), self.target_op, reverse=True),
            _map_region(lambda op: adjoint(op, lazy=False), self.compute_op, reverse=True),
        )

    @override
    def queue(self, context=queuing.QueuingManager):
        if self.is_abstract:
            return self
        if context.recording():
            for region in self:
                for op in _region_ops(region):
                    context.remove(op)
            context.append(self)
        return self

    @override
    def map_wires(self, wire_map):
        return type(self)(
            _map_region(lambda op: op.map_wires(wire_map), self.compute_op),
            _map_region(lambda op: op.map_wires(wire_map), self.target_op),
            _map_region(lambda op: op.map_wires(wire_map), self.uncompute_op),
        )

    def _build_pauli_rep(self):
        """PauliSentence representation of the Product of operations."""
        matrix_order_ops = [op for region in self.operands for op in reversed(_region_ops(region))]
        if all(operand_pauli_reps := [op.pauli_rep for op in matrix_order_ops]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None


def _change_op_basis_resources(compute_op, target_op, uncompute_op):
    resources = Counter()

    for region in (compute_op, target_op, uncompute_op):
        resources.update(_region_ops(region))

    return resources


def _controlled_change_op_basis_resources(
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):  # pylint: disable=unused-argument
    resources = defaultdict(int)
    for op in _region_ops(base.compute_op):
        resources[op] += 1
    for op in _region_ops(base.target_op):
        resources[
            _ctrl_abstract(
                op,
                Wire[len(control_wires)],
                Wire[len(work_wires)],
                work_wire_type,
            )
        ] += 1
    for op in _region_ops(base.uncompute_op):
        resources[op] += 1
    return resources


@register_resources(_controlled_change_op_basis_resources)
def _controlled_change_op_basis_decomposition(
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    for op in _region_ops(base.compute_op):
        queuing.apply(op)
    for op in _region_ops(base.target_op):
        ctrl(
            queuing.apply(op),
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    for op in _region_ops(base.uncompute_op):
        queuing.apply(op)


@register_resources(_change_op_basis_resources)
def _change_op_basis_decomp(compute_op, target_op, uncompute_op):
    for region in (compute_op, target_op, uncompute_op):
        for op in _region_ops(region):
            queuing.apply(op)


add_decomps(ChangeOpBasis, _change_op_basis_decomp)
add_decomps("C(ChangeOpBasis)", flip_zero_control(_controlled_change_op_basis_decomposition))
