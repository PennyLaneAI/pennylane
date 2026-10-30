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

import inspect
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import reduce

from pennylane import capture, math
from pennylane.core import queuing
from pennylane.core.apply import apply
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.exceptions import (
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    MatrixUndefinedError,
    SparseMatrixUndefinedError,
)
from pennylane.ops.op_math import adjoint, ctrl, prod
from pennylane.ops.op_math.adjoint import _adjoint_eager
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import ChangeOpBasis2
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.pytrees import flatten, unflatten
from pennylane.typing import Wire

from .composite import CompositeOp, handle_recursion_error


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
    return math.is_tracer(op) and isinstance(op.aval, capture.AbstractOperator)


def _convert_to_prod(op_or_func):
    if callable(op_or_func):
        _validate_callable(op_or_func)
        return prod(op_or_func)()
    if isinstance(op_or_func, Operator):
        return op_or_func
    raise TypeError(
        f"The parameters to change_op_basis must be Operator or Callable, not {type(op_or_func)}"
    )


def change_op_basis(
    compute_op: Operator | Callable,
    target_op: Operator | Callable,
    uncompute_op: Operator | Callable | None = None,
):
    """Construct an operator representing a compute-uncompute pattern.

    Args:
        compute_op (~.Operator or Callable): operator applied before the target
        target_op (~.Operator or Callable): operator applied in the changed basis
        uncompute_op (~.Operator or Callable or None): Operator applied after the target.
            ``None`` uses the adjoint of ``compute_op``.

    Returns:
        ~ops.op_math.ChangeOpBasis or ~ops.op_math.ChangeOpBasis2: the operator representing the
        compute-uncompute pattern

    Raises:
        TypeError: if an operand is not an operator or a supported callable

    .. note::

        With program capture enabled, all operands must be operators. Use :func:`~.subcircuit`
        to turn a quantum function with registered resources into an operator class, then pass
        an instance of that class. The basis-change operator is preserved in the captured program.

        With program capture disabled, callables with no unbound mandatory parameters are also
        accepted.

    .. seealso:: :func:`~.subcircuit`, :class:`~.ops.op_math.ChangeOpBasis`,
        :class:`~.ops.op_math.ChangeOpBasis2`

    **Example**

    Consider the following example involving a ``change_op_basis``. The compute-uncompute pattern
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

        circuit2 = qp.transforms._tape_decompose(circuit, max_expansion=1)

    When this circuit is decomposed, the ``compute_op`` and ``uncompute_op`` are not controlled,
    resulting in a much more resource-efficient decomposition:

    >>> print(qp.draw(circuit2)())
    0: ──H──────╭●────────────────┤ ╭State
    1: ─╭●─╭QFT─├PhaseAdder─╭QFT†─┤ ├State
    2: ─╰X─╰QFT─╰PhaseAdder─╰QFT†─┤ ╰State

    With program capture disabled, a callable can also be provided as an argument. The callable
    applies a series of operations and must have no unbound mandatory parameters.
    ``functools.partial`` can be used to bind any necessary parameters.

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

        circuit3 = qp.transforms._tape_decompose(circuit, max_expansion=1)

    >>> print(qp.draw(circuit3)())
    0: ─╭RX(0.10)@QFT@|Ψ⟩──X─╭(RX(0.10)@QFT@|Ψ⟩)†─┤ ╭State
    1: ─╰RX(0.10)@QFT@|Ψ⟩────╰(RX(0.10)@QFT@|Ψ⟩)†─┤ ╰State

    .. details::
        :title: Program capture

        Instantiate a :func:`~.subcircuit` to use a quantum function as an operand:

        .. code-block:: python

            @qp.subcircuit(dynamic_argnames=("angle",))
            @qp.register_resources({qp.H: 1, qp.RX: 1})
            def Compute(angle, wires):
                qp.H(wires)
                qp.RX(angle, wires)

            qp.capture.enable()
            op = qp.change_op_basis(Compute(0.2, wires=0), qp.RZ(0.3, 0))

        >>> isinstance(op, qp.ops.ChangeOpBasis2)
        True
        >>> qp.capture.disable()

    """

    if capture.enabled():
        operands = (compute_op, target_op)
        if uncompute_op is not None:
            operands += (uncompute_op,)
        for op in operands:
            if callable(op):
                raise TypeError(
                    "change_op_basis does not support callables as inputs when using program "
                    "capture. To create an operator from a quantum function, use qp.subcircuit."
                )
            if not (isinstance(op, Operator) or _is_abstract_operator(op)):
                raise TypeError(
                    "The parameters to change_op_basis must be Operator when using program "
                    f"capture, not {type(op)}"
                )
        compute, target, uncompute = compute_op, target_op, uncompute_op
    else:
        compute = _convert_to_prod(compute_op)
        target = _convert_to_prod(target_op)
        uncompute = _convert_to_prod(uncompute_op) if uncompute_op is not None else None
    if (
        isinstance(compute, Operator2)
        and isinstance(target, Operator2)
        and (isinstance(uncompute, Operator2) or uncompute is None)
    ):
        return ChangeOpBasis2(
            compute,
            target,
            uncompute,
        )
    return ChangeOpBasis(compute, target, uncompute)


class ChangeOpBasis(CompositeOp):
    """
    Composite operator representing a compute-uncompute pattern of operators, which constitutes changing the basis in
    which an operator is applied.

    Args:
        compute_op (:class:`~.Operator`): A single operator or product that applies quantum operations.
        target_op (:class:`~.Operator`): A single operator or a product that applies quantum operations.
        uncompute_op (:class:`~.Operator`): A single operator or a product that applies quantum operations.
            Default is uncompute_op=qp.adjoint(compute_op).

    Returns:
        (Operator): Returns an Operator which is the change_op_basis of the provided Operators: compute_op, target_op, uncompute_op.

    .. note::
        When a ``ChangeOpBasis`` operator is iterated over, its factors are iterated in the reverse order. This is to
        have a similar behaviour to ``Prod`` which applies its factors in reverse order.

    .. seealso:: :func:`~.change_op_basis`
    """

    def __init__(self, compute_op: Operator, target_op: Operator, uncompute_op: Operator = None):
        if uncompute_op is None:
            uncompute_op = adjoint(compute_op)
        super().__init__(uncompute_op, target_op, compute_op)

    def _flatten(self):
        return tuple(reversed(self.operands)), tuple()

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, compute_op, target_op, uncompute_op=None):
        if uncompute_op is None:
            uncompute_op = adjoint(compute_op)

        leaves, structure = flatten(
            (compute_op, target_op, uncompute_op), is_leaf=lambda x: isinstance(x, Operator)
        )

        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, Operator2):
                if leaf.tracer is None:
                    # pylint: disable-next=protected-access
                    leaf._bind_primitive()
                new_leaves.append(leaf if leaf.tracer is None else leaf.tracer)
                # The equation is now an input to the legacy primitive. Detach its tracer
                # so prebuilt operands can be reused in another trace.
                leaf.tracer = None
            elif isinstance(leaf, Operator):
                new_leaves.append(apply(leaf))
            else:
                new_leaves.append(leaf)

        compute_op, target_op, uncompute_op = unflatten(new_leaves, structure)

        return cls._primitive.bind(compute_op, target_op, uncompute_op)

    resource_keys = frozenset({"compute_op", "target_op", "uncompute_op"})

    has_matrix = False
    has_sparse_matrix = False

    _op_symbol = "@"
    _math_op = staticmethod(math.prod)

    def matrix(self, wire_order=None):
        raise MatrixUndefinedError

    def sparse_matrix(self, wire_order=None, format="csr"):
        raise SparseMatrixUndefinedError

    def diagonalizing_gates(self):
        raise DiagGatesUndefinedError

    def eigvals(self):
        raise EigvalsUndefinedError

    @property
    @handle_recursion_error
    def resource_params(self):
        return {
            "compute_op": abstractify(self[2]),
            "target_op": abstractify(self[1]),
            "uncompute_op": abstractify(self[0]),
        }

    grad_method = None

    @classmethod
    def _sort(cls, op_list: list, wire_map: dict = None) -> list[Operator]:
        """
        We do not sort the ops. The order is guaranteed to matter since if the compute operator
        and the base operator commute, the pattern would simplify to just being the base operator.

        Args:
            op_list (List[.Operator]): list of operators to be sorted
            wire_map (dict): Dictionary containing the wire values as keys and its indexes as values.
                Defaults to None.

        Returns:
            List[.Operator]: sorted list of operators
        """
        return op_list

    @property
    def is_verified_hermitian(self):
        """Check if the product operator is hermitian.

        Note, this check is not exhaustive. There can be hermitian operators for which this check
        yields false, which ARE hermitian. So a false result only implies that a more explicit check
        must be performed.
        """
        return self[1].is_verified_hermitian

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self):
        return True

    def decomposition(self):
        r"""Decomposition of the product operator is given by each of compute_op, target_op, compute_op† applied in succession."""
        if queuing.QueuingManager.recording():
            _ = [apply(op) for op in reversed(self)]
        return list(self[::-1])

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_adjoint(self):
        return True

    def adjoint(self):
        return ChangeOpBasis(*(_adjoint_eager(factor, update_queue=True) for factor in self))

    def _build_pauli_rep(self):
        """PauliSentence representation of the Product of operations."""
        if all(operand_pauli_reps := [op.pauli_rep for op in self.operands[::-1]]):
            return reduce(lambda a, b: a @ b, operand_pauli_reps) if operand_pauli_reps else None
        return None


def _change_op_basis_resources(compute_op, target_op, uncompute_op):
    resources = Counter()

    resources[compute_op] += 1
    resources[target_op] += 1
    resources[uncompute_op] += 1

    return resources


def _adjoint_change_op_basis_resources(base_params, **_):
    resources = defaultdict(int)
    resources[base_params["compute_op"]] += 1
    resources[base_params["uncompute_op"]] += 1
    target_op = base_params["target_op"]
    resources[_adjoint_abstract(target_op)] += 1
    return resources


# pylint: disable=protected-access
@register_resources(_adjoint_change_op_basis_resources)
def _adjoint_change_op_basis_decomp(*_, base, **__):
    apply(base.operands[2])
    adjoint(apply(base.operands[1]))
    apply(base.operands[0])


add_decomps("Adjoint(ChangeOpBasis)", _adjoint_change_op_basis_decomp)


def _controlled_change_op_basis_resources(
    *_,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
    base_class,
    base_params,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    resources = defaultdict(int)
    resources[base_params["compute_op"]] += 1
    resources[
        _ctrl_abstract(
            base_params["target_op"],
            Wire[num_control_wires],
            Wire[num_work_wires],
            work_wire_type,
            num_zero_control_values,
        )
    ] += 1
    resources[base_params["uncompute_op"]] += 1
    return resources


@register_resources(_controlled_change_op_basis_resources)
def _controlled_change_op_basis_decomposition(
    *_,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
    base,
    **__,
):
    apply(base.operands[2])
    ctrl(
        apply(base.operands[1]),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    apply(base.operands[0])


# pylint: disable=unused-argument
@register_resources(_change_op_basis_resources)
def _change_op_basis_decomp(*_, wires=None, operands, **__):
    for op in operands[::-1]:
        apply(op)


add_decomps(ChangeOpBasis, _change_op_basis_decomp)
add_decomps("C(ChangeOpBasis)", _controlled_change_op_basis_decomposition)
