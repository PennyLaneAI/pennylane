# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the ``DecompositionRule`` class to represent a decomposition rule."""

from __future__ import annotations

import inspect
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from textwrap import dedent
from typing import overload

from pennylane.operation import Operator

from .resources import Resources, auto_wrap
from .utils import translate_op_alias


@dataclass(frozen=True)
class WorkWireSpec:
    """The number of each type of work wires that a decomposition rule requires."""

    zeroed: int = 0
    r"""Zeroed wires are guaranteed to be in the :math:`|0\rangle` state initially, and they
    must be restored to the :math:`|0\rangle>` state before deallocation."""

    borrowed: int = 0
    """Borrowed wires could be allocated in any state, and they must be restored to their
    initial state before deallocation."""

    burnable: int = 0
    r"""Burnable wires are guaranteed to be in the :math:`|0\rangle` state initially, and they
    could be deallocated in any arbitrary state."""

    garbage: int = 0
    """Garbage wires could be allocated in any state, and can be deallocated in any state."""

    @property
    def total(self) -> int:
        """The total number of work wires."""
        return self.zeroed + self.borrowed + self.burnable + self.garbage


@overload
def register_condition(condition: Callable) -> Callable[[Callable], DecompositionRule]: ...
@overload
def register_condition(condition: Callable, qfunc: Callable) -> DecompositionRule: ...
def register_condition(
    condition: Callable[..., bool], qfunc: Callable | None = None
) -> Callable[[Callable], DecompositionRule] | DecompositionRule:
    """Binds a condition to a decomposition rule for when it is applicable.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via :func:`~pennylane.decomposition.enable_graph`. This new way of
        performing decompositions is generally more resource-efficient and accommodates multiple alternative
        decomposition rules for an operator. In this new system, custom decomposition rules are
        defined as quantum functions, and it is currently required that every decomposition rule
        declares its required resources using :func:`~.register_resources`.

    Args:
        condition (Callable): a function which takes the resource parameters of an operator as
            arguments and returns ``True`` or ``False`` based on whether the decomposition rule
            is applicable to an operator with the given resource parameters.
        qfunc (Callable): the quantum function that implements the decomposition. If ``None``,
            returns a decorator for acting on a function.

    Returns:
        DecompositionRule:
            a data structure that represents a decomposition rule, which contains a PennyLane
            quantum function representing the decomposition, and its resource function.

    **Example**

    This function can be used as a decorator to bind a condition function to a quantum function
    that implements a decomposition rule.

    .. code-block:: python

        import pennylane as qml
        from pennylane.math.decomposition import zyz_rotation_angles

        # The parameters must be consistent with ``qml.QubitUnitary.resource_keys``
        def _zyz_condition(num_wires):
            return num_wires == 1

        @qml.register_condition(_zyz_condition)
        @qml.register_resources({qml.RZ: 2, qml.RY: 1, qml.GlobalPhase: 1})
        def zyz_decomposition(U, wires, **__):
            # Assumes that U is a 2x2 unitary matrix
            phi, theta, omega, phase = zyz_rotation_angles(U, return_global_phase=True)
            qml.RZ(phi, wires=wires[0])
            qml.RY(theta, wires=wires[0])
            qml.RZ(omega, wires=wires[0])
            qml.GlobalPhase(-phase)

        # This decomposition will be ignored for `QubitUnitary` on more than one wire.
        qml.add_decomps(qml.QubitUnitary, zyz_decomposition)

    """

    def _decorator(_qfunc) -> DecompositionRule:
        if not isinstance(_qfunc, DecompositionRule):
            _qfunc = DecompositionRule(_qfunc)
        _qfunc.add_condition(condition)
        return _qfunc

    return _decorator(qfunc) if qfunc else _decorator


@overload
def register_resources(
    ops: Callable | dict, *, work_wires: Callable | dict | None = None, exact: bool = True
) -> Callable[[Callable], DecompositionRule]: ...
@overload
def register_resources(
    ops: Callable | dict,
    qfunc: Callable,
    *,
    work_wires: Callable | dict | None = None,
    exact: bool = True,
) -> DecompositionRule: ...
def register_resources(
    ops: Callable | dict,
    qfunc: Callable | None = None,
    *,
    work_wires: Callable | dict | None = None,
    exact: bool = True,
) -> Callable[[Callable], DecompositionRule] | DecompositionRule:
    r"""Binds a quantum function to its required resources.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via :func:`~pennylane.decomposition.enable_graph`. This new way of
        doing decompositions is generally more resource efficient and accommodates multiple alternative
        decomposition rules for an operator. In this new system, custom decomposition rules are
        defined as quantum functions, and it is currently required that every decomposition rule
        declares its required resources using ``qml.register_resources``.

    Args:
        ops (dict or Callable): a dictionary mapping unique operators within the given ``qfunc``
            to their number of occurrences therein. If a function is provided instead of a static
            dictionary, a dictionary must be returned from the function. For more information,
            consult the "Quantum Functions as Decomposition Rules" section below.
        qfunc (Callable): the quantum function that implements the decomposition. If ``None``,
            returns a decorator for acting on a function.

    Keyword Args:
        work_wires (dict or Callable): a dictionary declaring the number of work wires of each type
            required to perform this decomposition. Accepted work wire types include ``"zeroed"``,
            ``"borrowed"``, ``"burnable"``, and ``"garbage"``. For more information, consult the
            "Dynamic Allocation of Work Wires" section below.
        exact (bool): whether the resources are computed exactly (``True``, default) or
            estimated heuristically (``False``). This information is only relevant for testing
            and validation purposes.

    Returns:
        DecompositionRule:
            a data structure that represents a decomposition rule, which contains a PennyLane
            quantum function representing the decomposition, and its resource function.


    **Example**

    This function can be used as a decorator to bind a quantum function to its required resources
    so that it can be used as a decomposition rule within the new graph-based decomposition system.

    .. code-block:: python

        from functools import partial
        import pennylane as qml

        qml.decomposition.enable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires, **_):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(qml.transforms.decompose, gate_set={qml.CZ, qml.H}, fixed_decomps={qml.CNOT: my_cnot})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()


    >>> print(qml.draw(circuit, level="device")())
    0: ────╭●────┤  State
    1: ──H─╰Z──H─┤  State

    Alternatively, the decomposition rule can be created in-line:

    >>> my_cnot = qml.register_resources({qml.H: 2, qml.CZ: 1}, my_cnot)

    .. details::
        :title: Quantum Functions as Decomposition Rules

        Quantum functions representing decomposition rules within the new decomposition system
        are expected to take ``(*op.parameters, op.wires, **op.hyperparameters)`` as arguments,
        where ``op`` is an instance of the operator type that the decomposition is for.

    .. details::
        :title: Operators with Dynamic Resource Requirements

        In many cases, the resource requirement of an operator's decomposition is not static; some
        operators have properties that directly affect the resource estimate of its decompositions,
        i.e., the types of gates that exist in the decomposition and their number of occurrences.

        For each operator class, the set of parameters that affects the type of gates and their
        number of occurrences in its decompositions is given by the ``resource_keys`` attribute.
        For example, the number of gates in the decomposition for ``qml.MultiRZ`` changes based
        on the number of wires it acts on, in contrast to the decomposition for ``qml.CNOT``:

        >>> qml.CNOT.resource_keys
        set()
        >>> qml.MultiRZ.resource_keys
        {'num_wires'}

        The output of ``resource_keys`` indicates that custom decompositions for the operator
        should be registered to a resource function (as opposed to a static dictionary) that
        accepts those exact arguments and returns a dictionary.

        .. code-block:: python

            def _multi_rz_resources(num_wires):
                return {
                    qml.CNOT: 2 * (num_wires - 1),
                    qml.RZ: 1
                }

            @qml.register_resources(_multi_rz_resources)
            def multi_rz_decomposition(theta, wires, **__):
                for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                    qml.CNOT(wires=(w0, w1))
                qml.RZ(theta, wires=wires[0])
                for w0, w1 in zip(wires[1:], wires[:-1]):
                    qml.CNOT(wires=(w0, w1))

        Additionally, if a custom decomposition for an operator contains gates that, in turn,
        have properties that affect their own decompositions, this information must also be
        included in the resource function. For example, if a decomposition rule produces a
        ``MultiRZ`` gate, it is not sufficient to declare the existence of a ``MultiRZ`` in the
        resource function; the number of wires it acts on must also be specified.

        Consider a fictitious operator with the following decomposition:

        .. code-block:: python

            def my_decomp(theta, wires):
                qml.MultiRZ(theta, wires=wires[:-1])
                qml.MultiRZ(theta, wires=wires)
                qml.MultiRZ(theta, wires=wires[1:])

        It contains two ``MultiRZ`` gates acting on ``len(wires) - 1`` wires (the first and last
        ``MultiRZ``) and one ``MultiRZ`` gate acting on exactly ``len(wires)`` wires. This
        distinction must be reflected in the resource function:

        .. code-block:: python

            def my_resources(num_wires):
                return {
                    qml.resource_rep(qml.MultiRZ, num_wires=num_wires - 1): 2,
                    qml.resource_rep(qml.MultiRZ, num_wires=num_wires): 1
                }

            my_decomp = qml.register_resources(my_resources, my_decomp)

        where :func:`~pennylane.resource_rep` is a utility function that wraps an operator type and any
        additional information relevant to its resource estimate into a compressed data structure.
        To check what (if any) additional information is required to declare an operator type
        in a resource function, refer to the ``resource_keys`` attribute of the :class:`~pennylane.operation.Operator`
        class. Operators with non-empty ``resource_keys`` must be declared using ``qml.resource_rep``,
        with keyword arguments matching its ``resource_keys`` exactly.

        .. seealso::

            :func:`~pennylane.resource_rep`

    .. details::
       :title: Dynamically Allocated Wires as a Resource

       Some decomposition rules make use of work wires, which can be dynamically requested within
       the quantum function using :func:`~pennylane.allocation.allocate`. Such decomposition rules
       should register the number of work wires they require so that the decomposition algorithm
       is able to budget the use of work wires across decomposition rules.

       There are four types of work wires:

       - "zeroed" wires are guaranteed to be in the :math:`|0\rangle` state initially, and they
         must be restored to the :math:`|0\rangle` state before deallocation.

       - "borrowed" wires are allocated in an arbitrary state, but they must be restored to the same initial state before deallocation.

       - "burnable" wires are guaranteed to be in the :math:`|0\rangle` state initially, but they
         can be deallocated in any arbitrary state.

       - "garbage" wires can be allocated in any state, and can be deallocated in any state.

       Here's a decomposition for a multi-controlled ``Rot`` that uses a zeroed work wire:

       .. code-block:: python

          from functools import partial
          import pennylane as qml
          from pennylane.allocation import allocate
          from pennylane.decomposition import controlled_resource_rep

          qml.decomposition.enable_graph()

          def _ops_fn(num_control_wires, **_):
              return {
                  controlled_resource_rep(qml.X, {}, num_control_wires): 2,
                  qml.CRot: 1
              }

          @qml.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
          @qml.register_resources(ops=_ops_fn, work_wires={"zeroed": 1})
          def _controlled_rot_decomp(*params, wires, **_):
              with allocate(1, require_zeros=True, restored=True) as work_wires:
                  qml.ctrl(qml.X(work_wires[0]), control=wires[:-1])
                  qml.CRot(*params, wires=[work_wires[0], wires[-1]])
                  qml.ctrl(qml.X(work_wires[0]), control=wires[:-1])

          @partial(qml.transforms.decompose, fixed_decomps={"C(Rot)": _controlled_rot_decomp})
          @qml.qnode(qml.device("default.qubit"))
          def circuit():
              qml.ctrl(qml.Rot(0.1, 0.2, 0.3, wires=3), control=[0, 1, 2])
              return qml.probs(wires=[0, 1, 2, 3])

       >>> print(qml.draw(circuit)())
       <DynamicWire>: ──Allocate─╭X─╭●───────────────────╭X──Deallocate─┤
                   0: ───────────├●─│────────────────────├●─────────────┤ ╭Probs
                   1: ───────────├●─│────────────────────├●─────────────┤ ├Probs
                   2: ───────────╰●─│────────────────────╰●─────────────┤ ├Probs
                   3: ──────────────╰Rot(0.10,0.20,0.30)────────────────┤ ╰Probs

    """

    def _decorator(_qfunc) -> DecompositionRule:
        if isinstance(_qfunc, DecompositionRule):
            _qfunc.set_resources(ops, exact_resources=exact)
            if work_wires:
                _qfunc.set_work_wire_spec(work_wires)
            return _qfunc
        return DecompositionRule(
            _qfunc, resources=ops, work_wires=work_wires, exact_resources=exact
        )

    return _decorator(qfunc) if qfunc else _decorator


class DecompositionRule:
    """Represents a decomposition rule for an operator."""

    def __init__(
        self,
        func: Callable,
        resources: Callable | dict | None = None,
        work_wires: Callable | dict | None = None,
        exact_resources: bool = True,
    ):

        self._impl = func

        try:
            self._source = inspect.getsource(func)
        except OSError:  # pragma: no cover
            # OSError is raised if the source code cannot be retrieved
            self._source = ""  # pragma: no cover

        if isinstance(resources, dict):

            def resource_fn(*_, **__):
                return resources

            self._compute_resources = resource_fn
        else:
            self._compute_resources = resources

        self._conditions = []
        self._work_wire_spec = work_wires or {}
        self.exact_resources = exact_resources

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)

    def __str__(self):
        return dedent(self._source).strip()

    def compute_resources(self, *args, **kwargs) -> Resources:
        """Computes the resources required to implement this decomposition rule."""
        if self._compute_resources is None:
            raise NotImplementedError("No resource estimation found for this decomposition rule.")
        raw_gate_counts = self._compute_resources(*args, **kwargs)
        assert isinstance(raw_gate_counts, dict), "Resource function must return a dictionary."
        gate_counter = Counter()
        for op, count in raw_gate_counts.items():
            if count > 0:
                gate_counter.update({auto_wrap(op): count})
        return Resources(dict(gate_counter))

    def is_applicable(self, *args, **kwargs) -> bool:
        """Checks whether this decomposition rule is applicable."""
        return all(condition(*args, **kwargs) for condition in self._conditions)

    def get_work_wire_spec(self, *args, **kwargs) -> WorkWireSpec:
        """Gets the work wire requirements of this decomposition rule"""
        if isinstance(self._work_wire_spec, dict):
            return WorkWireSpec(**self._work_wire_spec)
        return WorkWireSpec(**self._work_wire_spec(*args, **kwargs))

    def add_condition(self, condition: Callable[..., bool]) -> None:
        """Adds a condition for this decomposition rule."""
        self._conditions.append(condition)

    def set_resources(self, resources: Callable | dict, exact_resources: bool = True) -> None:
        """Sets the resources for this decomposition rule."""

        if isinstance(resources, dict):

            def resource_fn(*_, **__):
                return resources

            self._compute_resources = resource_fn
        else:
            self._compute_resources = resources
        self.exact_resources = exact_resources

    def set_work_wire_spec(self, work_wires: Callable | dict) -> None:
        """Sets the work wire usage of this decomposition rule."""
        self._work_wire_spec = work_wires


_decompositions = defaultdict(list)
"""dict[str, list[DecompositionRule]]: A dictionary mapping operator names to decomposition rules."""


def add_decomps(op_type: type[Operator] | str, *decomps: DecompositionRule) -> None:
    """Globally registers new decomposition rules with an operator class.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via :func:`~pennylane.decomposition.enable_graph`. This new way of
        doing decompositions is generally more resource efficient and accommodates multiple alternative
        decomposition rules for an operator. In this new system, custom decomposition rules are
        defined as quantum functions, and it is currently required that every decomposition rule
        declares its required resources using :func:`~pennylane.register_resources`

    In the new system of decompositions, multiple decomposition rules can be registered for the
    same operator class. The specified decomposition rules in ``add_decomps`` serve as alternative
    decomposition rules that may be chosen if they lead to a more resource-efficient decomposition.

    Args:
        op_type (type or str): the operator type for which new decomposition rules are specified.
            For symbolic operators, use strings such as ``"Adjoint(RY)"``, ``"Pow(H)"``, ``"C(RX)"``, etc.
        decomps (DecompositionRule): new decomposition rules to add to the given ``op_type``.
            A decomposition is a quantum function registered with a resource estimate using
            ``qml.register_resources``.

    .. seealso:: :func:`~pennylane.register_resources` and :class:`~pennylane.list_decomps`

    **Example**

    This example demonstrates adding two new decomposition rules to the ``qml.Hadamard`` operator.

    .. code-block:: python

        import pennylane as qml
        import numpy as np

        @qml.register_resources({qml.RZ: 2, qml.RX: 1, qml.GlobalPhase: 1})
        def my_hadamard1(wires):
            qml.RZ(np.pi / 2, wires=wires)
            qml.RX(np.pi / 2, wires=wires)
            qml.RZ(np.pi / 2, wires=wires)
            qml.GlobalPhase(-np.pi / 2, wires=wires)

        @qml.register_resources({qml.RZ: 1, qml.RY: 1, qml.GlobalPhase: 1})
        def my_hadamard2(wires):
            qml.RZ(np.pi, wires=wires)
            qml.RY(np.pi / 2, wires=wires)
            qml.GlobalPhase(-np.pi / 2)

        qml.add_decomps(qml.Hadamard, my_hadamard1, my_hadamard2)

    These two new decomposition rules for ``qml.Hadamard`` will be subsequently stored within the
    scope of this program, and they will be taken into account for all circuit decompositions
    for the duration of the session. To add alternative decompositions for a particular circuit
    as opposed to globally, use the ``alt_decomps`` argument of the :func:`~pennylane.transforms.decompose` transform.

    Custom decomposition rules can also be specified for symbolic operators. In this case, the
    operator type can be specified as a string. For example,

    .. code-block:: python

        @register_resources({qml.RY: 1})
        def adjoint_ry(phi, wires, **_):
            qml.RY(-phi, wires=wires)

        qml.add_decomps("Adjoint(RY)", adjoint_ry)

    .. seealso:: :func:`~pennylane.transforms.decompose`

    """
    if not all(isinstance(d, DecompositionRule) for d in decomps):
        raise TypeError(
            "A decomposition rule must be a qfunc with a resource estimate "
            "registered using qml.register_resources"
        )
    if isinstance(op_type, type):
        op_type = op_type.__name__
    _decompositions[translate_op_alias(op_type)].extend(decomps)


def list_decomps(op: type[Operator] | Operator | str) -> list[DecompositionRule]:
    """Lists all stored decomposition rules for an operator class.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via :func:`~pennylane.decomposition.enable_graph`. This new way of
        doing decompositions is generally more resource efficient and accommodates multiple alternative
        decomposition rules for an operator.

    Args:
        op (type or Operator or str): the operator or operator type to retrieve decomposition
            rules for. For symbolic operators, use strings like ``"Adjoint(RY)"``, ``"Pow(H)"``,
            ``"C(RX)"``, etc.

    Returns:
        list[DecompositionRule]: a list of decomposition rules registered for the given operator.

    **Example**

    >>> import pennylane as qml
    >>> qml.list_decomps(qml.CRX)
    [<pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9de0>,
     <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9db0>,
     <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9f00>]

    Each decomposition rule can be inspected:

    >>> print(qml.list_decomps(qml.CRX)[0])
    @register_resources(_crx_to_rx_cz_resources)
    def _crx_to_rx_cz(phi, wires, **__):
        qml.RX(phi / 2, wires=wires[1]),
        qml.CZ(wires=wires),
        qml.RX(-phi / 2, wires=wires[1]),
        qml.CZ(wires=wires),
    >>> print(qml.draw(qml.list_decomps(qml.CRX)[0])(0.5, wires=[0, 1]))
    0: ───────────╭●────────────╭●─┤
    1: ──RX(0.25)─╰Z──RX(-0.25)─╰Z─┤

    """
    if isinstance(op, Operator):
        return _decompositions[op.name][:]
    if isinstance(op, type):
        op = op.__name__
    return _decompositions[translate_op_alias(op)][:]


def has_decomp(op: type[Operator] | Operator | str) -> bool:
    """Checks whether an operator has decomposition rules defined.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via :func:`~pennylane.decomposition.enable_graph`. This new way of
        doing decompositions is generally more resource efficient and accommodates multiple alternative
        decomposition rules for an operator.

    Args:
        op (type or Operator or str): the operator or operator type to check for
            decomposition rules. For symbolic operators, use strings like ``"Adjoint(RY)"``,
            ``"Pow(H)"``, ``"C(RX)"``, etc.

    Returns:
        bool: whether decomposition rules are defined for the given operator.

    """
    if isinstance(op, Operator):
        return op.name in _decompositions and len(_decompositions[op.name]) > 0
    if isinstance(op, type):
        op = op.__name__
    op = translate_op_alias(op)
    return op in _decompositions and len(_decompositions[op]) > 0


@register_resources({})
def null_decomp(*_, **__):
    """A decomposition rule that can be assigned to an operator so that the operator decomposes to nothing.

    **Example**

    .. code-block:: python

        from functools import partial
        import pennylane as qml
        from pennylane.decomposition import null_decomp

        qml.decomposition.enable_graph()

        @partial(
            qml.transforms.decompose,
            gate_set={qml.RZ},
            fixed_decomps={qml.GlobalPhase: null_decomp}
        )
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Z(0)

    >>> print(qml.draw(circuit)())
    0: ──RZ(3.14)─┤

    """
    return
