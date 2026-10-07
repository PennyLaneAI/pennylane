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
"""
This module contains the commands for allocating and deallocating wires dynamically.
"""

from collections.abc import Sequence
from enum import StrEnum
from numbers import Integral
from typing import Literal

import jax

from pennylane.capture import QpPrimitive
from pennylane.capture import enabled as capture_enabled
from pennylane.core.operator import Operator
from pennylane.math import is_abstract
from pennylane.pytrees import register_pytree
from pennylane.wires import AbstractQubit, DynamicWire, Wires


class AllocateState(StrEnum):
    """An enumeration for the different types of states a dynamic wire can start in."""

    ZERO = "zero"
    ANY = "any"
    MAGIC_T = "magic-T"  # |m⟩ = TH|0⟩
    MAGIC_T_ADJ = "magic-T-adj"  # |m̄⟩ = T†H|0⟩


_MAGIC_STATES = frozenset({AllocateState.MAGIC_T, AllocateState.MAGIC_T_ADJ})


class AbstractRegister(jax.core.AbstractValue):
    """An abstract value representing a register of dynamically-allocated qubits."""

    def __init__(self, num_wires):
        self.num_wires = num_wires

    def __eq__(self, other):
        return isinstance(other, AbstractRegister) and self.num_wires == other.num_wires

    def __hash__(self):
        return hash(("AbstractRegister", self.num_wires))


allocate_prim = QpPrimitive("allocate")


@allocate_prim.def_impl
def _allocate_primitive_impl(
    *, num_wires, state=AllocateState.ZERO, restored=False
):  # pylint: disable=unused-argument
    raise NotImplementedError("jaxpr containing qubit allocation cannot be executed.")


@allocate_prim.def_abstract_eval
def _allocate_primitive_abstract_eval(
    *, num_wires, state=AllocateState.ZERO, restored=False
):  # pylint: disable=unused-argument
    return AbstractRegister(num_wires)


extract_prim = QpPrimitive("extract")


@extract_prim.def_impl
def _extract_primitive_impl(idx, register):  # pylint: disable=unused-argument
    raise NotImplementedError("jaxpr containing qubit extraction cannot be executed.")


@extract_prim.def_abstract_eval
def _extract_primitive_abstract_eval(idx, register):  # pylint: disable=unused-argument
    return AbstractQubit()


deallocate_prim = QpPrimitive("deallocate")
deallocate_prim.multiple_results = True


@deallocate_prim.def_impl
def _deallocate_primitive_impl(register):  # pylint: disable=unused-argument
    raise NotImplementedError("jaxpr containing qubit deallocation cannot be executed.")


@deallocate_prim.def_abstract_eval
def _deallocate_primitive_abstract_eval(register):  # pylint: disable=unused-argument
    return []


class Allocate(Operator):
    """An instruction to request dynamic wires.

    Args:
        wires (list[DynamicWire]): a list of dynamic wire values.

    Keyword Args:
        state (Literal["any", "zero", "magic-T", "magic-T-adj"]): the state that the wires need to start in.
        restored (bool): Whether or not the qubit will be restored to the original state before being deallocated.

    ..see-also:: :func:`~.allocate`.

    """

    def __init__(self, wires, state: AllocateState = AllocateState.ZERO, restored=False):
        super().__init__(wires=wires)
        self._hyperparameters = {"state": state, "restored": restored}

    @property
    def state(self) -> AllocateState:
        """The initial state requested for the allocated wires."""
        return self.hyperparameters["state"]

    @property
    def restored(self) -> bool:
        """Whether the allocated wires will be restored to their original state before deallocation."""
        return self.hyperparameters["restored"]

    @classmethod
    def from_num_wires(
        cls, num_wires: int, state: AllocateState = AllocateState.ZERO, restored=False
    ) -> "Allocate":
        """Initialize an ``Allocate`` op from a number of wires instead of already constructed dynamic wires."""
        wires = tuple(DynamicWire() for _ in range(num_wires))
        return cls(wires=wires, state=state, restored=restored)


class Deallocate(Operator):
    """An instruction to deallocate the provided ``DynamicWire``'s.

    Args:
        wires (DynamicWire, Sequence[DynamicWire]): one or more dynamic wires to deallocate.

    """

    def __init__(self, wires: DynamicWire | Sequence[DynamicWire]):
        super().__init__(wires=wires)


def deallocate(wires: DynamicWire | Wires | Sequence[DynamicWire]) -> Deallocate | None:
    """Deallocates wires that have previously been allocated with :func:`~.allocate`.
    Upon deallocating, those wires is available to be allocated thereafter.

    Args:
        wires (DynamicWire, Wires, Sequence[DynamicWire]): one or more dynamic wires.

    .. seealso:: :func:`~.allocate`

    .. note::
        The :func:`~.allocate` function can be used as a context manager with automatic deallocation
        (recommended for most cases) upon exiting the scope.

    **Example**

    .. code-block:: python

        import pennylane as qp

        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.H(0)

            wire = qp.allocate(1, state="zero", restored=True)[0]
            qp.CNOT((0, wire))
            qp.CNOT((0, wire))
            qp.deallocate(wire)

            new_wires = qp.allocate(2, state="zero", restored=True)
            qp.SWAP((new_wires[1], new_wires[0]))
            qp.deallocate(new_wires)

            return qp.expval(qp.Z(0))

    >>> print(qp.draw(circuit)())
    0: ──H────╭●────╭●────┤  <Z>
         |0>├─╰X────╰X──┤
         |0>├─╭SWAP──┤
         |0>├─╰SWAP──┤


    Here, three dynamic wires were allocated in the circuit originally. When PennyLane determines
    which concrete values to use for dynamic wires to send to the device for execution, we can see
    that the first dynamic wire is already deallocated back into the zero state. This allows us to
    use it as one of the wires requested in the second allocation, resulting in a total of three wires
    being required from the device, including two dynamically allocated wires:

    >>> print(qp.draw(circuit, level="device")())
    0: ──H─╭●─╭●───────┤  <Z>
    1: ────╰X─╰X─╭SWAP─┤
    2: ──────────╰SWAP─┤
    """
    if isinstance(wires, Sequence) and len(wires) == 0:
        return None
    if capture_enabled():
        # Under capture, a dynamically-allocated register is deallocated via its register tracer.
        return deallocate_prim.bind(wires._register)  # pylint: disable=protected-access
    wires = Wires(wires)
    if not_dynamic_wires := [w for w in wires if not isinstance(w, DynamicWire)]:
        raise ValueError(f"deallocate only accepts DynamicWire wires. Got {not_dynamic_wires}")
    return Deallocate(wires)


class DynamicRegister(Wires):
    """A specialized ``Wires`` class for dynamically-allocated wires, with a context manager for
    automatic deallocation.

    A ``DynamicRegister`` can be initialized either with a sequence of wire labels (``DynamicWire``s
    or abstract qubits) or, under program capture, with an :class:`AbstractRegister` tracer. When
    constructed from a register tracer, the individual qubits are produced lazily via the ``extract``
    primitive the first time the wire labels are accessed, and then cached. Flattening the register
    therefore yields its individual qubits, and unflattening produces a ``Wires`` holding them.
    """

    def __init__(self, wires, _override=False):
        self._register = None
        if is_abstract(wires) and isinstance(wires.aval, AbstractRegister):
            # Wrap a register tracer: the qubit labels are materialized lazily from it.
            self._register = wires
            self._labels = None
            self._lazy_labels = None
            self._hash = None
            return

        super().__init__(wires, _override=_override)

    @property
    def _labels(self):
        if self._lazy_labels is None and self._register is not None:
            # Materialize the register's qubits once, by extracting each of them, then cache.
            self._lazy_labels = tuple(
                extract_prim.bind(i, self._register) for i in range(self._register.aval.num_wires)
            )
        return self._lazy_labels

    @_labels.setter
    def _labels(self, value):
        self._lazy_labels = value

    def __getitem__(self, idx):
        # When backed by a register tracer, each (non-slice) index extracts a qubit directly, since
        # ``select_n`` (used by ``Wires.__getitem__`` for dynamic indices) does not accept qubits.
        if self._register is not None:
            return extract_prim.bind(idx, self._register)
        return super().__getitem__(idx)

    def __len__(self):
        # Avoid materializing the qubits just to report the size of an un-materialized register.
        if self._register is not None:
            return self._register.aval.num_wires
        return len(self._labels)

    def __repr__(self):
        size = len(self._labels) if self._register is None else self._register.aval.num_wires
        return f"<DynamicRegister: size={size}>"

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        deallocate(self)

    def __hash__(self):
        raise TypeError("unhashable type 'DynamicRegister'")


# pylint: disable=protected-access
register_pytree(DynamicRegister, DynamicRegister._flatten, DynamicRegister._unflatten)


def allocate(
    num_wires: int,
    state: Literal["any", "zero", "magic-T", "magic-T-adj"] | AllocateState = AllocateState.ZERO,
    restored: bool = False,
) -> DynamicRegister:
    r"""Dynamically allocates new wires in-line,
    or as a context manager which also safely deallocates the new wires upon exiting the context.

    Args:
        num_wires (int):
            The number of wires to dynamically allocate.

    Keyword Args:
        state (Literal["any", "zero", "magic-T", "magic-T-adj"]):
            Specifies the initial state of the allocated wires. ``"zero"`` and ``"any"`` request
            wires in the all-zeros state or an arbitrary state, respectively. ``"magic-T"`` and
            ``"magic-T-adj"`` request magic states with :math:`|m\rangle = TH|0\rangle` or
            :math:`|\bar{m}\rangle = T^\dagger H|0\rangle`. For ``num_wires > 1``, a product
            state is created. The default value is ``state="zero"``.

        restored (bool):
            Whether or not the dynamically allocated wires are returned to the same state they
            started in. ``restored=True`` indicates that the user promises to restore the
            dynamically allocated wires to their original state before being deallocated.
            ``restored=False`` indicates that the user does not promise to restore the dynamically
            allocated wires before being deallocated. The default value is ``False``.

    Returns:
        DynamicRegister: an object, behaving similarly to ``Wires``, that represents the dynamically
        allocated wires.

    .. note::
        The ``allocate`` function can be used as a context manager with automatic deallocation
        (recommended for most cases) or with manual deallocation via :func:`~.deallocate`.

    .. note::
        The ``num_wires`` argument must be static when capture is enabled.

    .. seealso::
        :func:`~.deallocate`

    **Example**

    Using ``allocate`` to dynamically request wires returns an array of wires
    (``DynamicRegister``) that can be indexed into:

    >>> wires = qp.allocate(3)
    >>> wires
    <DynamicRegister: size=3>
    >>> wires[1]
    <DynamicWire>

    Note that allocating just one wire still requires indexing into:

    >>> wire = qp.allocate(1)
    >>> wire
    <DynamicRegister: size=1>
    >>> wire[0]
    <DynamicWire>

    Most use cases for ``allocate`` are covered by using it as a context manager, which ensures
    that allocation and safe deallocation are controlled within a localized scope.

    .. code-block:: python

        import pennylane as qp

        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.H(0)
            qp.H(1)

            with qp.allocate(2, state="zero", restored=False) as new_wires:
                qp.H(new_wires[0])
                qp.H(new_wires[1])

            return qp.expval(qp.Z(0))

    >>> print(qp.draw(circuit)())
    0: ──H──────────┤  <Z>
    1: ──H──────────┤
         |0>├──H──┤
         |0>├──H──┤


    Equivalently, ``allocate`` can be used in-line along with :func:`~.deallocate` for manual
    handling:

    .. code-block:: python

        new_wires = qp.allocate(2, state="zero", restored=False)
        qp.H(new_wires[0])
        qp.H(new_wires[1])
        qp.deallocate(new_wires)


    .. details::
        :title: Usage details

        **Efficient wire management**

        For more complex dynamic allocation in circuits, PennyLane will resolve the dynamic
        allocation calls in a resource-efficient manner before sending the program to the
        device. Consider the following circuit, which contains two dynamic allocations within a
        ``for`` loop.

        .. code-block:: python

            @qp.qnode(qp.device("default.qubit"), mcm_method="tree-traversal")
            def circuit():
                qp.H(0)

                for i in range(2):
                    with qp.allocate(1, state="zero", restored=True) as new_qubit1:
                        with qp.allocate(1, state="any", restored=False) as new_qubit2:
                            m0 = qp.measure(new_qubit1[0], reset=True)
                            qp.cond(m0 == 1, qp.Z)(new_qubit2[0])
                            qp.CNOT((0, new_qubit2[0]))

                return qp.expval(qp.Z(0))

        >>> print(qp.draw(circuit)())
        0: ──H─────────────────────╭●───────────╭●────┤  <Z>
             |0>├──┤↗│  │0⟩────────│──────────┤ │
             ├──────║────────Z─────╰X─────────┤ │
                    ║        ║|0>├──┤↗│  │0⟩────│───┤
                    ║        ║├──────║────────Z─╰X──┤
                    ╚════════╝       ╚════════╝

        The user-level circuit drawing shows four separate allocations and deallocations (two per
        loop iteration). However, the circuit that the device receives gets automatically compiled
        to only use **two** additional wires (wires labelled ``1`` and ``2`` in the diagram below). This
        is due to the fact that ``new_qubit1`` and ``new_qubit2`` can both be reused after they've been
        deallocated in the first iteration of the ``for`` loop:

        >>> print(qp.draw(circuit, level="device")())
        0: ──H───────────╭●──────────────╭●─┤  <Z>
        1: ──┤↗│  │0⟩────│───┤↗│  │0⟩────│──┤
        2: ───║────────Z─╰X───║────────Z─╰X─┤
              ╚════════╝      ╚════════╝

        Additionally, in circuits that deallocate a wire in `"any"` state, this wire can be reused
        as a `"zero"`. The arbitrary-state wire is reset back to a zero state by introducing a
        mid-circuit measurement. This is illustrated in the example below, where the first wire
        allocation is deallocated in an arbitrary state, but the only other dynamic wire allocation
        in the circuit requires a zero state:

        .. code-block:: python

            @qp.qnode(qp.device("default.qubit"), mcm_method="device")
            def circuit():
                with qp.allocate(1, state="zero", restored=False) as [wire]:
                    qp.H(wire)

                with qp.allocate(1, state="zero", restored=False) as [wire]:
                    qp.X(wire)

                return qp.expval(qp.Z(0))

        >>> print(qp.draw(circuit, level="user")())
        0: ─────────────┤  <Z>
            |0>├──H──┤
            |0>├──X──┤
        >>> print(qp.draw(circuit, level="device")())
        0: ─────────────────┤  <Z>
        1: ──H──┤↗│  │0⟩──X─┤
    """
    state = AllocateState(state)
    if state in _MAGIC_STATES and restored:
        raise ValueError(
            "restored=True is not supported for magic state allocations "
            f"(state={state!r}). Magic states cannot be restored to their initial state."
        )
    # Allocating nothing is a no-op: do not queue or bind ``Allocate``/``Deallocate``.
    if isinstance(num_wires, Integral) and not isinstance(num_wires, bool) and num_wires == 0:
        return DynamicRegister(())
    if capture_enabled():
        if is_abstract(num_wires):
            raise NotImplementedError(
                "Number of allocated wires must be static when capture is enabled."
            )
        register = allocate_prim.bind(num_wires=num_wires, state=state, restored=restored)
        return DynamicRegister(register)

    reg = DynamicRegister([DynamicWire() for _ in range(num_wires)])
    Allocate(reg, state=state, restored=restored)
    return reg
