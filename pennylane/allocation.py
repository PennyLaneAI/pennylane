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
from typing import Literal

from pennylane.capture import enabled as capture_enabled
from pennylane.operation import Operator
from pennylane.wires import DynamicWire, Wires

has_jax = True
try:
    import jax

    # pylint: disable=ungrouped-imports
    from pennylane.capture import QmlPrimitive
except ImportError:
    jax = None
    has_jax = False


class AllocateState(StrEnum):
    """An enumeration for the different types of states a dynamic wire can start in."""

    ZERO = "zero"
    ANY = "any"


if not has_jax:
    allocate_prim = None
    deallocate_prim = None
else:
    allocate_prim = QmlPrimitive("allocate")
    allocate_prim.multiple_results = True

    # pylint: disable=unused-argument
    @allocate_prim.def_impl
    def _(*, num_wires, state: AllocateState = AllocateState.ZERO, restored=False):
        raise NotImplementedError("jaxpr containing qubit allocation cannot be executed.")

    # pylint: disable=unused-argument
    @allocate_prim.def_abstract_eval
    def _(*, num_wires, state: AllocateState = AllocateState.ZERO, restored=False):
        return [jax.core.ShapedArray((), dtype=int) for _ in range(num_wires)]

    deallocate_prim = QmlPrimitive("deallocate")
    deallocate_prim.multiple_results = True

    # pylint: disable=unused-argument
    @deallocate_prim.def_impl
    def _(*wires):
        raise NotImplementedError("jaxpr containing qubit deallocation cannot be executed.")

    # pylint: disable=unused-argument
    @deallocate_prim.def_abstract_eval
    def _(*wires):
        return []


class Allocate(Operator):
    """An instruction to request dynamic wires.

    Args:
        wires (list[DynamicWire]): a list of dynamic wire values.

    Keyword Args:
        state (Literal["any", "zero"]): the state that the wires need to start in.
        restored (bool): Whether or not the qubit will be restored to the original state before being deallocated.

    ..see-also:: :func:`~.allocate`.

    """

    def __init__(self, wires, state: AllocateState = AllocateState.ZERO, restored=False):
        super().__init__(wires=wires)
        self._hyperparameters = {"state": state, "restored": restored}

    @property
    def state(self) -> AllocateState:
        """Whether or not the allocated wires are required to be in the zero state."""
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


def deallocate(wires: DynamicWire | Wires | Sequence[DynamicWire]) -> Deallocate:
    """Frees quantum memory that has previously been allocated with :func:`~.allocate`.
    Upon freeing quantum memory, that memory is available to be allocated thereafter.

    .. warning::
        This feature is experimental, and any workflows that include calls to ``deallocate`` cannot
        be executed on any device.

    Args:
        wires (DynamicWire, Wires, Sequence[DynamicWire]): one or more dynamic wires.

    .. seealso:: :func:`~.allocate`

    Using :func:`~.allocate` as a context manager is the recommended syntax, as it will automatically
    deallocate all dynamic wires at the end of the scope.

    .. code-block:: python

        def c():
            qml.H(0)

            wires = qml.allocation.allocate(1, require_zeros=True, restored=True)
            qml.CNOT((0, wires[0]))
            qml.CNOT((0, wires[0]))
            qml.allocation.deallocate(wires)

            new_wires = qml.allocation.allocate(1)
            qml.SWAP((0, new_wires[0]))
            qml.allocation.deallocate(new_wires)

    >>> print(qml.draw(c)())
                0: ──H────────╭●─╭●─────────────╭SWAP─────────────┤
    <DynamicWire>: ──Allocate─╰X─╰X──Deallocate─│─────────────────┤
    <DynamicWire>: ──Allocate───────────────────╰SWAP──Deallocate─┤

    Here, two dynamic wires are allocated in the circuit originally. When we are determining
    what concrete values to use for dynamic wires, we can see that the first dynamic wire is already
    deallocated back into the zero state. This allows us to use it for the second allocation used in
    the ``SWAP`` gate.

    """
    if capture_enabled():
        if not isinstance(wires, Sequence):
            wires = (wires,)
        return deallocate_prim.bind(*wires)
    wires = Wires(wires)
    if not_dynamic_wires := [w for w in wires if not isinstance(w, DynamicWire)]:
        raise ValueError(f"deallocate only accepts DynamicWire wires. Got {not_dynamic_wires}")
    return Deallocate(wires)


# pylint: disable=too-many-ancestors
class DynamicRegister(Wires):
    """A specialized ``Wires`` class for dynamic wires with a context manager for automatic deallocation."""

    def __repr__(self):
        return f"<DynamicRegister: size={len(self._labels)}>"

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        deallocate(self)

    def __hash__(self):
        raise TypeError("unhashable type 'DynamicRegister'")


def allocate(
    num_wires: int,
    state: Literal["any", "zero"] | AllocateState = AllocateState.ZERO,
    restored: bool = False,
) -> DynamicRegister:
    """Dynamically allocates new wires in-line,
    or as a context manager which also safely deallocates the new wires upon exiting the context.

    .. warning::
        This feature is experimental, and any workflows that include calls to ``allocate`` cannot be
        executed on any device.

    Args:
        num_wires (int): the number of wires to dynamically allocate.

    Keyword Args:
        state (Literal["any", "zero"]): the state that the wires need to start in.

        restored (bool):
            whether or not the dynamically allocated wires are returned to the same state they started
            in. ``restored=True`` indicates that the user promises to restore the dynamically allocated
            memory to its original state before :func:`~.deallocate` is called. ``restored=False`` indicates
            that the user does not promise to restore the dynamically allocated memory before :func:`~.deallocate`
            is called. The default value is ``False``.

    Returns:
        DynamicRegister: an object, behaving similarly to ``Wires``, that represents the dynamically
        allocated memory.

    .. note::
        The ``allocate`` function should be used with :func:`~.deallocate` if it is not used as a context
        manager.

    .. seealso::
        :func:`~.deallocate`

    This function can be used as a context manager with automatic deallocation (preferred) or with manual
    deallocation via :func:`~.deallocate`.

    .. code-block:: python

        def c():
            qml.H("a")
            qml.H("b")

            with qml.allocation.allocate(2, state="zero" restored=False) as wires:
                qml.CNOT(wires)

            wires = qml.allocation.allocate(2, state="zero", restored=False)
            qml.IsingXX(0.5, wires)
            qml.allocation.deallocate(wires)

    >>> print(qml.draw(c)())
                a: ──H───────────────────────────────────┤
                b: ──H───────────────────────────────────┤
    <DynamicWire>: ─╭Allocate─╭●─────────────╭Deallocate─┤
    <DynamicWire>: ─╰Allocate─╰X─────────────╰Deallocate─┤
    <DynamicWire>: ─╭Allocate─╭IsingXX(0.50)─╭Deallocate─┤
    <DynamicWire>: ─╰Allocate─╰IsingXX(0.50)─╰Deallocate─┤
    """
    state = AllocateState(state)
    if capture_enabled():
        wires = allocate_prim.bind(num_wires=num_wires, state=state, restored=restored)
    else:
        wires = [DynamicWire() for _ in range(num_wires)]
    reg = DynamicRegister(wires)
    if not capture_enabled():
        Allocate(reg, state=state, restored=restored)
    return reg
