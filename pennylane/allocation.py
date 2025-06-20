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
import uuid
from typing import Optional, Sequence

from pennylane.capture import enabled as capture_enabled
from pennylane.operation import Operator
from pennylane.wires import Wires

has_jax = True
try:
    import jax
except ImportError:
    jax = None
    has_jax = False


if not has_jax:
    allocate_prim = None
    deallocate_prim = None
else:
    allocate_prim = jax.extend.core.Primitive("allocate")
    allocate_prim.multiple_results = True

    # pylint: disable=unused-argument
    @allocate_prim.def_impl
    def _(*, num_wires, require_zeros=True, restored=False):
        raise NotImplementedError("jaxpr containing qubit allocation cannot be executed.")

    # pylint: disable=unused-argument
    @allocate_prim.def_abstract_eval
    def _(*, num_wires, require_zeros=True, restored=False):
        return [jax.core.ShapedArray((), dtype=int) for _ in range(num_wires)]

    deallocate_prim = jax.extend.core.Primitive("deallocate")
    deallocate_prim.multiple_results = True

    # pylint: disable=unused-argument
    @deallocate_prim.def_impl
    def _(*wires):
        raise NotImplementedError("jaxpr containing qubit deallocation cannot be executed.")

    # pylint: disable=unused-argument
    @deallocate_prim.def_abstract_eval
    def _(*wires):
        return []


class DynamicWire:
    """A wire whose concrete value will be determined later during a compilation step or execution.

    Multiple dynamic wires can correspond to the same device wire as long as they are properly allocated and
    deallocated.

    Args:
        key (Optional[str]): a ``uuid4`` string to uniquely identify the dynamic wire.
    """

    def __init__(self, key: Optional[uuid.UUID] = None):
        self.key = key or uuid.uuid4()

    def __eq__(self, other):
        if not isinstance(other, DynamicWire):
            return False
        return self.key == other.key

    def __hash__(self):
        return hash(("DynamicWire", self.key))

    def __repr__(self):
        return "<DynamicWire>"


class Allocate(Operator):
    """An instruction to request dynamic wires.

    Args:
        wires (list[DynamicWire]): a list of dynamic wire values.

    Keyword Args:
        require_zeros (bool): Whether or not the wire must start in a ``0`` state.
        restored (bool): Whether or not the qubit will be restored to the original state before being deallocated.

    ..see-also:: :func:`~.allocate`.

    """

    def __init__(self, wires, require_zeros=True, restored=False):
        super().__init__(wires=wires)
        self._hyperparameters = {"require_zeros": require_zeros, "restored": restored}

    @property
    def require_zeros(self):
        """Whether or not the allocated wires are required to be in the zero state."""
        return self.hyperparameters["require_zeros"]

    @property
    def restored(self):
        """Whether the allocated wires will be restored to their original state before deallocation."""
        return self.hyperparameters["restored"]

    @classmethod
    def from_num_wires(cls, num_wires: int, require_zeros=True, restored=False) -> "Allocate":
        """Initialize an ``Allocate`` op from a number of wires instead of already constructed dynamic wires."""
        wires = tuple(DynamicWire() for _ in range(num_wires))
        return cls(wires=wires, require_zeros=require_zeros, restored=restored)


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

        This feature is experimental and is not available on any device yet.

    Args:
        wires (DynamicWire, Wires, Sequence[DynamicWire]): One or more dynamic wires.

    .. seealso:: :func:`~.allocate`

    Using :func:`~.allocate` as a context manager is the recommended syntax, as it will automatically
    deallocate all dynamic wires at the end of the scope.

    .. code-block:: python

        @qml.qnode(qml.device('default.qubit'))
        def c():
            qml.H(0)

            wires = qml.allocation.allocate(1, require_zeros=True)
            qml.CNOT((0, wires[0]))
            qml.CNOT((0, wires[0]))
            qml.allocation.deallocate(wires, restored=True)

            new_wires = qml.allocation.allocate(1)
            qml.SWAP((0, new_wires[0]))
            qml.allocation.deallocate(new_wires)

            return qml.probs(wires=0)


    >>> print(qml.draw(c, level="user")())
                0: ──H────────╭●─╭●─────────────╭SWAP─────────────┤  Probs
    <DynamicWire>: ──Allocate─╰X─╰X──Deallocate─│─────────────────┤
    <DynamicWire>: ──Allocate───────────────────╰SWAP──Deallocate─┤
    >>> print(qml.draw(c, level="device")())
    0: ──H─╭●─╭●─╭SWAP─┤  Probs
    1: ────╰X─╰X─╰SWAP─┤


    Here two dynamic wires are allocated in the circuit originally. When we are determining
    what concrete values to use for dynamic wires, we can see that the first dynamic wire is already
    deallocated back into the zero state. This allows us to use it for the second allocation used in the ``SWAP``
    gate as well.

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


def allocate(num_wires: int, require_zeros: bool = True, restored: bool = False) -> DynamicRegister:
    """Dynamically allocates new wires in-line or as a context manager, which also safely deallocates the
    new wires upon exiting the context.

    .. warning::

        This feature is experimental and is not possible on any device yet.

    Args:
        num_wires (int): the number of dynamic wires to allocate.

    Keyword Args:
        require_zeros (bool): whether or not the wires must start in the ``0`` state
        restored (bool): whether or not the wires are returned to the same state they started in.

    This function can be used as a context manager with automatic deallocation (preferred) or with manual
    deallocation via :func:`~.deallocate`.

    .. code-block:: python

        @qml.qnode(qml.device('default.qubit', wires=("a", "b")))
        def c():
            with qml.allocation.allocate(2, require_zeros=True, restored=False) as wires:
                qml.CNOT(wires)

            wires = qml.allocation.allocate(2, require_zeros=True, restored=False)
            qml.IsingXX(0.5, wires)
            qml.allocation.deallocate(wires)

            return qml.probs()


    >>> print(qml.draw(c, level="user")())
    <DynamicWire>: ─╭Allocate─╭●─────────────╭Deallocate─┤  Probs
    <DynamicWire>: ─╰Allocate─╰X─────────────╰Deallocate─┤  Probs
    <DynamicWire>: ─╭Allocate─╭IsingXX(0.50)─╭Deallocate─┤  Probs
    <DynamicWire>: ─╰Allocate─╰IsingXX(0.50)─╰Deallocate─┤  Probs
    >>> print(qml.draw(c, level="device")())
    a: ─╭●──┤↗│  │0⟩─╭IsingXX(0.50)─┤ ╭Probs
    b: ─╰X──┤↗│  │0⟩─╰IsingXX(0.50)─┤ ╰Probs

    The initial circuit has the ``DynamicWire``'s present, but when executing on the device, those are converted into
    the device wires ``("a", "b")``. As the wires are not reset to their original state when deallocated, they are reset
    before being re-used again in the second block.

    """
    if capture_enabled():
        wires = allocate_prim.bind(
            num_wires=num_wires, require_zeros=require_zeros, restored=restored
        )
    else:
        wires = [DynamicWire() for _ in range(num_wires)]
    reg = DynamicRegister(wires)
    if not capture_enabled():
        Allocate(reg, require_zeros=require_zeros, restored=restored)
    return reg
