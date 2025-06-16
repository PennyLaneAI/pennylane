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
This module contains the commands for allocating and freeing wires dynamically.
"""
import uuid
from typing import Optional, Sequence

from pennylane.operation import Operator
from pennylane.wires import Wires


class DynamicWire:
    """A wire whose concrete value will be determined later during a compilation step or execution.

    Multiple dynamic wires can correspond to the same device wire as long as they are properly allocated and
    freed.

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
    """An instruction to values for dynamic wires.

    Args:
        wires (list[DynamicWire]): a list of dynamic wire values.

    Keyword Args:
        require_zeros (bool): Whether or not the wire must start in a ``0`` state.

    ..see-also:: :func:`~.allocate`, :func:`~.safe_allocate`.

    """

    def __init__(self, wires, require_zeros=True):
        super().__init__(wires=wires)
        self._hyperparameters = {"require_zeros": require_zeros}

    @property
    def require_zeros(self):
        """Whether or not the allocated wires are required to be in the zero state."""
        return self.hyperparameters["require_zeros"]

    @classmethod
    def from_num_wires(cls, num_wires: int, require_zeros=True) -> "Allocate":
        """Initialize an ``Allocate`` op from a number of wires instead of already constructed dynamic wires."""
        wires = tuple(DynamicWire() for _ in range(num_wires))
        return cls(wires=wires, require_zeros=require_zeros)


class Deallocate(Operator):
    """An instruction to deallocate the provided ``DynamicWire``'s.

    Args:
        wires (DynamicWire, Sequence[DynamicWire]): one or more dynamic wires to deallocate.
        restored (bool): Whether or not the qubit will be in the same state upon being freed.

    """

    def __init__(self, wires: DynamicWire | Sequence[DynamicWire], restored=False):
        super().__init__(wires=wires)
        self._hyperparameters = {"restored": restored}

    @property
    def restored(self):
        """Whether or not the dynamic wire will be returned to its original state."""
        return self.hyperparameters["restored"]


def allocate(num_wires: int, require_zeros: bool = True) -> Wires:
    """Allocate dynamic wires for temporary use within a circuit.

    .. warning::

        This feature is experimental and is not possible on any device yet.

    Args:
        num_wires (int): the number of wires to allocate

    Keyword Args:
        require_zeros (bool): Whether or not the wires must start in the ``0`` state.

    Returns:
        Wires: A wires object containing ``DynamicWire`` objects.

    .. seealso:: :class:`~.safe_allocate`

    :class:`~.safe_allocate` is recommended as the preferred way to allocate wires, as it enforces automatic deallocation.
    Manual use of ``allocate`` and ``deallocate`` should be used with caution.

    ..code-block:: python

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

        print(qml.draw(c, level="user")())


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
    op = Allocate.from_num_wires(num_wires, require_zeros=require_zeros)
    return op.wires


def deallocate(
    wires: DynamicWire | Wires | Sequence[DynamicWire], restored: bool = False
) -> Deallocate:
    """Deallocate dynamic wires back to the pool of wires available for allocation.

    .. warning::

        This feature is experimental and is not possible on any device yet.

    Args:
        wires (DynamicWire, Wires, Sequence[DynamicWire]): One or more dynamic wires.

    Keyword Args:
        restored (bool): Whether or not the qubits will be reset to their original state upon deallocation.


    .. seealso:: :class:`~.safe_allocate`

    :class:`~.safe_allocate` is recommended as the preferred way to allocate wires, as it enforces automatic deallocation.
    Manual use of ``allocate`` and ``deallocate`` should be used with caution.

    ..code-block:: python

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

        print(qml.draw(c, level="user")())


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
    wires = Wires(wires)
    if not_dynamic_wires := [w for w in wires if not isinstance(w, DynamicWire)]:
        raise ValueError(f"deallocate only accepts DynamicWire wires. Got {not_dynamic_wires}")
    return Deallocate(wires, restored=restored)


class safe_allocate:
    """Temporarily allocate dynamic wires while making sure to automatically deallocate them at the end.

    .. warning::

        This feature is experimental and is not possible on any device yet.

    Args:
        num_wires (int): the number of dynamic wires to allocate.

    Keyword Args:
        require_zeros (bool): whether or not the wires must start in the ``0`` state
        restored (bool): whether or not the wires return to the same state as they started.

    .. code-block:: python

        @qml.qnode(qml.device('default.qubit', wires=("a", "b")))
        def c():
            with qml.allocation.safe_allocate(2, require_zeros=True, restored=False) as wires:
                qml.CNOT(wires)
            with qml.allocation.safe_allocate(2, require_zeros=True, restored=False) as wires:
                qml.IsingXX(0.5, wires)
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

    def __init__(self, num_wires: int, require_zeros: bool = True, restored: bool = False):
        self.wires = allocate(num_wires, require_zeros=require_zeros)
        self._restored = restored

    def __enter__(self):
        return self.wires

    def __exit__(self, *_, **__):
        deallocate(self.wires, restored=self._restored)
