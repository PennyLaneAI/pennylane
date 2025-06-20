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
This submodule contains a transform for resolving dynamic wires into real wires.
"""
from collections.abc import Hashable, Sequence
from typing import Optional

from pennylane.allocation import Allocate, Deallocate
from pennylane.measurements import measure
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn, Result, ResultBatch

from .core import transform


class _WireManager:
    """Handles converting dynamic wires into concrete values."""

    def __init__(self, zeroed=(), any_state=(), min_int=None):
        self._zeroed = list(zeroed)
        self._any_state = list(any_state)
        self._loaned = {}  # wire to final register type
        self.min_int = min_int

    def get_wire(self, require_zeros, restored):
        """Retrieve a concrete wire label from available registers."""
        if not self._zeroed and not self._any_state:
            if self.min_int is None:
                raise ValueError("no wires left to allocate.")
            self._zeroed.append(self.min_int)
            self.min_int += 1
        if require_zeros:
            if self._zeroed:
                w = self._zeroed.pop()
                self._loaned[w] = "zeroed" if restored else "any_state"
                return w, []
            w = self._any_state.pop()
            self._loaned[w] = "zeroed" if restored else "any_state"
            m = measure(w, reset=True)
            return w, m.measurements

        if self._any_state:
            w = self._any_state.pop()
            self._loaned[w] = "any_state"
            return w, []
        w = self._zeroed.pop()
        self._loaned[w] = "zeroed" if restored else "any_state"
        return w, []

    def return_wire(self, wire):
        """Return a wire label back to be re-used."""
        reg_type = self._loaned.pop(wire)
        if reg_type == "zeroed":
            self._zeroed.append(wire)
        else:
            self._any_state.append(wire)


def null_postprocessing(results: ResultBatch) -> Result:
    """An empty postprocessing function returned by resolve_dynamic_wires"""
    return results[0]


@transform
def resolve_dynamic_wires(
    tape: QuantumScript,
    zeroed: Sequence[Hashable] = (),
    any_state: Sequence[Hashable] = (),
    min_int: Optional[int] = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Map dynamic wires to concrete values determined by the provided ``zeroed`` and ``any_state`` registers.

    Args:
        tape (QuantumScript): A circuit that may contain dynamic wire allocations and deallocations
        zeroed (Sequence[Hashable]): a register of wires known to be the zero state
        any_state (Sequence[Hashable]): a register of wires with any state
        min_int (Optional[int]): If not ``None``, new wire labels can be created starting at this
            integer and incrementing whenever a new wire is needed.

    Returns:
        tuple[QuantumScript], Callable[[ResultBatch], Result]: A batch of tapes and a postprocessing function

    .. node::

        This transform currently uses a "Last In, First Out" (LIFO) stack based approach to distributing wires.
        This minimizes the total number of wires used, at the cost of higher depth and more resets. Other
        approaches could be taken as well, such as a "First In, First out" algorithm that minimizes depth.

        This approach also means we pop wires from the *end* of the stack first.

    For a dynamic wire requested to be in the zero state (``require_zeros=True``), we try three things before erroring:

    #. If wires exist in the ``zeroed`` register, we take one from that register
    #. If no ``zeroed`` wires exist, we pull one from ``any_state`` and apply a reset operation
    #. If no wires exist in the ``zeroed`` or ``any_state`` registers, we increment ``min_int`` and
        add a new wire

    For a dynamic wire with ``require_zeros=False``, we try:

    #. If wires exist in the ``any_state``, we take one from that register
    #. If no wires exist in ``any_state``, we pull one from ``zeroed``
    #. If no wires exist in the ``zeroed`` or ``any_state`` registers, we increment ``min_int`` and
        add a new wire

    This approach minimizes the width of the circuit at the cost of more reset operations.

    .. code-block:: python

        def circuit(require_zeros=True):
            with qml.allocation.allocate(1, require_zeros=require_zeros) as wires:
                qml.X(wires)
            with qml.allocation.allocate(1, require_zeros=require_zeros) as wires:
                qml.Y(wires)

    >>> print(qml.draw(circuit)())
    <DynamicWire>: ──Allocate──X──Deallocate─┤
    <DynamicWire>: ──Allocate──Y──Deallocate─┤

    If we provide two zeroed qubits to the transform, we can see that the two operations have been
    assigned to both wires known to be in the zero state.

    >>> assigned_two_zeroed = qml.transforms.resolve_dynamic_wires(circuit, zeroed=("a", "b"))
    >>> print(qml.draw(assigned_two_zeroed)())
    a: ──Y─┤
    b: ──X─┤

    If we only provide one zeroed wire, we perform a reset on that wire before reusing for the ``Y`` operation.

    >>> assigned_one_zeroed = qml.transforms.resolve_dynamic_wires(circuit, zeroed=("a", "b"))
    >>> print(qml.draw(assigned_one_zeroed)())
    b: ──X──┤↗│  │0⟩──Y─┤

    If we only provide any_state qubits with unknown states, then they will be reset to zero before being used
    in an operation that requires a zero state.

    >>> assigned_any_state = qml.transforms.resolve_dynamic_wires(circuit, any_state=("a", "b"))
    >>> print(qml.draw(assigned_any_state)())
    b: ──┤↗│  │0⟩──X──┤↗│  │0⟩──Y─|

    If the wire allocations had ``require_zeros=False``, no reset operations would occur.

    >>> print(qml.draw(assigned_any_state)(require_zeros=False))
    b: ──X──Y─┤

    Instead of registers of available wires, a ``min_int`` can be specified instead.  The ``min_int`` indicates
    the first integer to start allocating wires to.  Whenever we have no qubits available to allocate, we increment the integer
    and add a new wire to the pool:

    >>> circuit_integers = qml.transforms.resolve_dynamic_wires(circuit, min_int=0)
    >>> print(qml.draw(circuit_integers)())
    0: ──X──┤↗│  │0⟩──Y─┤

    Note that we still prefer using already created wires over creating new wires.

    .. code-block:: python

        def multiple_allocations():
            with qml.allocation.allocate(1) as wires:
                qml.X(wires)
            with qml.allocation.allocate(3) as wires:
                qml.Toffoli(wires)

    >>> circuit_integers2 = qml.transforms.resolve_dynamic_wires(multiple_allocations, min_int=0)
    >>> print(qml.draw(circuit_integers2)())
    0: ──X──┤↗│  │0⟩─╭●─┤
    1: ──────────────├●─┤
    2: ──────────────╰X─┤

    If both an explicit register and ``min_int`` are specified, ``min_int`` will be used once all available
    explicit wires are loaned out. Below ``"a"`` is extracted and used first, but then later wires
    are extracted starting from ``0``.

    >>> zeroed_and_min_int = qml.transforms.resolve_dynamic_wires(multiple_allocations, zeroed=("a",), min_int=0)
    >>> print(qml.draw(zeroed_and_min_int)())
    a: ──X──┤↗│  │0⟩─╭●─┤
    0: ──────────────├●─┤
    1: ──────────────╰X─┤

    """
    manager = _WireManager(zeroed=zeroed, any_state=any_state, min_int=min_int)

    wire_map = {}
    deallocated = set()

    new_ops = []
    for op in tape.operations:
        if isinstance(op, Allocate):
            for w in op.wires:
                wire, ops = manager.get_wire(**op.hyperparameters)
                new_ops += ops
                wire_map[w] = wire
        elif isinstance(op, Deallocate):
            for w in op.wires:
                deallocated.add(w)
                manager.return_wire(wire_map.pop(w))
        else:
            op = op.map_wires(wire_map)
            if intersection := deallocated.intersection(set(op.wires)):
                raise ValueError(
                    f"Encountered deallocated wires {intersection} in {op}. Dynamic wires cannot be used after deallocation."
                )
            new_ops.append(op.map_wires(wire_map))

    mps = [mp.map_wires(wire_map) for mp in tape.measurements]
    for mp in mps:
        if intersection := deallocated.intersection(set(mp.wires)):
            raise ValueError(
                f"Encountered deallocated wires {intersection} in {mp}. Dynamic wires cannot be used after deallocation."
            )
    return (tape.copy(ops=new_ops, measurements=mps),), null_postprocessing
