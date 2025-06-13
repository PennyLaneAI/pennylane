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

    def __init__(self, zeroed=(), dirty=(), min_integer=None):
        self._zeroed = list(zeroed)
        self._dirty = list(dirty)
        self._loaned = {}  # wire to final register type
        self.min_integer = min_integer

    def get_wire(self, require_zeros):
        """Retrieve a concrete wire label from available registers."""
        if not self._zeroed and not self._dirty:
            if self.min_integer is None:
                raise ValueError("no wires left to allocate.")
            self._zeroed.append(self.min_integer)
            self.min_integer += 1
        if require_zeros:
            if self._zeroed:
                w = self._zeroed.pop()
                self._loaned[w] = "zeroed"
                return w, []
            w = self._dirty.pop()
            self._loaned[w] = "dirty"
            m = measure(w, reset=True)
            return w, m.measurements

        if self._dirty:
            w = self._dirty.pop()
            self._loaned[w] = "dirty"
            return w, []
        w = self._zeroed.pop()
        self._loaned[w] = "zeroed"
        return w, []

    def return_wire(self, wire, reset_to_original=False):
        """Return a wire label back to be re-used."""
        reg_type = self._loaned.pop(wire)
        if reg_type == "zeroed" and reset_to_original:
            self._zeroed.append(wire)
        else:
            self._dirty.append(wire)


def null_postprocessing(results: ResultBatch) -> Result:
    """An empty postprocessing function returned by resolve_dynamic_wires"""
    return results[0]


@transform
def resolve_dynamic_wires(
    tape: QuantumScript,
    zeroed: Sequence[Hashable] = (),
    dirty: Sequence[Hashable] = (),
    min_integer: Optional[int] = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Map dynamic wires to concrete values determined by the provided ``zeroed`` and ``dirty`` registers.

    Args:
        tape (QuantumScript): A circuit that may contain dynamic wire allocations and deallocations
        zeroed (Sequence[Hashable]): a register of wires known to be the zero state
        dirty (Sequence[Hashable]): a register of wires with any state

    Returns:
        tuple[QuantumScript], Callable[[ResultBatch], Result]: A batch of tapes and a postprocessing function

    .. code-block:: python

        @qml.qnode(qml.device('default.qubit'))
        def circuit(, require_zeros=True):
            with qml.allocation.safe_allocate(1, require_zeros=require_zeros) as wires:
                qml.X(wires)
            with qml.allocation.safe_allocate(1, require_zeros=require_zeros) as wires:
                qml.Y(wires)
            return qml.state()

    >>> print(qml.draw(circuit)())
    <DynamicWire>: ──Allocate──X──Deallocate─┤  State
    <DynamicWire>: ──Allocate──Y──Deallocate─┤  State

    If we provide two zeroed qubits to the transform, we can see that the two operations have been
    assigned to both wires known to be in the zero state.

    >>> assigned_two_zeroed = qml.transforms.resolve_dynamic_wires(circuit, zeroed=("a", "b"))
    >>> print(qml.draw(assigned_two_zeroed)())
    a: ──Y─┤  State
    b: ──X─┤  State

    If we only provide one zeroed wire, we perform a reset on that wire before reusing for the ``Y`` operation.

    >>> assigned_one_zeroed = qml.transforms.resolve_dynamic_wires(circuit, zeroed=("a", ))
    >>> print(qml.draw(assigned_one_zeroed)())
    a: ──X──┤↗│  │0⟩──Y─┤  State

    If we only provide dirty qubits with unknown states, then they will be reset to zero before being used
    in an operation that requires a zero state.

    >>> assigned_dirty = qml.transforms.resolve_dynamic_wires(circuit, dirty=("a", "b"))
    >>> print(qml.draw(assigned_dirty)())
    b: ──┤↗│  │0⟩──X──┤↗│  │0⟩──Y─┤  State

    If the wire allocations had ``require_zeros=False``, no reset operations would occur.

    >>> print(qml.draw(assigned_dirty)(require_zeros=False))
    b: ──X──Y─┤  State

    Instead of registers of available wires, a ``min_integer`` can be specified instead.  The ``min_integer`` indicates
    the first integer to start allocating wires to.  Whenever we have no qubits available to allocate, we increment the integer
    and add a new wire to the pool:

    >>> circuit_integers = qml.transforms.resolve_dynamic_wires(circuit, min_integer=0)
    >>> print(qml.draw(circuit_integers)())
    0: ──X──┤↗│  │0⟩──Y─┤  State

    Note that we still prefer using already created wires over creating new wires.

    .. code-block:: python

        @qml.qnode(qml.device('default.qubit'))
        def multiple_allocations():
            with qml.allocation.safe_allocate(1) as wires:
                qml.X(wires)
            with qml.allocation.safe_allocate(3) as wires:
                qml.Toffoli(wires)
            return qml.state()

    >>> circuit_integers2 = qml.transforms.resolve_dynamic_wires(multiple_allocations, min_integer=0)
    >>> print(qml.draw(circuit_integers2)())
    0: ──X──┤↗│  │0⟩─╭●─┤  State
    1: ──────────────├●─┤  State
    2: ──────────────╰X─┤  State

    """
    manager = _WireManager(zeroed=zeroed, dirty=dirty, min_integer=min_integer)

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
                manager.return_wire(wire_map.pop(w), **op.hyperparameters)
        else:
            op = op.map_wires(wire_map)
            if intersection := deallocated.intersection(set(op.wires)):
                raise ValueError(
                    f"Encountered deallocated wires {intersection}. Dynamic wires cannot be used after deallocation."
                )
            new_ops.append(op.map_wires(wire_map))

    mps = [mp.map_wires(wire_map) for mp in tape.measurements]
    return (tape.copy(ops=new_ops, measurements=mps),), null_postprocessing
