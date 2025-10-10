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

from pennylane.allocation import AllocateState
from pennylane.exceptions import AllocationError
from pennylane.measurements import measure
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn, Result, ResultBatch

from .core import transform


class _WireManager:
    """Handles converting dynamic wires into concrete values."""

    def __init__(self, zeroed=(), any_state=(), min_int=None, allow_resets: bool = True):
        self._registers = {AllocateState.ZERO: list(zeroed), AllocateState.ANY: list(any_state)}
        self._loaned = {}  # wire to final register type
        self.min_int = min_int
        self.allow_resets = allow_resets

    def _retrieval_method(self, state: AllocateState):
        _retrieval_map = {AllocateState.ZERO: self._get_zeroed, AllocateState.ANY: self._get_any}
        return _retrieval_map[state]

    @property
    def _zeroed(self):
        return self._registers[AllocateState.ZERO]

    @property
    def _any_state(self):
        return self._registers[AllocateState.ANY]

    def _get_zeroed(self, restored: bool):
        if self._zeroed:
            w = self._zeroed.pop()
            self._loaned[w] = AllocateState.ZERO if restored else AllocateState.ANY
            return w, []
        if self.allow_resets:
            w = self._any_state.pop()
            self._loaned[w] = AllocateState.ZERO if restored else AllocateState.ANY
            m = measure(w, reset=True)
            return w, m.measurements
        self._add_new_wire()
        return self._get_zeroed(restored=restored)

    def _add_new_wire(self):
        if self.min_int is None:
            raise AllocationError("no wires left to allocate.")
        self._zeroed.append(self.min_int)
        self.min_int += 1

    def _get_any(self, restored: bool):
        if self._any_state:
            w = self._any_state.pop()
            self._loaned[w] = AllocateState.ANY
            return w, []
        w = self._zeroed.pop()
        self._loaned[w] = AllocateState.ZERO if restored else AllocateState.ANY
        return w, []

    def get_wire(self, state: AllocateState, restored):
        """Retrieve a concrete wire label from available registers."""
        if not self._zeroed and not self._any_state:
            self._add_new_wire()
        return self._retrieval_method(state)(restored)

    def return_wire(self, wire):
        """Return a wire label back to be re-used."""
        reg_type = self._loaned.pop(wire)
        self._registers[reg_type].append(wire)


def null_postprocessing(results: ResultBatch) -> Result:
    """An empty postprocessing function returned by resolve_dynamic_wires"""
    return results[0]


def _new_ops(operations, manager, wire_map, deallocated):
    for op in operations:
        # check name faster than isinstance
        if op.name == "Allocate":
            for w in op.wires:
                wire, ops = manager.get_wire(**op.hyperparameters)
                yield from ops
                wire_map[w] = wire
        elif op.name == "Deallocate":
            for w in op.wires:
                deallocated.add(w)
                manager.return_wire(wire_map.pop(w))
        else:
            if wire_map:
                op = op.map_wires(wire_map)
            if deallocated and (intersection := deallocated.intersection(set(op.wires))):
                raise AllocationError(
                    f"Encountered deallocated wires {intersection} in {op}. Dynamic wires cannot be used after deallocation."
                )
            yield op


@transform
def resolve_dynamic_wires(
    tape: QuantumScript,
    zeroed: Sequence[Hashable] = (),
    any_state: Sequence[Hashable] = (),
    min_int: int | None = None,
    allow_resets: bool = True,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Map dynamic wires to concrete values determined by the provided ``zeroed`` and ``any_state`` registers.

    Args:
        tape (QuantumScript): A circuit that may contain dynamic wire allocations and deallocations
        zeroed (Sequence[Hashable]): a register of wires known to be in the :math:`|0\rangle` state
        any_state (Sequence[Hashable]): a register of wires with any state
        min_int (Optional[int]): If not ``None``, new wire labels can be created starting at this
            integer and incrementing whenever a new wire is needed.
        allow_resets (boo): Whether or not mid circuit measurements with ``reset=True`` can be added
            to turn any state wires into zeroed wires.

    Returns:
        tuple[QuantumScript], Callable[[ResultBatch], Result]: A batch of tapes and a postprocessing function

    .. note::

        This transform currently uses a "Last In, First Out" (LIFO) stack based approach to distributing wires.
        This minimizes the total number of wires used, at the cost of higher depth and more resets. Other
        approaches could be taken as well, such as a "First In, First out" algorithm that minimizes depth.

        This approach also means we pop wires from the *end* of the stack first.

    For a dynamic wire requested to be in the zero state (``state="zero"``), we try three things before raising an error:

      #. If wires exist in the ``zeroed`` register, we take one from that register
      #. If no ``zeroed`` wires exist and we are allowed to use resets, we pull one from ``any_state`` and apply a reset operation
      #. If no wires exist in the ``zeroed`` or ``any_state`` registers and ``min_int`` is not ``None``,
         we increment ``min_int`` and add a new wire.

    For a dynamic wire with ``state="any"``, we try:

      #. If wires exist in the ``any_state`` register, we take one from there
      #. If no wires exist in ``any_state``, we pull one from ``zeroed``
      #. If no wires exist in the ``zeroed`` or ``any_state`` registers and ``min_int`` is not ``None``,
         we increment ``min_int`` and add a new wire

    This transform uses a combination of two different modes: one with fixed registers specified by ``zeroed`` and
    ``any_state``, and one with a dynamically sized register characterized by the integer ``min_int``.  We assume
    that the upfront cost associated with using more wires has already been paid for anything in ``zeroed`` and
    ``any_state``. Whether or not we use them, they will still be there. In this case, using a fresh wire is cheaper
    than reset.  For the dynamically sized register, we assume that we have to pay an
    additional cost each time we allocate a new wire. For the dynamically sized register, applying a reset
    operation is therefor cheaper than allocating a new wire.

    This approach minimizes the width of the circuit at the cost of more reset operations.

    .. code-block:: python

        def circuit(state="zero"):
            with qml.allocation.allocate(1, state=state) as wires:
                qml.X(wires)
            with qml.allocation.allocate(1, state=state) as wires:
                qml.Y(wires)

    >>> print(qml.draw(circuit)())
    <DynamicWire>: ──Allocate──X──Deallocate─┤
    <DynamicWire>: ──Allocate──Y──Deallocate─┤

    If we provide two zeroed qubits to the transform, we can see that the two operations have been
    assigned to both wires known to be in the zero state.

    >>> from pennylane.transforms import resolve_dynamic_wires
    >>> assigned_two_zeroed = resolve_dynamic_wires(circuit, zeroed=("a", "b"))
    >>> print(qml.draw(assigned_two_zeroed)())
    a: ──Y─┤
    b: ──X─┤

    If we only provide one zeroed wire, we perform a reset on that wire before reusing for the ``Y`` operation.

    >>> assigned_one_zeroed = resolve_dynamic_wires(circuit, zeroed=("a",))
    >>> print(qml.draw(assigned_one_zeroed)())
    a: ──X──┤↗│  │0⟩──Y─┤

    This reset behavior can be turned off with ``allow_resets=False``.

    >>> no_resets = resolve_dynamic_wires(circuit, zeroed=("a",), allow_resets=False)
    >>> print(qml.draw(no_resets)())
    AllocationError: no wires left to allocate.

    If we only provide ``any_state`` qubits with unknown states, then they will be reset to zero before being used
    in an operation that requires a zero state.

    >>> assigned_any_state = resolve_dynamic_wires(circuit, any_state=("a", "b"))
    >>> print(qml.draw(assigned_any_state)())
    b: ──┤↗│  │0⟩──X──┤↗│  │0⟩──Y─|


    Note that the last provided wire with label ``"b"`` is used first.
    If the wire allocations had ``state="any"``, no reset operations would occur:

    >>> print(qml.draw(assigned_any_state)(state="any"))
    b: ──X──Y─┤

    Instead of registers of available wires, a ``min_int`` can be specified instead.  The ``min_int`` indicates
    the first integer to start allocating wires to.  Whenever we have no qubits available to allocate, we increment the integer
    and add a new wire to the pool:

    >>> circuit_integers = resolve_dynamic_wires(circuit, min_int=0)
    >>> print(qml.draw(circuit_integers)())
    0: ──X──┤↗│  │0⟩──Y─┤

    Note that we still prefer using already created wires over creating new wires.

    .. code-block:: python

        def multiple_allocations():
            with qml.allocation.allocate(1) as wires:
                qml.X(wires)
            with qml.allocation.allocate(3) as wires:
                qml.Toffoli(wires)

    >>> circuit_integers2 = resolve_dynamic_wires(multiple_allocations, min_int=0)
    >>> print(qml.draw(circuit_integers2)())
    0: ──X──┤↗│  │0⟩─╭●─┤
    1: ──────────────├●─┤
    2: ──────────────╰X─┤

    If both an explicit register and ``min_int`` are specified, ``min_int`` will be used once all available
    explicit wires are loaned out. Below, ``"a"`` is extracted and used first, but then wires
    are extracted starting from ``0``.

    >>> zeroed_and_min_int = resolve_dynamic_wires(multiple_allocations, zeroed=("a",), min_int=0)
    >>> print(qml.draw(zeroed_and_min_int)())
    a: ──X──┤↗│  │0⟩─╭●─┤
    0: ──────────────├●─┤
    1: ──────────────╰X─┤

    """
    manager = _WireManager(
        zeroed=zeroed, any_state=any_state, min_int=min_int, allow_resets=allow_resets
    )

    wire_map = {}
    deallocated = set()

    # note that manager, wire_map, and deallocated updated in place
    new_ops = list(_new_ops(tape.operations, manager, wire_map, deallocated))

    if wire_map:
        mps = [mp.map_wires(wire_map) for mp in tape.measurements]
    else:
        mps = tape.measurements
    for mp in mps:
        if intersection := deallocated.intersection(set(mp.wires)):
            raise AllocationError(
                f"Encountered deallocated wires {intersection} in {mp}. Dynamic wires cannot be used after deallocation."
            )

    if not wire_map and not deallocated:
        return (tape,), null_postprocessing
    # use private trainable params to avoid calculating them if they haven't already been set
    # pylint: disable=protected-access
    return (
        tape.copy(ops=new_ops, measurements=mps, trainable_params=tape._trainable_params),
    ), null_postprocessing
