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
Contains an implementation of a QuantumScriptSequence that can contain a series
of tapes to be executed sequentially without resetting the devic state. For the
FTQC/MBQC device prototype.
"""
import copy
from functools import cached_property

from pennylane.devices.preprocess import null_postprocessing
from pennylane.ftqc import RotXZX
from pennylane.measurements import Shots
from pennylane.ops import RZ
from pennylane.ops.functions import map_wires
from pennylane.tape.qscript import QuantumScript
from pennylane.transforms import transform


@transform
def split_at_non_clifford_gates(tape):
    """Split the tape into multiple tapes, stored in order. The sequence represents the same
    circuit if the tapes included in it are executed in sequence (without resetting the device
    state), but only contains a single non-Clifford gate in each tape. This allows the
    Pauli tracker to flush its buffer before each non-Clifford gate.

    ..note ::
        This implementation pays no attention to wires/commutation, and therefore splits the tapes
        up more than necessary.

    Args:
        tape (QNode or QuantumScript or Callable): The quantum circuit to modify the mid-circuit measurements of.

    Returns:
        tuple[List[QuantumScriptSequence], function]: A new representation of the tape that contains multiple segments to be executed in order without resetting the device.

    We split the list of operations at each non-Clifford gate, while maintaining the
    operation order. The result is a list of lists, where a new sub-list begins each
    time a non-Clifford gate is encountered. Flattening this list of lists should result
    in the original list of operations.

    **Example**

    >>> tape = QuantumScript([RZ(1.2, 0), X(1), S(2), RotXZX(0.1, 0.2, 0.3, 1), RZ(1.2, 2)])
    >>> (seq,), null_postprocessing_fn = split_at_non_clifford_gates(tape)
    >>> seq.operations
    [[RZ(1.2, wires=[0]), X(1), S(2)],
    [RotXZX(0.1, 0.2, 0.3, wires=[1])],
    [RZ(1.2, wires=[2])]]


    """
    all_operations = []
    current_ops = [tape.operations[0]]

    for op in tape.operations[1:]:
        # if its a non-Clifford gate, start a new list
        if isinstance(op, (RotXZX, RZ)):
            all_operations.append(current_ops)
            current_ops = []
        current_ops.append(op)
    all_operations.append(current_ops)

    tapes = []
    for ops_list in all_operations[:-1]:
        tapes.append(tape.copy(operations=ops_list, measurements=[]))
    tapes.append(tape.copy(operations=all_operations[-1]))

    return [
        QuantumScriptSequence(tapes),
    ], null_postprocessing


class QuantumScriptSequence:
    """A sequence of tapes meant to be executed in order without resetting the system state.
    Intermediate tapes may return mid-circuit measurements, or nothing. This is not currently
    validated, but it is assumed. The final tape returns terminal measurements."""

    def __init__(self, tapes, shots=None):

        if shots is None:
            shots = [tape.shots for tape in tapes]
            if len(set(shots)) != 1:
                raise RuntimeError(
                    "All scripts in a QuantumScriptSequence must have the same shots"
                )
            shots = shots[0]
        else:
            if not isinstance(shots, Shots):
                shots = Shots(shots)
        self._shots = shots

        self._tapes = []

        for tape in tapes:
            aux_tape = QuantumScript(
                tape.operations,
                tape.measurements,
                shots=[1],
            )
            self._tapes.append(aux_tape)

    @property
    def tapes(self):
        """Returns all the tapes in the sequence.

        Returns:
            list[QuantumScript]: list of all tapes
        """
        return self._tapes

    @property
    def final_tape(self):
        """Returns the final tape in the sequence.

        Returns:
            QuantumScript: the final tape in the sequence, with terminal measurements
        """
        return self._tapes[-1]

    @property
    def intermediate_tapes(self):
        """Returns all but the final tape in the sequence.

        Returns:
            list[QuantumScript]: list of all tapes except the tape with terminal measurements
        """

        return self._tapes[:-1]

    @property
    def measurements(self):
        """Returns the final measurements for the sequence.

        Returns:
            list[.MeasurementProcess]: list of measurement processes
        """
        return self.final_tape.measurements

    @property
    def intermediate_measurements(self):
        """Returns the intermediate measurements for all but the final tape. Since these
        are in the middle of an execution, they are expected to be empty, or to be mid-circuit
        measurements.

        Returns:
            list[list[MidMeasureMP]]: nested list of the returned MCMs for all but the final tape
        """
        return [tape.measurements for tape in self.intermediate_tapes]

    @property
    def operations(self):
        """Returns the operations for each tape

        Returns:
            list[list[Operation]]: a nested list of the operations for each tape
        """
        return [tape.operations for tape in self.tapes]

    @cached_property
    def wires(self):
        """Returns the wires used in the quantum script process

        Returns:
            ~.Wires: wires in quantum script process
        """
        wires = self.tapes[0].wires
        for tape in self.tapes[1:]:
            wires += tape.wires
        return wires

    @property
    def num_wires(self) -> int:
        """Returns the number of wires in the quantum script process

        Returns:
            int: number of wires in quantum script process
        """
        return len(self.wires)

    @property
    def shots(self):
        """Returns a ``Shots`` object containing information about the number
        and batches of shots

        Returns:
            ~.Shots: Object with shot information
        """
        return self._shots

    def __repr__(self) -> str:
        return f"<QuantumScriptSequence: wires={list(self.wires)}>"

    def map_to_standard_wires(self) -> "QuantumScriptSequence":
        """Wrapper to apply qscript.map_to_standard_wires to each segment contained in the Sequence"""
        wire_map = self._get_standard_wire_map()
        if wire_map is None:
            return self
        new_tapes = []
        for tape in self.tapes:
            tapes, fn = map_wires(tape, wire_map)
            new_tapes.append(fn(tapes))

        return self.copy(tapes=new_tapes)

    def _get_standard_wire_map(self) -> dict:
        """Helper function to produce the wire map for map_to_standard_wires. Wire map
        is the same as if the sequence were a flat tape"""
        flat_ops = []
        for ops in self.operations:
            flat_ops.extend(ops)

        as_tape = QuantumScript(flat_ops, self.measurements)
        return as_tape._get_standard_wire_map()  # pylint: disable=protected-access

    def copy(self, copy_operations: bool = False, **update):
        """Make it copyable as if it were a tape where possible. This allows transforms
        that only affect measurements or shots to be applied directly to a QuantumScriptSequnce
        as if it were a normal QuantumScript. Does not allow modifications to operations or
        trainable parameters like tape.copy does, because transforms or functions modifying
        operations on a tape will not work without modification on a sequence of tapes.

        Allows updating tapes as a whole as an alternative for QuantumScriptSquence-specific
        functions to deal with modifying operations on tapes."""

        if copy_operations is True:
            raise RuntimeError("Can't use copy_operations when copying a QuantumScriptSequence")

        if update:
            if "ops" in update:
                update["operations"] = update["ops"]
            for k in update:
                if k not in ["tapes", "measurements", "shots"]:
                    raise TypeError(f"{self.__class__}.copy() cannot update '{k}'")
            if "tapes" in update and "measurements" in update:
                raise RuntimeError(
                    "Can't update tapes and measurements at the same time, as tapes include measurements"
                )

        _tapes = update.get("tapes", [copy.copy(tape) for tape in self.tapes])
        _shots = update.get("shots", self.shots)

        if "measurements" in update:
            old_final_tape = _tapes.pop()
            _new_final_tape = old_final_tape.copy(measurements=update["measurements"])
            _tapes.append(_new_final_tape)

        new_sequence = QuantumScriptSequence(
            tapes=_tapes,
            shots=_shots,
        )

        return new_sequence
