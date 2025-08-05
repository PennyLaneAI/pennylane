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

import pennylane as qml
from pennylane.measurements import Shots
from pennylane.tape.qscript import QuantumScript


class QuantumScriptSequence:
    """A sequence of tapes meant to be executed in order without resetting the system state.
    Intermediate tapes may return mid-circuit measurements, or nothing. This is not currently
    validated, but it is assumed. The final tape returns terminal measurements."""

    shortname = "QuantumScriptSequence"

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
    def wires(self) -> qml.wires.Wires:
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
        return f"<{self.shortname}: wires={list(self.wires)}>"

    def map_to_standard_wires(self) -> "QuantumScriptSequence":
        """Wrapper to apply qscript.map_to_standard_wires to each segment contained in the Sequence"""
        wire_map = self._get_standard_wire_map()
        if wire_map is None:
            return self
        new_tapes = []
        for tape in self.tapes:
            tapes, fn = qml.map_wires(tape, wire_map)
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
        """Make it copy-able as if it were a tape where possible. Do not allow
        modifications to operations or trainable parameters, because any transform
        or function modifying operations on a tape will not work on a sequence of
        tapes. Allow updating tapes as a whole as an alternative for
        QuantumScriptSquence-specific functions to deal with modifying operations
        on tapes.

        This is not able to support trainable parameters and almost certainly also
        has other flaws. It is not a thorough implementation of the desired behaviour."""
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
