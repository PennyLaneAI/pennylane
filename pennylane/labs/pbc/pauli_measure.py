# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Workaround function for Mid-circuit Pauli measurement with correct post-measurement state"""

# pylint: disable=protected-access
import uuid

import pennylane as qp
from pennylane.drawer._add_obj import _add_grouping_symbols, _add_obj
from pennylane.ops.mid_measure.measurement_value import MeasurementValue
from pennylane.ops.mid_measure.mid_measure import MidMeasure

from .ops import _dequeue


class MeasurePauliWord(qp.operation.Operation):
    """Mid-circuit measurement of an arbitrary Pauli word."""

    num_wires = None
    num_params = 0

    def __init__(self, P):
        _dequeue(P)
        with qp.QueuingManager.stop_recording():
            # Extract sign/phase from P (handles SProd like -X @ Y)
            P = qp.simplify(P)
            if isinstance(P, qp.ops.SProd):
                self._phase = P.scalar
                P_bare = P.base
            else:
                self._phase = 1.0
                P_bare = P

            # Determine if measurement result should be flipped
            # For Hermitian observables, phase must be ±1
            # Flip when phase is -1 (or has negative real part for robustness)
            self._sign_flip = qp.math.isclose(self._phase, -1)

            self._pauli_word = qp.pauli.pauli_word_to_string(P_bare)
            wires = P_bare.wires
            measurement_id = str(uuid.uuid4())
            self._mid_measure = MidMeasure(wires=[wires[0]], meas_uid=measurement_id)
            self._measurement_value = MeasurementValue([self._mid_measure])
        super().__init__(wires=wires)

    def label(self, decimals=None, base_label=None, cache=None, wire=None):
        sign = "-" if self._sign_flip else ""
        if wire is None:
            return sign + "".join(f"{p}↗" for p in self._pauli_word)
        idx = list(self.wires).index(wire)
        prefix = sign if idx == 0 else ""
        return f"{prefix}{self._pauli_word[idx]}↗"

    @property
    def has_decomposition(self):
        return True

    def decomposition(self):
        wires = self.wires
        pw = self._pauli_word
        ops = []
        for pauli, wire in zip(pw, wires):
            if pauli == "X":
                ops.append(qp.Hadamard(wire))
            elif pauli == "Y":
                ops.append(qp.adjoint(qp.S)(wire))
                ops.append(qp.Hadamard(wire))
        target = wires[0]
        for w in wires[1:]:
            ops.append(qp.CNOT((w, target)))

        if self._sign_flip:
            ops.append(qp.X(target))

        # Include MidMeasure AND its measurements for proper MCM handling
        ops.append(self._mid_measure)

        if self._sign_flip:
            ops.append(qp.X(target))

        for w in reversed(list(wires[1:])):
            ops.append(qp.CNOT((w, target)))
        for pauli, wire in zip(pw, wires):
            if pauli == "X":
                ops.append(qp.Hadamard(wire))
            elif pauli == "Y":
                ops.append(qp.Hadamard(wire))
                ops.append(qp.S(wire))
        return ops


# Register custom drawer dispatch
@_add_obj.register
def _add_measure_pauli_word(
    op: MeasurePauliWord, layer_str, config, tape_cache=None, skip_grouping_symbols=False
):
    layer_str = _add_grouping_symbols(op.wires, layer_str, config)
    for w in op.wires:
        label = op.label(wire=w, decimals=config.decimals, cache=config.cache)
        layer_str[config.wire_map[w]] += label
    return layer_str


def measure(P):
    """Projectively measure a Pauli word mid-circuit. Returns MeasurementValue.
    Collapses the system into the +/- 1 eigenstate."""
    op = MeasurePauliWord(P)
    return op._measurement_value
