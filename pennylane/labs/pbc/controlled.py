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
"""Controlled Pauli operations"""

import numpy as np

import pennylane as qp
from pennylane.drawer._add_obj import _add_grouping_symbols, _add_obj

from .ops import _dequeue


class ControlledPauli(qp.operation.Operation):
    """Controlled Pauli operation C(P0, P1) via PPR decomposition."""

    num_wires = None
    num_params = 0

    def __init__(self, P0, P1):
        _dequeue(P0)
        _dequeue(P1)
        with qp.QueuingManager.stop_recording():
            P0 = qp.simplify(P0) if not isinstance(P0, qp.operation.Operation) else P0
            P1 = qp.simplify(P1) if not isinstance(P1, qp.operation.Operation) else P1

            self._P0 = P0
            self._P1 = P1
            self._pw0 = qp.pauli.pauli_word_to_string(P0)
            self._pw1 = qp.pauli.pauli_word_to_string(P1)
            self._wires0 = P0.wires
            self._wires1 = P1.wires
            self._P0P1 = P0 @ P1

        all_wires = qp.wires.Wires.all_wires([self._wires0, self._wires1])
        super().__init__(wires=all_wires)

    def label(self, decimals=None, base_label=None, cache=None, wire=None):
        if wire is None:
            return f"C({''.join(self._pw0)},{''.join(self._pw1)})"
        if wire in self._wires0:
            idx = list(self._wires0).index(wire)
            return f"{self._pw0[idx]}●"

        idx = list(self._wires1).index(wire)
        return f"{self._pw1[idx]}○"

    @property
    def has_decomposition(self):
        return True

    def decomposition(self):
        ops = []
        with qp.QueuingManager.stop_recording():
            pw_P0P1 = qp.pauli.pauli_word_to_string(self._P0P1)
            wires_P0P1 = self._P0P1.wires

        ops.append(qp.PauliRot(np.pi / 2, pw_P0P1, wires_P0P1))
        ops.append(qp.PauliRot(-np.pi / 2, self._pw0, self._wires0))
        ops.append(qp.PauliRot(-np.pi / 2, self._pw1, self._wires1))
        return ops


# Register custom drawer dispatch
@_add_obj.register
def _add_controlled_pauli(
    op: ControlledPauli, layer_str, config, tape_cache=None, skip_grouping_symbols=False
):
    layer_str = _add_grouping_symbols(op.wires, layer_str, config)
    for w in op.wires:
        label = op.label(wire=w, decimals=config.decimals, cache=config.cache)
        layer_str[config.wire_map[w]] += label
    return layer_str


def controlled(P0, P1):
    """Controlled operation C(P0, P1) in terms of its Clifford PPR decomposition."""
    ControlledPauli(P0, P1)
