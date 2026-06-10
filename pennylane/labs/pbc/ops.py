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
"""Operator shotcuts"""

import pennylane as qp


def _dequeue(op):
    """Remove an operator from the active queue if recording."""
    if qp.QueuingManager.recording():
        try:
            qp.QueuingManager.remove(op)
        except KeyError:
            pass  # already removed or not queued
    # return op


def ppr(angle, P):
    """Pauli rotation gate. Dequeues P so it's used purely as data."""
    _dequeue(P)
    wires = P.wires
    pw = qp.pauli.pauli_word_to_string(P)
    qp.PauliRot(angle, pw, wires)
