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

"""Built-in decomposition rules for op_math.evolution."""

import pennylane as qp
from pennylane import math
from pennylane.core import queuing
from pennylane.ops.op_math.evolution import Evolution
from pennylane.typing import Float, Wire


def _pauli_rot_decomp_condition(base, **__):
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    # The PauliRot decomposition is only applicable when the base is a Pauli word
    return qp.pauli.is_pauli_word(base)


def _pauli_rot_decomp_resource(base, **__):
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    pauli_word = qp.pauli.pauli_word_to_string(base)
    return {qp.PauliRot(Float, pauli_word=pauli_word, wires=Wire[len(pauli_word)]): 1}


@qp.register_condition(_pauli_rot_decomp_condition)
@qp.register_resources(_pauli_rot_decomp_resource)
def pauli_rot_decomp(*params, wires, base, **_):  # pylint: disable=unused-argument
    """Decompose the operator into a single PauliRot operator."""
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    coeff = -1j * params[0]
    if isinstance(base, qp.ops.SProd):
        coeff, base = coeff * base.scalar, base.base
    coeff = 2j * coeff  # The 2j cancels the coefficient added by PauliRot
    coeff = (
        math.real_if_close(coeff)
        if math.get_interface(coeff) not in ("torch", "jax")
        else math.real(coeff)
    )
    pauli_word = qp.pauli.pauli_word_to_string(base)
    qp.PauliRot(coeff, pauli_word, base.wires)


qp.add_decomps(Evolution, pauli_rot_decomp)
