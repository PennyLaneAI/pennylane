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

from pennylane import decomposition, math, ops, pauli
from pennylane.core import queuing
from pennylane.ops.op_math.evolution import Evolution
from pennylane.typing import Float, Wire


def _pauli_rot_decomp_condition(base, **__):
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    # The PauliRot decomposition is only applicable when the base is a Pauli word
    return pauli.is_pauli_word(base)


def _exp_to_pauli_rot_resources(base, **__):
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    pauli_word = pauli.pauli_word_to_string(base)
    return {ops.PauliRot(Float, pauli_word=pauli_word, wires=Wire[len(pauli_word)]): 1}


@decomposition.register_condition(_pauli_rot_decomp_condition)
@decomposition.register_resources(_exp_to_pauli_rot_resources, name="to_pauli_rot")
def exp_to_pauli_rot(*params, wires, base, **_):  # pylint: disable=unused-argument
    """Decompose the operator into a single PauliRot operator."""
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    coeff = -1j * params[0]
    if isinstance(base, ops.SProd):
        coeff, base = coeff * base.scalar, base.base
    coeff = 2j * coeff  # The 2j cancels the coefficient added by PauliRot
    coeff = (
        math.real_if_close(coeff)
        if math.get_interface(coeff) not in ("torch", "jax")
        else math.real(coeff)
    )
    pauli_word = pauli.pauli_word_to_string(base)
    ops.PauliRot(coeff, pauli_word, base.wires)


decomposition.add_decomps(Evolution, exp_to_pauli_rot)
