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


"""Built-in decomposition rules for qubit.special_unitary."""

from pennylane import ops
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.qubit.special_unitary import SpecialUnitary, TmpPauliRot
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import WiresLike


@register_resources({SpecialUnitary: 1}, name="negate_parameters")
def adjoint_special_unitary_standard(theta, wires, **__):
    """Decompose ``Adjoint(SpecialUnitary)`` using the ``negate_parameters`` strategy."""
    SpecialUnitary(-theta, wires=wires)


add_decomps("Adjoint(SpecialUnitary)", adjoint_special_unitary_standard)


# pylint: disable-next=unused-argument
def _tmp_paulirot_resources(theta, wires, pauli_word: str):
    return {ops.PauliRot(Float, pauli_word=pauli_word, wires=Wire[len(pauli_word)]): 1}


@register_resources(_tmp_paulirot_resources, name="to_pauli_rot")
def tmp_paulirot_standard(theta: TensorLike, wires: WiresLike, pauli_word: str):
    """Decompose ``TmpPauliRot`` using the ``to_pauli_rot`` strategy."""
    ops.PauliRot(theta, pauli_word=pauli_word, wires=wires)


add_decomps(TmpPauliRot, tmp_paulirot_standard)


@register_resources(lambda base: _tmp_paulirot_resources(**base.arguments), name="negate_angle")
def adjoint_tmp_paulirot_standard(base):
    """Decompose ``Adjoint(TmpPauliRot)`` using the ``negate_angle`` strategy."""
    ops.PauliRot(-base.theta, pauli_word=base.pauli_word, wires=base.wires)


add_decomps("Adjoint(TmpPauliRot)", adjoint_tmp_paulirot_standard)
