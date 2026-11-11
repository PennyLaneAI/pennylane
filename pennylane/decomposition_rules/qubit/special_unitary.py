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

# pylint: disable=arguments-differ, import-outside-toplevel

import pennylane as qp
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot
from pennylane.ops.qubit.special_unitary import SpecialUnitary, TmpPauliRot
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import WiresLike


@register_resources({SpecialUnitary: 1})
def _adjoint_special_unitary(theta, wires, **__):
    SpecialUnitary(-theta, wires=wires)


add_decomps("Adjoint(SpecialUnitary)", _adjoint_special_unitary)


# pylint: disable-next=unused-argument
def _tmp_paulirot_decomp_resources(theta, wires, pauli_word: str):
    return {qp.PauliRot(Float, pauli_word=pauli_word, wires=Wire[len(pauli_word)]): 1}


@register_resources(_tmp_paulirot_decomp_resources)
def _tmp_paulirot_decomp(theta: TensorLike, wires: WiresLike, pauli_word: str):
    PauliRot(theta, pauli_word=pauli_word, wires=wires)


add_decomps(TmpPauliRot, _tmp_paulirot_decomp)


@register_resources(lambda base: _tmp_paulirot_decomp_resources(**base.arguments))
def _adjoint_tmp_paulirot(base):
    PauliRot(-base.theta, pauli_word=base.pauli_word, wires=base.wires)


add_decomps("Adjoint(TmpPauliRot)", _adjoint_tmp_paulirot)
