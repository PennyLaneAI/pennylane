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

"""Built-in decomposition rules for templates.state_preparations.multiplexer_state_prep."""

import pennylane as qp
from pennylane import math
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.decomposition_rule import register_condition
from pennylane.templates.state_preparations.mottonen import _get_alpha_y
from pennylane.templates.state_preparations.multiplexer_state_prep import (
    MultiplexerStatePreparation,
)
from pennylane.typing import Complex, Float, Wire


def _select_pauli_rot_resources(num_wires) -> dict:
    return {
        qp.SelectPauliRot(Float[2**i], control_wires=Wire[i], target_wire=Wire[1], rot_axis="Y"): 1
        for i in range(num_wires)
    }


def _select_pauli_rots(amplitudes, wires):
    """Queue the SelectPauliRot("Y") gates that prepare the given (signed or absolute)
    amplitudes. For signed amplitudes, the sign is encoded directly into the leaf level
    (k=1) angle by ``_get_alpha_y``, eliminating the need for SelectPauliRot("Z") gates."""
    n = len(wires)
    for k in range(n):
        alpha_y_k = _get_alpha_y(amplitudes, n, n - k)
        qp.SelectPauliRot(alpha_y_k, target_wire=wires[k], control_wires=wires[:k], rot_axis="Y")


# pylint: disable=unused-argument
def _real_multiplexer_state_prep_resources(state_vector, wires, check=False) -> dict:
    r"""Computes the resources of '_real_multiplexer_state_prep_decomposition'."""
    return _select_pauli_rot_resources(len(wires))


@register_condition(lambda state_vector, **_: not math.is_complex_dtype(state_vector))
@register_resources(_real_multiplexer_state_prep_resources)
def _real_multiplexer_state_prep_decomposition(state_vector, wires, **_):
    r"""Decomposition of MultiplexerStatePreparation for a real-valued state vector."""
    _select_pauli_rots(state_vector, wires)


# pylint: disable=unused-argument
def _complex_multiplexer_state_prep_resources(state_vector, wires, check=False) -> dict:
    r"""Upper bound on the gates emitted by '_complex_multiplexer_state_prep_decomposition'.
    The ``DiagonalQubitUnitary`` is skipped if the state is (close to) real-valued or all
    phases are zero."""
    num_wires = len(wires)
    resources = _select_pauli_rot_resources(num_wires)
    resources[qp.DiagonalQubitUnitary(Complex[2**num_wires], wires=Wire[num_wires])] = 1
    return resources


@register_condition(lambda state_vector, **_: math.is_complex_dtype(state_vector))
@register_resources(_complex_multiplexer_state_prep_resources, exact=False)
def _complex_multiplexer_state_prep_decomposition(state_vector, wires, **_):
    r"""Decomposition of MultiplexerStatePreparation for a complex-valued state vector."""

    # A complex state vector that is (close to) real-valued can be prepared from its signed
    # real amplitudes, without the DiagonalQubitUnitary.
    is_real = math.is_real_obj_or_close(state_vector) and not math.requires_grad(state_vector)
    a = math.real(state_vector) if is_real else math.abs(state_vector)

    _select_pauli_rots(a, wires)

    if not is_real:
        omega = math.angle(state_vector)
        if math.is_tracer(omega) or math.requires_grad(omega) or not math.allclose(omega, 0):
            qp.DiagonalQubitUnitary(math.exp(1j * omega), wires=wires)


add_decomps(
    MultiplexerStatePreparation,
    _real_multiplexer_state_prep_decomposition,
    _complex_multiplexer_state_prep_decomposition,
)
