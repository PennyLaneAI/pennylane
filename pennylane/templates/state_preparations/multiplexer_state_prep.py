# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Contains the MultiplexerStatePreparation template."""

import pennylane as qp
from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.decomposition_rule import register_condition
from pennylane.templates.state_preparations.mottonen import _get_alpha_y
from pennylane.typing import AbstractArray, Complex, Float, Wire
from pennylane.wires import Wires


class MultiplexerStatePreparation(Operator2):
    r"""Prepares a quantum state using multiplexed rotations.

    This operation implements the state preparation method described
    in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

    Args:
        state_vector (tensor_like): The state vector of length :math:`2^n` to be prepared on
            :math:`n` wires.
        wires (Sequence[int]): The wires on which to prepare the state.
        check (bool): whether to check that the input state vector has norm 1.0. Defaults to ``False``.

    Raises:
        ValueError: If the length of the input state vector array is not :math:`2^n`, where
            :math:`n` is the number of wires, or if ``check=True`` and the norm of the input
            state is not unity.

    **Example**

    .. code-block:: python

        probs_vector = np.array([0.5, 0., 0.25, 0.25])

        dev = qp.device("default.qubit", wires = 2)

        wires = [0, 1]

        @qp.qnode(dev)
        def circuit():
            qp.MultiplexerStatePreparation(np.sqrt(probs_vector), wires)
            return qp.probs(wires)

    .. code-block:: pycon

        >>> np.round(circuit(), 2)
        array([0.5 , 0.  , 0.25, 0.25])

    .. seealso::

        :class:`~.SelectPauliRot` for a description of the main building blocks used to
        implement this operation.

    """

    dynamic_argnames = ("state_vector",)
    compilable_argnames = ("check",)

    # NOTE: 'state_vector' is deliberately left out of the 'arg_specs' as we want
    # a MultiplexerStatePreparation operator with a real-valued state vector to decompose
    # differently than a complex-valued one.
    arg_specs = {"wires": Wire[-1]}
    wire_sizes = (None,)

    def __init__(self, state_vector, wires, check=False):

        wires = Wires(wires)
        n_amplitudes = math.shape(state_vector)[0]
        if n_amplitudes != 2 ** len(wires):
            raise ValueError(
                f"State vector must be of length {2 ** len(wires)}; got length {n_amplitudes}."
            )

        if check and not math.is_abstract(state_vector):
            norm = math.linalg.norm(state_vector)
            if not math.allclose(norm, 1.0, atol=1e-3):
                raise ValueError(
                    f"State vector must have norm 1.0; the input state vector has norm {norm}"
                )
        # Resource signatures distinguish real and complex states, but not their precision.
        # Canonicalize abstract inputs without casting concrete arrays or tracers.
        if isinstance(state_vector, AbstractArray):
            state_type = Complex if math.is_complex_dtype(state_vector) else Float
            state_vector = state_type[state_vector.shape]

        super().__init__(state_vector, wires=wires)


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
        if math.is_abstract(omega) or math.requires_grad(omega) or not math.allclose(omega, 0):
            qp.DiagonalQubitUnitary(math.exp(1j * omega), wires=wires)


add_decomps(
    MultiplexerStatePreparation,
    _real_multiplexer_state_prep_decomposition,
    _complex_multiplexer_state_prep_decomposition,
)
