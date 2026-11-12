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

"""Built-in decomposition rules for templates.layers.particle_conserving_u1."""

import numpy as np

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, CZ, BasisState, CRot, PhaseShift
from pennylane.templates.layers.particle_conserving_u1 import ParticleConservingU1
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires, WiresLike


def _particle_conserving_u1_resources(n_layers: int, num_wires: int):
    num_nm_wires = num_wires - 1  # number of pairs of even-indexed of wires
    return {
        BasisState(Bool[num_wires], Wire[num_wires]): 1,
        CZ: 3 * num_nm_wires * n_layers,
        CRot: 3 * num_nm_wires * n_layers,
        PhaseShift: 6 * num_nm_wires * n_layers,
        CNOT: 4 * num_nm_wires * n_layers,
    }


def _decompose_ua_qfunc(phi: float, wires: WiresLike):
    r"""Appends the circuit decomposing the controlled application of the unitary
    :math:`U_A(\phi)`

    .. math::

        U_A(\phi) = \left(\begin{array}{cc} 0 & e^{-i\phi} \\ e^{-i\phi} & 0 \\ \end{array}\right)

    in terms of the quantum operations supported by PennyLane.

    :math:`U_A(\phi)` is used in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_,
    to define two-qubit exchange gates required to build particle-conserving
    VQE ansatze for quantum chemistry simulations. See :func:`~.ParticleConservingU1`.

    :math:`U_A(\phi)` is expressed in terms of ``PhaseShift``, ``Rot`` and ``PauliZ`` operations
    :math:`U_A(\phi) = R_\phi(-2\phi) R(-\phi, \pi, \phi) \sigma_z`.

    Args:
        phi (float): angle :math:`\phi` defining the unitary :math:`U_A(\phi)`
        wires (Iterable): the wires ``n`` and ``m`` the circuit acts on

    Returns:
          list[.Operator]: sequence of operators defined by this function
    """
    n, m = wires

    CZ(wires=wires)
    CRot(-phi, np.pi, phi, wires=wires)

    # decomposition of C-PhaseShift(2*phi) gate
    PhaseShift(-phi, wires=m)
    CNOT(wires=wires)
    PhaseShift(phi, wires=m)
    CNOT(wires=wires)
    PhaseShift(-phi, wires=n)


@register_resources(_particle_conserving_u1_resources)
def _particle_conserving_u1_decomposition(
    weights: list, wires: WiresLike, init_state: tuple[int]
):  # pylint: disable=arguments-differ
    nm_wires = [wires[l : l + 2] for l in range(0, len(wires) - 1, 2)]
    nm_wires += [wires[l : l + 2] for l in range(1, len(wires) - 1, 2)]
    n_layers = math.shape(weights)[0]

    if isinstance(wires, Wires):
        wires = wires.labels
        # If we are using capture, then it is going to be problematic to use Wires.
        # Instead, we need to use the wire labels. Here we have a collection of Wires,
        # and this line goes through each of them and turns them into their labels.
        nm_wires = list(map(lambda w: w.labels if isinstance(w, Wires) else w, nm_wires))

    if capture.enabled():
        nm_wires, weights, wires, init_state = (
            math.array(nm_wires, like="jax"),
            math.array(weights, like="jax"),
            math.array(wires, like="jax"),
            math.array(init_state, like="jax"),
        )

    BasisState(init_state, wires=wires)

    @for_loop(n_layers)
    def layers_loop(l):

        @for_loop(len(nm_wires))
        def nm_loop(i):
            wires_ = nm_wires[i]
            phi, theta = weights[l, i, 0], weights[l, i, 1]

            # C-UA(phi)
            _decompose_ua_qfunc(phi, wires_)

            CZ(wires=wires_[::-1])
            CRot(0, 2 * theta, 0, wires=wires_[::-1])

            # C-UA(-phi)
            _decompose_ua_qfunc(-phi, wires_)

        nm_loop()  # pylint: disable=no-value-for-parameter

    layers_loop()  # pylint: disable=no-value-for-parameter


add_decomps(ParticleConservingU1, _particle_conserving_u1_decomposition)
