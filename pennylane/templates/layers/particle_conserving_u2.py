# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the hardware efficient ``ParticleConservingU2`` template.
"""
import pennylane as qml

# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, CRX, RZ
from pennylane.templates.utils import (
    check_shape,
    get_shape,
)
from pennylane.wires import Wires


def u2_ex_gate(phi, wires=None):
    r"""Implements the two-qubit exchange gate :math:`U_{2,\mathrm{ex}}` proposed in
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ to build particle-conserving VQE ansatze
    for Quantum Chemistry simulations.

    The unitary matrix :math:`U_{2, \mathrm{ex}}` acts on the Hilbert space of two qubits

    .. math::

        U_{2, \mathrm{ex}}(\phi) = \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & \mathrm{cos}(\phi) & -i\;\mathrm{sin}(\phi) & 0 \\
        0 & -i\;\mathrm{sin}(\phi) & \mathrm{cos}(\phi) & 0 \\
        0 & 0 & 0 & 1 \\
        \end{array}\right).

    Args:
        phi (float): angle entering the controlled-RX operator :math:`CRX(2\phi)`
        wires (list[Wires]): the two wires ``n`` and ``m`` the circuit acts on
    """

    CNOT(wires=wires)
    CRX(2 * phi, wires=wires[::-1])
    CNOT(wires=wires)


@template
def ParticleConservingU2(weights, wires, init_state=None):
    r"""Implements the heuristic VQE ansatz for Quantum Chemistry simulations using the
    particle-conserving entangler :math:`U_\mathrm{ent}(\vec{\theta}, \vec{\phi})` proposed in
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_.

    This template prepares :math:`N`-qubit trial states by applying :math:`D` layers of the entangler
    block :math:`U_\mathrm{ent}(\vec{\theta}, \vec{\phi})` to the Hartree-Fock state

    .. math::

        \vert \Psi(\vec{\theta}, \vec{\phi}) \rangle = \hat{U}^{(D)}_\mathrm{ent}(\vec{\theta}_D,
        \vec{\phi}_D) \dots \hat{U}^{(2)}_\mathrm{ent}(\vec{\theta}_2, \vec{\phi}_2)
        \hat{U}^{(1)}_\mathrm{ent}(\vec{\theta}_1, \vec{\phi}_1) \vert \mathrm{HF}\rangle,

    where :math:`\hat{U}^{(i)}_\mathrm{ent}(\vec{\theta}_i, \vec{\phi}_i) =
    \hat{R}_\mathrm{z}(\vec{\theta}_i) \hat{U}_\mathrm{2,\mathrm{ex}}(\vec{\phi}_i)`.
    The circuit implementing the entangler blocks is shown in the figure below:

    |

    .. figure:: ../../_static/templates/layers/particle_conserving_u2.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    Each layer contains :math:`N` rotation gates :math:`R_\mathrm{z}(\vec{\theta})` and
    :math:`N-1` particle-conserving exchange gates :math:`U_{2,\mathrm{ex}}(\phi)`
    that act on pairs of nearest-neighbors qubits. The repeated units across several qubits are
    shown in dotted boxes.  The unitary matrix representing :math:`U_{2,\mathrm{ex}}(\phi)`
    (`arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_) is decomposed into its elementary
    gates and implemented in the :func:`~.u2_ex_gate` function using PennyLane quantum operations.

    |

    .. figure:: ../../_static/templates/layers/u2_decomposition.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |


    Args:
        weights (array[float]): Array of weights of shape ``(D, M)`` where ``D`` is the number of
            layers and ``M`` = ``2N-1`` is the total number of rotation ``(N)`` and exchange
            ``(N-1)`` gates per layer.
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers
            or strings, or a Wires object.
        init_state (array[int]): length ``len(wires)`` vector representing the Hartree-Fock state
            used to initialize the wires.

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::


        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The number of trainable parameters scales with the number of layers :math:`D` as
           :math:`D(2N-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import ParticleConservingU2

            from functools import partial

            # Build the electronic Hamiltonian from a local .xyz file
            h, qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")

            # Define the HF state
            ref_state = qml.qchem.hf_state(2, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            ansatz = partial(ParticleConservingU2, init_state=ref_state)

            # Define the cost function
            cost_fn = qml.ExpvalCost(ansatz, h, dev)

            # Compute the expectation value of 'h' for a given set of parameters
            layers = 1
            params = qml.init.particle_conserving_u2_normal(layers, qubits)
            print(cost_fn(params))
    """

    wires = Wires(wires)

    layers = weights.shape[0]

    if len(wires) < 2:
        raise ValueError(
            "This template requires the number of qubits to be greater than one;"
            "got a wire sequence with {} elements".format(len(wires))
        )

    expected_shape = (layers, 2 * len(wires) - 1)
    check_shape(
        weights,
        expected_shape,
        msg="'weights' must be of shape {}; got {}".format(expected_shape, get_shape(weights)),
    )

    nm_wires = [wires.subset([l, l + 1]) for l in range(0, len(wires) - 1, 2)]
    nm_wires += [wires.subset([l, l + 1]) for l in range(1, len(wires) - 1, 2)]

    qml.BasisState(init_state, wires=wires)

    for l in range(layers):

        for j, _ in enumerate(wires):
            RZ(weights[l, j], wires=wires[j])

        for i, wires_ in enumerate(nm_wires):
            u2_ex_gate(weights[l, len(wires) + i], wires=wires_)
