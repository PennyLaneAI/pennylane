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
    r"""Implement the two-qubit exchange gate :math:`U_{2,\mathrm{ex}}` proposed
    in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ to build
    particle-conserving VQE ansatze for Quantum Chemistry simulations.

    The unitary matrix :math:`U_{2, \mathrm{ex}}` acts on the Hilbert space of two qubits

    .. math::

        U_{2, \mathrm{ex}}(\phi) = \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & \mathrm{cos}(\phi) & -i\;\mathrm{sin}(\phi) & 0 \\
        0 & -i\;\mathrm{sin}(\phi) & \mathrm{cos}(\phi) & 0 \\
        0 & 0 & 0 & 1 \\
        \end{array}\right).

    The figure below shows the circuit used to decompose :math:`U_{2, \mathrm{ex}}` in
    elementary gates

    |

    .. figure:: ../../_static/templates/layers/u2_decomposition.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

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
    rotation gate :math:`R_\mathrm{z}(\vec{\theta})` and particle-conserving gate
    :math:`U_{2,\mathrm{ex}}` proposed in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ .

    This template prepares :math:`N`-qubit trial states by applying :math:`D` layers of
    :math:`R_\mathrm{z}(\vec{\theta})` and entangler block :math:`U_{2,\mathrm{ex}}(\vec{\phi})`
    to the Hartree-Fock state

    .. math::

        \vert \Psi(\vec{\phi}, \vec{\theta}) \rangle = \hat{R}^{(D)}_\mathrm{z}(\vec{\theta}_D)
        \hat{U}^{(D)}_\mathrm{2,\mathrm{ex}}(\vec{\phi}_D) \dots \hat{R}^{(2)}_\mathrm{z}(\vec{\theta}_2)
        \hat{U}^{(2)}_\mathrm{2,\mathrm{ex}}(\vec{\phi}_2) \hat{R}^{(1)}_\mathrm{z}(\vec{\theta}_1)
        \hat{U}^{(1)}_\mathrm{2,\mathrm{ex}}(\vec{\phi}_1) \vert \mathrm{HF}\rangle.

    The circuit implementing the operation blocks is shown in the figure below:

    |

    .. figure:: ../../_static/templates/layers/particle_conserving_u2.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    The repeated units across several qubits are shown in dotted boxes. Each layer contains
    :math:`N` rotation gates :math:`R_\mathrm{z}(\vec{\theta})` and
    :math:`N-1` particle-conserving exchange gates :math:`U_{2,\mathrm{ex}}(\phi)`
    that act on pairs of nearest-neighbors qubits. The unitary matrix representing
    :math:`U_{2,\mathrm{ex}}(\phi)` (`arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)
    is decomposed into its elementary gates as:

    |

    .. figure:: ../../_static/templates/layers/u2_decomposition.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    The :math:`U_{2,\mathrm{ex}}(\phi)` gate is implemented in the :func:`~.u2_ex_gate` function
    using PennyLane quantum operations.

    Args:
        weights (array[float]): Array of weights of shape ``(D, M)`` where ``D`` is the number of
            layers and ``M`` = :math:`2N-1` is the number of rotation and exchange gates per layer.
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers
            or strings, or a Wires object.
        init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
            HF state. ``init_state`` is used to initialize the wires.

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        Notice that:

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The number of trainable parameters scales with the number of layers :math:`D` as
           :math:`D(2N-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane import qchem
            from pennylane.templates import ParticleConservingU2

            from functools import partial

            # Build the electronic Hamiltonian
            h, qubits = qchem.molecular_hamiltonian("h2", "h2.xyz")

            # Define the HF state
            ref_state = qchem.hf_state(electrons=2, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            ansatz = partial(ParticleConservingU2, init_state=ref_state)

            # Define the cost function
            cost_fn = qml.VQECost(ansatz, h, dev)

            # Compute the expectation value of 'h' for given set of parameters
            layers = 1
            params = np.random.normal(0, 2*np.pi, (layers, 2*qubits-1))
            print(cost_fn(params))

    """

    wires = Wires(wires)

    layers = weights.shape[0]

    if len(wires) < 2:
        raise ValueError(
            "This template requires the number of qubits to be >= 2; got 'len(wires)' = {}".format(
                len(wires)
            )
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
