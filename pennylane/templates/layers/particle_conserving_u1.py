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
Contains the hardware efficient ``ParticleConservingU1`` template.
"""
import numpy as np

import pennylane as qml

# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, CRot, PhaseShift, CZ
from pennylane.wires import Wires


def _preprocess(weights, wires, init_state):
    """Validate and pre-process inputs as follows:

    * Check that the weights tensor has the correct shape.
    * Extract a wire list for the subroutines of this template.
    * Cast initial state to a numpy array.

    Args:
        weights (tensor_like): trainable parameters of the template
        wires (Wires): wires that template acts on
        init_state (tensor_like): shape ``(len(wires),)`` tensor

    Returns:
        int, list[Wires], array: number of times that the ansatz is repeated, wires pattern,
            and preprocessed initial state
    """
    if len(wires) < 2:
        raise ValueError(
            "This template requires the number of qubits to be greater than one;"
            "got a wire sequence with {} elements".format(len(wires))
        )

    shape = qml.math.shape(weights)

    if len(shape) != 3:
        raise ValueError(f"Weights tensor must be 3-dimensional; got shape {shape}")

    if shape[1] != len(wires) - 1:
        raise ValueError(
            f"Weights tensor must have second dimension of length {len(wires) - 1}; got {shape[1]}"
        )

    if shape[2] != 2:
        raise ValueError(f"Weights tensor must have third dimension of length 2; got {shape[2]}")

    repeat = shape[0]

    nm_wires = [wires.subset([l, l + 1]) for l in range(0, len(wires) - 1, 2)]
    nm_wires += [wires.subset([l, l + 1]) for l in range(1, len(wires) - 1, 2)]
    # we can extract the numpy representation here
    # since init_state can never be differentiable
    init_state = qml.math.toarray(init_state)
    return repeat, nm_wires, init_state


def decompose_ua(phi, wires=None):
    r"""Implements the circuit decomposing the controlled application of the unitary
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
        wires (list[Wires]): the wires ``n`` and ``m`` the circuit acts on
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


def u1_ex_gate(phi, theta, wires=None):
    r"""Implements the two-qubit exchange gate :math:`U_{1,\mathrm{ex}}` proposed
    in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ to build
    a hardware-efficient particle-conserving VQE ansatz for quantum chemistry
    simulations.

    Args:
        phi (float): angle entering the unitary :math:`U_A(\phi)`
        theta (float): angle entering the rotation :math:`R(0, 2\theta, 0)`
        wires (list[Wires]): the two wires ``n`` and ``m`` the circuit acts on
    """

    # C-UA(phi)
    decompose_ua(phi, wires=wires)

    qml.CZ(wires=wires[::-1])
    qml.CRot(0, 2 * theta, 0, wires=wires[::-1])

    # C-UA(-phi)
    decompose_ua(-phi, wires=wires)


@template
def ParticleConservingU1(weights, wires, init_state=None):
    r"""Implements the heuristic VQE ansatz for quantum chemistry simulations using the
    particle-conserving gate :math:`U_{1,\mathrm{ex}}` proposed by Barkoutsos *et al.* in
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_.

    This template prepares :math:`N`-qubit trial states by applying :math:`D` layers of the
    entangler block :math:`U_\mathrm{ent}(\vec{\phi}, \vec{\theta})` to the Hartree-Fock
    state

    .. math::

        \vert \Psi(\vec{\phi}, \vec{\theta}) \rangle = \hat{U}^{(D)}_\mathrm{ent}(\vec{\phi}_D,
        \vec{\theta}_D) \dots \hat{U}^{(2)}_\mathrm{ent}(\vec{\phi}_2, \vec{\theta}_2)
        \hat{U}^{(1)}_\mathrm{ent}(\vec{\phi}_1, \vec{\theta}_1) \vert \mathrm{HF}\rangle.

    The circuit implementing the entangler blocks is shown in the figure below:

    |

    .. figure:: ../../_static/templates/layers/particle_conserving_u1.png
        :align: center
        :width: 50%
        :target: javascript:void(0);

    |

    The repeated units across several qubits are shown in dotted boxes. Each layer
    contains :math:`N-1` particle-conserving two-parameter exchange gates
    :math:`U_{1,\mathrm{ex}}(\phi, \theta)` that act on pairs of nearest neighbors qubits.
    The unitary matrix representing :math:`U_{1,\mathrm{ex}}(\phi, \theta)`
    is given by (see `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_),

    .. math::

        U_{1, \mathrm{ex}}(\phi, \theta) = \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & \mathrm{cos}(\theta) & e^{i\phi} \mathrm{sin}(\theta) & 0 \\
        0 & e^{-i\phi} \mathrm{sin}(\theta) & -\mathrm{cos}(\theta) & 0 \\
        0 & 0 & 0 & 1 \\
        \end{array}\right).

    The figure below shows the circuit decomposing :math:`U_{1, \mathrm{ex}}` in
    elementary gates. The Pauli matrix :math:`\sigma_z` and single-qubit rotation
    :math:`R(0, 2 \theta, 0)` apply the Pauli Z operator and an arbitrary rotation
    on the qubit ``n`` with qubit ``m`` bein the control qubit,

    |

    .. figure:: ../../_static/templates/layers/u1_decomposition.png
        :align: center
        :width: 80%
        :target: javascript:void(0);

    |

    :math:`U_A(\phi)` is the unitary matrix

    .. math::

        U_A(\phi) = \left(\begin{array}{cc} 0 & e^{-i\phi} \\ e^{-i\phi} & 0 \\ \end{array}\right),

    which is applied controlled on the state of qubit ``m`` and can be further decomposed in
    terms of the
    `quantum operations <https://pennylane.readthedocs.io/en/stable/introduction/operations.html>`_
    supported by Pennylane,

    |

    .. figure:: ../../_static/templates/layers/ua_decomposition.png
        :align: center
        :width: 70%
        :target: javascript:void(0);

    |

    where,

    |

    .. figure:: ../../_static/templates/layers/phaseshift_decomposition.png
        :align: center
        :width: 65%
        :target: javascript:void(0);

    |

    The quantum circuits above decomposing the unitaries :math:`U_{1,\mathrm{ex}}(\phi, \theta)`
    and :math:`U_A(\phi)` are implemented by the ``u1_ex_gate`` and ``decompose_ua``
    functions, respectively. :math:`R_\phi` refers to the ``PhaseShift`` gate in the
    circuit diagram.

    Args:
        weights (array[float]): Array of weights of shape ``(D, M, 2)``.
            ``D`` is the number of entangler block layers and :math:`M=N-1`
            is the number of exchange gates :math:`U_{1,\mathrm{ex}}` per layer.
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers
            or strings, or a Wires object.
        init_state (array[int]): length ``len(wires)`` vector representing the Hartree-Fock state
            used to initialize the wires

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        #. The number of wires :math:`N` has to be equal to the number of
           spin orbitals included in the active space.

        #. The number of trainable parameters scales linearly with the number of layers as
           :math:`2D(N-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import ParticleConservingU1
            from functools import partial

            # Build the electronic Hamiltonian from a local .xyz file
            h, qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")

            # Define the Hartree-Fock state
            electrons = 2
            ref_state = qml.qchem.hf_state(electrons, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            ansatz = partial(ParticleConservingU1, init_state=ref_state)

            # Define the cost function
            cost_fn = qml.ExpvalCost(ansatz, h, dev)

            # Compute the expectation value of 'h'
            layers = 2
            params = qml.init.particle_conserving_u1_normal(layers, qubits)
            print(cost_fn(params))
    """

    wires = Wires(wires)
    repeat, nm_wires, init_state = _preprocess(weights, wires, init_state)

    qml.BasisState(init_state, wires=wires)

    for l in range(repeat):
        for i, wires_ in enumerate(nm_wires):
            u1_ex_gate(weights[l][i][0], weights[l][i][1], wires=wires_)
