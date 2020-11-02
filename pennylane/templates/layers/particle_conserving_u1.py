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
from pennylane.templates.utils import (
    check_shape,
    check_type,
    get_shape,
)
from pennylane.wires import Wires


def decompose_ua(phi, wires=None):
    r"""Implement the two-qubit circuit decomposing the controlled application of the unitary
    :math:`U_A(\phi)`

    .. math::

        U_A(\phi) = \left(\begin{array}{cc} 0 & e^{-i\phi} \\ e^{-i\phi} & 0 \\ \end{array}\right)

    in terms of the quantum operations supported by PennyLane.

    # :math:`U_A(\phi)` is used in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_),
    # to define two-qubit exchange gates that are used to build particle-conserving
    # VQE ansatze for Quantum Chemistry simulations. See :func:`~.ParticleConservingU1`.

    # This unitary can be expressed in terms of ``PhaseShift``, ``Rot`` and ``PauliZ`` operations
    # which are supported by PennyLane
    # :math:`U_A(\phi) = R_\phi(-2\phi) R(-\phi, \pi, \phi) \sigma_z`. The figures below shows
    # the decomposition of controlled-:math:`U_A` in terms of these operations:

    # |

    # .. figure:: ../../_static/templates/layers/ua_decomposition.png
    #     :align: center
    #     :width: 60%
    #     :target: javascript:void(0);


    # .. figure:: ../../_static/templates/layers/phaseshift_decomposition.png
    #     :align: center
    #     :width: 60%
    #     :target: javascript:void(0);

    # |

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
    r"""Implement the two-qubit exchange gate :math:`U_{1,\mathrm{ex}}` proposed
    in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ to build
    particle-conserving VQE ansatze for Quantum Chemistry simulations.

    # The unitary matrix :math:`U_{1, \mathrm{ex}}` acts on the Hilbert space of two qubits

    # .. math::

    #     U_{1, \mathrm{ex}}(\phi, \theta) = \left(\begin{array}{cccc}
    #     1 & 0 & 0 & 0 \\
    #     0 & \mathrm{cos}(\theta) & e^{i\phi} \mathrm{sin}(\theta) & 0 \\
    #     0 & e^{-i\phi} \mathrm{sin}(\theta) & -\mathrm{cos}(\theta) & 0 \\
    #     0 & 0 & 0 & 1 \\
    #     \end{array}\right).

    # The figure below shows the circuit use to decompose of :math:`U_{1, \mathrm{ex}}` in
    # elementary gates

    # |

    # .. figure:: ../../_static/templates/layers/u1_decomposition.png
    #     :align: center
    #     :width: 60%
    #     :target: javascript:void(0);

    # |

    # The unitary :math:`U_A(\phi)`

    # .. math::

    #     U_A(\phi) = \left(\begin{array}{cc} 0 & e^{-i\phi} \\ e^{-i\phi} & 0 \\ \end{array}\right).

    # is further decomposed in terms of the quantum operations supported by PennyLane as shown
    # in the circuits below

    # |

    # .. figure:: ../../_static/templates/layers/ua_decomposition.png
    #     :align: center
    #     :width: 60%
    #     :target: javascript:void(0);


    # .. figure:: ../../_static/templates/layers/phaseshift_decomposition.png
    #     :align: center
    #     :width: 60%
    #     :target: javascript:void(0);

    # |

    # The quantum circuits decomposing the unitaries :math:`U_A(\phi)` and :math:`U_{1, \mathrm{ex}}`
    # are implemented by the :func:`~.u1_ex_gate` and :func:`~.decompose_ua` functions.



    # See the :func:`~.decompose_ua` function for more details.

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
    r"""Implements the heuristic VQE ansatz for Quantum Chemistry simulations using the
    particle-conserving gate :math:`U_{1,\mathrm{ex}}` proposed in
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ .

    This template prepares N-qubit trial states by applying :math:`D` layers of the
    entangler block :math:`U_\mathrm{ent}(\vec{\phi}, \vec{\theta})` to the Hartree-Fock
    state

    .. math::

        \vert \Psi(\vec{\phi}, \vec{\theta}) = \hat{U}^{(D)}_\mathrm{ent}(\vec{\phi}_D,
        \vec{\theta}_D) \dots \hat{U}^{(2)}_\mathrm{ent}(\vec{\phi}_2, \vec{\theta}_2)
        \hat{U}^{(1)}_\mathrm{ent}(\vec{\phi}_1, \vec{\theta}_1) \vert \mathrm{HF}\rangle.

    The circuit implementing the entangler blocks is shown in the figure below. The repeated
    units across several qubits are shown in dotted boxes. Each layer
    contains :math:`N_\mathrm{qubits}-1` particle-conserving two-parameter exchange gates
    :math:`U_{1,\mathrm{ex}}(\phi, \theta)` that act on pairs of nearest neighbors qubits.
    
    The unitary matrix representing :math:`U_{1,\mathrm{ex}}(\phi, \theta)`
    (`arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_) acts on the Hilbert space of
    two qubits

    .. math::

        U_{1, \mathrm{ex}}(\phi, \theta) = \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & \mathrm{cos}(\theta) & e^{i\phi} \mathrm{sin}(\theta) & 0 \\
        0 & e^{-i\phi} \mathrm{sin}(\theta) & -\mathrm{cos}(\theta) & 0 \\
        0 & 0 & 0 & 1 \\
        \end{array}\right).

    The figure below shows the circuit decomposing :math:`U_{1, \mathrm{ex}}` in
    elementary gates. :math:`\sigma_z` and :math:`R(0, 2 \theta, 0)` apply the Pauli Z operator
    and an arbitrary rotation on the qubit ``n``.

    |

    .. figure:: ../../_static/templates/layers/u1_decomposition.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    On the other hand, :math:`U_A(\phi)` is the unitary matrix

    .. math::

        U_A(\phi) = \left(\begin{array}{cc} 0 & e^{-i\phi} \\ e^{-i\phi} & 0 \\ \end{array}\right)

    acting on the state of qubit ``n`` which is further decomposed in terms of the
    quantum operations supported by PennyLane as shown in the circuits below.

    |

    .. figure:: ../../_static/templates/layers/ua_decomposition.png
        :align: center
        :width: 60%
        :target: javascript:void(0);


    .. figure:: ../../_static/templates/layers/phaseshift_decomposition.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    The quantum circuits decomposing the unitaries :math:`U_{1,\mathrm{ex}}(\phi, \theta)` and
    :math:`U_A(\phi)` are implemented by the :func:`~.u1_ex_gate` and :func:`~.decompose_ua`
    functions, respectively.

    Args:
        weights (array[float]): Array of weights of shape ``(D, M, 2)``.
            ``D`` is the number of entangler block layers and ``M`` = :math:`N_\mathrm{qubits}-1`
            is the number of exchange gates :math:`U_{1,\mathrm{ex}}` per layer.
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
           :math:`2D(N_\mathrm{qubits}-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import ParticleConservingU1
            from functools import partial

            # Build the electronic Hamiltonian
            h, qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")

            # Define the HF state
            ref_state = qml.qchem.hf_state(electrons=2, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            ansatz = partial(ParticleConservingU1, init_state=ref_state)

            # Define the cost function
            cost_fn = qml.VQECost(ansatz, h, dev)

            # Compute the expectation value of 'h'
            layers = 2
            params = qml.init.particle_conserving_u1_normal(layers, qubits)
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

    expected_shape = (layers, len(wires) - 1, 2)
    check_shape(
        weights,
        expected_shape,
        msg="'weights' must be of shape {}; got {}".format(expected_shape, get_shape(weights)),
    )

    check_type(
        init_state,
        [np.ndarray],
        msg="'init_state' must be a Numpy array; got {}".format(init_state),
    )
    for i in init_state:
        check_type(
            i,
            [int, np.int64, np.ndarray],
            msg="Elements of 'init_state' must be integers; got {}".format(init_state),
        )

    nm_wires = [wires.subset([l, l + 1]) for l in range(0, len(wires) - 1, 2)]
    nm_wires += [wires.subset([l, l + 1]) for l in range(1, len(wires) - 1, 2)]

    qml.BasisState(init_state, wires=wires)

    for l in range(layers):
        for i, wires_ in enumerate(nm_wires):
            u1_ex_gate(*weights[l, i], wires=wires_)
