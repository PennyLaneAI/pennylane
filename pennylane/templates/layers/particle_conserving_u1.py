# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the hardware-efficient ParticleConservingU1 template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


def decompose_ua(phi, wires=None):
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
    op_list = []
    n, m = wires

    op_list.append(qml.CZ(wires=wires))
    op_list.append(qml.CRot(-phi, np.pi, phi, wires=wires))

    # decomposition of C-PhaseShift(2*phi) gate
    op_list.append(qml.PhaseShift(-phi, wires=m))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.PhaseShift(phi, wires=m))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.PhaseShift(-phi, wires=n))

    return op_list


def u1_ex_gate(phi, theta, wires=None):
    r"""Appends the two-qubit exchange gate :math:`U_{1,\mathrm{ex}}` proposed
    in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ to build
    a hardware-efficient particle-conserving VQE ansatz for quantum chemistry
    simulations.

    Args:
        phi (float): angle entering the unitary :math:`U_A(\phi)`
        theta (float): angle entering the rotation :math:`R(0, 2\theta, 0)`
        wires (list[Iterable]): the two wires ``n`` and ``m`` the circuit acts on

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    op_list = []

    # C-UA(phi)
    op_list.extend(decompose_ua(phi, wires=wires))

    op_list.append(qml.CZ(wires=wires[::-1]))
    op_list.append(qml.CRot(0, 2 * theta, 0, wires=wires[::-1]))

    # C-UA(-phi)
    op_list.extend(decompose_ua(-phi, wires=wires))

    return op_list


class ParticleConservingU1(Operation):
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
        weights (tensor_like): Array of weights of shape ``(D, M, 2)``.
            ``D`` is the number of entangler block layers and :math:`M=N-1`
            is the number of exchange gates :math:`U_{1,\mathrm{ex}}` per layer.
        wires (Iterable): wires that the template acts on.
        init_state (tensor_like): iterable or shape ``(len(wires),)`` tensor representing the Hartree-Fock state
            used to initialize the wires. If ``None``, a tuple of zeros is selected as initial state.

    .. details::
        :title: Usage Details

        #. The number of wires :math:`N` has to be equal to the number of
           spin orbitals included in the active space.

        #. The number of trainable parameters scales linearly with the number of layers as
           :math:`2D(N-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            import numpy as np
            from functools import partial

            # Build the electronic Hamiltonian
            symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
            h, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

            # Define the Hartree-Fock state
            electrons = 2
            ref_state = qml.qchem.hf_state(electrons, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            ansatz = partial(qml.ParticleConservingU1, init_state=ref_state, wires=dev.wires)

            # Define the cost function
            @qml.qnode(dev)
            def cost_fn(params):
                ansatz(params)
                return qml.expval(h)

            # Compute the expectation value of 'h'
            layers = 2
            shape = qml.ParticleConservingU1.shape(layers, qubits)
            params = np.random.random(shape)
            print(cost_fn(params))

        **Parameter shape**

        The shape of the trainable weights tensor can be computed by the static method
        :meth:`~.ParticleConservingU1.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = qml.ParticleConservingU1.shape(n_layers=2, n_wires=2)
            params = np.random.random(size=shape)
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, init_state=None, id=None):
        if len(wires) < 2:
            raise ValueError(
                f"Expected the number of qubits to be greater than one; " f"got wires {wires}"
            )

        shape = qml.math.shape(weights)

        if len(shape) != 3:
            raise ValueError(f"Weights tensor must be 3-dimensional; got shape {shape}")

        if shape[1] != len(wires) - 1:
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(wires) - 1}; got {shape[1]}"
            )

        if shape[2] != 2:
            raise ValueError(
                f"Weights tensor must have third dimension of length 2; got {shape[2]}"
            )

        init_state = tuple(0 for _ in wires) if init_state is None else init_state

        self._hyperparameters = {"init_state": tuple(init_state)}

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires, init_state):  # pylint: disable=arguments-differ
        r"""Representation of the ParticleConservingU1operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.ParticleConservingU1.decomposition`.

        Args:
            weights (tensor_like): Array of weights of shape ``(D, M, 2)``.
                ``D`` is the number of entangler block layers and :math:`M=N-1`
                is the number of exchange gates :math:`U_{1,\mathrm{ex}}` per layer.
            wires (Any or Iterable[Any]): wires that the operator acts on
            init_state (tensor_like): iterable or shape ``(len(wires),)`` tensor representing the Hartree-Fock state
                used to initialize the wires

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> weights = torch.tensor([[[0.3, 1.]]])
        >>> qml.ParticleConservingU1.compute_decomposition(weights, wires=["a", "b"], init_state=[0, 1])
        [BasisEmbedding(wires=['a', 'b']),
         CZ(wires=['a', 'b']),
         CRot(tensor(-0.3000), 3.141592653589793, tensor(0.3000), wires=['a', 'b']),
         PhaseShift(tensor(-0.3000), wires=['b']), CNOT(wires=['a', 'b']),
         PhaseShift(tensor(0.3000), wires=['b']), CNOT(wires=['a', 'b']),
         PhaseShift(tensor(-0.3000), wires=['a']), CZ(wires=['b', 'a']),
         CRot(0, tensor(2.), 0, wires=['b', 'a']), CZ(wires=['a', 'b']),
         CRot(tensor(0.3000), 3.141592653589793, tensor(-0.3000), wires=['a', 'b']),
         PhaseShift(tensor(0.3000), wires=['b']),
         CNOT(wires=['a', 'b']),
         PhaseShift(tensor(-0.3000), wires=['b']),
         CNOT(wires=['a', 'b']),
         PhaseShift(tensor(0.3000), wires=['a'])]
        """

        nm_wires = [wires[l : l + 2] for l in range(0, len(wires) - 1, 2)]
        nm_wires += [wires[l : l + 2] for l in range(1, len(wires) - 1, 2)]
        n_layers = qml.math.shape(weights)[0]
        op_list = [qml.BasisEmbedding(init_state, wires=wires)]

        for l in range(n_layers):
            for i, wires_ in enumerate(nm_wires):
                op_list.extend(u1_ex_gate(weights[l, i, 0], weights[l, i, 1], wires=wires_))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns the shape of the weight tensor required for this template.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of qubits

        Returns:
            tuple[int]: shape
        """

        if n_wires < 2:
            raise ValueError(
                f"The number of qubits must be greater than one; got 'n_wires' = {n_wires}"
            )
        return n_layers, n_wires - 1, 2
