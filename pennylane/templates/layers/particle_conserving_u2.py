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
Contains the hardware-efficient ParticleConservingU2 template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.operation import Operation, AnyWires


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

    qml.CNOT(wires=wires)
    qml.CRX(2 * phi, wires=wires[::-1])
    qml.CNOT(wires=wires)


class ParticleConservingU2(Operation):
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
        weights (tensor_like): Weight tensor of shape ``(D, M)`` where ``D`` is the number of
            layers and ``M`` = ``2N-1`` is the total number of rotation ``(N)`` and exchange
            ``(N-1)`` gates per layer.
        wires (Iterable): wires that the template acts on
        init_state (tensor_like): iterable or shape ``(len(wires),)`` tensor representing the Hartree-Fock state
            used to initialize the wires.

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

        **Parameter shape**

        The shape of the weights argument can be computed by the static method
        :meth:`~.ParticleConservingU2.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = ParticleConservingU2.shape(n_layers=2, n_wires=2)
            weights = np.random.random(size=shape)
    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, weights, wires, init_state=None, do_queue=True, id=None):

        if len(wires) < 2:
            raise ValueError(
                "This template requires the number of qubits to be greater than one;"
                "got a wire sequence with {} elements".format(len(wires))
            )

        shape = qml.math.shape(weights)

        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")

        if shape[1] != 2 * len(wires) - 1:
            raise ValueError(
                f"Weights tensor must have a second dimension of length {2 * len(wires) - 1}; got {shape[1]}"
            )

        self.n_layers = shape[0]
        # we can extract the numpy representation here
        # since init_state can never be differentiable
        self.init_state = qml.math.toarray(init_state)

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        nm_wires = [self.wires[l : l + 2] for l in range(0, len(self.wires) - 1, 2)]
        nm_wires += [self.wires[l : l + 2] for l in range(1, len(self.wires) - 1, 2)]

        with qml.tape.QuantumTape() as tape:

            qml.templates.BasisEmbedding(self.init_state, wires=self.wires)

            for l in range(self.n_layers):

                for j, _ in enumerate(self.wires):
                    qml.RZ(self.parameters[0][l, j], wires=self.wires[j])

                for i, wires_ in enumerate(nm_wires):
                    u2_ex_gate(self.parameters[0][l, len(self.wires) + i], wires=wires_)
        return tape

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
                "The number of qubits must be greater than one; got 'n_wires' = {}".format(n_wires)
            )
        return n_layers, 2 * n_wires - 1
