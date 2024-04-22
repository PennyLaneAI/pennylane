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

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    return [qml.CNOT(wires=wires), qml.CRX(2 * phi, wires=wires[::-1]), qml.CNOT(wires=wires)]


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
        wires (Iterable): wires that the template acts on.
        init_state (tensor_like): iterable or shape ``(len(wires),)`` tensor representing the Hartree-Fock state
            used to initialize the wires. If ``None``, a tuple of zeros is selected as initial state.

    .. details::
        :title: Usage Details

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The number of trainable parameters scales with the number of layers :math:`D` as
           :math:`D(2N-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            import numpy as np
            from functools import partial

            # Build the electronic Hamiltonian
            symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
            h, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

            # Define the HF state
            ref_state = qml.qchem.hf_state(2, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            ansatz = partial(qml.ParticleConservingU2, init_state=ref_state, wires=dev.wires)

            # Define the cost function
            @qml.qnode(dev)
            def cost_fn(params):
                ansatz(params)
                return qml.expval(h)

            # Compute the expectation value of 'h' for a given set of parameters
            layers = 1
            shape = qml.ParticleConservingU2.shape(layers, qubits)
            params = np.random.random(shape)
            print(cost_fn(params))

        **Parameter shape**

        The shape of the trainable weights tensor can be computed by the static method
        :meth:`~qml.ParticleConservingU2.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = qml.ParticleConservingU2.shape(n_layers=2, n_wires=2)
            params = np.random.random(size=shape)
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, init_state=None, id=None):
        if len(wires) < 2:
            raise ValueError(
                f"This template requires the number of qubits to be greater than one;"
                f"got a wire sequence with {len(wires)} elements"
            )

        shape = qml.math.shape(weights)

        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")

        if shape[1] != 2 * len(wires) - 1:
            raise ValueError(
                f"Weights tensor must have a second dimension of length {2 * len(wires) - 1}; got {shape[1]}"
            )

        init_state = tuple(0 for _ in wires) if init_state is None else init_state

        self._hyperparameters = {"init_state": tuple(init_state)}

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires, init_state):  # pylint: disable=arguments-differ
        r"""Representation of the ParticleConservingU2operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.ParticleConservingU2.decomposition`.

        Args:
            weights (tensor_like): Weight tensor of shape ``(D, M)`` where ``D`` is the number of
                layers and ``M`` = ``2N-1`` is the total number of rotation ``(N)`` and exchange
                ``(N-1)`` gates per layer.
            wires (Any or Iterable[Any]): wires that the operator acts on
            init_state (tensor_like): iterable or shape ``(len(wires),)`` tensor representing the Hartree-Fock state
                used to initialize the wires.

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> torch.tensor([[0.3, 1., 0.2]])
        >>> qml.ParticleConservingU2.compute_decomposition(weights, wires=["a", "b"], init_state=[0, 1])
        [BasisEmbedding(wires=['a', 'b']),
         RZ(tensor(0.3000), wires=['a']),
         RZ(tensor(1.), wires=['b']),
         CNOT(wires=['a', 'b']),
         CRX(tensor(0.4000), wires=['b', 'a']),
         CNOT(wires=['a', 'b'])]
        """
        nm_wires = [wires[l : l + 2] for l in range(0, len(wires) - 1, 2)]
        nm_wires += [wires[l : l + 2] for l in range(1, len(wires) - 1, 2)]
        n_layers = qml.math.shape(weights)[0]
        op_list = [qml.BasisEmbedding(init_state, wires=wires)]

        for l in range(n_layers):
            for j, wires_ in enumerate(wires):
                op_list.append(qml.RZ(weights[l, j], wires=wires_))

            for i, wires_ in enumerate(nm_wires):
                op_list.extend(u2_ex_gate(weights[l, len(wires) + i], wires=wires_))

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
        return n_layers, 2 * n_wires - 1
