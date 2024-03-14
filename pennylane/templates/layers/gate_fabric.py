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
Contains the quantum-number-preserving GateFabric template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class GateFabric(Operation):
    r"""Implements a local, expressive, and quantum-number-preserving ansatz proposed by
    `Anselmetti et al. (2021) <https://doi.org/10.1088/1367-2630/ac2cb3>`_.

    This template prepares the :math:`N`-qubit trial state by applying :math:`D` layers of gate-fabric blocks
    :math:`\hat{U}_{GF}(\vec{\theta},\vec{\phi})` to the Hartree-Fock state in the Jordan-Wigner basis

    .. math::

        \vert \Psi(\vec{\theta},\vec{\phi})\rangle =
        \hat{U}_{GF}^{(D)}(\vec{\theta}_{D},\vec{\phi}_{D}) \ldots
        \hat{U}_{GF}^{(2)}(\vec{\theta}_{2},\vec{\phi}_{2})
        \hat{U}_{GF}^{(1)}(\vec{\theta}_{1},\vec{\phi}_{1}) \vert HF \rangle,

    where each of the gate fabric blocks :math:`\hat{U}_{GF}(\vec{\theta},\vec{\phi})` is comprised of two-parameter four-qubit
    gates :math:`\hat{Q}(\theta, \phi)` that act on four nearest-neighbour qubits. The circuit implementing a
    single layer of the gate fabric block for :math:`N = 8` is shown in the figure below:

    .. figure:: ../../_static/templates/layers/gate_fabric_layer.png
        :align: center
        :width: 100%
        :target: javascript:void(0);

    The gate element :math:`\hat{Q}(\theta, \phi)` (`Anselmetti et al. (2021) <https://doi.org/10.1088/1367-2630/ac2cb3>`_)
    is composed of a four-qubit spin-adapted spatial orbital rotation gate, which is implemented by the :class:`~.OrbitalRotation()`
    operation and a four-qubit diagonal pair-exchange gate, which is equivalent to the :class:`~.DoubleExcitation()`
    operation. In addition to these two gates, the gate element :math:`\hat{Q}(\theta, \phi)` can also include an optional
    constant :math:`\hat{\Pi} \in \{\hat{I}, \text{OrbitalRotation}(\pi)\}` gate.

    .. figure:: ../../_static/templates/layers/q_gate_decompositon.png
        :align: center
        :width: 75%
        :target: javascript:void(0);

    |

    The four-qubit :class:`~.DoubleExcitation()` and :class:`~.OrbitalRotation()` gates given here are equivalent to the
    :math:`\text{QNP}_{PX}(\theta)` and :math:`\text{QNP}_{OR}(\phi)` gates presented in
    `Anselmetti et al. (2021) <https://doi.org/10.1088/1367-2630/ac2cb3>`_,
    respectively. Moreover, regardless of the choice of :math:`\hat{\Pi}`, this gate fabric will exactly preserve the number of particles
    and total spin of the state.

    Args:
        weights (tensor_like): Array of weights of shape ``(D, L, 2)``\,
            where ``D`` is the number of gate fabric layers and ``L = N/2-1``
            is the number of :math:`\hat{Q}(\theta, \phi)` gates per layer with N being the total number of qubits.
        wires (Iterable): wires that the template acts on
        init_state (tensor_like): init_state (tensor_like): iterable of shape ``(len(wires),)``\, representing the input Hartree-Fock state
            in the Jordan-Wigner representation.
        include_pi (boolean): If True, the optional constant :math:`\hat{\Pi}` gate  is set to :math:`\text{OrbitalRotation}(\pi)`.
            Default value is :math:`\hat{I}`.

    .. details::
        :title: Usage Details

        #. The number of wires :math:`N` has to be equal to the number of
           spin-orbitals included in the active space, and should be even.

        #. The number of trainable parameters scales linearly with the number of layers as
           :math:`2 D (N/2-1)`.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane import numpy as np

            # Build the electronic Hamiltonian
            symbols = ["H", "H"]
            coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
            H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

            # Define the Hartree-Fock state
            electrons = 2
            ref_state = qml.qchem.hf_state(electrons, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the ansatz
            @qml.qnode(dev)
            def ansatz(weights):
                qml.GateFabric(weights, wires=[0,1,2,3],
                            init_state=ref_state, include_pi=True)
                return qml.expval(H)

            # Get the shape of the weights for this template
            layers = 2
            shape = qml.GateFabric.shape(n_layers=layers, n_wires=qubits)

            # Initialize the weight tensors
            np.random.seed(42)
            weights = np.random.random(size=shape)

            # Define the optimizer
            opt = qml.GradientDescentOptimizer(stepsize=0.4)

            # Store the values of the cost function
            energy = [ansatz(weights)]

            # Store the values of the circuit weights
            angle = [weights]

            max_iterations = 100
            conv_tol = 1e-06

            for n in range(max_iterations):
                weights, prev_energy = opt.step_and_cost(ansatz, weights)
                energy.append(ansatz(weights))
                angle.append(weights)
                conv = np.abs(energy[-1] - prev_energy)

                if n % 2 == 0:
                    print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

                if conv <= conv_tol:
                    break

            print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
            print("\n" f"Optimal value of the circuit parameters = {angle[-1]}")

        .. code-block:: none

            Step = 0,  Energy = -0.87007254 Ha
            Step = 2,  Energy = -1.13107530 Ha
            Step = 4,  Energy = -1.13611971 Ha
            Step = 6,  Energy = -1.13618810 Ha

            Final value of the ground-state energy = -1.13618903 Ha

            Optimal value of the circuit parameters = [[[ 0.60328427  0.41850407]]
            [[ 0.85581129 -0.24522642]]]


        **Parameter shape**

        The shape of the weights argument can be computed by the static method
        :meth:`~.GateFabric.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = GateFabric.shape(n_layers=2, n_wires=4)
            weights = np.random.random(size=shape)

        >>> weights.shape
        (2, 1, 2)

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, init_state, include_pi=False, id=None):
        if len(wires) < 4:
            raise ValueError(
                f"This template requires the number of qubits to be greater than four; got wires {wires}"
            )
        if len(wires) % 2:
            raise ValueError(
                f"This template requires an even number of qubits; got {len(wires)} wires"
            )

        shape = qml.math.shape(weights)

        if len(shape) != 3:
            raise ValueError(f"Weights tensor must be 3-dimensional; got shape {shape}")

        len_wire_pattern = int((len(wires) / 2) - 1)
        if shape[1] != len_wire_pattern:
            raise ValueError(
                f"Weights tensor must have second dimension of length {len_wire_pattern}; got {shape[1]}"
            )

        if shape[2] != 2:
            raise ValueError(
                f"Weights tensor must have third dimension of length 2; got {shape[2]}"
            )

        self._hyperparameters = {
            "init_state": tuple(init_state),
            "include_pi": include_pi,
        }

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, init_state, include_pi
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.GateFabric.decomposition`.

        Args:
            weights (tensor_like): Array of weights of shape ``(D, L, 2)``,
                where ``D`` is the number of gate fabric layers and ``L = N/2-1``
                is the number of :math:`\hat{Q}(\theta, \phi)` gates per layer with N being the total number of qubits.
            wires (Any or Iterable[Any]): wires that the operator acts on
            init_state (tensor_like): init_state (tensor_like): iterable of shape ``(len(wires),)``\, representing the input Hartree-Fock state
                in the Jordan-Wigner representation.
            include_pi (boolean): If ``True``, the optional constant :math:`\hat{\Pi}` gate  is set to :math:`\text{OrbitalRotation}(\pi)`.
                Default value is :math:`\hat{I}`.

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> weights = torch.tensor([[[0.3, 1.]]])
        >>> qml.GateFabric.compute_decomposition(weights, wires=["a", "b", "c", "d"], init_state=[0, 1, 0, 1], include_pi=False)
        [BasisEmbedding(wires=['a', 'b', 'c', 'd']),
        DoubleExcitation(tensor(0.3000), wires=['a', 'b', 'c', 'd']),
        OrbitalRotation(tensor(1.), wires=['a', 'b', 'c', 'd'])]
        """
        op_list = []
        n_layers = qml.math.shape(weights)[0]
        wire_pattern = [
            wires[i : i + 4] for i in range(0, len(wires), 4) if len(wires[i : i + 4]) == 4
        ]
        if len(wires) > 4:
            wire_pattern += [
                wires[i : i + 4] for i in range(2, len(wires), 4) if len(wires[i : i + 4]) == 4
            ]

        op_list.append(qml.BasisEmbedding(init_state, wires=wires))

        for layer in range(n_layers):
            for idx, wires_ in enumerate(wire_pattern):
                if include_pi:
                    op_list.append(qml.OrbitalRotation(np.pi, wires=wires_))

                op_list.append(qml.DoubleExcitation(weights[layer][idx][0], wires=wires_))
                op_list.append(qml.OrbitalRotation(weights[layer][idx][1], wires=wires_))

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

        if n_wires < 4:
            raise ValueError(
                f"This template requires the number of qubits to be greater than four; got 'n_wires' = {n_wires}"
            )

        if n_wires % 2:
            raise ValueError(
                f"This template requires an even number of qubits; got 'n_wires' = {n_wires}"
            )

        return n_layers, n_wires // 2 - 1, 2
