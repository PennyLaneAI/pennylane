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
Contains the quantum number preserving GateFabric template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class GateFabric(Operation):
    r"""Implements a local, expressive, and quantum-number-preserving ansatz using VQE circuit fabrics
        proposed by Anselmetti *et al.* in `arXiv:2104.05692 <https://arxiv.org/abs/2104.05695>`_.

        This template prepares the :math:`N` qubits trial state by applying :math:`D` layers of gate fabric block
        :math:`\hat{U}_{GF}(\vec{\theta},\vec{\phi})` to the Hartree-Fock state

        .. math::

            \vert \Psi(\vec{\theta},\vec{\phi})\rangle =
            \hat{U}_{GF}^{(D)}(\vec{\theta}_{D},\vec{\phi}_{D}) \ldots
            \hat{U}_{GF}^{(2)}(\vec{\theta}_{2},\vec{\phi}_{2})
            \hat{U}_{GF}^{(1)}(\vec{\theta}_{1},\vec{\phi}_{1}) \vert HF \rangle

        where each of the gate fabric block :math:`\hat{U}_{GF}(\vec{\theta},\vec{\phi})` comprises of 2-parameter 4-qubit
        gate elements :math:`\hat{Q}(\theta, \phi)` that act on 4-nearest-neighboring qubits. The circuit implementing a
        single layer of the gate fabric block for :math:`N = 8` is shown in the figure below:

        .. figure:: ../../_static/templates/layers/gate_fabric_layer.png
            :align: center
            :width: 80%
            :target: javascript:void(0);

        The gate element :math:`\hat{Q}(\theta, \phi)` (`arXiv:2104.05692 <https://arxiv.org/abs/2104.05695>`_) is composed of
        a 1-parameter 4-qubit spin-adapted spatial orbital rotation gate, which is implemented by :class:`~.OrbitalRotation()`
        operation and 1-parameter 4-qubit diagonal pair-exchange gate, which is equivalent to the :class:`~.DoubleExcitation()`
        operation. In addition to these two gates, the gate element :math:`\hat{Q}(\theta, \phi)` can also include an optional
        constant :math:`\hat{\Pi} \in \{\hat{I}, \text{OrbitalRotation}(\pi)\}` gate.

        .. figure:: ../../_static/templates/layers/q_gate_decompositon.png
            :align: center
            :width: 75%
            :target: javascript:void(0);

        |

        The individual four qubit :class:`~.DoubleExcitation()` and :class:`~.OrbitalRotation()` gates given here are equivalent to the
        :math:`\text{QNP}_{PX}(\theta)` and :math:`\text{QNP}_{OR}(\phi)` gates presented in `arXiv:2104.05692 <https://arxiv.org/abs/2104.05695>`_
        respectively. Moreover, regardless of the choice of :math:`\hat{\Pi}`, this gate fabric will exactly preserve the quantum numbers
        :math:`\hat{N}_{\alpha}`, :math:`\hat{N}_{\beta}` and :math:`\hat{S}^{2}`.

        Args:
            weights (tensor_like): Array of weights of shape ``(D, L, 2)``\,
                where ``D`` is the number of gate fabric layers and ``L = N/2-1``
                is the number of :math:`\hat{Q}(\theta, \phi)` gates per layer.
            wires (Iterable): wires that the template acts on
            init_state (tensor_like): iterable of shape ``(len(wires),)``\, representing the Hartree-Fock state in Jordan-Wigner basis
                that is used to initialize the wires.
            include_pi (boolean): If ``include_pi = True``\, the optional constant :math:`\hat{\Pi}` gate  is set to :math:`\text{OrbitalRotation}(\pi)`.
                Default value is :math:`\hat{I}`.

        .. UsageDetails::

            #. The number of wires :math:`N` has to be equal to the number of
               spin orbitals included in the active space, and should be even.

            #. The number of trainable parameters scales linearly with the number of layers as
               :math:`2 D (N/2-1)`.

            An example of how to use this template is shown below:

            .. code-block:: python

                import pennylane as qml
                from pennylane.templates import GateFabric
                from functools import partial

                # Build the electronic Hamiltonian from a local .xyz file
                h, qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")

                # Define the Hartree-Fock state
                electrons = 2
                ref_state = qml.qchem.hf_state(electrons, qubits)

                # Define the device
                dev = qml.device('default.qubit', wires=qubits)

                # Define the ansatz
                ansatz = partial(GateFabric, init_state=ref_state, include_pi=True)

                # Define the cost function
                cost_fn = qml.ExpvalCost(ansatz, h, dev)

                # Compute the expectation value of 'h'
                layers = 2
                params = qml.init.gate_fabric_normal(layers, qubits)
                print(cost_fn(params))

            **Parameter shape**

            The shape of the weights argument can be computed by the static method
            :meth:`~.GateFabric.shape` and used when creating randomly
            initialised weight tensors:

            .. code-block:: python

                shape = GateFabric.shape(n_layers=2, n_wires=4)
                weights = np.random.random(size=shape)


    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, weights, wires, init_state, include_pi=False, do_queue=True, id=None):

        if len(wires) < 4:
            raise ValueError(
                f"This template requires the number of qubits to be greater than four; got wires {wires}"
            )
        if len(wires) % 2:
            raise ValueError(
                f"This template requires even number of qubits; got {len(wires)} wires"
            )

        self.qwires = [
            wires[i : i + 4] for i in range(0, len(wires), 4) if len(wires[i : i + 4]) == 4
        ]
        if len(wires) > 4:
            self.qwires += [
                wires[i : i + 4] for i in range(2, len(wires), 4) if len(wires[i : i + 4]) == 4
            ]

        shape = qml.math.shape(weights)

        if len(shape) != 3:
            raise ValueError(f"Weights tensor must be 3-dimensional; got shape {shape}")

        if shape[1] != len(self.qwires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(self.qwires)}; got {shape[1]}"
            )

        if shape[2] != 2:
            raise ValueError(
                f"Weights tensor must have third dimension of length 2; got {shape[2]}"
            )

        self.n_layers = shape[0]
        # we can extract the numpy representation here
        # since init_state can never be differentiable
        self.init_state = qml.math.toarray(init_state)

        self.include_pi = include_pi

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        with qml.tape.QuantumTape() as tape:

            qml.templates.BasisEmbedding(self.init_state, wires=self.wires)
            weight = self.parameters[0]

            for layer in range(self.n_layers):
                for idx, wires in enumerate(self.qwires):

                    if self.include_pi:
                        qml.OrbitalRotation(np.pi, wires=wires)

                    qml.DoubleExcitation(weight[layer][idx][0], wires=wires)
                    qml.OrbitalRotation(weight[layer][idx][1], wires=wires)

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

        if n_wires < 4:
            raise ValueError(
                f"This template requires the number of qubits to be greater than four; got 'n_wires' = {n_wires}"
            )

        if n_wires % 2:
            raise ValueError(
                f"This template requires even number of qubits; got 'n_wires' = {n_wires}"
            )

        return n_layers, n_wires // 2 - 1, 2
