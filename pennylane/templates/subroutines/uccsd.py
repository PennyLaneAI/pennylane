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
Contains the UCCSD template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import BasisState


class UCCSD(Operation):
    r"""Implements the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz.

    The UCCSD ansatz calls the
    :func:`~.FermionicSingleExcitation` and :func:`~.FermionicDoubleExcitation`
    templates to exponentiate the coupled-cluster excitation operator. UCCSD is a VQE ansatz
    commonly used to run quantum chemistry simulations.

    The UCCSD unitary, within the first-order Trotter approximation, is given by:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermionic annihilation and
    creation operators and the indices :math:`r, s` and :math:`p, q` run over the occupied and
    unoccupied molecular orbitals, respectively. Using the `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ the UCCSD unitary defined above can be written in terms
    of Pauli matrices as follows (for more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    .. math::

        \hat{U}(\vec{\theta}) = && \prod_{p > r} \mathrm{exp} \Big\{ \frac{i\theta_{pr}}{2}
        \bigotimes_{a=r+1}^{p-1} \hat{Z}_a (\hat{Y}_r \hat{X}_p - \mathrm{H.c.}) \Big\} \\
        && \times \prod_{p > q > r > s} \mathrm{exp} \Big\{ \frac{i\theta_{pqrs}}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p -
        \{\mathrm{H.c.}\}) \Big\}.

    Args:
        weights (tensor_like): Size ``(len(s_wires) + len(d_wires),)`` tensor containing the parameters
            :math:`\theta_{pr}` and :math:`\theta_{pqrs}` entering the Z rotation in
            :func:`~.FermionicSingleExcitation`
            and
            :func:`~.FermionicDoubleExcitation`. These parameters are the coupled-cluster
            amplitudes that need to be optimized for each single and double excitation generated
            with the :func:`~.excitations` function.
        wires (Iterable): wires that the template acts on
        s_wires (Sequence[Sequence]): Sequence of lists containing the wires ``[r,...,p]``
            resulting from the single excitation
            :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
            where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
            The first (last) entry ``r`` (``p``) is considered the wire representing the
            occupied (unoccupied) orbital where the electron is annihilated (created).
        d_wires (Sequence[Sequence[Sequence]]): Sequence of lists, each containing two lists that
            specify the indices ``[s, ...,r]`` and ``[q,..., p]`` defining the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r
            \hat{c}_s \vert \mathrm{HF} \rangle`. The entries ``s`` and ``r`` are wires
            representing two occupied orbitals where the two electrons are annihilated
            while the entries ``q`` and ``p`` correspond to the wires representing two unoccupied
            orbitals where the electrons are created. Wires in-between represent the occupied
            and unoccupied orbitals in the intervals ``[s, r]`` and ``[q, p]``, respectively.
        init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
            HF state. ``init_state`` is used to initialize the wires.

    .. details::
        :title: Usage Details

        Notice that:

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The single and double excitations can be generated with the function
           :func:`~.excitations`. See example below.

        #. The vector of parameters ``weights`` is a one-dimensional array of size
           ``len(s_wires)+len(d_wires)``


        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane import numpy as np

            # Define the molecule
            symbols  = ['H', 'H', 'H']
            geometry = np.array([[0.01076341,  0.04449877,  0.0],
                                 [0.98729513,  1.63059094,  0.0],
                                 [1.87262415, -0.00815842,  0.0]], requires_grad = False)
            electrons = 2
            charge = 1

            # Build the electronic Hamiltonian
            H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge)

            # Define the HF state
            hf_state = qml.qchem.hf_state(electrons, qubits)

            # Generate single and double excitations
            singles, doubles = qml.qchem.excitations(electrons, qubits)

            # Map excitations to the wires the UCCSD circuit will act on
            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

            # Define the device
            dev = qml.device("default.qubit", wires=qubits)

            # Define the qnode
            @qml.qnode(dev)
            def circuit(params, wires, s_wires, d_wires, hf_state):
                qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
                return qml.expval(H)

            # Define the initial values of the circuit parameters
            params = np.zeros(len(singles) + len(doubles))

            # Define the optimizer
            optimizer = qml.GradientDescentOptimizer(stepsize=0.5)

            # Optimize the circuit parameters and compute the energy
            for n in range(21):
                params, energy = optimizer.step_and_cost(circuit, params,
                wires=range(qubits), s_wires=s_wires, d_wires=d_wires, hf_state=hf_state)
                if n % 2 == 0:
                    print("step = {:},  E = {:.8f} Ha".format(n, energy))

        .. code-block:: none

            step = 0,  E = -1.24654994 Ha
            step = 2,  E = -1.27016844 Ha
            step = 4,  E = -1.27379541 Ha
            step = 6,  E = -1.27434106 Ha
            step = 8,  E = -1.27442311 Ha
            step = 10,  E = -1.27443547 Ha
            step = 12,  E = -1.27443733 Ha
            step = 14,  E = -1.27443761 Ha
            step = 16,  E = -1.27443765 Ha
            step = 18,  E = -1.27443766 Ha
            step = 20,  E = -1.27443766 Ha

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, s_wires=None, d_wires=None, init_state=None, id=None):
        if (not s_wires) and (not d_wires):
            raise ValueError(
                f"s_wires and d_wires lists can not be both empty; got ph={s_wires}, pphh={d_wires}"
            )

        for d_wires_ in d_wires:
            if len(d_wires_) != 2:
                raise ValueError(
                    f"expected entries of d_wires to be of size 2; got {d_wires_} of length {len(d_wires_)}"
                )

        shape = qml.math.shape(weights)
        if shape != (len(s_wires) + len(d_wires),):
            raise ValueError(
                f"Weights tensor must be of shape {(len(s_wires) + len(d_wires),)}; got {shape}."
            )

        init_state = qml.math.toarray(init_state)

        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {"init_state": init_state, "s_wires": s_wires, "d_wires": d_wires}

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, s_wires, d_wires, init_state
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.UCCSD.decomposition`.

        Args:
            weights (tensor_like): Size ``(len(s_wires) + len(d_wires),)`` tensor containing the parameters
                entering the Z rotation in :func:`~.FermionicSingleExcitation` and :func:`~.FermionicDoubleExcitation`.
            wires (Any or Iterable[Any]): wires that the operator acts on
            s_wires (Sequence[Sequence]): Sequence of lists containing the wires ``[r,...,p]``
                resulting from the single excitation.
            d_wires (Sequence[Sequence[Sequence]]): Sequence of lists, each containing two lists that
                specify the indices ``[s, ...,r]`` and ``[q,..., p]`` defining the double excitation.
            init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
                HF state. ``init_state`` is used to initialize the wires.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []

        op_list.append(BasisState(init_state, wires=wires))

        for i, (w1, w2) in enumerate(d_wires):
            op_list.append(
                qml.FermionicDoubleExcitation(weights[len(s_wires) + i], wires1=w1, wires2=w2)
            )

        for j, s_wires_ in enumerate(s_wires):
            op_list.append(qml.FermionicSingleExcitation(weights[j], wires=s_wires_))

        return op_list
