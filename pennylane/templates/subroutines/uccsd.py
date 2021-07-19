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
    :func:`~.SingleExcitationUnitary` and :func:`~.DoubleExcitationUnitary`
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
            :func:`~.SingleExcitationUnitary`
            and
            :func:`~.DoubleExcitationUnitary`. These parameters are the coupled-cluster
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

    .. UsageDetails::

        Notice that:

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The single and double excitations can be generated be generated with the function
           :func:`~.excitations`. See example below.

        #. The vector of parameters ``weights`` is a one-dimensional array of size
           ``len(s_wires)+len(d_wires)``


        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane import qchem
            from pennylane.templates import UCCSD

            from functools import partial

            # Build the electronic Hamiltonian
            name = "h2"
            geo_file = "h2.xyz"
            h, qubits = qchem.molecular_hamiltonian(name, geo_file)

            # Number of electrons
            electrons = 2

            # Define the HF state
            ref_state = qchem.hf_state(electrons, qubits)

            # Generate single and double excitations
            singles, doubles = qchem.excitations(electrons, qubits)

            # Map excitations to the wires the UCCSD circuit will act on
            s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            # Define the UCCSD ansatz
            ansatz = partial(UCCSD, init_state=ref_state, s_wires=s_wires, d_wires=d_wires)

            # Define the cost function
            cost_fn = qml.ExpvalCost(ansatz, h, dev)

            # Compute the expectation value of 'h' for given set of parameters 'params'
            params = np.random.normal(0, np.pi, len(singles) + len(doubles))
            print(cost_fn(params))

    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(
        self, weights, wires, s_wires=None, d_wires=None, init_state=None, do_queue=True, id=None
    ):

        if (not s_wires) and (not d_wires):
            raise ValueError(
                "s_wires and d_wires lists can not be both empty; got ph={}, pphh={}".format(
                    s_wires, d_wires
                )
            )

        for d_wires_ in d_wires:
            if len(d_wires_) != 2:
                raise ValueError(
                    "expected entries of d_wires to be of size 2; got {} of length {}".format(
                        d_wires_, len(d_wires_)
                    )
                )

        shape = qml.math.shape(weights)
        if shape != (len(s_wires) + len(d_wires),):
            raise ValueError(
                f"Weights tensor must be of shape {(len(s_wires) + len(d_wires),)}; got {shape}."
            )

        # we can extract the numpy representation here
        # since init_state can never be differentiable
        self.init_state = qml.math.toarray(init_state)
        self.s_wires = s_wires
        self.d_wires = d_wires

        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self.init_state_flipped = np.flip(init_state)

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        weights = self.parameters[0]

        with qml.tape.QuantumTape() as tape:

            BasisState(self.init_state_flipped, wires=self.wires)

            for i, (w1, w2) in enumerate(self.d_wires):
                qml.templates.DoubleExcitationUnitary(
                    weights[len(self.s_wires) + i], wires1=w1, wires2=w2
                )

            for j, s_wires_ in enumerate(self.s_wires):
                qml.templates.SingleExcitationUnitary(weights[j], wires=s_wires_)

        return tape
