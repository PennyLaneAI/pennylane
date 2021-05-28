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
        weights (tensor_like): Size ``(len(singles) + len(doubles),)`` tensor containing the
            angles :math:`\theta` entering the :class:`~.pennylane.SingleExcitation` and
            :class:`~.pennylane.DoubleExcitation` operations corresponding to the single and
            double excitations of the Hartree-Fock (HF) reference state generated with the
            :func:`~.excitations` function.
        wires (Iterable): wires that the template acts on
        singles (Sequence[Sequence]): Sequence of lists containing the wires indices ``[r, p]``
            encoding, respectively, the indices of the occupied and unoccupied orbitals
            involved in the single excitation
            :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`.
        doubles (Sequence[Sequence]): Sequence of lists containing the wires indices
            ``[s, r, q, p]``. The qubit indices ``s, r`` and ``q, p`` correspond, respectively,
            to the occupied and unoccupied orbitals involved in the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r
            \hat{c}_s \vert \mathrm{HF} \rangle`.
        hf_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
            HF state. ``hf_state`` is used to initialize the wires.

    .. UsageDetails::

        Notice that:

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The single and double excitations can be generated with the function
           :func:`~.excitations`. See example below.

        #. The vector of parameters ``weights`` is a one-dimensional array of size
           ``len(singles)+len(doubles)``

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml

            # Define the HF state
            hf_state = qml.qchem.hf_state(electrons=2, qubits=4)

            # Generate single and double excitations
            singles, doubles = qml.qchem.excitations(electrons=2, qubits=4)

            # Define the device
            dev = qml.device('default.qubit', wires=4)

            wires = range(4)

            @qml.qnode(dev)
            def circuit(weights, hf_state, singles, doubles):
            	qml.templates.UCCSD(weights, wires, hf_state, singles, doubles)
            	return qml.expval(qml.PauliZ(0))

            # Compute the expectation value of 'h' for given set of parameters 'params'
            params = np.random.normal(0, np.pi, len(singles) + len(doubles))
            circuit(params, hf_state, singles=singles, doubles=doubles)
    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, weights, wires, hf_state, singles=None, doubles=None, do_queue=True):

        if not singles and not doubles:
            raise ValueError(
                "'singles' and 'doubles' lists can not be both empty; got singles={}, doubles={}".format(
                    singles, doubles
                )
            )

        for d_wires in doubles:
            if len(d_wires) != 4:
                raise ValueError(
                    "Expected entries of 'doubles' to be of size 4; got {} of length {}".format(
                        d_wires, len(d_wires)
                    )
                )

        for s_wires in singles:
            if len(s_wires) != 2:
                raise ValueError(
                    "Expected entries of 'singles' to be of size 2; got {} of length {}".format(
                        s_wires, len(s_wires)
                    )
                )

        shape = qml.math.shape(weights)
        if shape != (len(singles) + len(doubles),):
            raise ValueError(
                f"'weights' tensor must be of shape {(len(singles) + len(doubles),)}; got {shape}."
            )

        # we can extract the numpy representation here
        # since hf_state can never be differentiable
        self.hf_state = qml.math.toarray(hf_state)
        self.singles = singles
        self.doubles = doubles

        if hf_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'hf_state' must be integers; got {hf_state.dtype}")

        super().__init__(weights, wires=wires, do_queue=do_queue)

    def expand(self):

        weights = self.parameters[0]

        with qml.tape.QuantumTape() as tape:

            BasisState(self.hf_state, wires=self.wires)

            for i, d_wires in enumerate(self.doubles):
                qml.DoubleExcitation(weights[len(self.singles) + i], wires=d_wires)

            for j, s_wires in enumerate(self.singles):
                qml.SingleExcitation(weights[j], wires=s_wires)

        return tape
