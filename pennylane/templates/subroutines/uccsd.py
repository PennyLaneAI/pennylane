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
    r"""Apply the :class:`~.pennylane.SingleExcitation` and :class:`~.pennylane.DoubleExcitation`
    operations, implemented as Givens rotations, to the :math:`n-`qubit system to prepare the
    correlated quantum state of a molecule.

    This ansatz is similar to the traditional `Unitary Coupled-Clusters Singles
    and Doubles (UCCSD) <https://arxiv.org/abs/1805.04340>`_ given by,

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`:math:`\hat{c}_p^\dagger \hat{c}_r` and
    :math:`\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s` are, respectively, the
    fermionic single- and double-excitation operators, and the indices :math:`r, s`
    and :math:`p, q` run over the Hartree-Fock occupied and unoccupied molecular orbitals.

    The main difference with respect to the original UCCSD template is that the gate
    decompositions used to exponentiate the single- and double-excitation operators
    are replaced with the application of `Givens rotations
    <https://en.wikipedia.org/wiki/Givens_rotation>`_ that act on the subspaces of two and
    four qubits. More specifically, the :class:`~.pennylane.SingleExcitation` operation performs
    a rotation in the two-dimensional subspace :math:`\{\vert 10 \rangle, \vert 01 \rangle \}`
    of the qubits :math:`r, p` associated with the single excitation
    `:math:`\hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle` of the Hartree-Fock (HF)
    state. Similarly, the :class:`~.pennylane.DoubleExcitation` operation implements
    a rotation in the subspace :math:`\{\vert 1100 \rangle, \vert 0011 \rangle \}` of the qubits
    :math:`s, r, q, p` involved in the double excitation
    :math:`\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s \vert \mathrm{HF} \rangle`.

    The resulting unitary conserves the number of particles and prepares the
    :math:`n`-qubit system in a superposition of the initial HF state and
    multiple-excited configurations.

    Args:
        weights (tensor_like): Size ``(len(singles) + len(doubles),)`` tensor containing the
            angles :math:`\theta` entering the :class:`~.pennylane.SingleExcitation` and
            :class:`~.pennylane.DoubleExcitation` operations. The indices of the qubits the
            operations act on are generated with the :func:`~.excitations` function.
        wires (Iterable): wires that the template acts on.
        singles (Sequence[Sequence]): sequence of lists containing the wires indices ``[r, p]``
        doubles (Sequence[Sequence]): sequence of lists containing the wires indices
            ``[s, r, q, p]``
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
            import numpy as np

            # Define the HF state
            electrons = 2
            qubits = 4
            hf_state = qml.qchem.hf_state(electrons, qubits)

            # Generate single and double excitations
            singles, doubles = qml.qchem.excitations(electrons, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            wires = range(qubits)

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

        if len(wires) < 2:
            raise ValueError(
                "The number of qubits (wires) can not be less than 2; got len(wires) = {}".format(
                    len(wires)
                )
            )

        if not singles and not doubles:
            raise ValueError(
                "'singles' and 'doubles' lists can not be both empty; got singles = {}, doubles = {}".format(
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
