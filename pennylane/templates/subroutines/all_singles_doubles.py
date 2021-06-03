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
Contains the AllSinglesDoubles template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import BasisState


class AllSinglesDoubles(Operation):
    r"""Apply the particle-conserving :class:`~.pennylane.SingleExcitation` and
    :class:`~.pennylane.DoubleExcitation` operations, implemented as Givens rotations,
    to prepare quantum states of molecules beyond the Hartree-Fock approximation.

    In the Jordan-Wigner representation the qubit state :math:`\vert 0 \rangle` or
    :math:`\vert 1 \rangle` encodes the occupation number of the molecular spin-orbitals.
    For example, the state :math:`\vert 1100 \rangle` represents the Hartree-Fock (HF) state of
    two electrons in a basis set consisting of four spin-orbitals. Other states with a fixed
    number of particles can be interpreted as excitations of the HF reference state. For example,
    the state :math:`\vert 0110 \rangle` is obtained by exciting a particle from the first to
    the third qubit. Similarly, the state :math:`\vert 0011 \rangle` corresponds to a double
    excitation involving the four qubits.

    This template initializes the qubit register to encode the Hartree-Fock state. Then,
    it applies :class:`~.pennylane.SingleExcitation` and
    :class:`~.pennylane.DoubleExcitation` operations corresponding to all possible
    single-excitations :math:`\hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`
    and double-excitations
    :math:`\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s \vert \mathrm{HF} \rangle`
    of the initial state where the indices :math:`r, s` and :math:`p, q` label, respectively,
    the occupied and unoccupied orbitals the particle annihilation (\hat{c}) and
    creation (\hat{c}^\dagger) operators act on.

    The quantum circuit for the case of two electrons and six spin-orbitals
    is sketched in the figure below,

    |

    .. figure:: ../../_static/templates/subroutines/all_singles_doubles.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    The :class:`~.pennylane.DoubleExcitation` gates :math:`G^{(2)}` perform
    `Givens rotations <https://en.wikipedia.org/wiki/Givens_rotation>`_
    in the two-dimensional subspace :math:`\{\vert 1100 \rangle, \vert 0011 \rangle \}` of
    the qubits ``[s, r, q, p]``. Similarly, the :class:`~.pennylane.SingleExcitation` gates
    :math:`G` acts on the subspace :math:`\{\vert 10 \rangle, \vert 01 \rangle \}` of the
    qubits ``[r, p]``. The resulting unitary conserves the number of particles and prepares the
    :math:`n`-qubit system in a superposition of the initial HF state and
    multiple-excited configurations.

    Args:
        weights (tensor_like): Size ``(len(singles) + len(doubles),)`` tensor containing the
            angles :math:`\theta` entering the :class:`~.pennylane.SingleExcitation` and
            :class:`~.pennylane.DoubleExcitation` operations. The indices of the qubits the
            operations act on are generated with the :func:`~.excitations` function.
        wires (Iterable): wires that the template acts on
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

            electrons = 2
            qubits = 4

            # Define the HF state
            hf_state = qml.qchem.hf_state(electrons, qubits)

            # Generate all single and double excitations
            singles, doubles = qml.qchem.excitations(electrons, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            wires = range(qubits)

            @qml.qnode(dev)
            def circuit(weights, hf_state, singles, doubles):
                qml.templates.AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
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
