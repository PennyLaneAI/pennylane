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
    r"""Builds a quantum circuit to prepare correlated states of molecules
    by applying all :class:`~.pennylane.SingleExcitation` and
    :class:`~.pennylane.DoubleExcitation` operations to
    the initial Hartree-Fock state.

    The template initializes the :math:`n`-qubit system to encode
    the input Hartree-Fock state and applies the particle-conserving
    :class:`~.pennylane.SingleExcitation` and
    :class:`~.pennylane.DoubleExcitation` operations which are implemented as
    `Givens rotations <https://en.wikipedia.org/wiki/Givens_rotation>`_ that act
    on the subspace of two and four qubits, respectively. The total number of
    excitation gates and the indices of the qubits they act on are obtained
    using the :func:`~.excitations` function.

    For example, the quantum circuit for the case of two electrons and six qubits
    is sketched in the figure below:

    |

    .. figure:: ../../_static/templates/subroutines/all_singles_doubles.png
        :align: center
        :width: 70%
        :target: javascript:void(0);

    |

    In this case, we have four single and double excitations that preserve the total-spin
    projection of the Hartree-Fock state. The :class:`~.pennylane.SingleExcitation` gate
    :math:`G` act on the qubits ``[0, 2], [0, 4], [1, 3], [1, 5]`` as indicated by the
    squares, while the :class:`~.pennylane.DoubleExcitation` operation :math:`G^{(2)}` is
    applied to the qubits ``[0, 1, 2, 3], [0, 1, 2, 5], [0, 1, 2, 4], [0, 1, 4, 5]``.

    The resulting unitary conserves the number of particles and prepares the
    :math:`n`-qubit system in a superposition of the initial Hartree-Fock state and
    other states encoding multiply-excited configurations.

    Args:
        weights (tensor_like): size ``(len(singles) + len(doubles),)`` tensor containing the
            angles entering the :class:`~.pennylane.SingleExcitation` and
            :class:`~.pennylane.DoubleExcitation` operations, in that order
        wires (Iterable): wires that the template acts on
        hf_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
            Hartree-Fock state. ``hf_state`` is used to initialize the wires.
        singles (Sequence[Sequence]): sequence of lists with the indices of the two qubits
            the :class:`~.pennylane.SingleExcitation` operations act on
        doubles (Sequence[Sequence]): sequence of lists with the indices of the four qubits
            the :class:`~.pennylane.DoubleExcitation` operations act on

    .. UsageDetails::

        Notice that:

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The single and double excitations can be generated with the function
           :func:`~.excitations`. See example below.

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

            # Evaluate the QNode for a given set of parameters
            params = np.random.normal(0, np.pi, len(singles) + len(doubles))
            circuit(params, hf_state, singles=singles, doubles=doubles)
    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(
        self, weights, wires, hf_state, singles=None, doubles=None, do_queue=True, id=None
    ):

        if len(wires) < 2:
            raise ValueError(
                "The number of qubits (wires) can not be less than 2; got len(wires) = {}".format(
                    len(wires)
                )
            )

        if doubles is not None:
            for d_wires in doubles:
                if len(d_wires) != 4:
                    raise ValueError(
                        "Expected entries of 'doubles' to be of size 4; got {} of length {}".format(
                            d_wires, len(d_wires)
                        )
                    )

        if singles is not None:
            for s_wires in singles:
                if len(s_wires) != 2:
                    raise ValueError(
                        "Expected entries of 'singles' to be of size 2; got {} of length {}".format(
                            s_wires, len(s_wires)
                        )
                    )

        weights_shape = qml.math.shape(weights)
        exp_shape = self.shape(singles, doubles)
        if weights_shape != exp_shape:
            raise ValueError(f"'weights' tensor must be of shape {exp_shape}; got {weights_shape}.")

        # we can extract the numpy representation here since hf_state can never be differentiable
        self.hf_state = qml.math.toarray(hf_state)
        self.singles = singles
        self.doubles = doubles

        if hf_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'hf_state' must be integers; got {hf_state.dtype}")

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        weights = self.parameters[0]

        with qml.tape.QuantumTape() as tape:

            BasisState(self.hf_state, wires=self.wires)

            for i, d_wires in enumerate(self.doubles):
                qml.DoubleExcitation(weights[len(self.singles) + i], wires=d_wires)

            for j, s_wires in enumerate(self.singles):
                qml.SingleExcitation(weights[j], wires=s_wires)

        return tape

    @staticmethod
    def shape(singles, doubles):
        r"""Returns the expected shape of the tensor that contains the circuit parameters.

        Args:
            singles (Sequence[Sequence]): sequence of lists with the indices of the two qubits
                the :class:`~.pennylane.SingleExcitation` operations act on
            doubles (Sequence[Sequence]): sequence of lists with the indices of the four qubits
                the :class:`~.pennylane.DoubleExcitation` operations act on

        Returns:
            tuple(int): shape of the tensor containing the circuit parameters
        """
        if singles is None or not singles:
            if doubles is None or not doubles:
                raise ValueError(
                    "'singles' and 'doubles' lists can not be both empty;"
                    " got singles = {}, doubles = {}".format(singles, doubles)
                )
            if doubles is not None:
                shape_ = (len(doubles),)
        elif doubles is None:
            shape_ = (len(singles),)
        else:
            shape_ = (len(singles) + len(doubles),)

        return shape_
