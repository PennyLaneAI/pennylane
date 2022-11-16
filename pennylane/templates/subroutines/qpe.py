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
"""
Contains the QuantumPhaseEstimation template.
"""
# pylint: disable=too-many-arguments,arguments-differ
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import Hadamard, ControlledQubitUnitary, ctrl


class QuantumPhaseEstimation(Operation):
    r"""Performs the
    `quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`__
    circuit.

    Given a unitary matrix :math:`U`, this template applies the circuit for quantum phase
    estimation. The unitary is applied to the qubits specified by ``target_wires`` and :math:`n`
    qubits are used for phase estimation as specified by ``estimation_wires``.

    .. figure:: ../../_static/templates/subroutines/qpe.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    This circuit can be used to perform the standard quantum phase estimation algorithm, consisting
    of the following steps:

    #. Prepare ``target_wires`` in a given state. If ``target_wires`` are prepared in an eigenstate
       of :math:`U` that has corresponding eigenvalue :math:`e^{2 \pi i \theta}` with phase
       :math:`\theta \in [0, 1)`, this algorithm will measure :math:`\theta`. Other input states can
       be prepared more generally.
    #. Apply the ``QuantumPhaseEstimation`` circuit.
    #. Measure ``estimation_wires`` using :func:`~.probs`, giving a probability distribution over
       measurement outcomes in the computational basis.
    #. Find the index of the largest value in the probability distribution and divide that number by
       :math:`2^{n}`. This number will be an estimate of :math:`\theta` with an error that decreases
       exponentially with the number of qubits :math:`n`.

    Note that if :math:`\theta \in (-1, 0]`, we can estimate the phase by again finding the index
    :math:`i` found in step 4 and calculating :math:`\theta \approx \frac{1 - i}{2^{n}}`. The
    usage details below give an example of this case.

    Args:
        unitary (array or QNode): the phase estimation unitary, specified as a matrix or a
            quantum function in the form of a :class:`.QNode` returning a :func:`~.state`
        target_wires (Union[Wires, Sequence[int], or int]): the target wires to apply the unitary
        estimation_wires (Union[Wires, Sequence[int], or int]): the wires to be used for phase
            estimation

    Raises:
        QuantumFunctionError: if the ``target_wires`` and ``estimation_wires`` share a common
            element, or if the quantum function is invalid

    .. details::
        :title: Usage Details

        Consider the matrix corresponding to a rotation from an :class:`~.RX` gate:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import QuantumPhaseEstimation
            from pennylane import numpy as np

            phase = 5
            target_wires = [0]
            unitary = qml.RX(phase, wires=0).matrix()

        The ``phase`` parameter can be estimated using ``QuantumPhaseEstimation``. An example is
        shown below using a register of five phase-estimation qubits:

        .. code-block:: python

            n_estimation_wires = 5
            estimation_wires = range(1, n_estimation_wires + 1)

            dev = qml.device("default.qubit", wires=n_estimation_wires + 1)

            @qml.qnode(dev)
            def circuit():
                # Start in the |+> eigenstate of the unitary
                qml.Hadamard(wires=target_wires)

                QuantumPhaseEstimation(
                    unitary,
                    target_wires=target_wires,
                    estimation_wires=estimation_wires,
                )

                return qml.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

            # Need to rescale phase due to convention of RX gate
            phase_estimated = 4 * np.pi * (1 - phase_estimated)

        We can also perform phase estimation on a quantum function. Note that the quantum function
        must only consist of unitary operators, have the same number of wires as the target wires,
        and not parameterized. The exact wire labels used in the quantum function need not be the
        same as the target wires, as they will be mapped. For example, if the quantum function is
        defined to act on wires 0 and 1, but the target wires are 2 and 3 of a larger circuit, the
        operators in the quantum function will be automatically mapped to wires 2 and 3.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def unitary():
                qml.RX(np.pi / 2, wires=[0])
                qml.CNOT(wires=[0, 1])
                return qml.state()

            eigenvector = np.array([-1/2, -1/2, 1/2, 1/2])

            n_estimation_wires = 5
            estimation_wires = range(2, n_estimation_wires + 2)
            target_wires = [0, 1]

            dev = qml.device("default.qubit", wires=n_estimation_wires + 2)

            @qml.qnode(dev)
            def circuit():
                qml.QubitStateVector(eigenvector, wires=target_wires)
                QuantumPhaseEstimation(
                    unitary,
                    target_wires=target_wires,
                    estimation_wires=estimation_wires,
                )
                return qml.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, unitary, target_wires, estimation_wires, do_queue=True, id=None):

        target_wires = list(target_wires)
        estimation_wires = list(estimation_wires)
        wires = target_wires + estimation_wires

        if any(wire in target_wires for wire in estimation_wires):
            raise qml.QuantumFunctionError(
                "The target wires and estimation wires must be different"
            )

        self._hyperparameters = {
            "target_wires": target_wires,
            "estimation_wires": estimation_wires,
        }

        super().__init__(unitary, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def _compute_decomposition_qnode(qnode, target_wires, estimation_wires):
        """Construct the decomposition of QPE for a :class:`.QNode` returning a :func:`~.state`

        Args:
            qnode (QNode): the quantum function for which to build the decomposition
            target_wires (Any or Iterable[Any]): the target wires to apply the unitary
            estimation_wires (Any or Iterable[Any]): the wires to be used for phase estimation

        Returns:
            list[.Operator]: decomposition of the phase estimation subroutine

        """

        @qml.QueuingManager.stop_recording()
        def get_operations():
            """Extract operations from a QNode, without recording these operations"""

            # Construct tape with qnode and extract all the operations
            qnode.construct([], {})

            # Handle wires
            if qnode.tape.wires.labels != tuple(range(qnode.tape.num_wires)):
                # Custom wire labels are not supported as it is difficult to accurately map
                # the function wires to the target wires.
                raise qml.QuantumFunctionError(
                    "QPE is not supported for quantum functions using custom wire labels"
                )
            if len(qnode.tape.wires) != len(target_wires):
                raise qml.QuantumFunctionError(
                    "The number of target wires does not match the number of wires the "
                    "quantum function uses."
                )

            # The quantum function can have different wire labels than the target wires, this
            # makes sure that the quantum function is applied correctly to the target wires.
            wire_map = dict(zip(qnode.tape.wires, target_wires))
            tape = qml.map_wires(qnode.tape, wire_map)
            return tape.operations

        # Construct the powers of the unitary
        operations = get_operations()
        powers = [operations]
        for _ in range(len(estimation_wires) - 1):
            # U^2 is achieved by doing the same operations twice
            new_power = powers[-1] + powers[-1]
            powers.append(new_power)

        # Construct the phase estimation circuit
        op_list = []

        for wire in estimation_wires:
            op_list.append(Hadamard(wire))
            sequence = powers.pop()
            for op in sequence:
                op_list.append(ctrl(op, wire))

        op_list.append(qml.adjoint(qml.templates.QFT(wires=estimation_wires)))

        return op_list

    @staticmethod
    def compute_decomposition(
        unitary, wires, target_wires, estimation_wires
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the QPE circuit as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.QuantumPhaseEstimation.decomposition`.

        Args:
            unitary (array or QNode): the phase estimation unitary, specified as a matrix or
                quantum function expressed as a QNode.
            wires (Any or Iterable[Any]): wires that the operator acts on
            target_wires (Any or Iterable[Any]): the target wires to apply the unitary
            estimation_wires (Any or Iterable[Any]): the wires to be used for phase estimation

        Returns:
            list[.Operator]: decomposition of the operator
        """

        if isinstance(unitary, qml.QNode):
            return QuantumPhaseEstimation._compute_decomposition_qnode(
                unitary, target_wires, estimation_wires
            )

        unitary_powers = [unitary]

        for _ in range(len(estimation_wires) - 1):
            new_power = unitary_powers[-1] @ unitary_powers[-1]
            unitary_powers.append(new_power)

        op_list = []

        for wire in estimation_wires:
            op_list.append(Hadamard(wire))
            op_list.append(
                ControlledQubitUnitary(unitary_powers.pop(), control_wires=wire, wires=target_wires)
            )

        op_list.append(qml.adjoint(qml.templates.QFT(wires=estimation_wires)))

        return op_list
