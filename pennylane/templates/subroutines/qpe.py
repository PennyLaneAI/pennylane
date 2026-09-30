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

from pennylane import capture, compiler, math, ops
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator, Operator2
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.exceptions import QuantumFunctionError
from pennylane.ops import adjoint
from pennylane.ops import pow as qp_pow
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.ops.op_math.pow2 import _pow_abstract
from pennylane.typing import Wire
from pennylane.wires import Wires, _filter_abstract_and_traced_wires

from .qft import QFT


class QuantumPhaseEstimation(Operator2):
    r"""Performs the
    `quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`__
    circuit.

    Given a unitary operator :math:`U`, this template applies the circuit for quantum phase
    estimation. The unitary is applied to the wires it is defined on (the target wires) and
    :math:`n` qubits are used for phase estimation as specified by ``estimation_wires``.

    .. figure:: ../../_static/templates/subroutines/qpe.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        unitary (Operator): the phase estimation unitary. The target wires of the phase estimation
            are the wires of this operator. To use a unitary matrix, wrap it in a
            :class:`~.QubitUnitary`.
        estimation_wires (Union[Wires, Sequence[int], or int]): the wires to be used for phase
            estimation

    Raises:
        TypeError: if ``unitary`` is not an :class:`~.Operator`
        QuantumFunctionError: if the wires of ``unitary`` and ``estimation_wires`` share a common
            element

    .. details::
        :title: Usage Details

        This circuit can be used to perform the standard quantum phase estimation algorithm, consisting
        of the following steps:

        #. Prepare the target wires (the wires of ``unitary``) in a given state. If they are prepared
           in an eigenstate of :math:`U` that has corresponding eigenvalue :math:`e^{2 \pi i \theta}`
           with phase :math:`\theta \in [0, 1)`, this algorithm will measure :math:`\theta`. Other input
           states can be prepared more generally.
        #. Apply the ``QuantumPhaseEstimation`` circuit.
        #. Measure ``estimation_wires`` using :func:`~.probs`, giving a probability distribution over
           measurement outcomes in the computational basis.
        #. Find the index of the largest value in the probability distribution and divide that number by
           :math:`2^{n}`. This number will be an estimate of :math:`\theta` with an error that decreases
           exponentially with the number of qubits :math:`n`.

        Note that if :math:`\theta \in (-1, 0]`, we can estimate the phase by again finding the index
        :math:`i` found in step 4 and calculating :math:`\theta \approx \frac{1 - i}{2^{n}}`. An example
        of this case is below.

        Consider the unitary corresponding to a rotation from an :class:`~.RX` gate:

        .. code-block:: python

            import pennylane as qp
            from pennylane.templates import QuantumPhaseEstimation
            from pennylane import numpy as np

            phase = 5
            target_wires = [0]
            unitary = qp.RX(phase, wires=target_wires)

        The ``phase`` parameter can be estimated using ``QuantumPhaseEstimation``. An example is
        shown below using a register of five phase-estimation qubits:

        .. code-block:: python

            n_estimation_wires = 5
            estimation_wires = range(1, n_estimation_wires + 1)

            dev = qp.device("default.qubit", wires=n_estimation_wires + 1)

            @qp.qnode(dev)
            def circuit():
                # Start in the |+> eigenstate of the unitary
                qp.Hadamard(wires=target_wires)

                QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)

                return qp.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

            # Need to rescale phase due to convention of RX gate
            phase_estimated = 4 * np.pi * (1 - phase_estimated)

        Compound operators can be specified using operator arithmetic, and a unitary matrix can be
        used by wrapping it in a :class:`~.QubitUnitary`:

        .. code-block:: python

            # use the product to specify compound operators
            unitary = qp.RX(np.pi / 2, wires=[0]) @ qp.CNOT(wires=[0, 1])
            # equivalently, as a matrix
            unitary_from_matrix = qp.QubitUnitary(qp.matrix(unitary), wires=[0, 1])
            eigenvector = np.array([-1/2, -1/2, 1/2, 1/2])

            n_estimation_wires = 5
            estimation_wires = range(2, n_estimation_wires + 2)
            target_wires = [0, 1]

            dev = qp.device("default.qubit", wires=n_estimation_wires + 2)

            @qp.qnode(dev)
            def circuit():
                qp.StatePrep(eigenvector, wires=target_wires)
                QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)
                return qp.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

    """

    wire_argnames = ("estimation_wires",)
    hybrid_argnames = ("unitary",)

    def __init__(self, unitary, estimation_wires):
        if not isinstance(unitary, Operator):
            raise TypeError(
                "The unitary of QuantumPhaseEstimation must be an Operator, got "
                f"{type(unitary).__name__}. To use a unitary matrix, wrap it in a "
                "QubitUnitary: qp.QubitUnitary(matrix, wires=target_wires)."
            )

        super().__init__(unitary, estimation_wires)

        if Wires.shared_wires(
            [
                _filter_abstract_and_traced_wires(self.target_wires),
                _filter_abstract_and_traced_wires(self.estimation_wires),
            ]
        ):
            raise QuantumFunctionError("The target wires and estimation wires must not overlap.")

    @property
    def target_wires(self) -> Wires:
        """The wires the unitary acts on."""
        return self.unitary.wires

    @property
    def wires(self) -> Wires:
        """All wires involved in the operation: the target wires followed by the estimation wires."""
        return self.target_wires + self.estimation_wires


def _qpe_decomp_resource(unitary, estimation_wires):
    num_estimation_wires = len(estimation_wires)
    gate_count = {
        ops.Hadamard: num_estimation_wires,
        adjoint(QFT(Wire[num_estimation_wires])): 1,
    }

    # NOTE: Need abstract resource representations just in case
    # the unitary is an operator1.
    for i in range(num_estimation_wires):
        pow_rep = _pow_abstract(
            unitary,
            2**i,
        )
        gate_count[_ctrl_abstract(pow_rep, control_wires=Wire[1])] = 1
    return gate_count


@register_resources(_qpe_decomp_resource)
def _qpe_decomp(unitary, estimation_wires):
    if compiler.active() or capture.enabled():
        estimation_wires = math.array(estimation_wires, like="jax")

    num_estimation_wires = len(estimation_wires)

    @for_loop(0, num_estimation_wires)
    def _apply_h(i):
        ops.Hadamard(estimation_wires[i])

    # pylint: disable=no-value-for-parameter
    _apply_h()

    # NOTE: Must be pythonic for loop as 'z' argument in 'pow'
    # is a static argument.
    for i, w in enumerate(estimation_wires):
        ops.ctrl(qp_pow(unitary, 2 ** (len(estimation_wires) - 1 - i)), w)

    ops.adjoint(QFT(wires=estimation_wires))


add_decomps(QuantumPhaseEstimation, _qpe_decomp)
