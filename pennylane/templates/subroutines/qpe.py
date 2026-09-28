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

from pennylane import ops
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

# pylint: disable=arguments-differ
from pennylane.typing import Wire
from pennylane.wires import Wires

from .qft import QFT


class QuantumPhaseEstimation(Operator2):
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

    Args:
        unitary (array or Operator): the phase estimation unitary, specified as a matrix or an
            :class:`~.Operator`
        target_wires (Union[Wires, Sequence[int], or int]): the target wires to apply the unitary.
            If the unitary is specified as an operator, the target wires should already have been
            defined as part of the operator. In this case, target_wires should not be specified.
        estimation_wires (Union[Wires, Sequence[int], or int]): the wires to be used for phase
            estimation

    Raises:
        QuantumFunctionError: if the ``target_wires`` and ``estimation_wires`` share a common
            element, or if ``target_wires`` are specified for an operator unitary.

    .. details::
        :title: Usage Details

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
        :math:`i` found in step 4 and calculating :math:`\theta \approx \frac{1 - i}{2^{n}}`. An example
        of this case is below.

        Consider the matrix corresponding to a rotation from an :class:`~.RX` gate:

        .. code-block:: python

            import pennylane as qp
            from pennylane.templates import QuantumPhaseEstimation
            from pennylane import numpy as np

            phase = 5
            target_wires = [0]
            unitary = qp.RX(phase, wires=0).matrix()

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

                QuantumPhaseEstimation(
                    unitary,
                    target_wires=target_wires,
                    estimation_wires=estimation_wires,
                )

                return qp.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

            # Need to rescale phase due to convention of RX gate
            phase_estimated = 4 * np.pi * (1 - phase_estimated)

        We can also perform phase estimation on an operator. Note that since operators are defined
        with target wires, the target wires should not be provided for the QPE.

        .. code-block:: python


            # use the product to specify compound operators
            unitary = qp.RX(np.pi / 2, wires=[0]) @ qp.CNOT(wires=[0, 1])
            eigenvector = np.array([-1/2, -1/2, 1/2, 1/2])

            n_estimation_wires = 5
            estimation_wires = range(2, n_estimation_wires + 2)
            target_wires = [0, 1]

            dev = qp.device("default.qubit", wires=n_estimation_wires + 2)

            @qp.qnode(dev)
            def circuit():
                qp.StatePrep(eigenvector, wires=target_wires)
                QuantumPhaseEstimation(
                    unitary,
                    estimation_wires=estimation_wires,
                )
                return qp.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

    """

    wire_argnames = ("target_wires", "estimation_wires")
    hybrid_argnames = ("unitary",)

    grad_method = None

    def __init__(self, unitary, target_wires=None, estimation_wires=None):
        if isinstance(unitary, Operator):
            # If the unitary is expressed in terms of operators, do not provide target wires
            if target_wires is not None and Wires(target_wires) != unitary.wires:
                raise QuantumFunctionError(
                    "The unitary is expressed as an operator, which already has target wires "
                    "defined, do not additionally specify target wires."
                )
            target_wires = unitary.wires

        elif target_wires is None:
            raise QuantumFunctionError(
                "Target wires must be specified if the unitary is expressed as a matrix."
            )

        else:
            unitary = ops.QubitUnitary(unitary, wires=target_wires)

        # Estimation wires are required, but kept as an optional argument so that it can be
        # placed after target_wires for backwards compatibility.
        if estimation_wires is None:
            raise QuantumFunctionError("No estimation wires specified.")

        super().__init__(unitary, target_wires, estimation_wires)

        if not self.is_fully_abstract and any(
            wire in self.target_wires for wire in self.estimation_wires
        ):
            raise QuantumFunctionError("The target wires and estimation wires must not overlap.")


def _qpe_decomp_resource(
    unitary, target_wires, estimation_wires
):  # pylint: disable=unused-argument
    num_estimation_wires = len(estimation_wires)
    gate_count = {
        ops.Hadamard: num_estimation_wires,
        adjoint(QFT(Wire[num_estimation_wires])): 1,
    }
    for i in range(num_estimation_wires):
        pow_rep = _pow_abstract(
            unitary,
            2**i,
        )
        gate_count[_ctrl_abstract(pow_rep, control_wires=Wire[1])] = 1
    return gate_count


@register_resources(_qpe_decomp_resource)
def _qpe_decomp(unitary, target_wires, estimation_wires):  # pylint: disable=unused-argument
    for w in estimation_wires:
        ops.Hadamard(w)
    for i, w in enumerate(estimation_wires):
        ops.ctrl(qp_pow(unitary, 2 ** (len(estimation_wires) - 1 - i)), w)
    ops.adjoint(QFT(wires=estimation_wires))


add_decomps(QuantumPhaseEstimation, _qpe_decomp)
