# Copyright 2025 Xanadu Quantum Technologies Inc.

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
A transform for decomposing RZ rotations using a phase gradient catalyst state.
"""

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires


def _binary_repr_int(phi, precision):
    # Reasoning for +1e-10 term:
    # due to the division by pi, we obtain 14.999.. instead of 15 for, e.g., (1, 1, 1, 1) pi
    # at the same time, we want to floor off any additional floats when converting to the desired precision,
    # e.g. representing (1, 1, 1, 1) with only 3 digits we want to obtain (1, 1, 1)
    # so overall we floor but make sure we add a little term to not accidentally write 14 when the result is 14.999..
    return int(np.floor(2**precision * phi / (2 * np.pi) + 1e-10))


@QueuingManager.stop_recording()
def _rz_phase_gradient(
    phi: float, wire: Wires, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> Operator:
    """Function that transforms the RZ gate to the phase gradient circuit
    The precision is implicitly defined by the length of ``angle_wires``
    Note that the global phases are collected and added as one big global phase in the main function
    """

    precision = len(angle_wires)
    # BasisEmbedding can handle integer inputs, no need to actually translate to binary
    binary_int = _binary_repr_int(-phi, precision)

    compute_op = qml.ctrl(qml.BasisEmbedding(features=binary_int, wires=angle_wires), control=wire)
    target_op = qml.SemiAdder(angle_wires, phase_grad_wires, work_wires)

    return qml.change_op_basis(compute_op, target_op, compute_op)


@transform
def rz_phase_gradient(
    tape: QuantumScript, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to decompose all instances of :class:`~.RZ` gates into additions
    using a phase gradient resource state.

    For example, an :class:`~.RZ` gate with angle :math:`\phi = (0 \cdot 2^{-1} + 1 \cdot 2^{-2} + 0 \cdot 2^{-3}) 2\pi`
    is translated into the following routine, where the angle is conditionally prepared on the ``angle_wires`` in binary
    and added to a ``phase_grad_wires`` register semi-inplace via :class:`~.SemiAdder`.

    .. code-block::

        target: ─RZ(ϕ)─ = ────╭●──────────────╭●────exp(iϕ/2)─┤
         ang_0:           ────├|0⟩─╭SemiAdder─├|0⟩────────────┤
         ang_1:           ────├|1⟩─├SemiAdder─├|1⟩────────────┤
         ang_2:           ────╰|0⟩─├SemiAdder─╰|0⟩────────────┤
         phg_0:           ─────────├SemiAdder─────────────────┤
         phg_1:           ─────────├SemiAdder─────────────────┤
         phg_2:           ─────────╰SemiAdder─────────────────┤

    For this routine to work, the provided ``phase_grad_wires`` need to hold a phase gradient
    state :math:`|\nabla Z\rangle = \frac{1}{\sqrt{2^n}} \sum_{m=0}^{2^n-1} e^{2 \pi i \frac{m}{2^n}} |m\rangle`.
    Because this state is not modified and can be re-used at a later stage, the transform does not prepare it but
    rather assumes it has been prepared on those wires at an earlier stage.


    Note that :class:`~.SemiAdder` requires additional ``work_wires`` (not shown in the diagram) for the semi-in-place addition
    :math:`\text{SemiAdder}|x\rangle_\text{ang} |y\rangle_\text{phg} = |x\rangle_\text{ang} |x + y\rangle_\text{phg}`.

    More details can be found on page 4 in `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`__
    and Figure 17a in `arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__ (a generalization to
    multiplexed :class:`~.RZ` rotations is provided in Figure 4 in
    `arXiv:2409.07332 <https://arxiv.org/abs/2409.07332>`__).

    Note that technically, this circuit realizes :class:`~.PhaseShift`, i.e. :math:`R_\phi(\phi) = R_Z(\phi) e^{i\phi/2}`.
    The additional global phase is taken into account in the decomposition.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit containing :class:`~.RZ` gates.
        angle_wires (Wires): The qubits that conditionally load the angle :math:`\phi` of
            the :class:`~.RZ` gate in binary as a multiple of :math:`2\pi`.
            The length of the ``angle_wires`` implicitly determines the precision
            with which the angle is represented.
            E.g., :math:`(2^{-1} + 2^{-2} + 2^{-3}) 2\pi` is exactly represented by three bits as ``111``.
        phase_grad_wires (Wires): The catalyst qubits with a phase gradient state prepared on them.
            Needs to be at least the length of ``angle_wires`` and will only
            use the first ``len(angle_wires)``.
        work_wires (Wires): Additional work wires to realize the :class:`~.SemiAdder` between the ``angle_wires`` and
            ``phase_grad_wires``. Needs to be at least ``b-1`` wires, where ``b=len(phase_grad_wires)`` is
            the precision of the angle :math:`\phi`.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    .. code-block:: python

        from functools import partial

        import numpy as np

        import pennylane as qml
        from pennylane.transforms.rz_phase_gradient import rz_phase_gradient

        precision = 3
        phi = (1 / 2 + 1 / 4 + 1 / 8) * 2 * np.pi
        wire = "targ"
        angle_wires = [f"ang_{i}" for i in range(precision)]
        phase_grad_wires = [f"phg_{i}" for i in range(precision)]
        work_wires = [f"work_{i}" for i in range(precision - 1)]
        wire_order = [wire] + angle_wires + phase_grad_wires + work_wires


        def phase_gradient(wires):
            # prepare phase gradient state
            qml.X(wires[-1])
            qml.QFT(wires)


        @partial(
            rz_phase_gradient,
            angle_wires=angle_wires,
            phase_grad_wires=phase_grad_wires,
            work_wires=work_wires,
        )
        @qml.qnode(qml.device("default.qubit"))
        def rz_circ(phi, wire):
            phase_gradient(phase_grad_wires)  # prepare phase gradient state

            qml.Hadamard(wire)  # transform rotation
            qml.RZ(phi, wire)
            qml.Hadamard(wire)  # transform rotation

            return qml.probs(wire)


    In this example we perform the rotation of an angle of :math:`\phi = (0.111)_2 2\pi`. Because phase shifts
    are trivial on computational basis states, we transform the :math:`R_Z` rotation to :math:`R_X = H R_Z H` via two
    :class:`~.Hadamard` gates.

    Note that for the transform to work, we need to also prepare a phase gradient state on the ``phase_grad_wires``.

    Overall, the full circuit looks like the following:

    >>> print(qml.draw(rz_circ, wire_order=wire_order)(phi, wire))
      targ: ──H──────╭(|Ψ⟩)@SemiAdder@(|Ψ⟩)──H─╭GlobalPhase(2.75)─┤  Probs
     ang_0: ─────────├(|Ψ⟩)@SemiAdder@(|Ψ⟩)────├GlobalPhase(2.75)─┤
     ang_1: ─────────├(|Ψ⟩)@SemiAdder@(|Ψ⟩)────├GlobalPhase(2.75)─┤
     ang_2: ─────────├(|Ψ⟩)@SemiAdder@(|Ψ⟩)────├GlobalPhase(2.75)─┤
     phg_0: ────╭QFT─├(|Ψ⟩)@SemiAdder@(|Ψ⟩)────├GlobalPhase(2.75)─┤
     phg_1: ────├QFT─├(|Ψ⟩)@SemiAdder@(|Ψ⟩)────├GlobalPhase(2.75)─┤
     phg_2: ──X─╰QFT─├(|Ψ⟩)@SemiAdder@(|Ψ⟩)────├GlobalPhase(2.75)─┤
    work_0: ─────────├(|Ψ⟩)@SemiAdder@(|Ψ⟩)────├GlobalPhase(2.75)─┤
    work_1: ─────────╰(|Ψ⟩)@SemiAdder@(|Ψ⟩)────╰GlobalPhase(2.75)─┤

    The additional work wires are required by the :class:`~.SemiAdder`.
    Executing the circuit, we get the expected result:

    >>> rz_circ(phi, wire)
    array([0.85355339, 0.14644661])

    """

    if len(phase_grad_wires) < len(angle_wires):
        raise ValueError(
            f"phase_grad_wires needs to be at least as large as angle_wires. Got {len(phase_grad_wires)} phase_grad_wires, which is fewer than the {len(angle_wires)} angle wires."
        )

    operations = []
    global_phases = []
    for op in tape.operations:
        if isinstance(op, qml.RZ):
            wire = op.wires
            phi = op.parameters[0]
            global_phases.append(phi / 2)

            operations.append(
                _rz_phase_gradient(
                    phi,
                    wire,
                    angle_wires=angle_wires,
                    phase_grad_wires=phase_grad_wires,
                    work_wires=work_wires,
                )
            )
        else:
            operations.append(op)

    operations.append(qml.GlobalPhase(sum(global_phases)))

    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
