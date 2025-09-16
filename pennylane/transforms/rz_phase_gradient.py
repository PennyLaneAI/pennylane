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
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires


def _binary_repr_int(phi, precision):
    # reasoning for +1e-10 term:
    # due to the devision by pi, we obtain 14.999.. instead of 15 for, e.g., (1, 1, 1, 1) pi
    # at the same time, we want to floor off any additional floats when converting to the desired precision,
    # e.g. representing (1, 1, 1, 1) with only 3 digits we want to obtain (1, 1, 1)
    # so overall we floor but make sure we add a little term to not accidentally write 14 when the result is 14.999..
    return int(np.floor(2**precision * phi / (2 * np.pi) + 1e-10))


@QueuingManager.stop_recording()
def _rz_phase_gradient(
    RZ_op: qml.RZ, aux_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Function that transforms the RZ gate to the phase gradient circuit

    The precision is implicitly defined by the length of ``aux_wires``"""
    wire = RZ_op.wires
    phi = -1 * RZ_op.parameters[0]
    precision = len(aux_wires)
    binary_int = _binary_repr_int(
        phi, precision
    )  # BasisEmbedding can handle integer inputs, no need to actually translate to binary
    ops = [
        qml.ctrl(qml.BasisEmbedding(features=binary_int, wires=aux_wires), control=wire),
        qml.SemiAdder(aux_wires, phase_grad_wires, work_wires),
        qml.ctrl(qml.BasisEmbedding(features=binary_int, wires=aux_wires), control=wire),
        qml.GlobalPhase(-phi / 2),
    ]
    return ops


@transform
def rz_phase_gradient(
    tape: QuantumScript, aux_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to decompose all instances of :class:`~RZ` gates into additions
    using a phase gradient resource state.

    For example, a :class:`~.RZ` gate with angle $\phi = (0 \cdot 2^{-1} + 1 \cdot 2^{-2} + 0 \cdot 2^{-3}) 2\pi$
    is translated into the following routine, where the angle is conditionally prepared on the ``aux_wires`` in binary
    and added to a ``phase_grad_wires`` register semi-inplace via :class:`~.SemiAdder`.

    .. code-block::

        target: ─RZ(ϕ)─ = ────╭●──────────────╭●────────┤
         aux_0:           ────├|0⟩─╭SemiAdder─├|0⟩──────┤
         aux_1:           ────├|1⟩─├SemiAdder─├|1⟩──────┤
         aux_2:           ────╰|0⟩─├SemiAdder─╰|0⟩──────┤
         phg_0:           ─────────├SemiAdder───────────┤
         phg_1:           ─────────├SemiAdder───────────┤
         phg_2:           ─────────╰SemiAdder───────────┤

    For this routine to work, the provided ``phase_gradient_wires`` need to hold a phase gradient
    state :math:`|\nabla Z\rangle = \frac{1}{\sqrt{2^n}} \sum_{m=0}^{2^n-1} e^{2 \pi i \frac{m}{2^n}} |m\rangle`.
    The state is not modified and can be re-used at a later stage.
    It is important to stress that this transform does not prepare the state.


    Note that :class`~SemiAdder` we requires additional ``work_wires`` (not shown in the diagram) for the semi-in-place addition
    :math:`\text{SemiAdder}|x\rangle_\text{aux} |y\rangle_\text{qft} = |x\rangle_\text{aux} |x + y\rangle_\text{qft}`.

    More details can be found on page 4 in `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`__
    and Figure 17a in `arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__ (a generalization to
    multiplexed :class:`~RZ` rotations is provided in Figure 4 in
    `arXiv:2409.07332 <https://arxiv.org/abs/2409.07332>`__).

    Note that, technically, this circuit realizes :class:`~PhaseShift`, i.e. :math:`R_\phi(\phi) = R_(\phi) e^{\phi/2}`.
    The additional global phase is taken into account in the decomposition.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit containing :class:`~RZ` gates.
        aux_wires (Wires): The auxiliary qubits that conditionally load the angle :math:`\phi` of
            the :class:`~RZ` gate in binary as a multiple of :math:`2\pi`.
            The length of the ``aux_wires`` implicitly determine the precision
            with which the angle is represented.
            E.g., :math:`(2^{-1} + 2^{-2} + 2^{-3}) * 2\pi` is exactly represented by three bits as ``111``.
        phase_grad_wires (Wires): The catalyst qubits with a phase gradient state prepared on them. Will only use the first ``len(aux_wires)`` according to the precision with which the angle is decomposed.
        work wires (Wires): Additional work wires to realize the :class`~SemiAdder` between the ``aux_wires`` and
            ``phase_grad_wires``. Needs to be at least ``b-1`` wires, where ``b`` is the number of
            phase gradient wires, hence the precision of the angle :math:`\phi`.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    .. code-block:: python

        import pennylane as qml
        import numpy as np
        from functools import partial
        from pennylane.transforms.rz_phase_gradient import rz_phase_gradient

        precision = 3
        phi = (1/2 + 1/4 + 1/8) * 2 * np.pi
        wire="targ"
        aux_wires = [f"aux_{i}" for i in range(precision)]
        phase_grad_wires = [f"phg_{i}" for i in range(precision)]
        work_wires = [f"work_{i}" for i in range(precision-1)]
        wire_order = [wire] + aux_wires + phase_grad_wires + work_wires

        def phase_gradient(wires):
            # prepare phase gradient state
            qml.X(wires[-1])
            qml.QFT(wires)

        @partial(qml.transforms.rz_phase_gradient, aux_wires=aux_wires, phase_grad_wires=phase_grad_wires, work_wires=work_wires)
        @qml.qnode(qml.device("default.qubit"))
        def rz_circ(phi, wire):
            phase_gradient(phase_grad_wires) # prepare phase gradient state

            qml.Hadamard(wire) # transform rotation
            qml.RZ(phi, wire)
            qml.Hadamard(wire) # transform rotation

            return qml.probs(wire)

    In this example we perform the rotation of an angle of :math:`\phi = (0.111)_2 2\pi`. Because phase shifts
    are trivial on computational basis states, we transform the :math:`R_Z` rotation to `R_X = H R_Z H` via two
    :class:`~.Hadamard` gates.

    Note that for the transform to work, we need to also prepare a phase gradient state on the ``phase_grad_wires``.

    Overall, the full circuit looks like the following:

    >>> print(qml.draw(rz_circ, wire_order=wire_order)(phi, wire))
      targ: ──H─╭●──────────────╭●───╭GlobalPhase(2.75)──H─┤  Probs
     aux_0: ────├|Ψ⟩─╭SemiAdder─├|Ψ⟩─├GlobalPhase(2.75)────┤
     aux_1: ────├|Ψ⟩─├SemiAdder─├|Ψ⟩─├GlobalPhase(2.75)────┤
     aux_2: ────╰|Ψ⟩─├SemiAdder─╰|Ψ⟩─├GlobalPhase(2.75)────┤
     phg_0: ────╭QFT─├SemiAdder──────├GlobalPhase(2.75)────┤
     phg_1: ────├QFT─├SemiAdder──────├GlobalPhase(2.75)────┤
     phg_2: ──X─╰QFT─├SemiAdder──────├GlobalPhase(2.75)────┤
    work_0: ─────────├SemiAdder──────├GlobalPhase(2.75)────┤
    work_1: ─────────╰SemiAdder──────╰GlobalPhase(2.75)────┤

    The additional work wires are required by the :class:`~.SemiAdder`.
    Executing the circuit, we get the expected result:

    >>> rz_circ(phi, wire)
    array([0.85355339, 0.14644661])

    """
    operations = []
    for op in tape.operations:
        if isinstance(op, qml.RZ):
            operations.extend(
                _rz_phase_gradient(
                    op,
                    aux_wires=aux_wires,
                    phase_grad_wires=phase_grad_wires,
                    work_wires=work_wires,
                )
            )
        else:
            operations.append(op)

    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
