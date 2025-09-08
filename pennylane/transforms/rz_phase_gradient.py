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
A transform for decomposing an RZ rotation using a phase gradient catalyst state
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
    return int(np.floor(2**precision * phi / np.pi + 1e-10))


def _rz_phase_gradient(
    RZ_op: qml.RZ, aux_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Function that transforms the RZ gate to the phase gradient circuit

    The precision is implicitly defined by the length of ``aux_wires``"""
    wire = RZ_op.wires
    phi = RZ_op.parameters[0]
    precision = len(aux_wires)
    binary_int = _binary_repr_int(
        phi, precision
    )  # BasisEmbedding can handle integer inputs, no need to actually translate to binary
    ops = [
        qml.ctrl(qml.BasisEmbedding(features=binary_int, wires=aux_wires), control=wire),
        qml.SemiAdder(aux_wires, phase_grad_wires, work_wires),
        qml.ctrl(qml.BasisEmbedding(features=binary_int, wires=aux_wires), control=wire),
    ]
    return ops


@transform
def rz_phase_gradient(
    tape: QuantumScript, aux_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to decomposes all instances of :class:`~RZ` gates into additions
    using a phase gradient resource state.

    This method is described on page 4 in `1709.06648 <https://arxiv.org/abs/1709.06648>`__
    and Figure 17a in `2211.15465 <https://arxiv.org/abs/2211.15465>`__ (a generalization to
    multiplexed :class:`~RZ` rotations is provided in Figure 4 in
    `2409.07332 <https://arxiv.org/abs/2409.07332>`__).

    The routine looks as follows.

    .. code-block::

        alghorithm level
          targ: ────╭●──────────────╭●────────┤
         aux_0: ────├|0⟩─╭SemiAdder─├|0⟩──────┤
         aux_1: ────├|1⟩─├SemiAdder─├|1⟩──────┤
         aux_2: ────╰|0⟩─├SemiAdder─╰|0⟩──────┤
         qft_0: ────╭QFT─├SemiAdder───────────┤
         qft_1: ────├QFT─├SemiAdder───────────┤
         qft_2: ──X─╰QFT─├SemiAdder───────────┤
        work_0: ─────────├SemiAdder───────────┤
        work_1: ─────────╰SemiAdder───────────┤

    The ``targ`` qubit is where :class:`~RZ` is applied. The ``aux`` qubits
    conditionally load the binary representation of the angle :math:`\phi = (0.010)_2 \pi`. The ``qft`` qubits
    hold a phase gradient state :math:`\text{QFT}|0..01\rangle = \frac{1}{\sqrt{2^n}} \sum_{m=0}^{2^n-1} e^{2 \pi i \frac{m}{2^n}} |m\rangle`.
    This state is not modified and can be re-used at a later stage. Because we
    are using :class`~SemiAdder`, we require additional ``work_wires`` wires for the semi-in-place addition
    :math:`\text{SemiAdder}|x\rangle_\text{aux} |y\rangle_\text{qft} = |x\rangle_\text{aux} |x + y\rangle_\text{qft}`.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit containing :class:`~RZ` gates.
        aux_wires (Wires): The auxiliary qubits that conditionally load the angle :math:`\phi` of the :class:`~RZ` gate.
        phase_grad_wires (Wires): The catalyst qubits with a phase gradient state prepared on them
        work wires (Wires): Additional work wires to realize the :class`~SemiAdder` between the ``aux_wires`` and
            ``phase_grad_wires``. Needs to be at least ``b-1`` wires, where ``b`` is the number of aux wires and
            phase gradient wires, hence the precision of the angle :math:`\phi`.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    asd

    .. details::
        :title: Usage Details

        asd
    """
    operations = []
    for op in tape.operations:
        if isinstance(op, qml.RZ):
            with QueuingManager.stop_recording():
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
