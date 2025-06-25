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
Transforms lowering gates and series of gates involving relative phases.
Transformations fist published in:

Amy, M. and Ross, N. J., “Phase-state duality in reversible circuit design”,
Physical Review A, vol. 104, no. 5, Art. no. 052602, APS, 2021. doi:10.1103/PhysRevA.104.052602.
"""

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.transforms import pattern_matching_optimization, transform
from pennylane.typing import PostprocessingFn

relative_phases_patterns = [
    [
        qml.CCZ([0, 1, 3]),
        qml.ctrl(qml.S(1), control=[0]),
        qml.ctrl(qml.S(2), control=[0, 1]),
        qml.MultiControlledX([0, 1, 2, 3]),
        # ------------
        qml.Hadamard(3),
        qml.T(3),
        qml.CNOT([2, 3]),
        qml.adjoint(qml.T(3)),
        qml.Hadamard(3),
        qml.T(3),
        qml.CNOT([1, 3]),
        qml.adjoint(qml.T(3)),
        qml.CNOT([0, 3]),
        qml.T(3),
        qml.CNOT([1, 3]),
        qml.adjoint(qml.T(3)),
        qml.CNOT([0, 3]),
        qml.Hadamard(3),
        qml.T(3),
        qml.CNOT([2, 3]),
        qml.adjoint(qml.T(3)),
        qml.Hadamard(3),
    ],
]


@transform
def replace_relative_phase_toffoli(
    tape: QuantumScript,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to replace 4-qubit relative phase toffoli gates, given in
    figure three of (Amy, M. and Ross, N. J., 2021).

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    The transform can be applied on :class:`QNode` directly.

    .. code-block:: python

        @replace_relative_phase_toffoli
        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.X(0)

            # begin relative phase 4-qubit Toffoli

            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])

            # end relative phase 4-qubit Toffoli

            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.X(0)
            return qml.expval(qml.Z(0))

    The relative phase 4-qubit Toffoli is then replaced before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc(x, y):
                qml.CCZ(wires=[0, 1, 3])
                qml.ctrl(qml.S(wires=[1]), control=[0])
                qml.ctrl(qml.S(wires=[2]), control=[0, 1])
                qml.MultiControlledX(wires=[0, 1, 2, 3])
                return qml.expval(qml.Z(0))

        The circuit before decomposition:

        >>> dev = qml.device('default.qubit', wires=4)
        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)())
            0: ─╭●─╭●─╭●─╭●─┤  <Z>
            1: ─├●─╰S─├●─├●─┤
            2: ─│─────╰S─├●─┤
            3: ─╰Z───────╰X─┤

        We can replace the relative phase 4-qubit Toffoli by running the transform:

        >>> lowered_qfunc = replace_relative_phase_toffoli(qfunc)
        >>> lowered_qnode = qml.QNode(lowered_qfunc, dev)
        >>> print(qml.draw(lowered_qnode)())

        0: ─────────────────╭●───────────╭●───────────────────────────┤  <Z>
        1: ─────────────────│─────╭●─────│─────╭●─────────────────────┤
        2: ───────╭●────────│─────│──────│─────│────────────╭●────────┤
        3: ──H──T─╰X──T†──H─╰X──T─╰X──T†─╰X──T─╰X──T†──H──T─╰X──T†──H─┤

    """
    pattern_ops = relative_phases_patterns[0]
    pattern = QuantumTape(pattern_ops)
    return pattern_matching_optimization(tape, pattern_tapes=[pattern])


@transform
def replace_controlled_iX_gate(
    tape: QuantumScript, num_controls=1
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to replace controlled iX gates. An iX gate is a CS and a Toffoli. The
    equivalency used is given in figure two of (Amy, M. and Ross, N. J., 2021) and the simple case
    of one num_controls=1 in given in figure one.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    The transform can be applied on :class:`QNode` directly.

    .. code-block:: python

        @replace_iX_gate
        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.X(0)

            # begin multi-controlled iX gate

            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])

            # end multi-controlled iX gate

            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.X(0)
            return qml.expval(qml.Z(0))

    The relative multi-controlled iX gate (CS, Toffoli) is then replaced before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc(x, y):
                qml.ctrl(qml.S(wires=[2]), control=[0, 1])
                qml.MultiControlledX(wires=[0, 1, 2, 3])
                return qml.expval(qml.Z(0))

        The circuit before decomposition:

        >>> dev = qml.device('default.qubit', wires=4)
        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)())
            0: ─╭●─╭●─┤  <Z>
            1: ─├●─├●─┤
            2: ─╰S─├●─┤
            3: ────╰X─┤

        We can replace the multi-controlled iX gate by running the transform:

        >>> lowered_qfunc = replace_multi_controlled_iX_gate(qfunc)
        >>> lowered_qnode = qml.QNode(lowered_qfunc, dev)
        >>> print(qml.draw(lowered_qnode)())

        0: ──────────────╭●───────────╭●────┤  <Z>
        1: ──────────────├●───────────├●────┤
        2: ────────╭●────│──────╭●────│─────┤
        3: ──H──T†─╰X──T─╰X──T†─╰X──T─╰X──H─┤

    """
    if num_controls < 1:
        raise ValueError(
            "There must be at least one control wire for the controlled iX gate decomposition."
        )
    pattern_ops = [
        qml.ctrl(qml.S(num_controls), control=list(range(num_controls))),
        qml.MultiControlledX(list(range(num_controls + 2))),
        # ------------
        qml.Hadamard(num_controls + 1),
        qml.MultiControlledX(list(range(num_controls)) + [num_controls + 1]),
        qml.adjoint(qml.T(num_controls + 1)),
        qml.CNOT([num_controls, num_controls + 1]),
        qml.T(num_controls + 1),
        qml.MultiControlledX(list(range(num_controls)) + [num_controls + 1]),
        qml.adjoint(qml.T(num_controls + 1)),
        qml.CNOT([num_controls, num_controls + 1]),
        qml.T(num_controls + 1),
        qml.Hadamard(num_controls + 1),
    ]
    pattern = QuantumTape(pattern_ops)
    return pattern_matching_optimization(tape, pattern_tapes=[pattern])
