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
Transformations first published in:

Maslov, Dmitri. "On the Advantages of Using Relative Phase Toffolis with an Application to
Multiple Control Toffoli Optimization", arXiv:1508.03273, arXiv, 2016.
doi:10.48550/arXiv.1508.03273.

Giles, Brett, and Peter Selinger. "Exact Synthesis of Multiqubit Clifford+T Circuits",
arXiv:1212.0506, arXiv, 2013. doi:10.48550/arXiv.1212.0506.
"""

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .pattern_matching import pattern_matching_optimization


@transform
def match_relative_phase_toffoli(
    tape: QuantumScript,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to replace 4-qubit relative phase Toffoli gates, given in
    Maslov, Dmitri. "On the Advantages of Using Relative Phase Toffolis with an Application to
    Multiple Control Toffoli Optimization", arXiv:1508.03273, arXiv, 2016.
    `doi:10.48550/arXiv.1508.03273 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.022311>`_.

    .. note::

        Will also replace any subcircuits from the full pattern (composed of the 4-qubit relative phase Toffoli
        and its decomposition) that can replaced by the rest of the pattern.

    Args:
        tape (QNode or QuantumScript or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumScript], function]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    .. code-block:: python

        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

    The circuit (a 4-qubit relative phase Toffoli) before decomposition:

    >>> dev = qml.device('default.qubit', wires=4)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> lowered_qnode = match_relative_phase_toffoli(qnode)
    >>> print(qml.draw(lowered_qnode, level=0)())
        0: ─╭●─╭●─╭●─╭●─┤  <Z>
        1: ─├●─╰S─├●─├●─┤
        2: ─│─────╰S─├●─┤
        3: ─╰Z───────╰X─┤

    We can replace the relative phase 4-qubit Toffoli by running the transform:

    >>> print(qml.draw(lowered_qnode, level=1)())
        0: ─────────────────╭●───────────╭●───────────────────────────┤  <Z>
        1: ─────────────────│─────╭●─────│─────╭●─────────────────────┤
        2: ───────╭●────────│─────│──────│─────│────────────╭●────────┤
        3: ──H──T─╰X──T†──H─╰X──T─╰X──T†─╰X──T─╰X──T†──H──T─╰X──T†──H─┤

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            @match_relative_phase_toffoli
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

    """
    pattern_ops = [
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
    ]
    pattern = QuantumScript(pattern_ops)
    return pattern_matching_optimization(tape, pattern_tapes=[pattern])


@transform
def match_controlled_iX_gate(
    tape: QuantumScript, num_controls=1
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to replace controlled iX gates. An iX gate is a controlled-S and a Toffoli. The
    equivalency used is given in Giles, Brett, and Peter Selinger. "Exact Synthesis of Multiqubit Clifford+T Circuits",
    arXiv:1212.0506, arXiv, 2013. `doi:10.48550/arXiv.1212.0506 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.032332>`_.

    .. note::

        Will also replace any subcircuits from the full pattern (composed of the controlled iX gate
        and its decomposition) that can replaced by the rest of the pattern.

    Args:
        tape (QNode or QuantumScript or Callable): A quantum circuit.
        num_controls (int): The number of controls on the CS gate.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumScript], function]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

    The circuit (a controlled iX gate) before decomposition:

    >>> dev = qml.device('default.qubit', wires=4)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> lowered_qnode = match_controlled_iX_gate(qnode, 2)
    >>> print(qml.draw(lowered_qnode, level=0)())
        0: ─╭●─╭●─┤  <Z>
        1: ─├●─├●─┤
        2: ─╰S─├●─┤
        3: ────╰X─┤

    We can replace the multi-controlled iX gate by running the transform:

    >>> print(qml.draw(lowered_qnode, level=1)())
        0: ──────────────╭●───────────╭●────┤  <Z>
        1: ──────────────├●───────────├●────┤
        2: ────────╭●────│──────╭●────│─────┤
        3: ──H──T†─╰X──T─╰X──T†─╰X──T─╰X──H─┤

    .. details::
        :title: Usage Details

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
    pattern = QuantumScript(pattern_ops)
    return pattern_matching_optimization(tape, pattern_tapes=[pattern])
