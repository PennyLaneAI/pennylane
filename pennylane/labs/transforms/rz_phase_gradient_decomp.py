# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Decomposition rule for RZ in terms of `phase gradient states <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__
"""

import pennylane as qml
from pennylane.decomposition import change_op_basis_resource_rep, controlled_resource_rep
from pennylane.transforms.rz_phase_gradient import _rz_phase_gradient


def make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires):
    r"""
    Custom decomposition rule for :class:`~RZ` gates

    This is a temporary workaround as long as capture does not work, which blocks usage of dynamic allocation.
    Here, we explicitly provide the necessary wires for the `phase gradient decomposition of RZ <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__.
    This way, this function can be used in a workflow context that explicitly uses those wires to generate this decomposition rule, which can then be used
    as ``alt_decomps`` or ``fixed_decomp`` within :func:`~decompose`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state
        work_wires (Wires): additional work wires for :class:`~SemiAdder` decomposition

    Returns:
        func: decomposition rule to be used within :func:`~decompose`.

    .. seealso:: :func:`~make_selectpaulirot_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~RZ` gate using the custom decomposition rule
    that we generate from within the context of the example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.labs.transforms import make_rz_to_phase_gradient_decomp
        import numpy as np

        seed = 0

        qp.decomposition.enable_graph()

        prec = 3
        phi = (1/2 + 1/4 + 1/8) * 2 * np.pi # binary rep is (111)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)

        @qp.transforms.decompose(
                gate_set={"C(BasisEmbedding)", "SemiAdder", "CNOT", "GlobalPhase"},
                fixed_decomps={qp.RZ: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].gate_types

    The resulting circuit corresponds to the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__ of RZ,
    containing two CNOT fanouts corresponding to the binary representation of the angle (111 in this case), the :class:`~SemiAdder`, and a :class:`~GlobalPhase`.

    >>> specs
    {'GlobalPhase': 1, 'C(BasisEmbedding)': 2, 'SemiAdder': 1}
    >>> print(qp.draw(circuit)())
         0: ─╭GlobalPhase(2.75)─╭●──────────────╭●───┤  State
     aux_0: ─├GlobalPhase(2.75)─├|Ψ⟩─╭SemiAdder─├|Ψ⟩─┤  State
     aux_1: ─├GlobalPhase(2.75)─├|Ψ⟩─├SemiAdder─├|Ψ⟩─┤  State
     aux_2: ─├GlobalPhase(2.75)─╰|Ψ⟩─├SemiAdder─╰|Ψ⟩─┤  State
     qft_0: ─├GlobalPhase(2.75)──────├SemiAdder──────┤  State
     qft_1: ─├GlobalPhase(2.75)──────├SemiAdder──────┤  State
     qft_2: ─├GlobalPhase(2.75)──────├SemiAdder──────┤  State
    work_0: ─├GlobalPhase(2.75)──────├SemiAdder──────┤  State
    work_1: ─╰GlobalPhase(2.75)──────╰SemiAdder──────┤  State

    """
    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    def _resource_fn():
        # rz decomposition costs, using information about angle_wires etc from the outer scope
        target_op = qml.resource_rep(qml.SemiAdder, num_y_wires=len(phase_grad_wires))
        compute_op = uncompute_op = controlled_resource_rep(
            qml.BasisEmbedding,
            base_params={"num_wires": len(angle_wires)},
            num_control_wires=1,
            num_zero_control_values=0,
        )
        change_basis_rep = change_op_basis_resource_rep(compute_op, target_op, uncompute_op)

        return {change_basis_rep: 1, qml.resource_rep(qml.GlobalPhase): 1}

    @qml.register_resources(_resource_fn)
    def _decomp_fn(phi, wires):
        target_wire = wires
        qml.GlobalPhase(phi / 2)

        pg_op = _rz_phase_gradient(phi, target_wire, **kwargs)
        qml.apply(pg_op)  # because _rz_phase_gradient is in non-queing context

    return _decomp_fn
