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

"""Tests the decomp_inspector transform."""

from textwrap import dedent

import pytest

import pennylane as qp
from pennylane.transforms import decomp_inspector


@pytest.mark.usefixtures("disable_graph_decomposition")
def test_error_raised_graph_disabled():
    """Tests that an error is raised if graph is disabled."""

    @decomp_inspector
    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        qp.CRX(0.5, [0, 1])
        return qp.probs()

    with pytest.raises(ValueError, match="only relevant with the new graph-based decomposition"):
        circuit()


# pylint: disable=protected-access
@pytest.mark.usefixtures("enable_graph_decomposition")
class TestInspectDecompGraph:
    """Tests the decomp_inspector transform."""

    def test_non_existent_op(self):
        """Tests that the correct message is produced for a non existent op."""

        @decomp_inspector(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.CRX(0.5, wires=[0, 1])
            return qp.probs()

        inspector = circuit()
        result = inspector.inspect_decomps(qp.CRY(0.5, wires=[0, 1]))

        expected = (
            "This operator is not found in the decomposition graph! This typically "
            "means that this operator was not part of the original circuit, nor is it "
            "produced by any of the operators' decomposition rules."
        )
        assert str(result) == expected
        assert result._repr_markdown_() == expected

    def test_op_type_error(self):
        """Tests that a proper error is raised when an operator type is provided."""

        @decomp_inspector(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.CRX(0.5, wires=[0, 1])
            return qp.probs()

        inspector = circuit()
        with pytest.raises(TypeError, match="takes a concrete operator instance as"):
            inspector.inspect_decomps(qp.CRX)

    def test_work_wire_budget_mismatch(self):
        """Tests that a correct message is produced when the work wire budget is wrong."""

        @decomp_inspector(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT, num_work_wires=1)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.MultiControlledX([0, 1, 2, 3, 4])
            return qp.probs()

        inspector = circuit()
        expected = (
            "The decomposition graph was solved with 1 work wires available for dynamic "
            "allocation at the top level. There is not a point where a MultiControlledX("
            "num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type"
            "=borrowed) is decomposed with a dynamic allocation budget of 2."
        )
        result = inspector.inspect_decomps(qp.MultiControlledX([0, 1, 2, 3, 4]), num_work_wires=2)
        assert str(result) == expected
        assert result._repr_markdown_() == expected

    def test_work_wire_budget(self):
        """Tests that the correct output is produced according to the work wire budget."""

        @decomp_inspector(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT, num_work_wires=0)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5])
            return qp.probs()

        inspector = circuit()

        op = qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5])
        result = inspector.inspect_decomps(op)
        assert str(result) == dedent("""
            Decomposition 0 (name: flip_zero_ctrl_values(_ctrl_single_work_wire))
            Insufficient work wires: requires 1 but only 0 available.

            Decomposition 1 (name: to_controlled_qubit_unitary)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            CHOSEN: Decomposition 2 (name: controlled(_multi_rz_decomposition))
            0: ─╭X─╭RZ(0.50)─╭X─┤  
            1: ─├●─│─────────├●─┤  
            3: ─├●─├●────────├●─┤  
            4: ─├●─├●────────├●─┤  
            5: ─╰●─╰●────────╰●─┤  
            First-Level Expansion Gates: {Controlled(RZ, num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1, MultiControlledX(num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 2}
            Full Expansion Gates: {GlobalPhase: 88, RZ: 136, CNOT: 160, RY: 28, RX: 8}
            Weighted Cost: 332.0
            """).strip()

        assert result._repr_markdown_() == dedent("""
            #### Decomposition 0 (name: flip_zero_ctrl_values(_ctrl_single_work_wire))

            _Insufficient work wires: requires 1 but only 0 available._

            ---

            #### Decomposition 1 (name: to_controlled_qubit_unitary)

            _Not applicable (provided operator instance does not meet all conditions for this rule)._

            ---

            #### **CHOSEN:** Decomposition 2 (name: controlled(_multi_rz_decomposition))

            ```
            0: ─╭X─╭RZ(0.50)─╭X─┤  
            1: ─├●─│─────────├●─┤  
            3: ─├●─├●────────├●─┤  
            4: ─├●─├●────────├●─┤  
            5: ─╰●─╰●────────╰●─┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Count |
            | :--- | :--- |
            | Controlled(RZ, num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed) | 1 |
            | MultiControlledX(num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed) | 2 |

            | Full Expansion | Count |
            | :--- | :--- |
            | CNOT | 160 |
            | GlobalPhase | 88 |
            | RX | 8 |
            | RY | 28 |
            | RZ | 136 |
            | **Weighted Cost** | 332.0 |
            </details>
            """).strip()

    def test_work_wires_available(self):
        """Tests that the correct output is produced when there are available work wires."""

        @decomp_inspector(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT, num_work_wires=2)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5])
            return qp.probs()

        inspector = circuit()

        op = qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5])
        result = inspector.inspect_decomps(op, num_work_wires=2)
        assert str(result) == dedent("""
            CHOSEN: Decomposition 0 (name: flip_zero_ctrl_values(_ctrl_single_work_wire))
            0: ──────────╭MultiRZ(0.50)───────┤  
            1: ──────────├MultiRZ(0.50)───────┤  
            3: ───────╭●─│──────────────╭●────┤  
            4: ───────├●─│──────────────├●────┤  
            5: ───────├●─│──────────────├●────┤  
                 |0>├─╰X─╰●─────────────╰X──┤    
            First-Level Expansion Gates: {MultiControlledX(num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 2, Controlled(MultiRZ(num_wires=2), num_control_wires=1, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1}
            Wire Allocations: {'zero': 1}
            Full Expansion Gates: {RZ: 58, CNOT: 34, GlobalPhase: 64, RY: 18, RX: 8, MidMeasure: 2}
            Weighted Cost: 120.0

            Decomposition 1 (name: to_controlled_qubit_unitary)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 2 (name: controlled(_multi_rz_decomposition))
            0: ─╭X─╭RZ(0.50)─╭X─┤  
            1: ─├●─│─────────├●─┤  
            3: ─├●─├●────────├●─┤  
            4: ─├●─├●────────├●─┤  
            5: ─╰●─╰●────────╰●─┤  
            First-Level Expansion Gates: {Controlled(RZ, num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1, MultiControlledX(num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 2}
            Full Expansion Gates: {GlobalPhase: 76, RX: 16, MidMeasure: 4, RY: 24, RZ: 80, CNOT: 72}
            Weighted Cost: 196.0
            """).strip()

        assert result._repr_markdown_() == dedent("""
            #### **CHOSEN:** Decomposition 0 (name: flip_zero_ctrl_values(_ctrl_single_work_wire))

            ```
            0: ──────────╭MultiRZ(0.50)───────┤  
            1: ──────────├MultiRZ(0.50)───────┤  
            3: ───────╭●─│──────────────╭●────┤  
            4: ───────├●─│──────────────├●────┤  
            5: ───────├●─│──────────────├●────┤  
                 |0>├─╰X─╰●─────────────╰X──┤    
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Count |
            | :--- | :--- |
            | Controlled(MultiRZ(num_wires=2), num_control_wires=1, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed) | 1 |
            | MultiControlledX(num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed) | 2 |

            | Wire Type | Num Allocated |
            | :--- | :--- |
            | zero | 1 |

            | Full Expansion | Count |
            | :--- | :--- |
            | CNOT | 34 |
            | GlobalPhase | 64 |
            | MidMeasure | 2 |
            | RX | 8 |
            | RY | 18 |
            | RZ | 58 |
            | **Weighted Cost** | 120.0 |
            </details>

            ---

            #### Decomposition 1 (name: to_controlled_qubit_unitary)

            _Not applicable (provided operator instance does not meet all conditions for this rule)._

            ---

            #### Decomposition 2 (name: controlled(_multi_rz_decomposition))

            ```
            0: ─╭X─╭RZ(0.50)─╭X─┤  
            1: ─├●─│─────────├●─┤  
            3: ─├●─├●────────├●─┤  
            4: ─├●─├●────────├●─┤  
            5: ─╰●─╰●────────╰●─┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Count |
            | :--- | :--- |
            | Controlled(RZ, num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed) | 1 |
            | MultiControlledX(num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed) | 2 |

            | Full Expansion | Count |
            | :--- | :--- |
            | CNOT | 72 |
            | GlobalPhase | 76 |
            | MidMeasure | 4 |
            | RX | 16 |
            | RY | 24 |
            | RZ | 80 |
            | **Weighted Cost** | 196.0 |
            </details>
            """).strip()

        op = qp.MultiControlledX([0, 1, 2, 3])
        assert str(inspector.inspect_decomps(op, num_work_wires=1)) == dedent("""
            Decomposition 0 (name: flip_zero_ctrl_values(_2cx_elbow_explicit))
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 1 (name: no_workers)
            0: ────╭●───────────────────╭●──────────────────────╭●──────────────────┤  
            1: ────├●───────────────────├●──────────────────────├●──────────────────┤  
            2: ────│─────────╭●─────────│─────────╭●────────────├●──────────────────┤  
            3: ──H─╰X──U(M0)─╰X──U(M0)†─╰X──U(M0)─╰X──U(M0)†──H─╰GlobalPhase(-1.57)─┤  
            M0 = 
            [[ 9.23879533e-01+0.38268343j -5.34910791e-34+0.j        ]
             [ 5.34910791e-34+0.j          9.23879533e-01-0.38268343j]]
            First-Level Expansion Gates: {Hadamard: 2, QubitUnitary(num_wires=1): 2, CNOT: 2, Toffoli: 2, Adjoint(QubitUnitary(num_wires=1)): 2, Controlled(GlobalPhase, num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1}
            Full Expansion Gates: {CNOT: 24, GlobalPhase: 25, RY: 10, RZ: 31, RX: 4}
            Weighted Cost: 69.0

            CHOSEN: Decomposition 2 (name: one_zeroed_worker)
            0: ───────╭●─────●╮────┤  
            1: ───────├●─────●┤────┤  
            2: ───────│──╭●───│────┤  
            3: ───────│──├X───│────┤  
                 |0>├─╰⊕─╰●──⊕╯──┤    
            First-Level Expansion Gates: {Toffoli: 1, TemporaryAND: 1, Adjoint(TemporaryAND): 1}
            Wire Allocations: {'zero': 1}
            Full Expansion Gates: {GlobalPhase: 23, RX: 4, MidMeasure: 1, RY: 7, RZ: 19, CNOT: 10}
            Weighted Cost: 41.0

            Decomposition 3 (name: one_borrowed_worker)
            0: ────╭●────╭●───────┤  
            1: ────├●────├●───────┤  
            2: ────│──╭●─│──╭●────┤  
            3: ────│──├X─│──├X────┤  
                 ├─╰X─╰●─╰X─╰●──┤    
            First-Level Expansion Gates: {Toffoli: 4}
            Wire Allocations: {'any': 1}
            Full Expansion Gates: {CNOT: 24, GlobalPhase: 36, RZ: 36, RY: 8}
            Weighted Cost: 68.0

            Decomposition 4 (name: one_explicit_worker)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 5 (name: two_zeroed_workers)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 6 (name: two_borrowed_workers)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 7 (name: two_explicit_workers)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 8 (name: many_zeroed_workers)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 9 (name: many_borrowed_workers)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 10 (name: many_explicit_workers)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 11 (name: _mcx_to_cnot_or_toffoli)
            Not applicable (provided operator instance does not meet all conditions for this rule).
            """).strip()

    def test_missing_ops(self):
        """Tests that missing operators are correctly reported."""

        @decomp_inspector(gate_set={"RZ", "RX", "CNOT"}, num_work_wires=2)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.PauliRot(0.5, "XYZ", [0, 1, 2])
            return qp.probs()

        inspector = circuit()
        op = qp.PauliRot(0.5, "XYZ", [0, 1, 2])
        result = inspector.inspect_decomps(op)
        assert str(result) == dedent("""
            Decomposition 0 (name: _pauli_rot_decomposition)
            0: ──H────────╭MultiRZ(0.50)──H─────────┤  
            1: ──RX(1.57)─├MultiRZ(0.50)──RX(-1.57)─┤  
            2: ───────────╰MultiRZ(0.50)────────────┤  
            First-Level Expansion Gates: {Hadamard: 2, RX: 2, MultiRZ(num_wires=3): 1}
            Missing Ops: {Hadamard}
            """).strip()

        assert result._repr_markdown_() == dedent("""
            #### Decomposition 0 (name: _pauli_rot_decomposition)

            ```
            0: ──H────────╭MultiRZ(0.50)──H─────────┤  
            1: ──RX(1.57)─├MultiRZ(0.50)──RX(-1.57)─┤  
            2: ───────────╰MultiRZ(0.50)────────────┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Count |
            | :--- | :--- |
            | Hadamard | 2 |
            | MultiRZ(num_wires=3) | 1 |
            | RX | 2 |
            </details>

            | Missing Ops |
            | :--- |
            | Hadamard |
            """).strip()

        assert str(inspector.inspect_decomps(qp.H(0))) == dedent("""
            Decomposition 0 (name: _hadamard_to_rz_ry)
            0: ──RZ(3.14)──RY(1.57)──GlobalPhase(-1.57)─┤  
            First-Level Expansion Gates: {RZ: 1, RY: 1, GlobalPhase: 1}
            Missing Ops: {GlobalPhase}

            Decomposition 1 (name: _hadamard_to_rz_rx)
            0: ──RZ(1.57)──RX(1.57)──RZ(1.57)──GlobalPhase(-1.57)─┤  
            First-Level Expansion Gates: {RZ: 2, RX: 1, GlobalPhase: 1}
            Missing Ops: {GlobalPhase}
            """).strip()

    def test_inexact_count(self):
        """Tests that the output is correct when the gate count is inexact."""

        @decomp_inspector(gate_set={"RZ", "RX", "CNOT", "GlobalPhase"})
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.QubitUnitary([[1, 0], [0, 1]], wires=0)
            return qp.probs()

        inspector = circuit()
        op = qp.QubitUnitary([[1, 0], [0, 1]], wires=0)
        result = inspector.inspect_decomps(op)
        assert str(result) == dedent("""
            Decomposition 0 (name: multi_qubit_decomp_rule)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 1 (name: two_qubit_decomp_rule)
            Not applicable (provided operator instance does not meet all conditions for this rule).

            Decomposition 2 (name: rot)
            0: ──RZ(0.00)─┤  
            Estimated First-Level Expansion Gates: {Rot: 1, RZ: 1, GlobalPhase: 1}
            Actual First-Level Expansion Gates: {RZ: 1}
            Full Expansion Gates: {GlobalPhase: 1, RZ: 5, RX: 1}
            Weighted Cost: 7.0

            Decomposition 3 (name: xyx)
            0: ──RX(0.00)──RY(0.00)──RX(0.00)─┤  
            Estimated First-Level Expansion Gates: {RX: 2, RY: 1, GlobalPhase: 1}
            Actual First-Level Expansion Gates: {RX: 2, RY: 1}
            Full Expansion Gates: {GlobalPhase: 1, RX: 3, RZ: 2}
            Weighted Cost: 6.0

            CHOSEN: Decomposition 4 (name: xzx)
            0: ──RX(0.00)──RZ(0.00)──RX(0.00)─┤  
            Estimated First-Level Expansion Gates: {RX: 2, RZ: 1, GlobalPhase: 1}
            Actual First-Level Expansion Gates: {RX: 2, RZ: 1}
            Full Expansion Gates: {GlobalPhase: 1, RX: 2, RZ: 1}
            Weighted Cost: 4.0

            Decomposition 5 (name: zxz)
            0: ──RZ(0.00)──RX(0.00)──RZ(0.00)─┤  
            Estimated First-Level Expansion Gates: {RZ: 2, RX: 1, GlobalPhase: 1}
            Actual First-Level Expansion Gates: {RZ: 2, RX: 1}
            Full Expansion Gates: {GlobalPhase: 1, RX: 1, RZ: 2}
            Weighted Cost: 4.0

            Decomposition 6 (name: zyz)
            0: ──RZ(0.00)──RY(0.00)──RZ(0.00)─┤  
            Estimated First-Level Expansion Gates: {RZ: 2, RY: 1, GlobalPhase: 1}
            Actual First-Level Expansion Gates: {RZ: 2, RY: 1}
            Full Expansion Gates: {GlobalPhase: 1, RZ: 4, RX: 1}
            Weighted Cost: 6.0
            """).strip()

        assert result._repr_markdown_() == dedent("""
            #### Decomposition 0 (name: multi_qubit_decomp_rule)

            _Not applicable (provided operator instance does not meet all conditions for this rule)._

            ---

            #### Decomposition 1 (name: two_qubit_decomp_rule)

            _Not applicable (provided operator instance does not meet all conditions for this rule)._

            ---

            #### Decomposition 2 (name: rot)

            ```
            0: ──RZ(0.00)─┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Estimated | Actual |
            | :--- | :--- | :--- |
            | GlobalPhase | 1 | 0 |
            | RZ | 1 | 1 |
            | Rot | 1 | 0 |

            | Full Expansion | Count |
            | :--- | :--- |
            | GlobalPhase | 1 |
            | RX | 1 |
            | RZ | 5 |
            | **Weighted Cost** | 7.0 |
            </details>

            ---

            #### Decomposition 3 (name: xyx)

            ```
            0: ──RX(0.00)──RY(0.00)──RX(0.00)─┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Estimated | Actual |
            | :--- | :--- | :--- |
            | GlobalPhase | 1 | 0 |
            | RX | 2 | 2 |
            | RY | 1 | 1 |

            | Full Expansion | Count |
            | :--- | :--- |
            | GlobalPhase | 1 |
            | RX | 3 |
            | RZ | 2 |
            | **Weighted Cost** | 6.0 |
            </details>

            ---

            #### **CHOSEN:** Decomposition 4 (name: xzx)

            ```
            0: ──RX(0.00)──RZ(0.00)──RX(0.00)─┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Estimated | Actual |
            | :--- | :--- | :--- |
            | GlobalPhase | 1 | 0 |
            | RX | 2 | 2 |
            | RZ | 1 | 1 |

            | Full Expansion | Count |
            | :--- | :--- |
            | GlobalPhase | 1 |
            | RX | 2 |
            | RZ | 1 |
            | **Weighted Cost** | 4.0 |
            </details>

            ---

            #### Decomposition 5 (name: zxz)

            ```
            0: ──RZ(0.00)──RX(0.00)──RZ(0.00)─┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Estimated | Actual |
            | :--- | :--- | :--- |
            | GlobalPhase | 1 | 0 |
            | RX | 1 | 1 |
            | RZ | 2 | 2 |

            | Full Expansion | Count |
            | :--- | :--- |
            | GlobalPhase | 1 |
            | RX | 1 |
            | RZ | 2 |
            | **Weighted Cost** | 4.0 |
            </details>

            ---

            #### Decomposition 6 (name: zyz)

            ```
            0: ──RZ(0.00)──RY(0.00)──RZ(0.00)─┤  
            ```
            <details><summary>Gate Counts and Wire Allocations</summary>

            | First-Level Expansion | Estimated | Actual |
            | :--- | :--- | :--- |
            | GlobalPhase | 1 | 0 |
            | RY | 1 | 1 |
            | RZ | 2 | 2 |

            | Full Expansion | Count |
            | :--- | :--- |
            | GlobalPhase | 1 |
            | RX | 1 |
            | RZ | 4 |
            | **Weighted Cost** | 6.0 |
            </details>
            """).strip()

    def test_gate_set(self):
        """Tests that the output is correct when querying an op in the gate set."""

        @decomp_inspector(gate_set={"RZ", "RX", "CNOT", "GlobalPhase"})
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.CNOT([0, 1])
            return qp.probs()

        inspector = circuit()

        msg = "The operator does not have decompositions as it is in the target gate set."
        result = inspector.inspect_decomps(qp.CNOT([0, 1]))
        assert str(result) == msg
        assert result._repr_markdown_() == msg
