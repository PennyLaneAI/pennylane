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
Tests for the TemporaryAND template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.templates.subroutines.arithmetic.temporary_and import (
    _adjoint_temporary_and,
    _adjoint_temporary_and_ppm,
    _adjoint_temporary_and_to_toffoli,
)
from pennylane.testing import assert_valid_decomp_rule
from pennylane.typing import Bool, Wire

expected_adjoint_temporary_and_rule_ops = [
    (
        _adjoint_temporary_and,
        [
            "H(2)",
            "MidMeasure(wires=[2], postselect=None, reset=True)",
            "Conditional(CZ)(wires=[0, 1])",
        ],
    ),
    (_adjoint_temporary_and_to_toffoli, ["Toffoli(wires=[0, 1, 2])"]),
    (
        _adjoint_temporary_and_ppm,
        [
            "PauliMeasure('X', wires=[2])",
            "Conditional(CZ)(wires=[0, 1])",
            "PauliMeasure('Z', wires=[2])",
            "Conditional(PauliX)(wires=[2])",
        ],
    ),
]


class TestTemporaryAND:
    """Tests specific to the TemporaryAND operation"""

    def compare_to_toffoli_on_zero(self, matrix, zeroed, cvals=None):
        """Compare a given matrix to the matrix of Toffoli on a constrained subspace.
        The constraint is either that the input state is |0> on the target qubit, or that the
        output is |0>. This is determined by the argument ``zeroed``.
        """
        cvals = cvals or [1, 1]
        toffoli_mat = qp.matrix(qp.Toffoli(wires=[0, 1, 2]))
        if not cvals[0]:
            x_mat = qp.matrix(qp.X(0), wire_order=[0, 1, 2])
            toffoli_mat = x_mat @ toffoli_mat @ x_mat
        if not cvals[1]:
            x_mat = qp.matrix(qp.X(1), wire_order=[0, 1, 2])
            toffoli_mat = x_mat @ toffoli_mat @ x_mat
        if zeroed == "input":
            # When the third qubit starts in |0>, we only check the odd columns
            isometry = matrix[:, ::2]
            isometry_toffoli = toffoli_mat[:, ::2]

        # When the third qubit ends in |0>, we only check the odd rows
        else:
            isometry = matrix[::2, :]
            isometry_toffoli = toffoli_mat[::2, :]

        assert qp.math.allclose(isometry, isometry_toffoli)

    def test_repr_and_str(self):
        """Test the repr and str of TemporaryAND."""
        assert repr(qp.TemporaryAND(wires=[0, "a", 2])) == "TemporaryAND(wires=[0, 'a', 2])"
        assert str(qp.TemporaryAND(wires=[0, "a", 2])) == "TemporaryAND(wires=[0, 'a', 2])"
        assert (
            repr(qp.TemporaryAND(wires=[0, "a", 2], control_values=(0, 1)))
            == "TemporaryAND(wires=[0, 'a', 2], control_values=[False, True])"
        )
        assert (
            repr(qp.TemporaryAND(wires=Wire[3], control_values=Bool[2]))
            == "TemporaryAND(wires=AbstractWires(3), control_values=AbstractArray((2,), bool, weak_type=True))"
        )
        assert str(qp.TemporaryAND(wires=Wire[3], control_values=Bool[2])) == "TemporaryAND"

    def test_abstract(self):
        """Tests that abstract TemporaryAND can be created."""

        full = qp.TemporaryAND(Wire[3], Bool[2])
        assert full.wires == Wire[3]
        assert full.control_values == Bool[2]

        op = abstractify(qp.TemporaryAND)
        assert op == full

        op2 = abstractify(qp.TemporaryAND(Wire[3]))
        assert op2 == full

    def test_alias(self):
        """Test that Elbow is an alias of TemporaryAND"""
        op1 = qp.TemporaryAND(wires=[0, "a", 2], control_values=(0, 0))
        op2 = qp.Elbow(wires=[0, "a", 2], control_values=(0, 0))
        qp.assert_equal(op1, op2)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""
        cvals = qp.math.array([0, 0], like="jax") if qp.capture.enabled() else [0, 0]
        op = qp.TemporaryAND(wires=[0, 1, 2], control_values=cvals)
        # Skip matrix check because the decomposition to Toffoli,and the adjoint decomposition
        # to mcm + cond(CZ) do not reproduce the matrix of the op. Skip bind_new_parameters
        # now that control_values is part of dynamic data
        qp.testing.assert_valid(
            op,
            skip_decomp_matrix_check=True,
            skip_bind_new_parameters=True,
        )

    def test_correctness(self):
        """Tests the correctness of the TemporaryAND operator.
        This is done by comparing the results with the Toffoli operator
        """

        dev = qp.device("default.qubit", wires=4)

        qs_and = qp.tape.QuantumScript(
            [
                qp.RY(-2.6321, 0),
                qp.RY(0.612, 1),
                qp.TemporaryAND([0, 1, 2], control_values=[0, 1]),
                qp.CNOT([2, 3]),
                qp.RX(1.2, 3),
                qp.adjoint(qp.TemporaryAND([0, 1, 2], control_values=[0, 1])),
            ],
            [qp.state()],
        )

        qs_toffoli = qp.tape.QuantumScript(
            [
                qp.RY(-2.6321, 0),
                qp.RY(0.612, 1),
                qp.X(0),
                qp.Toffoli([0, 1, 2]),
                qp.X(0),
                qp.CNOT([2, 3]),
                qp.RX(1.2, 3),
                qp.X(0),
                qp.Toffoli([0, 1, 2]),
                qp.X(0),
            ],
            [qp.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs_and])
        output_and = dev.execute(tape[0])[0]

        tape = program([qs_toffoli])
        output_toffoli = dev.execute(tape[0])[0]
        assert qp.math.allclose(output_toffoli, output_and)

        # Compare the contracted isometries with the third qubit fixed to |0>
        matrix_and = qp.matrix(qp.TemporaryAND(wires=[0, 1, 2]))
        matrix_and_adj = qp.matrix(qp.adjoint(qp.TemporaryAND(wires=[0, 1, 2])))
        self.compare_to_toffoli_on_zero(matrix_and, "input")
        self.compare_to_toffoli_on_zero(matrix_and_adj, "output")

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("cvals", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_temporary_and_decompositions(self, cvals):
        """Tests that TemporaryAND is decomposed properly."""
        wires = [0, 1, 2]
        op = qp.TemporaryAND(wires, control_values=cvals)
        for rule in qp.list_decomps(qp.TemporaryAND):
            assert rule.is_applicable(**op.arguments)
            assert_valid_decomp_rule(op, rule, check_matrix=False)
            with qp.capture.pause():
                matrix = qp.matrix(rule, wire_order=wires)(wires, control_values=cvals)
            self.compare_to_toffoli_on_zero(matrix, "input", cvals)

    @pytest.mark.parametrize("rule", qp.list_decomps("Adjoint(TemporaryAND)"))
    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_adjoint_temporary_and_decomposition(self, control_values, rule, seed):
        """Validate various decompositions of Adjoint(TemporaryAND), excluding the PPM-based rule,
        which is tested in test_adjoint_temporary_and_ppm_decomposition below."""
        if rule == _adjoint_temporary_and_ppm:
            pytest.skip(
                reason=(
                    "The PPM-based decomposition requires special handling and is tested separately "
                    "in test_adjoint_temporary_and_ppm_decomposition."
                )
            )
        sys_wires = [0, 1, 2]
        work_wires = [3]  # auxiliary qubit for deferred measure
        dev = qp.device("default.qubit", wires=sys_wires + work_wires)

        @qp.qnode(dev)
        def circuit(state):
            # Prepare control state
            qp.StatePrep(state, wires=sys_wires[:2])
            op = qp.TemporaryAND(wires=sys_wires, control_values=control_values)
            rule(op)
            # Unprepare control state
            qp.adjoint(qp.StatePrep)(state, wires=sys_wires[:2])
            return qp.probs(wires=sys_wires)

        rng = np.random.default_rng(seed)
        state = rng.random(4) + 1j * rng.random(4)
        state /= np.linalg.norm(state)
        probs = circuit(state)
        assert qp.math.allclose(probs, np.eye(1, 8))

    @pytest.mark.catalyst
    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_adjoint_temporary_and_ppm_decomposition(self, control_values, seed):
        """Validate the PPM-based decomposition of Adjoint(TemporaryAND)."""
        sys_wires = [0, 1, 2]
        dev = qp.device("lightning.qubit", wires=sys_wires, seed=seed)

        gate_set = {"Toffoli", "CNOT", "RX", "RY", "RZ", "PauliMeasure", "X", "CZ", "GlobalPhase"}

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=gate_set)
        @qp.qnode(dev, shots=100, mcm_method="one-shot")
        def circuit(state):
            # Prepare control state
            qp.StatePrep(state, wires=sys_wires[:2])
            op = qp.TemporaryAND(wires=sys_wires, control_values=control_values)
            _adjoint_temporary_and_ppm(op)
            qp.adjoint(qp.MottonenStatePreparation.compute_decomposition)(
                state, wires=sys_wires[:2]
            )
            return qp.probs(wires=sys_wires)

        rng = np.random.default_rng(seed)
        state = rng.random(4) + 1j * rng.random(4)
        state /= np.linalg.norm(state)
        probs = circuit(state)
        assert qp.math.allclose(probs, np.eye(1, 8))

    def test_all_adjoint_temporary_and_rules_are_tested(self):
        """Test that the test cases in `expected_adjoint_temporary_and_rule_ops` cover all
        implemented decomposition rules."""
        if len(expected_adjoint_temporary_and_rule_ops) != len(
            qp.list_decomps("Adjoint(TemporaryAND)")
        ):
            raise NotImplementedError(
                "Need to add the expected operators for at least one decomposition rule of "
                "Adjoint(TemporaryAND) to `expected_adjoint_temporary_and_rule_ops` in the tests."
            )

    @pytest.mark.parametrize(
        "rule, expected_ops_per_elbow", expected_adjoint_temporary_and_rule_ops
    )
    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_adjoint_temporary_and_decomposition_ops(self, rule, expected_ops_per_elbow):
        """Test that the expected operators are produced by a fixed decomposition of
        Adjoint(TemporaryAND)."""

        wires = [0, 1, 2]
        gate_set = {"X", "Hadamard", "CNOT", "CZ", "MidMeasureMP", "Toffoli", "PauliMeasure"}

        @qp.set_shots(1)
        @qp.qnode(qp.device("default.qubit", wires=wires))
        # pylint: disable-next=protected-access
        @qp.transforms._tape_decompose(
            gate_set=gate_set,
            fixed_decomps={"Adjoint(TemporaryAND)": rule},
        )
        def circuit():
            qp.adjoint(qp.TemporaryAND(wires, control_values=(0, 0)))
            qp.adjoint(qp.TemporaryAND(wires, control_values=(0, 1)))
            qp.adjoint(qp.TemporaryAND(wires, control_values=(1, 0)))
            qp.adjoint(qp.TemporaryAND(wires))
            return qp.sample(wires=wires)

        tape = qp.workflow.construct_tape(circuit)()
        expected_ops = (
            ["X(0)", "X(1)"]
            + expected_ops_per_elbow
            + ["X(0)", "X(1)"]
            + ["X(0)"]
            + expected_ops_per_elbow
            + ["X(0)"]
            + ["X(1)"]
            + expected_ops_per_elbow
            + ["X(1)"]
            + expected_ops_per_elbow
        )

        for op, exp_op in zip(tape.operations, expected_ops):
            assert repr(op) == exp_op

    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_compute_matrix_temporary_and(self, control_values):
        """Tests that the matrix of the TemporaryAND operator is correct."""

        matrix_base = qp.math.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, -1j, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=complex,
        )

        eye = qp.math.eye(2)
        single_qubit = [qp.math.array([[0, 1], [1, 0]]), eye]
        X_matrix = qp.math.kron(
            single_qubit[control_values[0]], qp.math.kron(single_qubit[control_values[1]], eye)
        )
        assert qp.math.allclose(
            X_matrix @ matrix_base @ X_matrix,
            qp.matrix(qp.TemporaryAND([0, 1, 2], control_values)),
        )

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests that TemporaryAND works with jax and jit"""
        import jax

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.Hadamard(1)
            qp.TemporaryAND(wires=[0, 1, 2], control_values=[0, 1])
            qp.CNOT(wires=[2, 3])
            qp.RY(1.2, 3)
            qp.adjoint(qp.TemporaryAND([0, 1, 2]))
            return qp.probs([0, 1, 2, 3])

        jit_circuit = jax.jit(circuit)

        assert qp.math.allclose(circuit(), jit_circuit())

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.catalyst
    @pytest.mark.parametrize("cvals", [(0, 0), (0, 1), (1, 1), (True, False)])
    def test_jax_qjit_control_values(self, cvals):
        """Tests that TemporaryAND works with jax and jit"""

        dev = qp.device("lightning.qubit")

        @qp.qnode(dev)
        def circuit(values):
            qp.Hadamard(0)
            qp.Hadamard(1)
            qp.TemporaryAND(wires=[0, 1, 2], control_values=values)
            return qp.probs([0, 1, 2])

        exp_probs = qp.math.array([1, 0, 1, 0, 1, 0, 1, 0]) / 4
        flip = 2 * int(cvals[0]) + int(cvals[1])
        exp_probs[2 * flip : 2 * flip + 2] = [0, 0.25]
        qjit_circuit = qp.qjit(circuit)
        values = qp.math.array(cvals, like="jax")
        out = circuit(values)
        qjit_out = qjit_circuit(values)
        assert qp.math.allclose(out, exp_probs)
        assert qp.math.allclose(out, qjit_out)
