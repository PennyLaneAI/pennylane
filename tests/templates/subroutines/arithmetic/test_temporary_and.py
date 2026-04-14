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

import pennylane as qml
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.temporary_and import (
    _adjoint_temporary_and,
    _adjoint_temporary_and_to_toffoli,
    _temporary_and_to_toffoli,
)


class TestTemporaryAND:
    """Tests specific to the TemporaryAND operation"""

    def compare_to_toffoli_on_zero(self, matrix, zeroed, cvals=None):
        """Compare a given matrix to the matrix of Toffoli on a constrained subspace.
        The constraint is either that the input state is |0> on the target qubit, or that the
        output is |0>. This is determined by the argument ``zeroed``.
        """
        cvals = cvals or [1, 1]
        toffoli_mat = qml.matrix(qml.Toffoli(wires=[0, 1, 2]))
        if not cvals[0]:
            x_mat = qml.matrix(qml.X(0), wire_order=[0, 1, 2])
            toffoli_mat = x_mat @ toffoli_mat @ x_mat
        if not cvals[1]:
            x_mat = qml.matrix(qml.X(1), wire_order=[0, 1, 2])
            toffoli_mat = x_mat @ toffoli_mat @ x_mat
        if zeroed == "input":
            # When the third qubit starts in |0>, we only check the odd columns
            iso = matrix[:, ::2]
            iso_toffoli = toffoli_mat[:, ::2]

        # When the third qubit ends in |0>, we only check the odd rows
        else:
            iso = matrix[::2, :]
            iso_toffoli = toffoli_mat[::2, :]

        assert qml.math.allclose(iso, iso_toffoli)

    def test_repr(self):
        """Test the repr of TemporaryAND."""
        assert repr(qml.TemporaryAND(wires=[0, "a", 2])) == "TemporaryAND(wires=Wires([0, 'a', 2]))"
        assert (
            repr(qml.TemporaryAND(wires=[0, "a", 2], control_values=(0, 1)))
            == "TemporaryAND(wires=Wires([0, 'a', 2]), control_values=(0, 1))"
        )

    def test_alias(self):
        """Test that Elbow is an alias of TemporaryAND"""
        op1 = qml.TemporaryAND(wires=[0, "a", 2], control_values=(0, 0))
        op2 = qml.Elbow(wires=[0, "a", 2], control_values=(0, 0))
        qml.assert_equal(op1, op2)

    @pytest.mark.jax
    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""
        op = qml.TemporaryAND(wires=[0, "a", 2], control_values=(0, 0))
        # Skip matrix check because the decomposition to Toffoli,and the adjoint decomposition
        # to mcm + cond(CZ) do not reproduce the matrix of the op.
        qml.ops.functions.assert_valid(op, skip_decomp_matrix_check=True)

    def test_correctness(self):
        """Tests the correctness of the TemporaryAND operator.
        This is done by comparing the results with the Toffoli operator
        """

        dev = qml.device("default.qubit", wires=4)

        qs_and = qml.tape.QuantumScript(
            [
                qml.RY(-2.6321, 0),
                qml.RY(0.612, 1),
                qml.TemporaryAND([0, 1, 2], control_values=[0, 1]),
                qml.CNOT([2, 3]),
                qml.RX(1.2, 3),
                qml.adjoint(qml.TemporaryAND([0, 1, 2], control_values=[0, 1])),
            ],
            [qml.state()],
        )

        qs_toffoli = qml.tape.QuantumScript(
            [
                qml.RY(-2.6321, 0),
                qml.RY(0.612, 1),
                qml.X(0),
                qml.Toffoli([0, 1, 2]),
                qml.X(0),
                qml.CNOT([2, 3]),
                qml.RX(1.2, 3),
                qml.X(0),
                qml.Toffoli([0, 1, 2]),
                qml.X(0),
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs_and])
        output_and = dev.execute(tape[0])[0]

        tape = program([qs_toffoli])
        output_toffoli = dev.execute(tape[0])[0]
        assert qml.math.allclose(output_toffoli, output_and)

        # Compare the contracted isometries with the third qubit fixed to |0>
        matrix_and = qml.matrix(qml.TemporaryAND(wires=[0, 1, 2]))
        matrix_and_adj = qml.matrix(qml.adjoint(qml.TemporaryAND(wires=[0, 1, 2])))
        self.compare_to_toffoli_on_zero(matrix_and, "input")
        self.compare_to_toffoli_on_zero(matrix_and_adj, "output")

    @pytest.mark.parametrize("cvals", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_temporary_and_decompositions(self, cvals):
        """Tests that TemporaryAND is decomposed properly."""
        wires = [0, 1, 2]
        for rule in qml.list_decomps(qml.TemporaryAND):
            _test_decomposition_rule(
                qml.TemporaryAND(wires, control_values=cvals), rule, skip_decomp_matrix_check=True
            )
            matrix = qml.matrix(rule, wire_order=wires)(wires, control_values=cvals)
            self.compare_to_toffoli_on_zero(matrix, "input", cvals)

    @pytest.mark.parametrize("rule", qml.list_decomps("Adjoint(TemporaryAND)"))
    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_adjoint_temporary_and_decomposition(self, control_values, rule, seed):
        """
        Validate the MCM-based decomposition of Adjoint(TemporaryAND).
        """
        sys_wires = [0, 1, 2]
        work_wires = [3]  # auxiliary qubit for deferred measure
        dev = qml.device("default.qubit", wires=sys_wires + work_wires)

        @qml.qnode(dev)
        def circuit(state):
            # Prepare control state
            qml.StatePrep(state, wires=sys_wires[:2])
            op = qml.TemporaryAND(wires=sys_wires, control_values=control_values)
            rule(sys_wires, base=op)
            # Unprepare control state
            qml.adjoint(qml.StatePrep)(state, wires=sys_wires[:2])
            return qml.probs(wires=sys_wires)

        rng = np.random.default_rng(seed)
        state = rng.random(4) + 1j * rng.random(4)
        state /= np.linalg.norm(state)
        probs = circuit(state)
        assert qml.math.allclose(probs, np.eye(8)[0])

    @pytest.mark.parametrize("rule", qml.list_decomps("Adjoint(TemporaryAND)"))
    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_adjoint_temporary_and_integration(self, rule):
        wires = [0, 1, "aux0", 2]
        gate_set = {"X", "Hadamard", "CNOT", "CZ", "MidMeasureMP", "Toffoli"}

        @qml.set_shots(1)
        @qml.qnode(qml.device("default.qubit", wires=wires), interface=None)
        @qml.transforms.decompose(
            gate_set=gate_set,
            fixed_decomps={
                qml.Select: qml.templates.subroutines.select._select_decomp_unary,  # pylint: disable=protected-access
                "Adjoint(TemporaryAND)": rule,
                "TemporaryAND": _temporary_and_to_toffoli,
            },
        )
        def circuit():
            ops = [qml.Z(2) for _ in range(4)]
            qml.Select(ops, control=[0, 1], work_wires=["aux0"], partial=True)
            return qml.sample(wires=wires)

        tape = qml.workflow.construct_tape(circuit)()
        expected_operators = [
            # Start of left elbow with cval [0, 0]
            qml.X(0),
            qml.X(1),
            qml.Toffoli([0, 1, "aux0"]),
            qml.X(0),
            qml.X(1),
            # End of left elbow
            qml.CZ(wires=["aux0", 2]),  # First target op
            # Merged right and left elbow (cvals [0, 0] to [0, 1])
            qml.CNOT(wires=[0, "aux0"]),
            qml.X("aux0"),
            qml.CZ(wires=["aux0", 2]),  # Second target op
            # Merged right and left elbow (cvals [0, 1] to [1, 0])
            qml.CNOT(wires=[0, "aux0"]),
            qml.CNOT(wires=[1, "aux0"]),
            qml.CZ(wires=["aux0", 2]),  # Third target op
            # Merged right and left elbow (cvals [1, 0] to [1, 1])
            qml.CNOT(wires=[0, "aux0"]),
            qml.CZ(wires=["aux0", 2]),  # Fourth target op
        ]
        if rule == _adjoint_temporary_and:
            expected_operators += [
                qml.H("aux0"),
                qml.measurements.MidMeasureMP(wires=["aux0"], postselect=None, reset=True),
                "ConditionalCZ",
            ]
        elif rule == _adjoint_temporary_and_to_toffoli:
            expected_operators += [qml.Toffoli([0, 1, "aux0"])]

        else:
            raise NotImplementedError(f"Please add expected operators for rule {rule}")

        for op, exp_op in zip(tape.operations, expected_operators):
            # manual check: each MidMeasure has a unique ID, which prevents
            # qml.equal from treating two MidMeasure as equal.
            if isinstance(op, qml.measurements.MidMeasureMP):
                assert op.wires == exp_op.wires
                assert op.postselect == exp_op.postselect
                assert op.reset == exp_op.reset

            # manual check for the conditional operator
            elif isinstance(op, qml.ops.op_math.condition.Conditional):
                assert exp_op == "ConditionalCZ"
                assert isinstance(op.base, qml.CZ)
                assert list(op.base.wires) == [0, 1]
                meas = op.meas_val  # same as the expr passed to qml.cond
                assert list(meas.wires) == ["aux0"]

            else:
                qml.assert_equal(op, exp_op)

    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_compute_matrix_temporary_and(self, control_values):
        """Tests that the matrix of the TemporaryAND operator is correct."""

        matrix_base = qml.math.array(
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

        eye = qml.math.eye(2)
        single_qubit = [qml.math.array([[0, 1], [1, 0]]), eye]
        X_matrix = qml.math.kron(
            single_qubit[control_values[0]], qml.math.kron(single_qubit[control_values[1]], eye)
        )
        assert qml.math.allclose(
            X_matrix @ matrix_base @ X_matrix,
            qml.matrix(qml.TemporaryAND([0, 1, 2], control_values)),
        )

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests that TemporaryAND works with jax and jit"""
        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.TemporaryAND(wires=[0, 1, 2], control_values=[0, 1])
            qml.CNOT(wires=[2, 3])
            qml.RY(1.2, 3)
            qml.adjoint(qml.TemporaryAND([0, 1, 2]))
            return qml.probs([0, 1, 2, 3])

        jit_circuit = jax.jit(circuit)

        assert qml.math.allclose(circuit(), jit_circuit())

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.external
    @pytest.mark.parametrize("cvals", [(0, 0), (0, 1), (1, 1), (True, False)])
    def test_jax_qjit_control_values(self, cvals):
        """Tests that TemporaryAND works with jax and jit"""

        dev = qml.device("lightning.qubit")

        @qml.qnode(dev)
        def circuit(values):
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.TemporaryAND(wires=[0, 1, 2], control_values=values)
            return qml.probs([0, 1, 2])

        exp_probs = qml.math.array([1, 0, 1, 0, 1, 0, 1, 0]) / 4
        flip = 2 * int(cvals[0]) + int(cvals[1])
        exp_probs[2 * flip : 2 * flip + 2] = [0, 0.25]
        qjit_circuit = qml.qjit(circuit)
        values = qml.math.array(cvals, like="jax")
        out = circuit(values)
        qjit_out = qjit_circuit(values)
        assert qml.math.allclose(out, exp_probs)
        assert qml.math.allclose(out, qjit_out)
