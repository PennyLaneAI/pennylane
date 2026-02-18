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

import pytest

import pennylane as qml
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.temporary_and import _adjoint_TemporaryAND


class TestTemporaryAND:
    """Tests specific to the TemporaryAND operation"""

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
        qml.ops.functions.assert_valid(op, skip_decomp_matrix_check=True)

    def test_correctness(self):
        """Tests the correctness of the TemporaryAND operator.
        This is done by comparing the results with the Toffoli operator
        """

        dev = qml.device("default.qubit", wires=4)

        qs_and = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.TemporaryAND([0, 1, 2], control_values=[0, 1]),
                qml.CNOT([2, 3]),
                qml.RX(1.2, 3),
                qml.adjoint(qml.TemporaryAND([0, 1, 2], control_values=[0, 1])),
            ],
            [qml.state()],
        )

        qs_toffoli = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
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
        M_and = qml.matrix(qml.TemporaryAND(wires=[0, 1, 2]))
        M_and_adj = qml.matrix(qml.adjoint(qml.TemporaryAND(wires=[0, 1, 2])))
        M_toffoli = qml.matrix(qml.Toffoli(wires=[0, 1, 2]))

        # When the third qubit starts in |0>, we only check the odd columns
        iso_and = M_and[:, ::2]
        iso_toffoli = M_toffoli[:, ::2]

        # When the third qubit ends in |0>, we only check the odd rows
        iso_M_and_adj = M_and_adj[::2, :]
        iso_toffoli_adj = M_toffoli[::2, :]

        assert qml.math.allclose(iso_and, iso_toffoli)
        assert qml.math.allclose(iso_M_and_adj, iso_toffoli_adj)

    def test_and_decompositions(self):
        """Tests that TemporaryAND is decomposed properly."""

        for rule in qml.list_decomps(qml.TemporaryAND):
            _test_decomposition_rule(qml.TemporaryAND([0, 1, 2], control_values=(0, 0)), rule)

    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_adjoint_temporary_and_decomposition(self, control_values):
        """
        Validate the MCM-based decomposition of Adjoint(TemporaryAND).
        """
        sys_wires = [0, 1, 2]
        work_wires = [3]  # auxiliary qubit for deferred measure
        dev = qml.device("default.qubit", wires=sys_wires + work_wires)

        @qml.qnode(dev)
        def circuit(a, b):
            qml.BasisState(qml.math.array([a, b, 0], dtype=int), wires=sys_wires)
            qml.TemporaryAND(wires=sys_wires, control_values=control_values)
            _adjoint_TemporaryAND(wires=sys_wires)
            return qml.probs(wires=sys_wires)

        for a in (0, 1):
            for b in (0, 1):
                probs = circuit(a, b)
                idx = (a << 2) | (b << 1)
                assert qml.math.allclose(
                    probs[idx], 1.0
                ), f"Failed for a={a}, b={b}, cv={control_values}"

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_adjoint_temporary_and_integration(self):
        wires = [0, 1, "aux0", 2]
        gate_set = {"X", "T", "Adjoint(T)", "Hadamard", "CX", "CZ", "MidMeasureMP", "Adjoint(S)"}

        @qml.set_shots(1)
        @qml.qnode(qml.device("default.qubit", wires=wires), interface=None)
        @qml.transforms.decompose(
            gate_set=gate_set,
            fixed_decomps={
                qml.Select: qml.templates.subroutines.select._select_decomp_unary  # pylint: disable=protected-access
            },
        )
        def circuit():
            ops = [qml.Z(2) for _ in range(4)]
            qml.Select(ops, control=[0, 1], work_wires=["aux0"], partial=True)
            return qml.sample(wires=wires)

        tape = qml.workflow.construct_tape(circuit)()
        expected_operators = [
            qml.X(0),
            qml.X(1),
            qml.H("aux0"),
            qml.T("aux0"),
            qml.H("aux0"),
            qml.CZ(wires=[1, "aux0"]),
            qml.H("aux0"),
            qml.adjoint(qml.T("aux0")),
            qml.H("aux0"),
            qml.CZ(wires=[0, "aux0"]),
            qml.H("aux0"),
            qml.T("aux0"),
            qml.H("aux0"),
            qml.CZ(wires=[1, "aux0"]),
            qml.H("aux0"),
            qml.adjoint(qml.T("aux0")),
            qml.H("aux0"),
            qml.adjoint(qml.S("aux0")),
            qml.X(0),
            qml.X(1),
            qml.CZ(wires=["aux0", 2]),
            qml.H("aux0"),
            qml.CZ(wires=[0, "aux0"]),
            qml.H("aux0"),
            qml.X("aux0"),
            qml.CZ(wires=["aux0", 2]),
            qml.H("aux0"),
            qml.CZ(wires=[0, "aux0"]),
            qml.H("aux0"),
            qml.H("aux0"),
            qml.CZ(wires=[1, "aux0"]),
            qml.H("aux0"),
            qml.CZ(wires=["aux0", 2]),
            qml.H("aux0"),
            qml.CZ(wires=[0, "aux0"]),
            qml.H("aux0"),
            qml.CZ(wires=["aux0", 2]),
            qml.H("aux0"),
            qml.measurements.MidMeasureMP(wires=["aux0"], postselect=None, reset=True),
            "ConditionalCZ",
        ]

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
        import qpjax

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

        jit_circuit = qpjax.jit(circuit)

        assert qml.math.allclose(circuit(), jit_circuit())
