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
Tests for the TemporaryAND template (3-wire and multi-wire cases).
"""

import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import (
    _check_decomposition_new,
    _test_decomposition_rule,
)
from pennylane.templates.subroutines.arithmetic.temporary_and import _adjoint_TemporaryAND


class TestTemporaryAND:
    """Tests specific to the TemporaryAND operation (3-wire and multi-wire)."""

    def test_repr(self):
        """Test the repr of TemporaryAND."""
        assert repr(qp.TemporaryAND(wires=[0, "a", 2])) == "TemporaryAND(wires=Wires([0, 'a', 2]))"
        assert (
            repr(qp.TemporaryAND(wires=[0, "a", 2], control_values=(0, 1)))
            == "TemporaryAND(wires=Wires([0, 'a', 2]), control_values=(0, 1))"
        )

    def test_is_controlled_op(self):
        """TemporaryAND should now be an instance of ControlledOp with an ``X`` base."""
        op = qp.TemporaryAND(wires=[0, 1, 2])
        assert isinstance(op, qp.ops.op_math.ControlledOp)
        assert isinstance(op.base, qp.X)
        assert op.target_wires == qp.wires.Wires([2])
        assert op.control_wires == qp.wires.Wires([0, 1])

    def test_is_controlled_op_multi(self):
        """TemporaryAND with more than 2 controls is a ControlledOp with an X base."""
        op = qp.TemporaryAND(wires=[0, 1, 2, 3, 4], work_wires=[5, 6])
        assert isinstance(op, qp.ops.op_math.ControlledOp)
        assert isinstance(op.base, qp.X)
        assert op.target_wires == qp.wires.Wires([4])
        assert op.control_wires == qp.wires.Wires([0, 1, 2, 3])
        assert op.work_wires == qp.wires.Wires([5, 6])

    def test_alias(self):
        """Test that Elbow is an alias of TemporaryAND"""
        op1 = qp.TemporaryAND(wires=[0, "a", 2], control_values=(0, 0))
        op2 = qp.Elbow(wires=[0, "a", 2], control_values=(0, 0))
        qp.assert_equal(op1, op2)

    def test_too_few_wires(self):
        """TemporaryAND requires at least 3 wires."""
        with pytest.raises(ValueError, match="wrong number of wires"):
            qp.TemporaryAND(wires=[0, 1])

    @pytest.mark.jax
    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""
        op = qp.TemporaryAND(wires=[0, "a", 2], control_values=(0, 0))
        qp.ops.functions.assert_valid(op, skip_decomp_matrix_check=True)

    def test_correctness(self):
        """Tests the correctness of the TemporaryAND operator.
        This is done by comparing the results with the Toffoli operator.
        """

        dev = qp.device("default.qubit", wires=4)

        qs_and = qp.tape.QuantumScript(
            [
                qp.Hadamard(0),
                qp.Hadamard(1),
                qp.TemporaryAND([0, 1, 2], control_values=[0, 1]),
                qp.CNOT([2, 3]),
                qp.RX(1.2, 3),
                qp.adjoint(qp.TemporaryAND([0, 1, 2], control_values=[0, 1])),
            ],
            [qp.state()],
        )

        qs_toffoli = qp.tape.QuantumScript(
            [
                qp.Hadamard(0),
                qp.Hadamard(1),
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
        M_and = qp.matrix(qp.TemporaryAND(wires=[0, 1, 2]))
        M_and_adj = qp.matrix(qp.adjoint(qp.TemporaryAND(wires=[0, 1, 2])))
        M_toffoli = qp.matrix(qp.Toffoli(wires=[0, 1, 2]))

        # When the third qubit starts in |0>, we only check the odd columns
        iso_and = M_and[:, ::2]
        iso_toffoli = M_toffoli[:, ::2]

        # When the third qubit ends in |0>, we only check the odd rows
        iso_M_and_adj = M_and_adj[::2, :]
        iso_toffoli_adj = M_toffoli[::2, :]

        assert qp.math.allclose(iso_and, iso_toffoli)
        assert qp.math.allclose(iso_M_and_adj, iso_toffoli_adj)

    @pytest.mark.parametrize("cvals", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_and_decompositions(self, cvals):
        """Tests that the 3-wire TemporaryAND decomposition matches the operator on
        the |0> target-subspace."""
        for rule in qp.list_decomps(qp.TemporaryAND):
            op = qp.TemporaryAND([0, 1, 2], control_values=cvals)
            # The phase-baked decomposition only matches the operator on the |0>-target
            # subspace, so skip the full matrix check.
            _test_decomposition_rule(op, rule, skip_decomp_matrix_check=True)

    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_adjoint_temporary_and_decomposition(self, control_values):
        """
        Validate the MCM-based decomposition of Adjoint(TemporaryAND).
        """
        sys_wires = [0, 1, 2]
        work_wires = [3]  # auxiliary qubit for deferred measure
        dev = qp.device("default.qubit", wires=sys_wires + work_wires)

        @qp.qnode(dev)
        def circuit(a, b):
            qp.BasisState(qp.math.array([a, b, 0], dtype=int), wires=sys_wires)
            qp.TemporaryAND(wires=sys_wires, control_values=control_values)
            _adjoint_TemporaryAND(wires=sys_wires)
            return qp.probs(wires=sys_wires)

        for a in (0, 1):
            for b in (0, 1):
                probs = circuit(a, b)
                idx = (a << 2) | (b << 1)
                assert qp.math.allclose(
                    probs[idx], 1.0
                ), f"Failed for a={a}, b={b}, cv={control_values}"

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_adjoint_temporary_and_integration(self):
        """Integration smoke-test: decomposing a ``Select`` with a ``TemporaryAND``
        intermediate uses the MCM-based adjoint pattern."""
        wires = [0, 1, "aux0", 2]
        gate_set = {"X", "T", "Adjoint(T)", "Hadamard", "CX", "CZ", "MidMeasureMP", "Adjoint(S)"}

        @qp.set_shots(1)
        @qp.qnode(qp.device("default.qubit", wires=wires), interface=None)
        @qp.transforms.decompose(
            gate_set=gate_set,
            fixed_decomps={
                qp.Select: qp.templates.subroutines.select._select_decomp_unary  # pylint: disable=protected-access
            },
        )
        def circuit():
            ops = [qp.Z(2) for _ in range(4)]
            qp.Select(ops, control=[0, 1], work_wires=["aux0"], partial=True)
            return qp.sample(wires=wires)

        tape = qp.workflow.construct_tape(circuit)()

        # The MCM-based adjoint decomposition of TemporaryAND contributes a
        # MidMeasureMP followed by a Conditional gate.
        assert any(isinstance(op, qp.measurements.MidMeasureMP) for op in tape.operations)
        assert any(isinstance(op, qp.ops.op_math.condition.Conditional) for op in tape.operations)
        # All final gates should come from the specified gate set (plus X == PauliX alias)
        allowed = gate_set | {"Conditional", "MidMeasureMP", "PauliX"}
        for op in tape.operations:
            name = op.name
            # Conditional ops carry a different top-level name
            if isinstance(op, qp.ops.op_math.condition.Conditional):
                continue
            assert name in allowed or name.startswith("Adjoint("), f"Unexpected op {name}"

    @pytest.mark.parametrize("control_values", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_compute_matrix_temporary_and(self, control_values):
        """Tests that the (multi-)controlled-X matrix of the TemporaryAND operator is correct."""

        # TemporaryAND is now a controlled-X, so its matrix is the MCX matrix
        expected = qp.matrix(
            qp.MultiControlledX(wires=[0, 1, 2], control_values=control_values),
            wire_order=[0, 1, 2],
        )
        actual = qp.matrix(
            qp.TemporaryAND([0, 1, 2], control_values=control_values),
            wire_order=[0, 1, 2],
        )
        assert qp.math.allclose(expected, actual)

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

    @pytest.mark.xfail  # generally, control_values of controlled ops are not jitable. However, this originally worked because we did not actually make TemporaryAND a ControlledOp.
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.external
    @pytest.mark.parametrize("cvals", [[0, 0], [0, 1], [1, 1], (True, False)])
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


class TestMultiTemporaryAND:
    """Tests specific to the multi-wire (``num_controls > 2``) case of
    :class:`~.TemporaryAND` (formerly ``MultiTemporaryAND``)."""

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_valid_decomp(self, n):
        """Test that the multi-control decomposition rule is valid as a fixed
        decomposition and yields the correct resources."""
        wires = list(range(n))
        work_wires = list(range(n, 2 * n - 2))  # num_controls - 2 == n - 2 wires
        op = qp.TemporaryAND(wires=wires, work_wires=work_wires)
        _check_decomposition_new(op, skip_decomp_matrix_check=True)

    @pytest.mark.parametrize("n", [4, 5])
    def test_multi_correctness_on_zero_target(self, n):
        """The multi-control TemporaryAND on a |0> target behaves as a
        multi-controlled X gate for all control basis states."""
        wires = list(range(n))
        work_wires = list(range(n, 2 * n - 2))
        all_wires = wires + work_wires

        dev = qp.device("default.qubit", wires=all_wires)

        @qp.qnode(dev)
        @qp.transforms.decompose(
            gate_set={"X", "T", "Adjoint(T)", "Hadamard", "CNOT", "S", "Adjoint(S)"}
        )
        def decomp_circuit(control_bits):
            basis = qp.math.array(
                list(control_bits) + [0] * (len(all_wires) - len(control_bits)), dtype=int
            )
            qp.BasisState(basis, wires=all_wires)
            qp.TemporaryAND(wires=wires, work_wires=work_wires)
            return qp.probs(wires=wires)

        @qp.qnode(dev)
        def reference_circuit(control_bits):
            basis = qp.math.array(
                list(control_bits) + [0] * (len(all_wires) - len(control_bits)), dtype=int
            )
            qp.BasisState(basis, wires=all_wires)
            qp.MultiControlledX(wires=wires)
            return qp.probs(wires=wires)

        # check all control configurations
        for bits in range(2 ** (n - 1)):
            control_bits = [(bits >> i) & 1 for i in range(n - 1)][::-1]
            got = decomp_circuit(control_bits)
            want = reference_circuit(control_bits)
            assert qp.math.allclose(got, want), f"Mismatch for n={n}, control_bits={control_bits}"

    @pytest.mark.parametrize("n", [4, 5])
    def test_multi_control_values(self, n):
        """Check that ``control_values`` flips the activation pattern."""
        wires = list(range(n))
        work_wires = list(range(n, 2 * n - 2))
        all_wires = wires + work_wires
        control_values = [i % 2 for i in range(n - 1)]  # mixed 0/1 controls

        dev = qp.device("default.qubit", wires=all_wires)

        @qp.qnode(dev)
        @qp.transforms.decompose(
            gate_set={"X", "T", "Adjoint(T)", "Hadamard", "CNOT", "S", "Adjoint(S)"}
        )
        def decomp_circuit(bits):
            basis = qp.math.array(list(bits) + [0] * (len(all_wires) - len(bits)), dtype=int)
            qp.BasisState(basis, wires=all_wires)
            qp.TemporaryAND(wires=wires, work_wires=work_wires, control_values=control_values)
            return qp.probs(wires=wires)

        @qp.qnode(dev)
        def reference_circuit(bits):
            basis = qp.math.array(list(bits) + [0] * (len(all_wires) - len(bits)), dtype=int)
            qp.BasisState(basis, wires=all_wires)
            qp.MultiControlledX(wires=wires, control_values=control_values)
            return qp.probs(wires=wires)

        for bits in range(2 ** (n - 1)):
            cbits = [(bits >> i) & 1 for i in range(n - 1)][::-1]
            got = decomp_circuit(cbits)
            want = reference_circuit(cbits)
            assert qp.math.allclose(got, want)

    def test_not_enough_work_wires_raises_no_match(self):
        """With fewer than ``num_controls - 2`` work wires the multi-control rule does not apply."""
        n = 5
        wires = list(range(n))
        # Provide only one work wire (need num_controls - 2 = 2)
        op = qp.TemporaryAND(wires=wires, work_wires=[n])

        # At least one registered decomposition should NOT be applicable here.
        rules = qp.list_decomps(qp.TemporaryAND)
        applicable = [r for r in rules if r.is_applicable(**op.resource_params)]
        # The 3-wire rule requires num_control_wires==2 and the multi rule requires enough work wires
        assert len(applicable) == 0

    def test_multi_is_controlled_op(self):
        """Multi-wire TemporaryAND is still a ControlledOp instance."""
        op = qp.TemporaryAND(wires=[0, 1, 2, 3, 4], work_wires=[5, 6])
        assert isinstance(op, qp.ops.op_math.ControlledOp)
        assert isinstance(op.base, qp.X)
