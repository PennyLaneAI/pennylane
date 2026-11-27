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
"""
Tests for the MultiControlledX decomposition rules.
"""

import pytest
from scipy import sparse

import pennylane as qp
from pennylane.decomposition.decomposition_rule import _fix_decomp
from pennylane.decomposition_rules.op_math.multi_controlled_x import (
    mcx_many_work_wires,
    mcx_no_work_wires,
    mcx_one_work_wire,
    mcx_two_work_wires,
)
from pennylane.testing import assert_decomposable, assert_valid_decomp_rule


def _tape_to_matrix(tape, wire_order):
    """Convert a tape to a sparse matrix representation."""
    result = sparse.eye(2 ** len(wire_order))
    for op in tape.operations:
        op_matrix = (
            op.sparse_matrix(wire_order=wire_order)
            if op.has_sparse_matrix
            else sparse.csr_matrix(op.matrix(wire_order=wire_order))
        )
        result = op_matrix @ result
    return result


class TestMCXDecomposition:
    """Tests the decomposition rules for MCX."""

    def test_wrong_work_wire_type(self):
        """Test that an error is raised if the work wire type is not 'zeroed' or 'borrowed'."""

        expected_msg = r"work_wire_type must be one of \('zeroed', 'borrowed'\). Got 'blah'."

        with pytest.raises(ValueError, match=expected_msg):
            qp.MultiControlledX(
                wires=[0, 1, 2],
                work_wires=3,
                work_wire_type="blah",
            )

        with pytest.raises(ValueError, match=expected_msg):
            qp.MultiControlledX.compute_decomposition(
                wires=[0, 1, 2],
                work_wires=3,
                work_wire_type="blah",
            )

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.unit
    @pytest.mark.parametrize("work_wire_type", ["zeroed", "borrowed"])
    @pytest.mark.parametrize("n_ctrl_wires", [3, 4, 5])
    def test_decomposition_with_many_workers(self, n_ctrl_wires, work_wire_type):
        """Test that the decomposed MultiControlledX gate produces the correct matrix."""

        control_wires = list(range(1, n_ctrl_wires + 1))
        target_wire = 0
        num_work_wires = n_ctrl_wires - 2
        work_wires = list(range(n_ctrl_wires + 1, n_ctrl_wires + 1 + num_work_wires))

        mcx = qp.MultiControlledX(
            wires=control_wires + [target_wire],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        with qp.queuing.AnnotatedQueue() as q:
            if work_wire_type == "zeroed":
                qp.Projector([0] * len(work_wires), wires=work_wires)
            with qp.capture.pause():
                mcx_many_work_wires(**mcx.arguments)

        # Verify that the resource estimate is correct.
        assert_valid_decomp_rule(mcx, mcx_many_work_wires, check_matrix=False)

        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = _tape_to_matrix(tape, wire_order=control_wires + work_wires + [target_wire])
        equivalent_op = mcx
        if work_wire_type == "zeroed":
            equivalent_op @= qp.Projector([0] * len(work_wires), wires=work_wires)
        expected_matrix = equivalent_op.sparse_matrix(
            wire_order=control_wires + work_wires + [target_wire]
        )

        assert qp.math.allclose(matrix, expected_matrix)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("work_wire_type", ["zeroed", "borrowed"])
    @pytest.mark.parametrize("n_ctrl_wires", [3, 4, 5, 6, 7, 8, 9])
    def test_decomposition_with_one_worker(self, n_ctrl_wires, work_wire_type):
        """Test that the decompose_mcx_with_one_worker is correct."""

        control_wires = list(range(1, n_ctrl_wires + 1))
        target_wire = 0
        work_wire = n_ctrl_wires + 1

        # The MultiControlledX instance to test.
        mcx = qp.MultiControlledX(
            wires=control_wires + [target_wire],
            work_wires=work_wire,
            work_wire_type=work_wire_type,
        )

        with qp.queuing.AnnotatedQueue() as q:
            if work_wire_type == "zeroed":
                qp.Projector([0], wires=work_wire)
            with qp.capture.pause():
                mcx_one_work_wire(**mcx.arguments)

        # Verify that the resource estimate is correct.
        assert_valid_decomp_rule(mcx, mcx_one_work_wire, check_matrix=False)

        # Verify that the decomposition produces an equivalent matrix.
        tape = qp.tape.QuantumScript.from_queue(q)
        all_wires = mcx.wires + [work_wire]
        matrix = _tape_to_matrix(tape, wire_order=all_wires)

        equivalent_op = mcx
        if work_wire_type == "zeroed":
            equivalent_op @= qp.Projector([0], wires=work_wire)
        expected_matrix = equivalent_op.sparse_matrix(wire_order=all_wires)

        assert qp.math.allclose(matrix, expected_matrix)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("work_wire_type", ["zeroed", "borrowed"])
    @pytest.mark.parametrize("n_ctrl_wires", [3, 4, 5, 6, 7, 8, 9, 10])
    def test_decomposition_with_two_workers(self, n_ctrl_wires, work_wire_type):
        """Test that the decomposed MCX gate using 2 work wires produce the correct matrix."""

        control_wires = list(range(1, n_ctrl_wires + 1))
        target_wire = 0
        work_wires = [n_ctrl_wires + 1, n_ctrl_wires + 2]

        # The MultiControlledX instance to test.
        mcx = qp.MultiControlledX(
            wires=control_wires + [target_wire],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

        with qp.queuing.AnnotatedQueue() as q:
            if work_wire_type == "zeroed":
                qp.Projector([0, 0], wires=work_wires)
            with qp.capture.pause():
                mcx_two_work_wires(**mcx.arguments)

        # Verify that the resource estimate is correct.
        assert_valid_decomp_rule(mcx, mcx_two_work_wires, check_matrix=False)

        # Verify that the decomposition produces an equivalent matrix.
        tape = qp.tape.QuantumScript.from_queue(q)
        all_wires = mcx.wires + work_wires
        matrix = _tape_to_matrix(tape, wire_order=all_wires)

        equivalent_op = mcx
        if work_wire_type == "zeroed":
            equivalent_op @= qp.Projector([0, 0], wires=work_wires)
        expected_matrix = equivalent_op.sparse_matrix(wire_order=all_wires)

        assert qp.math.allclose(matrix, expected_matrix)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("n_ctrl_wires", [4, 5, 6, 7, 8, 9, 10])
    def test_decomposition_with_no_workers(self, n_ctrl_wires):
        """Test that the decomposed MCX gate using 2 work wires produce the correct matrix."""

        control_wires = list(range(1, n_ctrl_wires + 1))
        target_wire = 0

        # The MultiControlledX instance to test.
        mcx = qp.MultiControlledX(wires=control_wires + [target_wire])

        with qp.queuing.AnnotatedQueue() as q:
            with qp.capture.pause():
                mcx_no_work_wires(**mcx.arguments)

        # Verify that the resource estimate is correct.
        assert_valid_decomp_rule(mcx, mcx_no_work_wires, check_matrix=False)

        # Verify that the decomposition produces an equivalent matrix.
        tape = qp.tape.QuantumScript.from_queue(q)
        matrix = _tape_to_matrix(tape, wire_order=mcx.wires)

        expected_matrix = mcx.sparse_matrix()
        assert qp.math.allclose(matrix, expected_matrix)

        if qp.capture.enabled():
            return  # the following check is not expected to work with capture

        # compute decomposition result
        old_decomps = mcx.decomposition()
        assert tape.operations == old_decomps

    @pytest.mark.parametrize(
        "params",
        [
            {"wires": [1, 0]},
            {"wires": [1, 0], "control_values": [0]},
            {"wires": [2, 1, 0]},
            {"wires": [2, 1, 0], "control_values": [0, 1]},
            {"wires": [1, 0], "work_wires": [2]},
            {"wires": [1, 0], "control_values": [0], "work_wires": [2]},
            {"wires": [2, 1, 0], "work_wires": [3, 4]},
            {"wires": [2, 1, 0], "control_values": [0, 1], "work_wires": [3, 4]},
            {"wires": [1, 0], "work_wires": [2], "work_wire_type": "zeroed"},
            {"wires": [1, 0], "control_values": [0], "work_wires": [2], "work_wire_type": "zeroed"},
            {"wires": [2, 1, 0], "work_wires": [3, 4], "work_wire_type": "zeroed"},
            {
                "wires": [2, 1, 0],
                "control_values": [0, 1],
                "work_wires": [3, 4],
                "work_wire_type": "zeroed",
            },
            {"wires": [2, 3, 1, 0], "work_wires": [5, 4, 6], "work_wire_type": "zeroed"},
            {
                "wires": [2, 1, 0, 3],
                "control_values": [0, 0, 1],
                "work_wires": [5, 4, 6],
                "work_wire_type": "zeroed",
            },
        ],
    )
    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_mcx_decompositions(self, params):
        """Tests that MCX can be resolved into CNOT and Toffoli properly."""

        mcx = qp.MultiControlledX(**params)
        assert_decomposable(mcx)

    def test_mcx_fixed_decomp(self):
        """Tests that a fixed decomposition rule is used instead of the stock ones."""

        mcx = qp.MultiControlledX(wires=[0, 1, 2, 3])
        _fix_decomp(qp.MultiControlledX, mcx_no_work_wires)
        assert list(qp.list_decomps(mcx)) == [mcx_no_work_wires]

    @pytest.mark.catalyst
    @pytest.mark.parametrize("num_control_wires, num_work_wires", [(4, 1), (4, 2)])
    @pytest.mark.parametrize("work_wire_type", ["zeroed", "borrowed"])
    def test_mcx_qjit(self, num_control_wires, num_work_wires, work_wire_type):
        """Test that MultiControlledX decomposition is QJIT compatible with JAX-traced wires."""
        jax = pytest.importorskip("jax")
        from catalyst.device.decomposition import catalyst_decompose

        jnp = jax.numpy

        gate_set = {
            "X",
            "CNOT",
            "Toffoli",
            "TemporaryAND",
            "Adjoint(TemporaryAND)",
            "Cond",
            "HybridAdjoint",
            "ForLoop",
            "S",
            "T",
            "Adjoint(S)",
            "Adjoint(T)",
            "RZ",
            "Hadamard",
            "GlobalPhase",
        }

        wires = jnp.arange(num_control_wires + 1)
        work_wires = jnp.arange(num_control_wires + 1, num_control_wires + 1 + num_work_wires)
        cvals = (0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0)[:num_control_wires]

        @qp.qjit(capture=False, static_argnums=2)
        @catalyst_decompose(capabilities=None, target_gates=gate_set)
        @qp.qnode(qp.device("lightning.qubit"))
        def circuit(wires, work_wires, cvals):
            qp.MultiControlledX(
                wires,
                work_wires=work_wires,
                control_values=cvals,
                work_wire_type=work_wire_type,
            )
            return qp.probs(wires=wires)

        result = circuit(wires, work_wires, cvals)
        assert result is not None

    @pytest.mark.parametrize(
        "n_ctrl, n_work, expected",
        [
            (2, 0, False),
            (3, 0, True),
            (6, 0, True),
            (3, 1, False),
            (5, 1, False),
            (5, 2, False),
            (8, 6, False),
        ],
    )
    def test_mcx_no_worker_condition(self, n_ctrl, n_work, expected):
        """The no-worker rule only applies to an MCX without work wires."""
        wires = list(range(n_ctrl + 1))
        work_wires = list(range(n_ctrl + 1, n_ctrl + 1 + n_work))
        op = qp.MultiControlledX(wires=wires, work_wires=work_wires)
        assert mcx_no_work_wires.is_applicable(**op.arguments) is expected
