# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the Adder template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity_Adder():
    """Check the operation using the assert_valid function."""
    k = 6
    mod = 11
    x_wires = [0, 1, 2, 3]
    work_wires = [4, 5]
    op = qml.Adder(k, x_wires=x_wires, mod=mod, work_wires=work_wires)
    qml.ops.functions.assert_valid(op)


class TestAdder:
    """Test the qml.Adder template."""

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wires", "x"),
        [
            (6, [0, 1, 2], 8, [4, 5], 1),
            (
                6,
                ["a", "b", "c"],
                8,
                ["d", "e"],
                2,
            ),
            (
                1,
                [0, 1, 2, 3],
                9,
                [4, 5],
                3,
            ),
            (
                2,
                [0, 1, 4],
                4,
                [3, 2],
                2,
            ),
            (
                0,
                [0, 1, 4],
                4,
                [3, 2],
                2,
            ),
            (
                1,
                [0, 1, 4],
                4,
                [3, 2],
                0,
            ),
            (
                -3,
                [0, 1, 4],
                4,
                [3, 2],
                1,
            ),
            (
                10,
                [0, 1, 2, 5],
                9,
                [3, 4],
                2,
            ),
            (
                1,
                [0, 1, 2],
                7,
                [3, 4],
                3,
            ),
            (
                6,
                [0, 1, 2, 3],
                None,
                [4, 5],
                3,
            ),
        ],
    )
    def test_operation_result(
        self, k, x_wires, mod, work_wires, x
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the PhaseAdder template output."""
        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Adder(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

        if mod is None:
            mod = 2 ** len(x_wires)

        # pylint: disable=bad-reversed-sequence
        out = list(circuit(x)[0, :])
        result = sum(bit * (2**i) for i, bit in enumerate(reversed(out)))
        assert np.allclose(result, (x + k) % mod)

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wires", "msg_match"),
        [
            (
                1,
                [0, 1, 2],
                9,
                [3, 4],
                ("Adder must have enough x_wires to represent mod."),
            ),
            (
                3,
                [0, 1, 2, 3, 4],
                12,
                [4, 5],
                "None of the wires in work_wires should be included in x_wires.",
            ),
        ],
    )
    def test_operation_and_test_wires_error(
        self, k, x_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""

        with pytest.raises(ValueError, match=msg_match):
            qml.Adder(k, x_wires, mod, work_wires)

    @pytest.mark.parametrize("work_wires", [None, [3], [3, 4, 5]])
    def test_validation_of_num_work_wires(self, work_wires):
        """Test that when mod is not 2**len(x_wires), validation confirms two
        work wires are present, while any work wires are accepted for mod=2**len(x_wires)"""

        # if mod=2**len(x_wires), anything goes
        qml.Adder(1, [0, 1, 2], mod=8, work_wires=work_wires)

        with pytest.raises(ValueError, match="two work wires should be provided"):
            qml.Adder(1, [0, 1, 2], mod=9, work_wires=work_wires)

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wires", "msg_match"),
        [
            (
                2.3,
                [0, 1, 2],
                9,
                [3, 4],
                ("Both k and mod must be integers"),
            ),
            (
                2,
                [0, 1, 2],
                3.2,
                [3, 4],
                ("Both k and mod must be integers"),
            ),
        ],
    )
    def test_types_error(
        self, k, x_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test errors are raised"""
        with pytest.raises(ValueError, match=msg_match):
            qml.Adder(k, x_wires, mod, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""

        k = 2
        mod = 7
        x_wires = [0, 1, 2]
        work_wires = [3, 4]
        adder_decomposition = (
            qml.Adder(k, x_wires, mod, work_wires)
            .compute_decomposition(k, x_wires, mod, work_wires)[0]
            .decomposition()
        )

        op_list = []
        op_list.append(qml.QFT(work_wires[:1] + x_wires))
        op_list.append(qml.PhaseAdder(k, work_wires[:1] + x_wires, mod, work_wires[1:]))
        op_list.append(qml.adjoint(qml.QFT)(work_wires[:1] + x_wires))

        for op1, op2 in zip(adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize("mod", [2, 4])
    def test_controlled_decomposition(self, mod):
        """Tests the decomposition works for the controlled adder."""

        k = 4
        x_wires = [2, 3]
        control_wires = [1]
        work_wires = [4, 5]
        wire_order = control_wires + x_wires + work_wires

        ctrl_op1 = qml.ops.Controlled(
            qml.change_op_basis(
                qml.QFT(work_wires[:1] + x_wires),
                qml.PhaseAdder(k, work_wires[:1] + x_wires, mod, work_wires[1:]),
            ),
            control_wires,
            [1],
        )

        ctrl_op2 = qml.prod(
            qml.adjoint(qml.QFT(work_wires[:1] + x_wires)),
            qml.ctrl(
                qml.PhaseAdder(k, work_wires[:1] + x_wires, mod, work_wires[1:]),
                control=control_wires,
            ),
            qml.QFT(work_wires[:1] + x_wires),
        )

        mat1, mat2 = qml.matrix(ctrl_op1, wire_order), qml.matrix(ctrl_op2, wire_order)
        assert qml.math.allclose(mat1, mat2)

    @pytest.mark.parametrize("mod", [7, 8])
    def test_decomposition_new(self, mod):
        """Tests the decomposition rule implemented with the new system."""

        k = 4
        x_wires = [2, 3, 4]
        work_wires = [0, 1]
        op = qml.Adder(k, x_wires, mod, work_wires)
        for rule in qml.list_decomps(qml.Adder):
            _test_decomposition_rule(op, rule)

    def test_work_wires_added_correctly(self):
        """Test that no work wires are added if work_wire = None"""
        wires = qml.Adder(1, x_wires=[1, 2]).wires
        assert wires == qml.wires.Wires([1, 2])

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)
        x = 2
        k = 6
        mod = 7
        x_wires = [0, 1, 2]
        work_wires = [3, 4]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Adder(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

        # pylint: disable=bad-reversed-sequence
        result = sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()[0, :])))
        assert jax.numpy.allclose(result, (x + k) % mod)
