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
Tests for the OutMultiplier template.
"""

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.out_multiplier import OutMultiplier, _add_plus_one


class TestBuildingBlocks:

    @pytest.mark.parametrize(
        "num_x_wires, num_y_wires, num_work_wires",
        [(1, 1, 0), (1, 1, 1), (2, 1, 3), (4, 1, 0), (2, 2, 1), (2, 4, 8)],
    )
    def test_add_plus_one_arithmetic(self, num_x_wires, num_y_wires, num_work_wires):
        """Test that the subroutine _add_plus_one computes the correct arithmetic."""
        x_wires = list(range(num_x_wires))
        y_wires = list(range(num_x_wires, num_x_wires + num_y_wires))
        work_wires = list(
            range(num_x_wires + num_y_wires, num_x_wires + num_y_wires + num_work_wires)
        )

        @qp.set_shots(10)
        @qp.qnode(qp.device("default.qubit"))
        def node(x, y):
            qp.BasisEmbedding(x, x_wires)
            qp.BasisEmbedding(y, y_wires)
            _add_plus_one(x_wires, y_wires, work_wires)
            if work_wires:
                return (
                    qp.counts(wires=x_wires),
                    qp.counts(wires=y_wires),
                    qp.counts(wires=work_wires),
                )
            return qp.counts(wires=x_wires), qp.counts(wires=y_wires)

        num_x = 2**num_x_wires
        num_y = 2**num_y_wires
        for x in range(num_x):
            for y in range(num_y):
                output = node(x, y)
                assert all(len(out) == 1 for out in output)
                output = tuple(int(list(out.keys())[0], 2) for out in output)
                if work_wires:
                    assert output == (x, (x + y + 1) % num_y, 0)
                else:
                    assert output == (x, (x + y + 1) % num_y)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("decompose_right_elbow", [False, True])
    def test_add_plus_one_phase(self, decompose_right_elbow, seed):
        """Test that the subroutine _add_plus_one does not incur any phases."""
        x_wires = list(range(3))
        y_wires = list(range(3, 7))
        work_wires = list(range(7, 10))

        gate_set = {
            "StatePrep",
            "X",
            "CNOT",
            "TemporaryAND",
            "T",
            "S",
            "Adjoint(T)",
            "Adjoint(S)",
            "MidMeasureMP",
            "H",
            "Cond",
            "CZ",
            "Adjoint(StatePrep)",
        }
        if not decompose_right_elbow:
            gate_set.add("Adjoint(TemporaryAND)")

        @qp.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("default.qubit"))
        def node(x_state, y_state, transformed_state):
            qp.StatePrep(x_state, x_wires)
            qp.StatePrep(y_state, y_wires)
            _add_plus_one(x_wires, y_wires, work_wires)
            qp.adjoint(qp.StatePrep)(transformed_state, x_wires + y_wires)
            return qp.probs(wires=x_wires + y_wires)

        rng = np.random.default_rng(seed)
        x_state = rng.random(8) + 1j * rng.random(8)
        x_state /= np.linalg.norm(x_state)
        y_state = rng.random(16) + 1j * rng.random(16)
        y_state /= np.linalg.norm(y_state)
        transformed_state = np.zeros(128, dtype=complex)
        for x in range(8):
            transformed_state[x * 16 : (x + 1) * 16] = np.roll(x_state[x] * y_state, x + 1)
        probs = node(x_state, y_state, transformed_state)
        assert np.isclose(probs[0], 1.0)
        assert np.allclose(probs[1:], 0.0)


@pytest.mark.jax
def test_standard_validity_out_multiplier():
    """Check the operation using the assert_valid function."""
    mod = 12
    x_wires = [0, 1]
    y_wires = [2, 3, 4]
    output_wires = [6, 7, 8, 9]
    work_wires = [5, 10]
    op = OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
    qp.ops.functions.assert_valid(op)


def _test_mult_correctness(all_wires, mod, rule, seed, output_wires_zeroed=False):
    """Test the correctness of a decomposition rule for an ``OutMultiplier`` op."""
    x_wires, y_wires, output_wires, work_wires = all_wires
    total_wires = x_wires + y_wires + output_wires
    if work_wires:
        total_wires += work_wires

    dev = qp.device("default.qubit")

    @qp.set_shots(10)
    @qp.qnode(dev)
    def circuit(x_state, y_state, z_state, output_state):
        qp.StatePrep(x_state, x_wires)
        qp.StatePrep(y_state, y_wires)
        qp.StatePrep(z_state, output_wires, pad_with=0.0)
        rule(x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed)
        qp.adjoint(qp.StatePrep)(output_state, x_wires + y_wires + output_wires)
        return qp.probs(wires=total_wires)

    if mod is None:
        mod = 2 ** len(output_wires)

    rng = np.random.default_rng(seed)

    num_x = 2 ** len(x_wires)
    x_state = rng.random(num_x) + 1j * rng.random(num_x)
    x_state /= np.linalg.norm(x_state)

    num_y = 2 ** len(y_wires)
    y_state = rng.random(num_y) + 1j * rng.random(num_y)
    y_state /= np.linalg.norm(y_state)

    if output_wires_zeroed:
        z_state = np.zeros(mod, dtype=complex)
        z_state[0] = 1.0
    else:
        z_state = rng.random(mod) + 1j * rng.random(mod)
        z_state /= np.linalg.norm(z_state)

    num_z = 2 ** len(output_wires)
    output_state = np.zeros((num_x, num_y, mod), dtype=complex)
    for x in range(num_x):
        for y in range(num_y):
            output_state[x, y] = x_state[x] * y_state[y] * np.roll(z_state, x * y)
    output_state = np.concatenate([output_state, np.zeros((num_x, num_y, num_z - mod))], axis=2)
    output_state = output_state.reshape(-1)

    probs = circuit(x_state, y_state, z_state, output_state)
    assert np.isclose(probs[0], 1.0)
    assert np.allclose(probs[1:], 0.0)


class TestOutMultiplier:
    """Test the qp.OutMultiplier template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            ([0, 1, 2], [3, 4, 5], [6, 7, 8], 7, [9, 10]),
            ([0, 1], [3, 4, 5], [6, 7, 8, 2], 14, [9, 10]),
            ([0, 1, 2], [3, 4], [5, 6, 7, 8], 8, [9, 10]),
            ([0, 1, 2, 3], [4, 5], [6, 7, 8, 9, 10], 22, [11, 12]),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8], None, [9, 10]),
            ([0, 1], [3, 4, 5], [6, 7, 8], None, None),
        ],
    )
    def test_operation_result(self, x_wires, y_wires, output_wires, mod, work_wires, seed):
        """Test the correctness of the OutMultiplier template output."""
        # pylint: disable=too-many-arguments
        all_wires = (x_wires, y_wires, output_wires, work_wires)
        _test_mult_correctness(all_wires, mod, OutMultiplier.compute_decomposition, seed)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [1, 10],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [3, 10],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                [0, 1, 2],
                [2, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in y_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 7, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in y_wires should be included in output_wires.",
            ),
            (
                [0, 1, 7],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in x_wires should be included in output_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                9,
                [9, 10],
                "OutMultiplier must have enough wires to represent mod.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                9,
                None,
                "If mod is not",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, y_wires, output_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)

    @pytest.mark.parametrize("work_wires", [None, [9]])
    def test_validation_of_num_work_wires(self, work_wires):
        """Test that when mod is not 2**len(output_wires), validation confirms at least two
        work wires are present, while any work wires are accepted for mod=2**len(output_wires)"""

        # if mod=2**len(output_wires), anything goes
        OutMultiplier(
            x_wires=[0, 1, 2],
            y_wires=[3, 4, 5],
            output_wires=[6, 7, 8],
            mod=8,
            work_wires=work_wires,
        )

        with pytest.raises(ValueError, match="two work wires should be provided"):
            OutMultiplier(
                x_wires=[0, 1, 2],
                y_wires=[3, 4, 5],
                output_wires=[6, 7, 8],
                mod=7,
                work_wires=work_wires,
            )

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, output_wires, mod, work_wires = (
            [0, 1, 2],
            [3, 5],
            [6, 8],
            3,
            [9, 10],
        )
        multiplier_decomposition = (
            OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            .compute_decomposition(
                x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
            )[0]
            .decomposition()
        )

        op_list = []
        if mod != 2 ** len(output_wires):
            qft_output_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:]
        else:
            qft_output_wires = output_wires
            work_wire = None
        op_list.append(qp.QFT(wires=qft_output_wires))
        op_list.append(
            qp.ControlledSequence(
                qp.ControlledSequence(
                    qp.PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires
                ),
                control=y_wires,
            )
        )
        op_list.append(qp.adjoint(qp.QFT)(wires=qft_output_wires))

        for op1, op2 in zip(multiplier_decomposition, op_list):
            qp.assert_equal(op1, op2)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "applicable_rules"),
        [
            ([0, 1, 2], [3, 5], [6, 8], 3, [9, 10, 11], [0]),
            ([0, 1, 2], [3, 6], [5, 8], 4, [], [0]),
            ([0, 1, 2], [3, 6], [5, 8], 4, [9], [0, 1]),
            ([0, 1, 2], [3, 6], [5, 8], 4, [9, 10, 11], [0, 1, 2]),
            ([0], [3, 6], [5, 8], 4, [], [0]),
            ([0], [3, 6], [5, 8], 4, [9], [0, 1]),
            ([0], [3, 6], [5, 8], 4, [9, 10, 11], [0, 1, 2]),
            ([0, 1, 2], [3], [5, 7, 8], None, [9], [0]),
            ([0, 1, 2], [3], [5, 7, 8], None, [9, 10], [0, 1]),
            ([0, 1, 2], [3], [5, 7, 8], None, [9, 10, 11, 12], [0, 1, 2]),
            ([0, 1, 2], [3], [5, 7, 8, 9, 10], None, [11], [0]),
            ([0, 1, 2], [3], [5, 7, 8, 9, 10], None, [11, 12], [0, 1]),
            ([0, 1, 2], [3], [5, 7, 8, 9, 10], None, [11, 12, 13, 14, 15, 16], [0, 1, 2]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], None, [9], [0]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], None, [9, 10, 13], [0, 1]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], None, [9, 10, 13, 14, 15, 16], [0, 1, 2]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], 16, [9, 10], [0]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], 16, [9, 10, 13, 14, 15, 16, 17, 18], [0]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9], [0]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9, 10, 11], [0, 1]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9, 10, 11, 12, 13], [0, 1, 2]),
        ],
    )
    def test_decomposition_new_output_wires_zeroed(
        self, x_wires, y_wires, output_wires, mod, work_wires, applicable_rules, seed
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system
        with output_wires_zeroed=True."""

        op = qp.OutMultiplier(
            x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=True
        )
        for j, rule in enumerate(qp.list_decomps(qp.OutMultiplier)):
            applicable = rule.is_applicable(**op.resource_params)
            assert applicable is (j in applicable_rules)
            _test_decomposition_rule(op, rule)
            if applicable:
                all_wires = (x_wires, y_wires, output_wires, work_wires)
                _test_mult_correctness(all_wires, mod, rule, seed, output_wires_zeroed=True)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "applicable_rules"),
        [
            ([0, 1, 2], [3, 5], [6, 8], 3, [9, 10], [0]),
            ([0, 1, 2], [3, 6], [5, 8], 4, [9], [0]),
            ([0, 1, 2], [3, 6], [5, 8], 4, [9, 10], [0, 1]),
            ([0, 1, 2], [3, 6], [5, 8], 4, [9, 10, 11], [0, 1, 2, 3]),
            ([0], [3, 6], [5, 8], 4, [9], [0]),
            ([0], [3, 6], [5, 8], 4, [9, 10], [0, 1]),
            ([0], [3, 6], [5, 8], 4, [9, 10, 11], [0, 1, 2, 3]),
            ([0, 1, 2], [3], [5, 7, 8], None, [9], [0]),
            ([0, 1, 2], [3], [5, 7, 8], None, [9, 10], [0]),
            ([0, 1, 2], [3], [5, 7, 8], None, [9, 10, 11, 12], [0, 1, 2]),
            ([0, 1, 2], [3], [5, 7, 8], None, [9, 10, 11, 12, 13], [0, 1, 2, 3]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], None, [9], [0]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], None, [9, 10, 13], [0]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], None, [9, 10, 13, 14, 15, 16], [0, 1, 2]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], 16, [9, 10], [0]),
            ([0, 1, 2], [3, 6], [5, 8, 4, 11, 12], 16, [9, 10, 13, 14, 15, 16, 17, 18], [0]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9], [0]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9, 10, 11], [0]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9, 10, 11, 12], [0, 1]),
            ([0, 1], [3, 6], [5, 8, 2, 4], 16, [9, 10, 11, 12, 13], [0, 1, 2]),
            ([0], [3, 6], [5, 8, 2, 4, 7, 9], None, [11, 12, 13, 14, 15, 16, 17], [0, 1, 2]),
        ],
    )
    def test_decomposition_new_non_zero_output_wires(
        self, x_wires, y_wires, output_wires, mod, work_wires, applicable_rules, seed
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system
        with output_wires_zeroed=False (default)."""
        op = qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
        for j, rule in enumerate(qp.list_decomps(qp.OutMultiplier)):
            applicable = rule.is_applicable(**op.resource_params)
            assert applicable is (j in applicable_rules)
            _test_decomposition_rule(op, rule)
            if applicable:
                all_wires = (x_wires, y_wires, output_wires, work_wires)
                _test_mult_correctness(all_wires, mod, rule, seed)

    def test_work_wires_added_correctly(self):
        """Test that no work wires are added if work_wire = None"""
        wires = qp.OutMultiplier(x_wires=[1, 2], y_wires=[3, 4], output_wires=[5, 6]).wires
        assert wires == qp.wires.Wires([1, 2, 3, 4, 5, 6])

    @pytest.mark.external
    def test_qjit_compatible(self):
        """Test that the template is compatible with the QJIT compiler."""
        x, y = 2, 3
        x_list = [1, 0]
        y_list = [1, 1]
        mod = 12
        x_wires = [0, 1]
        y_wires = [2, 3]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]
        dev = qp.device("lightning.qubit")

        @qp.qjit
        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            qp.BasisEmbedding(x_list, wires=x_wires)
            qp.BasisEmbedding(y_list, wires=y_wires)
            OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qp.sample(wires=output_wires)

        # pylint: disable=bad-reversed-sequence
        out = circuit()[0, :]

        assert np.allclose(2 ** np.arange(3, -1, -1) @ out, (x * y) % mod)
