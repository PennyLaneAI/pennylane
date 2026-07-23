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
Tests for the OutSquare template.
"""

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.ops.op_math import Adjoint
from pennylane.templates.subroutines.arithmetic.out_square import (
    OutSquare,
    _out_square_with_adder_zeroed,
    _out_square_with_caddsub,
)


@pytest.mark.parametrize("output_wires_zeroed", [False, True])
@pytest.mark.jax
def test_standard_validity_out_square(output_wires_zeroed):
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2, 3]
    output_wires = [4, 5, 6, 7, 8, 9, 10]
    work_wires = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    op = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
    qp.ops.functions.assert_valid(op)


def _test_square_correctness(all_wires, rule, seed, output_wires_zeroed, use_jit):
    """Test the correctness of a decomposition rule for an ``OutMultiplier`` op."""
    x_wires, output_wires, work_wires = all_wires
    total_wires = x_wires + output_wires
    if work_wires:
        total_wires += work_wires

    dev = qp.device("lightning.qubit")

    @qp.qnode(dev)
    def circuit(x_state, y_state, output_state):
        qp.StatePrep(x_state, x_wires)
        qp.StatePrep(y_state, output_wires)
        rule(x_wires, output_wires, work_wires, output_wires_zeroed)
        qp.adjoint(qp.StatePrep)(output_state, x_wires + output_wires)
        return qp.probs(wires=total_wires)

    if use_jit:
        circuit = qp.qjit(qp.decompose(circuit, max_expansion=2))

    rng = np.random.default_rng(seed)

    num_x = 2 ** len(x_wires)
    x_state = rng.random(num_x) + 1j * rng.random(num_x)
    x_state /= np.linalg.norm(x_state)

    num_y = 2 ** len(output_wires)

    if output_wires_zeroed:
        y_state = np.zeros(num_y, dtype=complex)
        y_state[0] = 1.0
    else:
        y_state = rng.random(num_y) + 1j * rng.random(num_y)
        y_state /= np.linalg.norm(y_state)

    output_state = np.zeros((num_x, num_y), dtype=complex)
    # To compute the output state, we populate the slice of the total state corresponding to
    # basis state `x` on `x_wires` with the amplitudes of `y_state`, multiplied by that particular
    # amplitude for `x` from `x_state` and permuted by the classical reversible logic of the
    # template, which is just to add the square of `x` to the computational basis state. np.roll
    # accomplished exactly this permutation, including the periodic behaviour across the end
    # of the state vector (represented by the fact that we compute everything modulus the size of
    # the output register).
    for x in range(num_x):
        output_state[x] = x_state[x] * np.roll(y_state, x**2)
    output_state = output_state.reshape(-1)

    probs = circuit(x_state, y_state, output_state)
    assert np.isclose(probs[0], 1.0)
    assert np.allclose(probs[1:], 0.0)


class TestOutSquare:
    """Test the OutSquare template."""

    @pytest.mark.catalyst
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    def test_qjit_dynamic_wires(self, output_wires_zeroed):
        """Test the OutSquare template with dynamic wires."""
        x_wires = np.array([0, 1, 2, 3])
        output_wires = np.array([4, 5, 6, 7, 8])
        work_wires = np.array([9, 10, 11, 12, 13, 14, 15])

        dev = qp.device("lightning.qubit")

        x = 13
        mod = 2 ** len(output_wires)
        if output_wires_zeroed:
            y = 0
        else:
            y = mod - 2  # Some number close to causing overflows

        fixed_decomps = {"C(SemiAdder)": qp.list_decomps("C(SemiAdder)")[0]}

        @qp.qjit
        @qp.set_shots(1)
        @qp.decompose(max_expansion=2, fixed_decomps=fixed_decomps)
        @qp.qnode(dev)
        def circuit(x, y, x_wires, work_wires):
            qp.BasisEmbedding(x, wires=x_wires)
            qp.BasisEmbedding(y, wires=output_wires)
            OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
            return (
                qp.sample(wires=x_wires),
                qp.sample(wires=output_wires),
                qp.sample(wires=work_wires),
            )

        output = circuit(x, y, x_wires, work_wires)
        out_ints = [int("".join(map(str, out[0])), 2) for out in output]
        assert np.allclose(out_ints, [x, (y + x**2) % mod, 0])

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [3, 10, 11],
                "None of the wires in work_wires should be included in output_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [1, 10, 9],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [2, 4, 5],
                [9, 10, 6],
                "None of the wires in output_wires should be included in x_wires.",
            ),
        ],
    )
    def test_wires_overlap_error(
        self, x_wires, output_wires, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some registers overlap."""
        with pytest.raises(ValueError, match=msg_match):
            OutSquare(x_wires, output_wires, work_wires)

    @pytest.mark.parametrize(
        "num_x_wires, num_output_wires, num_work_wires, zeroed, should_raise",
        [
            (6, 6, 2, True, False),
            (6, 6, 1, True, True),
            (5, 10, 4, True, False),
            (5, 10, 3, True, True),
            (3, 3, 3, False, False),
            (3, 3, 2, False, True),
            (2, 3, 3, False, False),
            (2, 3, 2, False, True),
            (1, 3, 3, False, False),
            (1, 3, 2, False, True),
            (1, 3, 1, False, True),
        ],
    )
    def test_work_wire_number(
        self, num_x_wires, num_output_wires, num_work_wires, zeroed, should_raise
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised (only) when too few work wires are supplied."""
        wires = qp.registers(
            {"x_wires": num_x_wires, "output_wires": num_output_wires, "work_wires": num_work_wires}
        )
        if should_raise:
            msg_match = "OutSquare requires at least"
            with pytest.raises(ValueError, match=msg_match):
                OutSquare(**wires, output_wires_zeroed=zeroed)
        else:
            OutSquare(**wires, output_wires_zeroed=zeroed)

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "output_wires_zeroed", "applicable_rules"),
        [
            ([0, 1, 2, 3], [4, 5, 6], [9], True, [0]),
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11], False, [1]),
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11, 12, 13], True, [0, 1]),
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11, 12, 13], False, [1]),
            ([0, 1], [3, 5], [], True, [0]),
            ([0, 1], [3, 5], [9, 10], False, [1]),
            ([0, 1], [3, 5], [9, 10, 11, 12], True, [0, 1]),
            ([0, 1], [3, 5], [9, 10, 11, 12], False, [1]),
            ([0, 1], [3, 4, 5, 6], [9], True, [0]),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12, 13], True, [0, 1]),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12, 13], False, [1]),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12, 13, 14], False, [1]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11], True, [0]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14], True, [0, 1]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14], False, [1]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14, 15], False, [1]),
        ],
    )
    @pytest.mark.parametrize("use_jit", [pytest.param(True, marks=(pytest.mark.catalyst,)), False])
    def test_decomposition_new(
        self,
        x_wires,
        output_wires,
        work_wires,
        output_wires_zeroed,
        applicable_rules,
        use_jit,
        seed,
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
        for j, rule in enumerate(qp.list_decomps(OutSquare)):
            applicable = rule.is_applicable(**op.resource_params)
            assert applicable is (j in applicable_rules)
            _test_decomposition_rule(op, rule)
            if applicable:
                all_wires = (x_wires, output_wires, work_wires)
                _test_square_correctness(all_wires, rule, seed, output_wires_zeroed, use_jit)

    def test_adder_decomposition_output_wires_zeroed(self):
        """Test that the controlled adder decomposition has the expected structure with
        ``output_wires_zeroed=True``."""
        x_wires, output_wires, work_wires = (
            [0, 1, 2],
            [3, 4, 5, 6],
            [7, 8, 9, 10],
        )
        with qp.queuing.AnnotatedQueue() as q:
            _out_square_with_adder_zeroed(x_wires, output_wires, work_wires)

        expected = [
            # controlled copy
            qp.TemporaryAND(wires=[2, 1, 4]),
            qp.TemporaryAND(wires=[2, 0, 3]),
            qp.TemporaryAND(wires=[1, 4, 7]),
            qp.MultiControlledX(
                wires=[1, 7, 3],
                control_values=[True, True],
                work_wires=[8, 9, 10, 6, 5],
                work_wire_type="zeroed",
            ),
            Adjoint(qp.TemporaryAND(wires=[1, 4, 7])),
            qp.CNOT(wires=[1, 4]),
            qp.CNOT([2, 6]),
        ]
        assert q.queue == expected

    def test_caddsub_decomposition_output_wires_zeroed(self):
        """Test that the controlled-add/subtract decomposition has the expected structure with
        ``output_wires_zeroed=True``."""
        x_wires, output_wires, work_wires = [0, 1], [2, 3, 4], [5, 6, 7, 8]

        with qp.queuing.AnnotatedQueue() as q:
            _out_square_with_caddsub(x_wires, output_wires, work_wires, output_wires_zeroed=True)

        expected = [
            # Controlled add-subtract block (contains decomposed adder)
            qp.MultiControlledX(wires=[1, 3], control_values=[False]),
            qp.TemporaryAND(wires=[0, 3, 5]),
            qp.MultiControlledX(wires=[1, 5], control_values=[False]),
            qp.CNOT(wires=[5, 2]),
            qp.MultiControlledX(wires=[1, 5], control_values=[False]),
            Adjoint(qp.TemporaryAND(wires=[0, 3, 5])),
            qp.CNOT(wires=[0, 3]),
            qp.MultiControlledX(wires=[1, 3], control_values=[False]),
            # Sparse adder
            qp.TemporaryAND(wires=[1, 4, 5]),
            qp.TemporaryAND(wires=[5, 3, 6]),
            qp.CNOT(wires=[6, 2]),
            Adjoint(qp.TemporaryAND(wires=[5, 3, 6])),
            qp.CNOT(wires=[5, 3]),
            Adjoint(qp.TemporaryAND(wires=[1, 4, 5])),
            qp.CNOT(wires=[1, 4]),
            # Subtractor
            qp.X(3),
            qp.X(2),
            qp.SemiAdder([0], [2, 3], [5, 6, 7, 8]),
            qp.X(3),
            qp.X(2),
            # Shifted adder
            qp.X(1),
            qp.X(2),
            qp.SemiAdder([1], [2], [5, 6, 7, 8]),
            qp.X(2),
            qp.X(1),
        ]
        assert q.queue == expected

    def test_caddsub_decomposition_output_wires_not_zeroed(self):
        """Test that the controlled-add/subtract decomposition has the expected structure with
        ``output_wires_zeroed=False``."""
        x_wires, output_wires, work_wires = [0, 1, 2], [3, 4, 5], [6, 7]

        with qp.queuing.AnnotatedQueue() as q:
            _out_square_with_caddsub(x_wires, output_wires, work_wires, output_wires_zeroed=False)

        expected = [
            # Controlled add-subtract block (contains decomposed adder)
            qp.ctrl(qp.BasisState([1], [0]), control=[2], control_values=[False]),
            qp.MultiControlledX(wires=[2, 4], control_values=[False]),
            qp.TemporaryAND(wires=[1, 4, 6]),
            qp.MultiControlledX(wires=[2, 6], control_values=[False]),
            qp.CNOT(wires=[6, 3]),
            qp.CNOT(wires=[0, 3]),
            qp.MultiControlledX(wires=[2, 6], control_values=[False]),
            Adjoint(qp.TemporaryAND(wires=[1, 4, 6])),
            qp.CNOT(wires=[1, 4]),
            qp.MultiControlledX(wires=[2, 4], control_values=[False]),
            qp.ctrl(qp.BasisState([1], [0]), control=[2], control_values=[False]),
            # Sparse adder
            qp.TemporaryAND(wires=[2, 5, 6]),
            qp.TemporaryAND(wires=[6, 4, 7]),
            qp.CNOT(wires=[7, 3]),
            Adjoint(qp.TemporaryAND(wires=[6, 4, 7])),
            qp.CNOT(wires=[6, 4]),
            Adjoint(qp.TemporaryAND(wires=[2, 5, 6])),
            qp.CNOT(wires=[2, 5]),
            # Subtractor
            qp.X(4),
            qp.X(3),
            qp.SemiAdder([0, 1], [3, 4], [6, 7]),
            qp.X(4),
            qp.X(3),
        ]
        assert q.queue == expected
