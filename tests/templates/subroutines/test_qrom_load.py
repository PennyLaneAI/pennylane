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
Tests for the QROMLoad template.
"""

from itertools import product

import hint_helpers
import jax
import numpy
import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.decomposition import DecompositionGraph
from pennylane.templates.subroutines.qrom_load import (
    _calculate_select_swap_sizes,
    _qrom_load_to_qrom,
    _select_swap,
    _swap_ops,
)
from pennylane.testing import assert_decomposable

BITSTRINGS = (
    (0, 0, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize(
    "work_wires",
    [
        [],  # falls back to QROM
        [6, 7, 8],  # Select-SWAP without a work wire for the CSWAPs
        [6, 7, 8, 9, 10, 11],  # Select-SWAP with a work wire for the CSWAPs
    ],
)
def test_assert_valid_qrom_load(work_wires):
    """Run standard validity tests."""
    op = qp.QROMLoad(BITSTRINGS, [0, 1, 2], [3, 4, 5], work_wires)
    qp.testing.assert_valid(op, skip_differentiation=True)


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_falsy_zero_as_work_wire():
    """Test that work wire is not treated as a falsy zero."""
    op = qp.QROMLoad(
        ((1,), (0,), (0,), (1,)),
        control_wires=[1, 2],
        target_wires=[3],
        work_wires=0,
    )
    qp.testing.assert_valid(op, skip_differentiation=True)


class TestQROMLoad:
    """Test the qp.QROMLoad template."""

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("bitstrings", "target_wires", "control_wires", "work_wires"),
        [
            (
                np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                    ]
                ),
                np.array([0, 1, 2]),
                np.array([3, 4]),
                None,
            ),
            (
                np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                    ]
                ),
                np.array([0, 1, 2]),
                np.array([3, 4]),
                None,
            ),
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5],
            ),
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5, 6, 7, 8, 9],
            ),
            (
                np.array([[0, 1], [0, 1], [0, 0], [0, 0]]),
                np.array(["a", "b"]),
                np.array([2, 3]),
                np.array([4, 5, 6]),
            ),
            (
                np.array(
                    [
                        [1, 1, 1],
                        [0, 0, 1],
                        [0, 0, 0],
                        [1, 0, 0],
                    ]
                ),
                np.array([0, 1, "b"]),
                np.array([2, 3]),
                None,
            ),
            (
                np.array(
                    [
                        [1, 1, 1, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 0],
                        [1, 0, 1, 0],
                    ]
                ),
                np.array([0, 1, "b", "d"]),
                np.array([2, 3]),
                np.array(["a", 5, 6, 7]),
            ),
            # Single bitstring with multiple controls and work wires (num_blocks=1 unary path).
            (
                [[1, 0]],
                [0, 1],
                [2, 3],
                [4],
            ),
            (
                np.array([[1, 0, 1]]),
                np.array([0, 1, 2]),
                np.array([3, 4, 5]),
                np.array([6]),
            ),
            (
                [[0, 1]],
                [0, 1],
                [2, 3, 4],
                [5, 6],
            ),
        ],
    )
    def test_operation_result(self, bitstrings, target_wires, control_wires, work_wires):
        """Test the correctness of the QROMLoad template output."""
        dev = qp.device("default.qubit")

        if not isinstance(bitstrings, numpy.ndarray):
            bitstrings, control_wires, target_wires, work_wires = (
                qp.math.array(bitstrings, like="jax"),
                qp.math.array(control_wires, like="jax"),
                qp.math.array(target_wires, like="jax"),
                qp.math.array(work_wires, like="jax"),
            )

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit(j):
            qp.BasisEmbedding(qp.math.int_to_binary(j, len(control_wires)), wires=control_wires)
            qp.QROMLoad(bitstrings, control_wires, target_wires, work_wires)
            return qp.sample(wires=target_wires)

        num_bitstrings = len(bitstrings)
        for j in range(2 ** len(control_wires)):
            expected = bitstrings[j] if j < num_bitstrings else [0] * len(target_wires)
            assert np.allclose(circuit(j), [int(bit) for bit in expected])

    @pytest.mark.parametrize("work_wires", [[6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11, 12]])
    def test_select_swap_with_excess_select_work_wires(self, work_wires):
        """Test that the Select-SWAP decomposition is correct when more work wires
        are provided than the cost-optimal depth uses."""
        bitstrings = [[(i // 3) % 2] for i in range(32)]
        control_wires = [0, 1, 2, 3, 4]
        target_wires = [5]
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(j):
            qp.BasisEmbedding(qp.math.int_to_binary(j, len(control_wires)), wires=control_wires)
            qp.QROMLoad(bitstrings, control_wires, target_wires, work_wires)
            return qp.probs(wires=target_wires), qp.probs(wires=work_wires)

        for j, bitstring in enumerate(bitstrings):
            target_probs, work_probs = circuit(j)
            assert np.isclose(target_probs[bitstring[0]], 1.0)
            assert np.isclose(work_probs[0], 1.0)

    def test_adjoint_unloads(self, seed):
        """Test that the adjoint of ``QROMLoad`` unloads the data again, returning the target and
        work wires to the zero state."""
        rng = np.random.default_rng(seed)
        bitstrings = rng.integers(0, 2, size=(32, 1))
        wires = qp.registers({"control_wires": 5, "target_wires": 1, "work_wires": 5})
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            for wire in wires["control_wires"]:
                qp.Hadamard(wire)
            qp.QROMLoad(bitstrings, **wires)
            qp.adjoint(qp.QROMLoad(bitstrings, **wires))
            return qp.probs(wires=wires["target_wires"] + wires["work_wires"])

        assert np.isclose(circuit()[0], 1.0)

    def test_fallback_always_applicable(self, seed):
        """Test that the plain QROM fallback is applicable even if Select-SWAP is."""
        bitstrings = np.random.default_rng(seed).integers(0, 2, size=(32, 1))
        wires = qp.registers({"control_wires": 5, "target_wires": 1, "work_wires": 5})
        op = qp.QROMLoad(bitstrings, **wires)
        assert _select_swap.is_applicable(**op.arguments)
        assert _qrom_load_to_qrom.is_applicable(**op.arguments)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        bitstrings = [[(i // 3) % 2] for i in range(32)]
        qrom_decomposition = qp.QROMLoad(
            bitstrings,
            control_wires=[0, 1, 2, 3, 4],
            target_wires=[5],
            work_wires=[6, 7, 8, 9, 10],
        ).decomposition()

        # Four bitstrings are loaded in parallel, into the target wire and three work wires
        columns = [sum(bitstrings[4 * i : 4 * i + 4], []) for i in range(8)]
        qrom = qp.QROM(
            columns, control_wires=[0, 1, 2], target_wires=[5, 6, 7, 8], work_wires=[9, 10]
        )
        cswaps = [
            qp.ctrl(qp.SWAP(wires), control=c, work_wires=[9], work_wire_type="zeroed")
            for c, wires in [(4, [5, 6]), (3, [5, 7]), (3, [6, 8])]
        ]
        expected_gates = ([qp.Hadamard(5)] + cswaps + [qrom] + cswaps[::-1]) * 2

        assert len(qrom_decomposition) == len(expected_gates)
        for op1, op2 in zip(qrom_decomposition, expected_gates):
            qp.assert_equal(op1, op2)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize(
        ("num_bitstrings", "control_wires", "target_wires", "work_wires"),
        [
            (1, [], [0], [1]),
            (1, [0], [1], []),
            (2, [0], [1], []),
            (1, [0, 1], [2, 3], [4]),
            (3, [0, 1], [2], [3]),
            (3, [0, 1], [2], [3, 4]),
            (4, [0, 1], [2], []),
            (4, [2, 3], [0, 1], [4, 5, 6, 7, 8, 9]),
            (8, [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11]),
            (8, [0, 1, 2], [3, 4], [5]),
            (8, [0, 1, 2], [3, 4], [5, 6]),
            (7, [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11]),
            (5, [0, 1, 2], [3, 4], [5]),
            (6, [0, 1, 2], [3, 4], [5, 6]),
            (1, [0, 1, 2], [3, 4, 5], [6]),
            (1, [0, 1, 2, 6], [3, 4, 5], [7, 8, 9]),
            (5, [0, 1, 2, 3], [4], []),
            (11, [0, 1, 2, 3], [4, 5], [6, 7]),
        ],
    )
    def test_decomposition_new(self, num_bitstrings, control_wires, target_wires, work_wires, seed):
        """Tests the decomposition rule implemented with the new system."""
        rng = np.random.default_rng(seed)
        bitstrings = rng.integers(0, 2, size=(num_bitstrings, len(target_wires)))
        op = qp.QROMLoad(
            bitstrings,
            control_wires=control_wires,
            target_wires=target_wires,
            work_wires=work_wires,
        )
        assert_decomposable(op)

    @pytest.mark.jax
    def test_traced_wires(self):
        """Test that QROMLoad construction and decomposition do not raise
        TracerBoolConversionError when wires are JAX tracers."""

        jax.config.update("jax_enable_x64", True)

        def build_and_decompose(bitstrings, control_wires, target_wires, work_wires):
            op = qp.QROMLoad(bitstrings, control_wires, target_wires, work_wires)
            op.decomposition()
            return True

        n, m, w = 2, 2, 1
        bitstrings = qp.math.array([[1, 0], [0, 1], [1, 1], [0, 0]], like="jax")
        control_wires = qp.math.arange(0, n, like="jax")
        target_wires = qp.math.arange(n, n + m, like="jax")
        work_wires = qp.math.arange(n + m, n + m + w, like="jax")

        jax.make_jaxpr(build_and_decompose)(bitstrings, control_wires, target_wires, work_wires)

    @pytest.mark.capture
    def test_decomposition_rule_capture_dynamic_wires(self):
        """Test that the Select-SWAP decomposition rule can be captured with dynamic wires."""
        wire_lens = {"control_wires": 5, "target_wires": 1, "work_wires": 5}
        registers = qp.registers(wire_lens)
        kwargs = {
            "bitstrings": qp.math.zeros(
                (2 ** wire_lens["control_wires"], wire_lens["target_wires"]),
                like="jax",
                dtype=int,
            ),
            **{
                name: qp.math.array(wires, like="jax", dtype=int)
                for name, wires in registers.items()
            },
        }

        # pylint: disable-next=protected-access
        rule = qp.capture.subroutine(_select_swap._impl)
        _ = jax.make_jaxpr(rule)(**kwargs)

    @pytest.mark.capture
    @pytest.mark.parametrize("adjoint", [True, False])
    @pytest.mark.parametrize(("num_swap_controls", "num_targets"), [(1, 2), (2, 1), (3, 2)])
    def test_swap_ops_num_iters_hint(self, num_swap_controls, num_targets, adjoint):
        """Test that the inner loop of the SWAP network carries a ``num-iters`` hint that
        reproduces the number of controlled SWAP gates."""
        depth = 2**num_swap_controls
        swap_control_wires = list(range(num_swap_controls))
        swap_wires = list(range(num_swap_controls, num_swap_controls + depth * num_targets))
        target_wires = swap_wires[:num_targets]

        plxpr = jax.make_jaxpr(
            lambda: _swap_ops(swap_control_wires, swap_wires, target_wires, [], adjoint=adjoint)
        )()
        ops = qp.testing.plxpr_to_tape(plxpr.jaxpr, plxpr.consts).operations
        num_cswaps = sum(isinstance(op, qp.ops.Controlled) for op in ops)
        assert num_cswaps == num_targets * (depth - 1)

        # Outer loop, inner loop, SWAP layer
        num_inner_iters = num_cswaps / num_targets / num_swap_controls
        assert hint_helpers.loop_hints(plxpr.jaxpr) == [
            None,
            pytest.approx(num_inner_iters),
            None,
        ]


@pytest.mark.parametrize(
    ("control_wires", "target_wires", "work_wires", "msg_match"),
    [
        (
            [0, 1, 2],
            [0, 3],
            [4, 5],
            "control_wires and target_wires must not overlap",
        ),
        (
            [0, 1, 2],
            [4],
            [2, 5],
            "control_wires and work_wires must not overlap",
        ),
        (
            [0, 1, 2],
            [4],
            [4],
            "target_wires and work_wires must not overlap",
        ),
    ],
)
def test_wires_error(control_wires, target_wires, work_wires, msg_match):
    """Test an error is raised when a control wire is in one of the ops"""
    with pytest.raises(ValueError, match=msg_match):
        qp.QROMLoad([[1]] * 8, control_wires, target_wires, work_wires)


@pytest.mark.parametrize(
    ("bitstrings", "control_wires", "target_wires", "msg_match"),
    [
        (
            [[1], [0], [0], [1]],
            [0],
            [2],
            (
                r"Not enough control wires \(1\) for the desired number of bitstrings \(4\). "
                "At least 2 control wires are required."
            ),
        ),
        (
            [[1], [0], [0], [1]],
            [0, 1],
            [2, 3],
            "Bitstring length must match the number of target wires.",
        ),
    ],
)
def test_wrong_wires_error(bitstrings, control_wires, target_wires, msg_match):
    """Test that error is raised if more ops are requested than can fit in control wires"""
    with pytest.raises(ValueError, match=msg_match):
        qp.QROMLoad(bitstrings, control_wires, target_wires, work_wires=None)


def test_too_many_work_wires_case():
    """Test that QROMLoad does not use more work wires than the cost-optimal depth requires."""
    bitstrings = np.array([[1], [0], [1], [1]] * 8)
    control_wires = [0, 1, 2, 3, 4]
    gates_few_work = qp.QROMLoad(bitstrings, control_wires, [5], list(range(6, 11))).decomposition()
    gates_many_work = qp.QROMLoad(
        bitstrings, control_wires, [5], list(range(6, 21))
    ).decomposition()

    assert len(gates_few_work) == len(gates_many_work)
    for op1, op2 in zip(gates_few_work, gates_many_work):
        qp.assert_equal(op1, op2)


@pytest.mark.parametrize(
    ("num_bitstrings", "n_ctrl", "n_target", "n_work", "expected"),
    [
        (16, 4, 10, 5, (4, 5, 0, 1, 1)),
        (7, 3, 2, 2, (3, 2, 0, 1, 1)),
        (256, 8, 2, 11, (6, 5, 6, 1, 4)),
        # Depth 8 would fit, but is more expensive than depth 4
        (32, 5, 1, 13, (3, 10, 3, 1, 4)),
        # Select-SWAP does not pay off for tiny examples, no matter the number of work wires
        (4, 2, 1, 1, (2, 1, 0, 1, 1)),
        (4, 2, 1, 5, (2, 5, 0, 1, 1)),
        # Quick-return path (n_work < n_ctrl-1): We get depth 1, no swap work wires and copied
        # control/work wires for the Select part
        (14, 4, 2, 2, (4, 2, 0, 0, 1)),
        (256, 8, 2, 6, (8, 6, 0, 0, 1)),
        (4, 2, 1, 0, (2, 0, 0, 0, 1)),
    ],
)
def test_calculate_select_swap_sizes(num_bitstrings, n_ctrl, n_target, n_work, expected):
    """Test the allocation logic for Select vs Swap work wires."""

    # result contains (num_control_wires_select, num_work_wires_select, num_work_wires_swap,
    # num_work_wires_cswap, depth)
    output = _calculate_select_swap_sizes(
        num_bitstrings=num_bitstrings,
        num_control_wires=n_ctrl,
        num_targets=n_target,
        num_work_wires=n_work,
    )
    assert output == expected
    assert _calculate_select_swap_sizes(num_bitstrings, n_ctrl, n_target, n_work) == output

    # Extra validation on top of hardcoded expected values
    (
        num_control_wires_select,
        num_work_wires_select,
        num_work_wires_swap,
        num_work_wires_cswap,
        depth,
    ) = output
    assert num_work_wires_select + num_work_wires_swap == n_work  # all work wires are allocated
    new_n_target = int(np.ceil(n_target / depth))
    new_n_ctrl = qp.math.ceil_log2(new_n_target)
    # Select has enough work wires for unary iteration if originally that was the case
    if n_work >= n_ctrl - 1:
        assert num_work_wires_select >= num_control_wires_select - 1
    else:
        assert num_work_wires_cswap == 0
    assert num_work_wires_select >= new_n_ctrl - 1


def test_cswap_work_wire_always_available():
    """Test that the cost-optimal Select-SWAP decomposition always leaves an idle work wire for
    the controlled SWAPs, which the cost model relies on to count them as elbow pairs."""
    for n_ctrl in range(1, 8):
        for num_bitstrings, n_target, n_work in product(
            range(1, 2**n_ctrl + 1), range(1, 4), range(20)
        ):
            *_, num_work_wires_cswap, depth = _calculate_select_swap_sizes(
                num_bitstrings, n_ctrl, n_target, n_work
            )
            assert depth == 1 or num_work_wires_cswap == 1


@pytest.mark.parametrize(
    "sizes", [(21, 5, 1, 5), (32, 5, 1, 5), (64, 6, 1, 9), (64, 6, 2, 20), (256, 8, 2, 11)]
)
def test_select_swap_t_count(sizes, seed):
    """Test that the Select-SWAP decomposition, if applicable, uses fewer T gates than a plain
    QROM."""
    num_bitstrings, n_ctrl, n_target, n_work = sizes
    bitstrings = numpy.random.default_rng(seed).integers(0, 2, size=(num_bitstrings, n_target))
    wires = qp.registers({"control_wires": n_ctrl, "target_wires": n_target, "work_wires": n_work})
    op = qp.QROMLoad(bitstrings, **wires)
    assert _select_swap.is_applicable(**op.arguments)
    # Weigh T gates heavily so that the graph minimizes the T count, which is what we compare
    gate_set = {"T": 100, "Adjoint(T)": 100, "H": 1, "S": 1, "Adjoint(S)": 1, "CNOT": 1, "X": 1}
    gate_set |= {"Z": 1, "CZ": 1, "GlobalPhase": 1, "MidMeasure": 1}

    def t_count(op):
        solution = DecompositionGraph([op], gate_set).solve()
        counts = solution.resource_estimate(op).gate_counts
        return solution, sum(
            num for gate, num in counts.items() if str(gate) in {"T", "Adjoint(T)"}
        )

    solution, select_swap_t_count = t_count(op)
    assert solution.decomposition(op) is _select_swap
    assert 0 < select_swap_t_count < t_count(qp.QROM(bitstrings, **wires))[1]
