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

from functools import partial

import hint_helpers
import jax
import numpy
import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.templates.subroutines.qrom_load import (
    _calculate_select_swap_sizes,
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
@pytest.mark.parametrize("clean", [True, False])
@pytest.mark.parametrize(
    "work_wires",
    [
        [],  # falls back to QROM
        [6, 7, 8],  # Select-SWAP without a work wire for the CSWAPs
        [6, 7, 8, 9, 10, 11],  # Select-SWAP with a work wire for the CSWAPs
    ],
)
def test_assert_valid_qrom_load(work_wires, clean):
    """Run standard validity tests."""
    op = qp.QROMLoad(BITSTRINGS, [0, 1, 2], [3, 4, 5], work_wires, clean=clean)
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
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        ("bitstrings", "target_wires", "control_wires", "work_wires", "clean"),
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
                False,
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
                True,
            ),
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5],
                True,
            ),
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5, 6, 7, 8, 9],
                True,
            ),
            (
                np.array([[0, 1], [0, 1], [0, 0], [0, 0]]),
                np.array(["a", "b"]),
                np.array([2, 3]),
                np.array([4, 5, 6]),
                False,
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
                False,
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
                True,
            ),
            # Single bitstring with multiple controls and work wires (num_blocks=1 unary path).
            (
                [[1, 0]],
                [0, 1],
                [2, 3],
                [4],
                True,
            ),
            (
                np.array([[1, 0, 1]]),
                np.array([0, 1, 2]),
                np.array([3, 4, 5]),
                np.array([6]),
                True,
            ),
            (
                [[0, 1]],
                [0, 1],
                [2, 3, 4],
                [5, 6],
                False,
            ),
        ],
    )
    def test_operation_result(
        self, bitstrings, target_wires, control_wires, work_wires, clean
    ):  # pylint: disable=too-many-arguments
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
            qp.QROMLoad(bitstrings, control_wires, target_wires, work_wires, clean)
            return qp.sample(wires=target_wires)

        num_bitstrings = len(bitstrings)
        for j in range(2 ** len(control_wires)):
            expected = bitstrings[j] if j < num_bitstrings else [0] * len(target_wires)
            assert np.allclose(circuit(j), [int(bit) for bit in expected])

    @pytest.mark.parametrize(
        ("bitstrings", "target_wires", "control_wires", "work_wires"),
        [
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5],
            ),
            (
                [[0, 1], [0, 1], [0, 0], [0, 0]],
                ["a", "b"],
                [2, 3],
                [4, 5, 6],
            ),
            (
                [
                    [1, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0],
                ],
                [0, 1, "b"],
                [2, 3],
                ["a", 5, 6],
            ),
            (
                [
                    [1, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 0],
                    [1, 0, 1, 0],
                ],
                [0, 1, "b", "d"],
                [2, 3],
                ["a", 5, 6, 7],
            ),
        ],
    )
    def test_work_wires_output(self, bitstrings, target_wires, control_wires, work_wires):
        """Tests that the ``clean = True`` version don't modify the initial state in work_wires."""
        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            # Initialize the work wires to a non-zero state
            for ind, wire in enumerate(work_wires):
                qp.RX(ind, wires=wire)

            for wire in control_wires:
                qp.Hadamard(wires=wire)

            qp.QROMLoad(bitstrings, control_wires, target_wires, work_wires)

            for ind, wire in enumerate(work_wires):
                qp.RX(-ind, wires=wire)

            return qp.probs(wires=work_wires)

        assert np.isclose(circuit()[0], 1.0)

    @pytest.mark.parametrize("work_wires", [[3, 4, 5, 6], [3, 4, 5, 6, 7]])
    def test_clean_with_excess_select_work_wires(self, work_wires):
        """Test that the ``clean=True`` Select-SWAP decomposition is correct when the nested Select
        QROM would have enough work wires to use a Select-SWAP decomposition itself."""
        bitstrings = [[1], [1], [0]]
        control_wires = [0, 1]
        target_wires = [2]
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(j):
            qp.BasisEmbedding(qp.math.int_to_binary(j, len(control_wires)), wires=control_wires)
            qp.QROMLoad(bitstrings, control_wires, target_wires, work_wires, clean=True)
            return qp.probs(wires=target_wires), qp.probs(wires=work_wires)

        for j, bitstring in enumerate(bitstrings):
            target_probs, work_probs = circuit(j)
            assert np.isclose(target_probs[bitstring[0]], 1.0)
            assert np.isclose(work_probs[0], 1.0)

    @pytest.mark.parametrize("clean", [True, False])
    def test_adjoint_unloads(self, clean, seed):
        """Test that the adjoint of ``QROMLoad`` unloads the data again, returning the target and
        work wires to the zero state."""
        rng = np.random.default_rng(seed)
        bitstrings = rng.integers(0, 2, size=(8, 2))
        wires = qp.registers({"control_wires": 3, "target_wires": 2, "work_wires": 4})
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            for wire in wires["control_wires"]:
                qp.Hadamard(wire)
            qp.QROMLoad(bitstrings, **wires, clean=clean)
            qp.adjoint(qp.QROMLoad(bitstrings, **wires, clean=clean))
            return qp.probs(wires=wires["target_wires"] + wires["work_wires"])

        assert np.isclose(circuit()[0], 1.0)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        qrom_decomposition = qp.QROMLoad(
            [[1], [0], [0], [1]],
            control_wires=[0, 1],
            target_wires=[2],
            work_wires=[3],
            clean=True,
        ).decomposition()

        expected_gates = [
            qp.Hadamard(wires=[2]),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.QROM([[1, 0], [0, 1]], control_wires=[0], target_wires=[2, 3], work_wires=[]),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.Hadamard(wires=[2]),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.QROM([[1, 0], [0, 1]], control_wires=[0], target_wires=[2, 3], work_wires=[]),
            qp.CSWAP(wires=[1, 2, 3]),
        ]

        for op1, op2 in zip(qrom_decomposition, expected_gates):
            qp.assert_equal(op1, op2)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize(
        ("num_bitstrings", "control_wires", "target_wires", "work_wires", "clean"),
        [
            (1, [], [0], [1], True),
            (1, [0], [1], [], True),
            (1, [0], [1], [], False),
            (2, [0], [1], [], True),
            (2, [0], [1], [], False),
            (4, [2, 3], [0, 1], [4, 5, 6, 7, 8, 9], True),
            (3, [0, 1], [2], [3], True),
            (4, [0, 1], [2], [], False),
            (3, [0, 1], [2], [3, 4], False),
            (8, [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11], False),
            (8, [0, 1, 2], [3, 4], [5], True),
            (8, [0, 1, 2], [3, 4], [5, 6], True),
            (8, [0, 1, 2], [3, 4], [5, 6], False),
            (7, [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11], False),
            (5, [0, 1, 2], [3, 4], [5], True),
            (2, [0, 1, 2], [3, 4], [5, 6], True),
            (6, [0, 1, 2], [3, 4], [5, 6], False),
            (1, [0, 1], [2, 3], [4], True),
            (1, [0, 1, 2], [3, 4, 5], [6], True),
            (1, [0, 1, 2, 6], [3, 4, 5], [7, 8, 9], True),
            (1, [0, 1, 2, 6], [3, 4, 5], [7, 8, 9], False),
            (11, [0, 1, 2, 3], [4, 5], [6, 7], True),
            (5, [0, 1, 2, 3], [4], [], False),
        ],  # pylint: disable=too-many-arguments
    )
    def test_decomposition_new(
        self, num_bitstrings, control_wires, target_wires, work_wires, clean, seed
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        rng = np.random.default_rng(seed)
        bitstrings = rng.integers(0, 2, size=(num_bitstrings, len(target_wires)))
        op = qp.QROMLoad(
            bitstrings,
            control_wires=control_wires,
            target_wires=target_wires,
            work_wires=work_wires,
            clean=clean,
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
    @pytest.mark.parametrize(
        ("wire_lens", "clean"),
        [
            ({"control_wires": 1, "target_wires": 1, "work_wires": 0}, False),
            ({"control_wires": 2, "target_wires": 1, "work_wires": 1}, True),
        ],
    )
    def test_decomposition_rule_capture_dynamic_wires(self, wire_lens, clean):
        """Test that the Select-SWAP decomposition rule can be captured with dynamic wires."""
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

        # ``clean`` is compilable and is therefore static when Catalyst captures this rule.
        # pylint: disable-next=protected-access
        rule = qp.capture.subroutine(partial(_select_swap._impl, clean=clean))
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


def test_none_work_wires_case():
    """Test that clean version is not applied if work wires are not used"""

    gates_clean = qp.QROMLoad(
        np.array([[1], [0], [0], [1]]), [0, 1], [2], [], clean=True
    ).decomposition()
    gates_not_clean = qp.QROMLoad(
        np.array([[1], [0], [0], [1]]), [0, 1], [2], [], clean=False
    ).decomposition()

    assert gates_clean == gates_not_clean


def test_too_many_work_wires_case():
    """Test that QROMLoad works when more work wires are given than necessary"""

    gates_few_work = qp.QROMLoad(
        np.array([[1], [0], [1], [1]]), [0, 1], [2], [3, 4, 5], clean=False
    ).decomposition()
    gates_many_work = qp.QROMLoad(
        np.array([[1], [0], [1], [1]]),
        [0, 1],
        [2],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        clean=False,
    ).decomposition()

    assert len(gates_few_work) == 4 == len(gates_many_work)
    # Compare the QROM manually because the work wires are expected to differ
    for qrom_op in [gates_few_work[0], gates_many_work[0]]:
        assert isinstance(qrom_op, qp.QROM)
        assert np.allclose(qrom_op.bitstrings, np.array([[1, 0, 1, 1]]))
        assert qrom_op.control_wires == qp.wires.Wires([])
        assert qrom_op.target_wires == qp.wires.Wires([2, 3, 4, 5])

    assert gates_few_work[1:] == gates_many_work[1:]


@pytest.mark.parametrize(
    ("num_bitstrings", "n_ctrl", "n_target", "n_work", "expected"),
    [
        (16, 4, 1, 3, (3, 2, 1, 1, 2)),
        (16, 4, 10, 5, (4, 5, 0, 1, 1)),
        (7, 3, 2, 2, (3, 2, 0, 1, 1)),
        (14, 4, 2, 10, (2, 4, 6, 1, 4)),
        (256, 8, 2, 10, (7, 8, 2, 1, 2)),
        (4, 2, 1, 1, (1, 0, 1, 0, 2)),
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

    # Extra validation on top of hardcoded expected values
    (
        num_control_wires_select,
        num_work_wires_select,
        num_work_wires_swap,
        num_work_wires_cswap,
        depth,
    ) = output
    assert num_work_wires_select + num_work_wires_swap == n_work  # all work wires are used
    new_n_target = int(np.ceil(n_target / depth))
    new_n_ctrl = qp.math.ceil_log2(new_n_target)
    # Select has enough work wires for unary iteration if originally that was the case
    if n_work >= n_ctrl - 1:
        assert num_work_wires_select >= num_control_wires_select - 1
    else:
        assert num_work_wires_cswap == 0
    assert num_work_wires_select >= new_n_ctrl - 1
