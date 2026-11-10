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

import itertools
import math
from collections import Counter
from functools import partial
from unittest import mock

import hint_helpers
import jax
import numpy
import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.core.operator import abstractify
from pennylane.decomposition import DecompositionGraph
from pennylane.templates.subroutines import qrom_load
from pennylane.templates.subroutines.qrom import (
    _qrom_measurement_decomposition,
    _qrom_unary_iteration,
)
from pennylane.templates.subroutines.qrom_load import (
    _adjoint_qrom_load_measurement,
    _adjoint_qrom_load_one_hot_size,
    _calculate_select_swap_sizes,
    _qrom_load_to_qrom,
    _select_swap,
    _swap_ops,
)
from pennylane.testing import assert_decomposable
from pennylane.testing.decompositions import _assert_counts_match
from pennylane.typing import Bool, Wire

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

    @pytest.mark.parametrize("work_wires", [[6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11, 12]])
    def test_clean_with_excess_select_work_wires(self, work_wires):
        """Test that the ``clean=True`` Select-SWAP decomposition is correct when more work wires
        are provided than the cost-optimal depth uses."""
        bitstrings = [[(i // 3) % 2] for i in range(32)]
        control_wires = [0, 1, 2, 3, 4]
        target_wires = [5]
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
        bitstrings = rng.integers(0, 2, size=(32, 1))
        wires = qp.registers({"control_wires": 5, "target_wires": 1, "work_wires": 5})
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            for wire in wires["control_wires"]:
                qp.Hadamard(wire)
            qp.QROMLoad(bitstrings, **wires, clean=clean)
            qp.adjoint(qp.QROMLoad(bitstrings, **wires, clean=clean))
            return qp.probs(wires=wires["target_wires"] + wires["work_wires"])

        assert np.isclose(circuit()[0], 1.0)

    @pytest.mark.parametrize("clean", [True, False])
    def test_fallback_not_applicable_with_garbage(self, clean, seed):
        """Test that the plain QROM fallback is not applicable for ``clean=False`` if Select-SWAP
        is. The two rules leave different garbage on the work wires, so the graph could otherwise
        decompose ``QROMLoad`` and its adjoint differently, which would not unload the data."""
        bitstrings = np.random.default_rng(seed).integers(0, 2, size=(32, 1))
        wires = qp.registers({"control_wires": 5, "target_wires": 1, "work_wires": 5})
        op = qp.QROMLoad(bitstrings, **wires, clean=clean)
        assert _select_swap.is_applicable(**op.arguments)
        assert _qrom_load_to_qrom.is_applicable(**op.arguments) == clean

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        bitstrings = [[(i // 3) % 2] for i in range(32)]
        qrom_decomposition = qp.QROMLoad(
            bitstrings,
            control_wires=[0, 1, 2, 3, 4],
            target_wires=[5],
            work_wires=[6, 7, 8, 9, 10],
            clean=True,
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
            ({"control_wires": 5, "target_wires": 1, "work_wires": 5}, True),
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


@pytest.mark.parametrize("clean", [True, False])
def test_too_many_work_wires_case(clean):
    """Test that QROMLoad does not use more work wires than the cost-optimal depth requires."""
    bitstrings = np.array([[1], [0], [1], [1]] * 8)
    control_wires = [0, 1, 2, 3, 4]
    gates_few_work = qp.QROMLoad(
        bitstrings, control_wires, [5], list(range(6, 11)), clean=clean
    ).decomposition()
    gates_many_work = qp.QROMLoad(
        bitstrings, control_wires, [5], list(range(6, 21)), clean=clean
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
    # These cases do not depend on whether the work wires are cleaned up
    assert _calculate_select_swap_sizes(num_bitstrings, n_ctrl, n_target, n_work, False) == output

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


@pytest.mark.parametrize(
    ("num_bitstrings", "n_ctrl", "n_target", "n_work", "expected"),
    [
        (16, 4, 1, 3, {True: (4, 3, 0, 1, 1), False: (3, 2, 1, 1, 2)}),
        (8, 3, 2, 4, {True: (3, 4, 0, 1, 1), False: (2, 2, 2, 1, 2)}),
        # For clean=False, depth 4 would fit, but is not cheaper than depth 2
        (14, 4, 2, 10, {True: (4, 10, 0, 1, 1), False: (3, 8, 2, 1, 2)}),
        (256, 8, 2, 10, {True: (8, 10, 0, 1, 1), False: (7, 8, 2, 1, 2)}),
        (64, 6, 1, 9, {True: (4, 6, 3, 1, 4), False: (3, 2, 7, 1, 8)}),
        (256, 8, 1, 18, {True: (5, 11, 7, 1, 8), False: (4, 3, 15, 1, 16)}),
    ],
)
def test_calculate_select_swap_sizes_clean(num_bitstrings, n_ctrl, n_target, n_work, expected):
    """Test that the cost-optimal depth is smaller for ``clean=True``, because the Select-SWAP
    circuit is applied twice whereas the plain QROM fallback is applied once."""
    for clean in (True, False):
        sizes = _calculate_select_swap_sizes(num_bitstrings, n_ctrl, n_target, n_work, clean)
        assert sizes == expected[clean]


_cswap_without_elbow = pytest.mark.xfail(
    strict=True,
    reason="Controlled SWAPs with a zeroed work wire are not yet decomposed into elbows",
)


@pytest.mark.parametrize(
    ("sizes", "clean"),
    [
        pytest.param((21, 5, 1, 5), True, marks=_cswap_without_elbow),
        pytest.param((32, 5, 1, 5), True, marks=_cswap_without_elbow),
        ((64, 6, 1, 9), True),
        pytest.param((64, 6, 2, 20), True, marks=_cswap_without_elbow),
        ((256, 8, 2, 11), True),
    ]
    + [
        (sizes, False)
        for sizes in [(21, 5, 1, 5), (32, 5, 1, 5), (64, 6, 1, 9), (64, 6, 2, 20), (256, 8, 2, 11)]
    ],
)
def test_select_swap_t_count(sizes, clean, seed):
    """Test that the Select-SWAP decomposition, if applicable, uses fewer T gates than a plain
    QROM."""
    num_bitstrings, n_ctrl, n_target, n_work = sizes
    bitstrings = numpy.random.default_rng(seed).integers(0, 2, size=(num_bitstrings, n_target))
    wires = qp.registers({"control_wires": n_ctrl, "target_wires": n_target, "work_wires": n_work})
    op = qp.QROMLoad(bitstrings, **wires, clean=clean)
    assert _select_swap.is_applicable(**op.arguments)
    gate_set = {"T", "Adjoint(T)", "H", "S", "Adjoint(S)", "CNOT", "X", "Z", "CZ", "GlobalPhase"}
    gate_set |= {"MidMeasure"}

    def t_count(op, fixed_decomps=None):
        graph = DecompositionGraph([op], gate_set, fixed_decomps=fixed_decomps)
        counts = graph.solve().resource_estimate(op).gate_counts
        return sum(num for gate, num in counts.items() if str(gate) in {"T", "Adjoint(T)"})

    select_swap_t_count = t_count(op, fixed_decomps={qp.QROMLoad: _select_swap})
    assert 0 < select_swap_t_count < t_count(qp.QROM(bitstrings, **wires))


def _adjoint_qrom_load_setup(L, n_target, n_extra, n_work):
    rng = numpy.random.default_rng(7 * L + n_target + 13 * n_extra)
    n_control = max(1, math.ceil(math.log2(L))) + n_extra
    wires = qp.registers({"control": n_control, "target": n_target, "work": n_work})
    bitstrings = rng.integers(0, 2, size=(L, n_target))
    return qp.QROMLoad(bitstrings, wires["control"], wires["target"], wires["work"]), rng


# (num_bitstrings, num_targets, num_extra_control_wires, num_work_wires)
ADJOINT_QROM_LOAD_CASES = [
    (1, 2, 0, 1),
    (1, 2, 2, 1),
    (2, 2, 0, 1),
    (3, 2, 0, 1),
    (4, 3, 0, 1),
    (5, 2, 0, 2),
    (7, 3, 0, 2),
    (8, 2, 0, 3),
    (11, 2, 0, 3),
    (16, 2, 0, 1),
    (16, 2, 0, 5),
    (16, 2, 0, 7),
    (20, 2, 0, 4),
    (4, 2, 1, 2),
    (5, 2, 2, 3),
]


class TestAdjointQROMLoadMeasurement:
    """Tests for the measurement-based uncomputation of QROMLoad from Appendix C of
    arXiv:1902.02134."""

    @pytest.mark.parametrize(
        ("num_bitstrings", "num_control_wires", "num_work_wires", "expected"),
        [
            # With enough work wires, the size minimizes ceil(L/k) + k.
            (1, 1, 8, 0),
            (2, 1, 8, 0),
            (8, 3, 8, 1),
            (16, 4, 8, 2),
            (20, 5, 16, 2),
            (64, 6, 16, 3),
            (256, 8, 32, 4),
            # Ties in the estimated T-count go to the smaller one-hot register.
            (4, 2, 8, 0),
            (5, 4, 16, 1),
            # With few work wires, the fixup QROM receives enough work wires first.
            (64, 6, 1, 0),
            (16, 4, 4, 1),
            (16, 4, 5, 2),
            (12, 4, 4, 1),
            (64, 6, 8, 2),
            (64, 6, 10, 3),
            (256, 8, 16, 3),
            (256, 8, 22, 4),
            (100, 7, 10, 2),
            (5, 4, 2, 0),
        ],
    )
    def test_one_hot_size(self, num_bitstrings, num_control_wires, num_work_wires, expected):
        """Test that the one-hot register size gives the fixup QROM enough work wires for its
        unary iteration first, and otherwise minimizes ceil(L/k) + k."""
        size = _adjoint_qrom_load_one_hot_size(num_bitstrings, num_control_wires, num_work_wires)
        assert size == expected

    def test_condition(self, mocker):
        """Test that the rule requires an active compiler with program capture, at least one
        control wire, at least one work wire, and ``clean=True``."""
        op = qp.adjoint(qp.QROMLoad(Bool[4, 2], Wire[2], Wire[2], Wire[1]))
        assert not _adjoint_qrom_load_measurement.is_applicable(base=op.base)
        with qp.capture.toggle_ctx(True):
            assert not _adjoint_qrom_load_measurement.is_applicable(base=op.base)

        mocker.patch("pennylane.templates.subroutines.qrom_load.compiler.active", return_value=True)
        assert not _adjoint_qrom_load_measurement.is_applicable(base=op.base)
        with qp.capture.toggle_ctx(True):
            assert _adjoint_qrom_load_measurement.is_applicable(base=op.base)
            op = qp.adjoint(qp.QROMLoad(Bool[1, 2], Wire[0], Wire[2], Wire[1]))
            assert not _adjoint_qrom_load_measurement.is_applicable(base=op.base)
            op = qp.adjoint(qp.QROMLoad(Bool[4, 2], Wire[2], Wire[2], Wire[0]))
            assert not _adjoint_qrom_load_measurement.is_applicable(base=op.base)
            op = qp.adjoint(qp.QROMLoad(Bool[4, 2], Wire[2], Wire[2], Wire[1], False))
            assert not _adjoint_qrom_load_measurement.is_applicable(base=op.base)

    @pytest.mark.parametrize("case", ADJOINT_QROM_LOAD_CASES)
    def test_resources(self, case):
        """Test that the resource function matches the decomposition when all measurement
        outcomes are 1, so that all conditional corrections are applied."""
        op, _ = _adjoint_qrom_load_setup(*case)

        def measure(pauli_word, wire):
            qp.pauli_measure(pauli_word, wire)
            return 1

        with mock.patch.object(qrom_load, "pauli_measure", measure):
            with qp.queuing.AnnotatedQueue() as q:
                _adjoint_qrom_load_measurement(base=op)

        tape = qp.tape.QuantumScript.from_queue(q)
        actual = Counter(
            abstractify(o.base if isinstance(o, qp.ops.Conditional) else o) for o in tape.operations
        )
        expected = _adjoint_qrom_load_measurement.compute_resources(base=op).gate_counts
        _assert_counts_match({k: v for k, v in expected.items() if v > 0}, actual)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("case", ADJOINT_QROM_LOAD_CASES)
    def test_correctness(self, case):
        """Test that applying QROMLoad followed by the measurement-based uncomputation acts as
        the identity on a random address state, for every outcome of the X-basis measurements.

        The measurements are replaced by postselected measurements, so that the
        measurement-dependent fixup table can be computed from concrete outcomes without capture.
        """
        op, rng = _adjoint_qrom_load_setup(*case)
        control_wires = list(op.control_wires)
        all_wires = control_wires + list(op.target_wires) + list(op.work_wires)
        state = rng.random(2 ** len(control_wires)) + 1j * rng.random(2 ** len(control_wires))
        state /= numpy.linalg.norm(state)

        @qp.qnode(qp.device("default.qubit"), mcm_method="deferred")
        def circuit():
            qp.StatePrep(state, wires=control_wires)
            _qrom_unary_iteration(**op.dynamic_args, **op.wire_args)
            _adjoint_qrom_load_measurement(base=op)
            qp.adjoint(qp.StatePrep(state, wires=control_wires))
            return qp.probs(wires=all_wires)

        def postselected_measure(outcome_iter, pauli_word, wire):
            if pauli_word == "Z":
                # Exercise the conditional reset of the target wire.
                qp.measure(wire, postselect=1)
                return 1
            outcome = next(outcome_iter)
            qp.H(wire)
            qp.measure(wire, postselect=outcome)
            qp.H(wire)
            return outcome

        for outcomes in itertools.product([0, 1], repeat=len(op.target_wires)):
            measure = partial(postselected_measure, iter(outcomes))
            with mock.patch.object(qrom_load, "pauli_measure", measure):
                probs = circuit()
            assert np.isclose(probs[0] / np.sum(probs), 1.0), f"outcomes={outcomes}"

    @pytest.mark.usefixtures("enable_capture")
    @pytest.mark.parametrize("case", ADJOINT_QROM_LOAD_CASES[::3])
    def test_capture(self, case):
        """Test that the decomposition can be captured, with an X-basis measurement and a Z-basis
        reset measurement per target wire."""
        op, _ = _adjoint_qrom_load_setup(*case)
        jaxpr = jax.make_jaxpr(partial(_adjoint_qrom_load_measurement, base=op))()
        measurements = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "pauli_measure"]
        assert [eqn.params["pauli_word"] for eqn in measurements] == ["X", "Z"] * len(
            op.target_wires
        )

    @pytest.mark.xfail(
        reason="Controlled MultiX needs work wire support for control regions in Catalyst "
        "(Catalyst #3376).",
        strict=True,
    )
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.catalyst
    @pytest.mark.parametrize("case", [(4, 2, 0, 1), (8, 2, 0, 2), (11, 2, 0, 3), (4, 2, 1, 2)])
    def test_qjit(self, case):
        """Test that loading and uncomputing a QROMLoad under ``qjit`` leaves the target and work
        wires in the zero state, and does not introduce relative phases on a uniform superposition
        of addresses."""
        op, _ = _adjoint_qrom_load_setup(*case)
        control_wires = list(op.control_wires)
        shots = 20

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=qp.gate_sets.CLIFFORD_T_PLUS_RZ)
        @qp.set_shots(shots)
        @qp.qnode(qp.device("lightning.qubit", wires=len(op.wires)))
        def circuit():
            for wire in control_wires:
                qp.H(wire)
            _qrom_measurement_decomposition(**op.dynamic_args, **op.wire_args)
            _adjoint_qrom_load_measurement(base=op)
            for wire in control_wires:
                qp.H(wire)
            return qp.sample(wires=op.wires)

        samples = circuit()
        assert samples.shape == (shots, len(op.wires))
        assert np.allclose(samples, 0)
