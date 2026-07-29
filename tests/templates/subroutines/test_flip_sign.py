# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the FlipSign template.
"""

import pytest

import pennylane as qp
from pennylane import math
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.wires import Wires


@pytest.mark.jax
def test_standard_checks():
    """Run standard checks with the assert_valid function."""
    op = qp.FlipSign([0, 1], wires=("a", "b"))
    qp.ops.functions.assert_valid(op)


def test_repr():
    """Test the repr for a flip sign operator."""
    op = qp.FlipSign([0, 1], wires=("a", "b"))
    expected = "FlipSign(state=(0, 1), wires=['a', 'b'])"
    assert repr(op) == expected


class TestFlipSign:
    """Tests that the template defines the correct sign flip."""

    @pytest.mark.parametrize(
        "state, wires",
        [
            (0, 0),
            (1, 3),
            (2, range(2)),
            (6, range(3)),
            (8, range(4)),
            ([1, 0], [1, 2]),
            ([1, 1, 0], [4, 1, 2]),
            ([1, 0, 1, 0], [0, 1, 5, 4]),
        ],
    )
    def test_eval(self, state, wires):
        if isinstance(wires, int):
            wires = [wires]

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            for wire in wires:
                qp.Hadamard(wire)
            qp.FlipSign(state, wires=wires)
            return qp.state()

        if isinstance(state, list):
            # convert the basis state from list of bits to integer number
            state = sum(2**i for i, bit in enumerate(state[::-1]) if bit)

        # check that only the indicated value has been changed
        out_state = circuit()
        signs_are_correct = [
            math.sign(x) == -1 if i == state else math.sign(x) == 1 for i, x in enumerate(out_state)
        ]
        assert all(signs_are_correct)

    @pytest.mark.parametrize(
        "state, wires",
        [
            (0, 0),
            (1, 3),
            (2, range(2)),
            (6, range(3)),
            ([1, 0], [1, 2]),
            ([1, 1, 0], [4, 1, 2]),
        ],
    )
    def test_wires(self, state, wires):
        """Test that the operation wires attribute is correct."""
        op = qp.FlipSign(state, wires=wires)
        assert op.wires == Wires(wires)

    @pytest.mark.parametrize("state, num_wires", [(-1, 1), (16, 4)])
    def test_invalid_state_error(self, state, num_wires):
        """Assert error raised when given negative or too large basis state"""
        with pytest.raises(ValueError, match="The given basis state must be a non-negative integ"):
            qp.FlipSign(state, wires=list(range(num_wires)))

    @pytest.mark.parametrize(
        "state, wires",
        [
            ([0, 1], [2]),
            ([1, 0, 0], [0, 1]),
            ((1, 0, 1, 1), [0, 2, 3]),
        ],
    )
    def test_length_not_match_error(self, state, wires):
        """Assert error raised when length of basis state and wires length does not match"""
        a = len(state)
        b = len(wires)
        with pytest.raises(
            ValueError,
            match=f"The basis state and wires must have equal length, but got {a} and {b}.",
        ):
            qp.FlipSign(state, wires)

    @pytest.mark.parametrize(
        "state, wires",
        [
            ([1, 0], []),
            (2, []),
            (3, ()),
            (1, {}),
        ],
    )
    def test_wire_empty_error(self, state, wires):
        """Assert error raised when given empty wires"""
        with pytest.raises(ValueError, match="At least one wire is required."):
            qp.FlipSign(state, wires=wires)

    @pytest.mark.jax
    def test_jax_jit(self):
        import jax

        num_wires = 2
        dev = qp.device("default.qubit", wires=num_wires)

        @qp.qnode(dev)
        def circuit():
            for wire in range(num_wires):
                qp.Hadamard(wire)
            qp.FlipSign([1, 0], wires=range(num_wires))
            return qp.state()

        jit_circuit = jax.jit(circuit)

        res = circuit()
        jit_res = jit_circuit()
        assert qp.math.allclose(res, jit_res)

    @pytest.mark.parametrize(
        "state, wires",
        [
            (0, 0),
            (1, 3),
            (2, range(2)),
            (6, range(3)),
            (8, range(4)),
            ([1, 0], [1, 2]),
            ([1, 1, 0], [4, 1, 2]),
            ([1, 0, 1, 0], [0, 1, 5, 4]),
        ],
    )
    def test_decomposition_new(self, state, wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qp.FlipSign(state, wires=wires)

        for rule in qp.list_decomps(qp.FlipSign):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        "state, wires",
        [
            (0, 0),
            (1, 3),
            (2, range(2)),
            (6, range(3)),
            (8, range(4)),
            ([1, 0], [1, 2]),
            ([1, 1, 0], [4, 1, 2]),
            ([1, 0, 1, 0], [0, 1, 5, 4]),
        ],
    )
    @pytest.mark.capture
    def test_decomposition_new_capture(self, state, wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qp.FlipSign(state, wires=wires)

        for rule in qp.list_decomps(qp.FlipSign):
            _test_decomposition_rule(op, rule)
