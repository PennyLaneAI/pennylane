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
import re

import pytest

import pennylane as qml
from pennylane import math
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.wires import Wires


@pytest.mark.jax
def test_standard_checks():
    """Run standard checks with the assert_valid function."""
    op = qml.FlipSign([0, 1], wires=("a", "b"))
    qml.ops.functions.assert_valid(op)


def test_repr():
    """Test the repr for a flip sign operator."""
    op = qml.FlipSign([0, 1], wires=("a", "b"))
    expected = "FlipSign((0, 1), wires=['a', 'b'])"
    assert repr(op) == expected


class TestFlipSign:
    """Tests that the template defines the correct sign flip."""

    @pytest.mark.parametrize(
        ("n, wires"),
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
    def test_eval(self, n, wires):
        if isinstance(wires, int):
            wires = [wires]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            for wire in wires:
                qml.Hadamard(wire)
            qml.FlipSign(n, wires=wires)
            return qml.state()

        if isinstance(n, list):
            # convert the basis state from list of bits to integer number
            n = sum(2**i for i, bit in enumerate(n[::-1]) if bit)

        # check that only the indicated value has been changed
        state = circuit()
        signs_are_correct = [
            math.sign(x) == -1 if i == n else math.sign(x) == 1 for i, x in enumerate(state)
        ]
        assert all(signs_are_correct)

    @pytest.mark.parametrize(
        ("n, wires"),
        [
            (0, 0),
            (1, 3),
            (2, range(2)),
            (6, range(3)),
            ([1, 0], [1, 2]),
            ([1, 1, 0], [4, 1, 2]),
        ],
    )
    def test_wires(self, n, wires):
        """Test that the operation wires attribute is correct."""
        op = qml.FlipSign(n, wires=wires)
        assert op.wires == Wires(wires)

    @pytest.mark.parametrize(
        ("n, wires"),
        [
            (-1, 0),
        ],
    )
    def test_invalid_state_error(self, n, wires):
        """Assert error raised when given negative basic state"""
        with pytest.raises(
            ValueError, match="The given basis state cannot be a negative integer number."
        ):
            qml.FlipSign(n, wires=wires)

    @pytest.mark.parametrize(
        ("n, wires"),
        [
            (2, 1),
            (5, 2),
            (3, [1]),
        ],
    )
    def test_number_wires_error(self, n, wires):
        """Assert error raised when given basis state length is less than number of wires"""
        num_wires = 1 if isinstance(wires, int) else len(wires)

        with pytest.raises(
            ValueError, match=f"Cannot encode basis state {n} on {num_wires} wires."
        ):
            qml.FlipSign(n, wires=wires)

    @pytest.mark.parametrize(
        ("n, wires"),
        [
            ([0, 1], [2]),
            ([1, 0, 0], [0, 1]),
            ([1, 0, 1, 1], [0, 2, 3]),
        ],
    )
    def test_length_not_match_error(self, n, wires):
        """Assert error raised when length of basis state and wires length does not match"""
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The basis state {tuple(n)} and wires {wires} must be of equal length."
            ),
        ):
            qml.FlipSign(n, wires=wires)

    @pytest.mark.parametrize(
        ("n, wires"),
        [
            ([1, 0], []),
            (2, []),
            (3, ()),
            (1, ""),
            (2, ""),
        ],
    )
    def test_wire_empty_error(self, n, wires):
        """Assert error raised when given empty wires"""
        with pytest.raises(ValueError, match="At least one valid wire is required."):
            qml.FlipSign(n, wires=wires)

    @pytest.mark.jax
    def test_jax_jit(self):
        import jax

        num_wires = 2
        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev)
        def circuit():
            for wire in range(num_wires):
                qml.Hadamard(wire)
            qml.FlipSign([1, 0], wires=range(num_wires))
            return qml.state()

        jit_circuit = jax.jit(circuit)

        res = circuit()
        jit_res = jit_circuit()
        assert qml.math.allclose(res, jit_res)

    @pytest.mark.parametrize(
        ("n, wires"),
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
    def test_decomposition_new(self, n, wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.FlipSign(n, wires=wires)

        for rule in qml.list_decomps(qml.FlipSign):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        ("n, wires"),
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
    def test_decomposition_new_capture(self, n, wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.FlipSign(n, wires=wires)

        for rule in qml.list_decomps(qml.FlipSign):
            _test_decomposition_rule(op, rule)
