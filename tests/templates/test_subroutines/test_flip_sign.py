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
from pennylane import numpy as np
import pennylane as qml


def test_standarad_checks():
    """Run standard checks with the assert_valid function."""
    op = qml.FlipSign([0, 1], wires=("a", "b"))
    qml.ops.functions.assert_valid(op)


def test_repr():
    """Test the repr for a flip sign operator."""
    op = qml.FlipSign([0, 1], wires=("a", "b"))
    expected = "FlipSign([0, 1], wires=['a', 'b'])"
    assert repr(op) == expected


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Test the flatten and unflatten methods."""
    op = qml.FlipSign([0, 1], wires=2)
    data, metadata = op._flatten()

    assert data == tuple()
    hyperparameters = (("n", (0, 1)),)
    assert metadata == (op.wires, hyperparameters)

    # make sure metadata hasable
    assert hash(metadata)

    new_op = type(op)._unflatten(*op._flatten())
    # data casted to tuple. unimportant difference
    assert qml.equal(qml.FlipSign((0, 1), wires=2), new_op)
    assert op is not new_op


class TestFlipSign:
    """Tests that the template defines the correct sign flip."""

    @pytest.mark.parametrize(
        ("n_status, n_wires"),
        [
            (2, 2),
            (6, 3),
            (8, 4),
            ([1, 0], 2),
            ([1, 1, 0], 3),
            ([1, 0, 0, 0], 4),
            ([1, 0, 0, 0], [0, 1, 2, 3]),
            (1, 0),
        ],
    )
    def test_eval(self, n_status, n_wires):
        if isinstance(n_wires, int):
            n_wires = 1 if n_wires == 0 else n_wires
            n_wires = list(range(n_wires))

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            for wire in n_wires:
                qml.Hadamard(wires=wire)

            qml.FlipSign(n_status, wires=n_wires)

            return qml.state()

        def to_number(status):
            return sum(status[i] * 2 ** (len(status) - i - 1) for i in range(len(status)))

        if isinstance(n_status, list):
            n_status = to_number(n_status)

        # we check that only the indicated value has been changed
        statuses = []
        for ind, x in enumerate(circuit()):
            if ind == n_status:
                statuses.append(bool(np.sign(x) == -1))
            else:
                statuses.append(bool(np.sign(x) == 1))

        assert np.all(np.array(statuses))

    @pytest.mark.parametrize(
        ("n_status, n_wires"),
        [
            (-1, 0),
        ],
    )
    def test_empty_wire_error(self, n_status, n_wires):
        """Assert error raised when given negative basic status"""
        with pytest.raises(
            ValueError,
            match="expected an integer equal or greater than zero for basic flipping state",
        ):
            qml.FlipSign(n_status, wires=n_wires)

    @pytest.mark.parametrize(
        ("n_status, n_wires"),
        [
            (2, 1),
        ],
    )
    def test_number_wires_error(self, n_status, n_wires):
        """Assert error raised when given basis state length is less than number of wires"""
        with pytest.raises(ValueError, match=f"cannot encode {n_status} with {n_wires} wires "):
            qml.FlipSign(n_status, wires=n_wires)

    @pytest.mark.parametrize(
        ("n_status, n_wires"),
        [
            ([1, 0, 0], [0, 1]),
        ],
    )
    def test_length_not_match_error(self, n_status, n_wires):
        """Assert error raised when length of basis state and wires length does not match"""
        with pytest.raises(
            ValueError,
            match="Wires length and flipping state length does not match, they must be equal length ",
        ):
            qml.FlipSign(n_status, wires=n_wires)

    @pytest.mark.parametrize(
        ("n_status, n_wires"),
        [
            ([1, 0], []),
            (2, []),
            (1, ""),
            (2, ""),
            (3, ""),
        ],
    )
    def test_wire_empty_error(self, n_status, n_wires):
        """Assert error raised when given empty wires"""
        with pytest.raises(ValueError, match="expected at least one wire representing the qubit "):
            qml.FlipSign(n_status, wires=n_wires)
