# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Tests for the SemiAdder template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity_SemiAdder():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2]
    y_wires = [4, 5, 6]
    work_wires = [7, 8]
    op = qml.SemiAdder(x_wires, y_wires, work_wires)
    qml.ops.functions.assert_valid(op)


class TestSemiAdder:
    """Test the qml.SemiAdder template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires", "x", "y"),
        [
            ([0], [1], None, 1, 0),
            ([0], [1], None, 1, 1),
            ([0, 1], [2], None, 0, 1),
            ([0, 1], [2], None, 1, 1),
            ([0, 1], [2], None, 1, 0),
            ([0, 1], [2, 3], [4], 2, 0),
            ([0, 1], [2, 3], [4], 1, 2),
            ([0, 1, 2], [3, 4, 5], [6, 7], 1, 2),
            ([0, 1, 2], [3, 4, 5], [6, 7], 5, 6),
            ([0, 1], [2, 3, 4], [5, 6], 3, 2),
            ([0, 1], [2, 3, 4, 5], [6, 7, 8], 3, 10),
            ([0, 1, 2], [3, 4, 5], [6, 7], 7, 7),
            ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9], 6, 5),
            ([0], [3, 4, 5, 6], [7, 8, 9], 1, 5),
            ([0, 1, 2, 3, 4], [5, 6], [7], 11, 2),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z"], 4, 2),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, work_wires, x, y
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the SemiAdder template output."""
        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.SemiAdder(x_wires, y_wires, work_wires)
            return qml.sample(wires=y_wires), qml.probs(wires=work_wires)

        output = circuit(x, y)

        sample = output[0]

        #  check that the output sample is the binary representation of x + y mod 2^len(y_wires)
        # pylint: disable=bad-reversed-sequence
        assert np.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(sample[0, :]))),
            (x + y) % 2 ** len(y_wires),
        )

        if work_wires:
            # check work_wires are in state |0>
            assert np.isclose(output[1][0], 1.0)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [1],
                "At least 2 work_wires should be provided.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [1, 6],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [3, 6],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                [1],
                [0],
                [0],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                [0, 1, 2],
                [2, 3, 4, 5],
                [6, 7, 8],
                "None of the wires in y_wires should be included in x_wires.",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, y_wires, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.SemiAdder(x_wires, y_wires, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, work_wires = (
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13],
        )

        adder_decomposition = qml.SemiAdder(x_wires, y_wires, work_wires).compute_decomposition(
            x_wires, y_wires, work_wires
        )

        names = [op.name for op in adder_decomposition]

        # Example in Fig 1.  https://arxiv.org/pdf/1709.06648
        assert names.count("TemporaryAND") == 4
        assert names.count("Adjoint(TemporaryAND)") == 4
        assert names.count("CNOT") == 21

    @pytest.mark.parametrize(
        ("x_wires"),
        [
            [0, 1, 2],
            [0, 1],
            [0, 1, 2, 3],
        ],
    )
    def test_decomposition_rule(self, x_wires):
        """Tests that SemiAdder is decomposed properly."""

        for rule in qml.list_decomps(qml.SemiAdder):
            _test_decomposition_rule(
                qml.SemiAdder(x_wires, [5, 6, 7, 8], [9, 10, 11]),
                rule,
            )

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3

        x_wires = [0, 1, 4]
        y_wires = [2, 3, 5]
        work_wires = [7, 8]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.SemiAdder(x_wires, y_wires, work_wires)
            return qml.sample(wires=y_wires)

        # pylint: disable=bad-reversed-sequence
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()[0, :]))),
            (x + y) % 2 ** len(y_wires),
        )
