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
        ("k", "x_wires", "mod", "work_wires"),
        [
            (
                6,
                [0, 1, 2],
                8,
                [4, 5],
            ),
            (
                1,
                [0, 1, 2, 3],
                9,
                [4, 5],
            ),
            (
                2,
                [0, 1, 4],
                4,
                [3, 2],
            ),
            (
                -3,
                [0, 1, 4],
                4,
                [3, 2],
            ),
            (
                10,
                [0, 1, 2, 5],
                9,
                [3],
            ),
            (
                1,
                [0, 1, 2],
                7,
                [3, 4],
            ),
            (
                6,
                [0, 1, 2, 3],
                None,
                [4, 5],
            ),
        ],
    )
    def test_operation_result(
        self, k, x_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the PhaseAdder template output."""
        dev = qml.device("default.qubit", shots=1)
        if mod is None:
            max = 2 ** len(x_wires)
        else:
            max = mod

        @qml.qnode(dev)
        def circuit(x):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Adder(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

        for x in range(0, max // 2):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x)))), (x + k) % max
            )

    def test_wires_error(self):  # pylint: disable=too-many-arguments
        """Test an error is raised when some wire in work_wires is in wires"""
        k, x_wires, mod, work_wires, msg_match = (
            3,
            [0, 1, 2, 3, 4],
            12,
            [4, 5],
            "None wire in work_wires should be included in x_wires.",
        )
        with pytest.raises(ValueError, match=msg_match):
            qml.Adder(k, x_wires, mod, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""

        k = 2
        mod = 7
        x_wires = [0, 1, 2]
        work_wires = [3, 4]
        adder_decomposition = qml.Adder(k, x_wires, mod, work_wires).compute_decomposition(
            k, x_wires, mod, work_wires
        )
        op_list = []
        op_list.append(qml.QFT(work_wires[:1] + x_wires))
        op_list.append(PhaseAdder(k, work_wires[:1] + x_wires, mod, work_wires[1:]))
        op_list.append(qml.adjoint(qml.QFT)(work_wires[:1] + x_wires))

        for op1, op2 in zip(adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)
        x = 2
        x_list = [0, 1, 0]
        k = 6
        mod = 7
        x_wires = [0, 1, 2]
        work_wires = [3, 4]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x_list, wires=x_wires)
            qml.Adder(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()))), (x + k) % mod
        )
