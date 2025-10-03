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
Tests for the ModExp template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity_ModExp():
    """Check the operation using the assert_valid function."""
    base = 6
    mod = 11
    x_wires = [0, 1, 2, 3]
    output_wires = [4, 5, 6, 7]
    work_wires = [8, 9, 10, 11, 12, 13]
    op = qml.ModExp(
        x_wires=x_wires, output_wires=output_wires, base=base, mod=mod, work_wires=work_wires
    )
    qml.ops.functions.assert_valid(op)


class TestModExp:
    """Test the qml.ModExp template."""

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "base", "mod", "work_wires", "x", "k"),
        [
            ([0, 1], [3, 4, 5], 2, 7, [7, 8, 9, 10, 11], 1, 1),
            ([0, 1, 2], [3, 4, 5], 3, 7, [7, 8, 9, 10, 11], 2, 2),
            ([0, 1, 2], [3, 4], 4, 3, [7, 8, 9, 10], 0, 0),
            (
                [0, 1, 2],
                [3, 4, 5],
                7,
                None,
                [9, 10, 11],
                3,
                2,
            ),
            ([0, 1, 2], [3, 4, 5], 5, 6, [6, 7, 8, 9, 10], 3, 0),
        ],
    )
    def test_operation_result(
        self, x_wires, output_wires, base, mod, work_wires, x, k
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the ModExp template output."""
        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x, k):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(k, wires=output_wires)
            qml.ModExp(x_wires, output_wires, base, mod, work_wires)
            return qml.sample(wires=output_wires)

        if mod is None:
            mod = 2 ** len(output_wires)

        # pylint: disable=bad-reversed-sequence
        assert np.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x, k)[0, :]))),
            (k * (base**x)) % mod,
        )

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "base", "mod", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                8,
                5,
                None,
                "Work wires must be specified for ModExp",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                8,
                9,
                [9, 10, 11, 12, 13],
                "ModExp must have enough wires to represent mod.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                5,
                6,
                [9, 10, 11, 12],
                "ModExp needs as many work_wires as output_wires plus two.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                5,
                8,
                [9, 10],
                "ModExp needs as many work_wires as output_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                6,
                7,
                [1, 10, 11, 12, 13],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                7,
                5,
                [3, 10, 11, 12, 13],
                "None of the wires in work_wires should be included in output_wires.",
            ),
            (
                [0, 1, 2],
                [2, 4, 5],
                3,
                7,
                [9, 10, 11, 12, 13],
                "None of the wires in x_wires should be included in output_wires.",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, output_wires, base, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.ModExp(x_wires, output_wires, base, mod, work_wires)

    def test_check_base_and_mod_are_coprime(self):
        """Test that an error is raised when base and mod are not coprime"""

        with pytest.raises(ValueError, match="base has no inverse modulo mod"):
            qml.ModExp(
                x_wires=[0, 1, 2],
                output_wires=[3, 4, 5],
                base=8,
                mod=6,
                work_wires=[6, 7, 8, 9, 10],
            )

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, output_wires, base, mod, work_wires = (
            [0, 1, 2],
            [3, 4, 5],
            6,
            7,
            [9, 10, 11, 12, 13],
        )
        adder_decomposition = qml.ModExp(
            x_wires, output_wires, base, mod, work_wires
        ).compute_decomposition(x_wires, output_wires, base, mod, work_wires)
        op_list = []
        op_list.append(
            qml.ControlledSequence(
                qml.Multiplier(base, output_wires, mod, work_wires), control=x_wires
            )
        )

        for op1, op2 in zip(adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    def test_decomposition_new(self):
        """Tests the decomposition rule implemented with the new system."""
        x_wires, output_wires, base, mod, work_wires = (
            [0, 1, 2],
            [3, 4, 5],
            6,
            7,
            [9, 10, 11, 12, 13],
        )
        op = qml.ModExp(x_wires, output_wires, base, mod, work_wires)
        for rule in qml.list_decomps(qml.ModExp):
            _test_decomposition_rule(op, rule)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x = 2
        x_list = [0, 1, 0]
        mod = 7
        x_wires = [0, 1, 2]
        base = 3
        output_wires = [3, 4, 5]
        work_wires = [11, 10, 12, 13, 14]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x_list, wires=x_wires)
            qml.BasisEmbedding([0, 0, 1], wires=output_wires)
            qml.ModExp(x_wires, output_wires, base, mod, work_wires)
            return qml.sample(wires=output_wires)

        # pylint: disable=bad-reversed-sequence
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()[0, :]))), (base**x) % mod
        )
