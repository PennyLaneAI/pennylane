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
Tests for the Multiplier template.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity_Multiplier():
    """Check the operation using the assert_valid function."""
    k = 6
    mod = 11
    x_wires = [0, 1, 2, 3]
    work_wires = [4, 5, 6, 7, 8, 9]
    op = qml.Multiplier(k, x_wires=x_wires, mod=mod, work_wires=work_wires)
    qml.ops.functions.assert_valid(op)


class TestMultiplier:
    """Test the qml.Multiplier template."""

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wires", "x"),
        [
            (
                5,
                [0, 1, 2],
                8,
                [4, 5, 6, 7, 8],
                3,
            ),
            (
                1,
                [0, 1, 2],
                3,
                [3, 4, 5, 6, 7],
                2,
            ),
            (
                -12,
                [0, 1, 2, 3, 4],
                23,
                [5, 6, 7, 8, 9, 10, 11],
                1,
            ),
            (
                5,
                [0, 1, 2, 3, 4],
                None,
                [5, 6, 7, 8, 9, 10, 11],
                0,
            ),
            (
                5,
                [0, 1, 2, 3, 4],
                None,
                [5, 6, 7, 8, 9],
                1,
            ),
        ],
    )
    def test_operation_result(
        self, k, x_wires, mod, work_wires, x
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the Multiplier template output."""
        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Multiplier(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

        if mod is None:
            mod = 2 ** len(x_wires)

        # pylint: disable=bad-reversed-sequence
        assert np.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x)[0, :]))), (x * k) % mod
        )

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wires", "msg_match"),
        [
            (
                3,
                [0, 1, 2, 3, 4],
                11,
                None,
                "Work wires must be specified for Multiplier",
            ),
            (
                6,
                [0, 1],
                7,
                [3, 4, 5, 6],
                "Multiplier must have enough wires to represent mod.",
            ),
            (
                2,
                [0, 1, 2],
                6,
                [3, 4, 5, 6, 7],
                "The operator cannot be built because k has no inverse modulo mod",
            ),
            (
                3,
                [0, 1, 2, 3, 4],
                11,
                [4, 5],
                "None of the wire in work_wires should be included in x_wires.",
            ),
            (
                3,
                [0, 1, 2, 3, 4],
                11,
                [5, 6, 7, 8, 9, 10],  # not enough
                "Multiplier needs as many work_wires as x_wires plus two.",
            ),
            (
                3,
                [0, 1, 2, 3, 4],
                11,
                [5, 6, 7, 8, 9, 10, 11, 12],  # too many
                "Multiplier needs as many work_wires as x_wires plus two.",
            ),
            (
                3,
                [0, 1, 2, 3],
                16,
                [5, 6, 7],
                "Multiplier needs as many work_wires as x_wires.",
            ),
        ],
    )
    def test_operation_and_wires_error(
        self, k, x_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when k or mod don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.Multiplier(k, x_wires, mod, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        k, x_wires, mod, work_wires = 4, [0, 1, 2], 7, [3, 4, 5, 6, 7]
        multiplier_decomposition = qml.transforms.decompose(
            qml.tape.QuantumScript([qml.Multiplier(k, x_wires, mod, work_wires)]), max_expansion=2
        )[0][0].operations

        op_list = []
        if mod != 2 ** len(x_wires):
            work_wire_aux = work_wires[:1]
            wires_aux = work_wires[1:]
            wires_aux_swap = wires_aux[1:]
        else:
            work_wire_aux = None
            wires_aux = work_wires[:3]
            wires_aux_swap = wires_aux

        op_list.append(qml.QFT(wires=wires_aux))
        op_list.append(
            qml.ControlledSequence(
                qml.PhaseAdder(k, wires_aux, mod, work_wire_aux), control=x_wires
            )
        )
        op_list.append(qml.adjoint(qml.QFT(wires=wires_aux)))

        for x_wire, aux_wire in zip(x_wires, wires_aux_swap):
            op_list.append(qml.SWAP(wires=[x_wire, aux_wire]))
        inv_k = pow(k, -1, mod)
        op_list.append(qml.QFT(wires=wires_aux))
        op_list.append(
            qml.adjoint(
                qml.ControlledSequence(
                    qml.PhaseAdder(inv_k, wires_aux, mod, work_wire_aux), control=x_wires
                )
            )
        )
        op_list.append(qml.adjoint(qml.QFT(wires=wires_aux)))

        for op1, op2 in zip(multiplier_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        ("k", "x_wire", "mod", "work_wires"), [(3, [1], 1, [2, 3, 4]), (3, [1], 2, [2, 3, 4])]
    )
    def test_decomposition_new(
        self, k, x_wire, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = qml.Multiplier(k, x_wire, mod, work_wires)
        for rule in qml.list_decomps(qml.Multiplier):
            _test_decomposition_rule(op, rule)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)
        x = 2
        k = 6
        mod = 7
        x_wires = [0, 1, 2]
        work_wires = [4, 5, 6, 7, 8]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Multiplier(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

        # pylint: disable=bad-reversed-sequence
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()[0, :]))), (x * k) % mod
        )
