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
Tests for the PhaseAdder template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines.phase_adder import _add_k_fourier


def test_standard_validity_Phase_Adder():
    """Check the operation using the assert_valid function."""
    k = 6
    mod = 11
    x_wires = [0, 1, 2, 3]
    work_wire = [4]
    op = qml.PhaseAdder(k, x_wires=x_wires, mod=mod, work_wire=work_wire)
    qml.ops.functions.assert_valid(op)


class TestPhaseAdder:
    """Test the PhaseAdder template."""

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wire"),
        [
            (
                6,
                [0, 1, 2],
                7,
                [4],
            ),
            (
                0,
                [0, 1, 2, 3],
                9,
                [4],
            ),
            (
                2,
                [0, 1, 4],
                4,
                [3],
            ),
            (
                -2,
                [0, 1, 4],
                4,
                [3],
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
                [3],
            ),
            (
                6,
                [0, 1, 2, 3],
                None,
                [4],
            ),
        ],
    )
    def test_operation_result(
        self, k, x_wires, mod, work_wire
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
            qml.QFT(wires=x_wires)
            qml.PhaseAdder(k, x_wires, mod, work_wire)
            qml.adjoint(qml.QFT)(wires=x_wires)
            return qml.sample(wires=x_wires)

        for x in range(0, max / 2):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x)))), (x + k) % max
            )

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wire", "msg_match"),
        [
            (
                1,
                [0, 1, 2],
                9,
                [3],
                ("PhaseAdder must have at least enough x_wires to represent mod."),
            ),
            (
                1,
                [0, 1, 2],
                9,
                None,
                (r"If mod is not"),
            ),
            (
                3,
                [0, 1, 2, 3, 4],
                12,
                [4],
                "work_wire should not be included in x_wires.",
            ),
        ],
    )
    def test_operation_and_wires_error(
        self, k, x_wires, mod, work_wire, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when mod doesn't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.PhaseAdder(k, x_wires, mod, work_wire)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        k = 4
        x_wires = [1, 2, 3]
        mod = 7
        work_wire = [0]

        phase_adder_decomposition = qml.PhaseAdder(
            k, x_wires, mod, work_wire
        ).compute_decomposition(k, x_wires, mod, work_wire)
        op_list = []

        if mod == 2 ** (len(x_wires)):
            op_list.extend(_add_k_fourier(k, x_wires))
        else:
            aux_k = x_wires[0]
            op_list.extend(_add_k_fourier(k, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(mod, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=1))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(qml.ctrl(op, control=work_wire) for op in _add_k_fourier(mod, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(k, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=0))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(_add_k_fourier(k, x_wires))

        for op1, op2 in zip(phase_adder_decomposition, op_list):
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
        work_wire = [4]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x_list, wires=x_wires)
            qml.QFT(wires=x_wires)
            qml.PhaseAdder(k, x_wires, mod, work_wire)
            qml.adjoint(qml.QFT)(wires=x_wires)
            return qml.sample(wires=x_wires)

        assert np.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()))), (x + k) % mod
        )
