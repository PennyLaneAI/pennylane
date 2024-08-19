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


def test_standard_validity_Phase_Adder():
    """Check the operation using the assert_valid function."""
    k = 6
    mod = 11
    wires = [0, 1, 2, 3]
    work_wires = [4, 5]
    op = qml.PhaseAdder(k, wires=wires, mod=mod, work_wires=work_wires)
    qml.ops.functions.assert_valid(op)


def _add_k_fourier(k, wires):
    """Adds k in the Fourier basis"""
    op_list = []
    for j, wire in enumerate(wires):
        op_list.append(qml.PhaseShift(k * np.pi / (2**j), wires=wire))
    return op_list


class TestPhaseAdder:
    """Test the qml.PhaseAdder template."""

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                6,
                [0, 1, 2],
                7,
                [3, 4],
            ),
            (
                0,
                [0, 1, 2, 3],
                9,
                [5, 4],
            ),
            (
                2,
                [0, 1, 4],
                4,
                [2, 3],
            ),
            (
                8,
                [0, 1, 2, 5],
                9,
                [3, 4],
            ),
            (
                1,
                [0, 1, 2],
                7,
                [3, 4],
            ),
        ],
    )
    def test_operation_result(
        self, k, wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the PhaseAdder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(m):
            qml.BasisEmbedding(m, wires=wires)
            qml.QFT(wires=work_wires[:1] + wires)
            qml.PhaseAdder(k, wires, mod, work_wires)
            qml.adjoint(qml.QFT)(wires=work_wires[:1] + wires)
            return qml.sample(wires=wires)

        if mod is None:
            max = 2 ** len(wires)
        else:
            max = mod
        for m in range(0, max):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(m)))), (m + k) % max
            )

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                6,
                [0, 1, 2, 3],
                10,
                None,
            ),
            (
                6,
                [0, 1, 2, 3],
                None,
                [4, 5],
            ),
            (
                6,
                [0, 1, 2, 3],
                None,
                None,
            ),
        ],
    )
    def test_operation_result_args_None(
        self, k, wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the PhaseAdder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(m):
            qml.BasisEmbedding(m, wires=wires)
            if mod is None:
                qml.QFT(wires=wires)
            else:
                qml.QFT(wires=[4] + wires)
            qml.PhaseAdder(k, wires, mod, work_wires)
            if mod is None:
                qml.adjoint(qml.QFT)(wires=wires)
            else:
                qml.adjoint(qml.QFT)(wires=[4] + wires)

            return qml.sample(wires=wires)

        if mod is None:
            max = 2 ** len(wires)
        else:
            max = mod
        for m in range(1, max):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(m)))), (m + k) % max
            )

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires", "msg_match"),
        [
            (6, [0, 1, 2, 3, 4], 6, [5, 6], "The module mod must be larger than k."),
            (
                1,
                [0, 1, 2],
                9,
                [3, 4],
                ("PhaseAdder must have at least enough wires to represent mod."),
            ),
        ],
    )
    def test_operation_error(
        self, k, wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when k or mod don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.PhaseAdder(k, wires, mod, work_wires)

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires", "msg_match"),
        [
            (
                3,
                [0, 1, 2, 3, 4],
                12,
                [4, 5],
                "None of the wires in work_wires should be included in wires.",
            ),
        ],
    )
    def test_wires_error(
        self, k, wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some wire in work_wires is in wires"""
        with pytest.raises(ValueError, match=msg_match):
            qml.PhaseAdder(k, wires, mod, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        k = 4
        wires = [1, 2, 3]
        mod = 7
        work_wires = [0, 4]

        phase_adder_decomposition = qml.PhaseAdder(k, wires, mod, work_wires).compute_decomposition(
            k, mod, work_wires, wires
        )
        op_list = []

        if mod == 2 ** len(wires):
            op_list.extend(_add_k_fourier(k, wires))
        else:
            new_wires = work_wires[:1] + wires
            work_wire = work_wires[1]
            aux_k = new_wires[0]
            op_list.extend(_add_k_fourier(k, new_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(mod, new_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=new_wires))
            op_list.append(qml.CNOT(wires=[aux_k, work_wire]))
            op_list.append(qml.QFT(wires=new_wires))
            op_list.extend(qml.ctrl(op, control=work_wire) for op in _add_k_fourier(mod, new_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(k, new_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=new_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=0))
            op_list.append(qml.QFT(wires=new_wires))
            op_list.extend(_add_k_fourier(k, new_wires))

        for op1, op2 in zip(phase_adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)
        m = 2
        # m in binary
        m_list = [0, 1, 0]
        k = 6
        mod = 7
        wires = [0, 1, 2]
        work_wires = [4, 5]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(m_list, wires=wires)
            qml.QFT(wires=work_wires[:1] + wires)
            qml.PhaseAdder(k, wires, mod, work_wires)
            qml.adjoint(qml.QFT)(wires=work_wires[:1] + wires)
            return qml.sample(wires=wires)

        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()))), (m + k) % mod
        )
