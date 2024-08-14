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
Tests for the InMultiplier template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_standard_validity_Multiplier():
    """Check the operation using the assert_valid function."""
    k = 6
    mod = 11
    wires=[0,1,2,3]
    work_wires=[4,5,6,7,8,9]
    op = qml.InMultiplier(k, wires=wires,mod=mod,work_wires=work_wires)
    qml.ops.functions.assert_valid(op)

def _mul_out_k_mod(k, wires_m,mod, work_wires_aux,wires_aux):
    """Performs m*k in the registers wires_aux"""
    op_list = []
    if mod == (2**len(wires_m)):
        qft_wires=wires_aux
    else:
        qft_wires=work_wires_aux[:1]+wires_aux
    op_list.append(qml.QFT(wires = qft_wires))
    op_list.append(qml.ControlledSequence(PhaseAdder(k, wires_aux,mod,work_wires_aux), control = wires_m))
    op_list.append(qml.adjoint(qml.QFT(wires = qft_wires)))
    return op_list

class TestMultiplier:
    """Test the qml.InMultiplier template."""

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                5,
                [0, 1, 2],
                8,
                [4, 5, 6, 7, 8],
            ),
            (
                1,
                [0, 1, 2],
                3,
                [3, 4, 5, 6, 7],
            ),
            (
                12,
                [0, 1, 2, 3, 4],
                23,
                [5, 6, 7, 8, 9, 10, 11],
            ),
        ],
    )
    def test_operation_result(
        self, k, wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the Multiplier template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(m):
            qml.BasisEmbedding(m, wires=wires)
            qml.InMultiplier(k,wires,mod,work_wires)
            return qml.sample(wires=wires)
        
        if mod == None:
            max = 2**len(wires)
        else:
            max=mod
        for m in range(max):
            assert np.allclose(sum(bit * (2 ** i) for i, bit in enumerate(reversed(circuit(m)))), (m*k) % max)

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                5,
                [0, 1, 2, 3, 4],
                12,
                None,
            ),
            (
                5,
                [0, 1, 2, 3, 4],
                None,
                [5, 6, 7, 8, 9, 10, 11],
            ),
            (
                7,
                [0, 1, 2, 3, 4],
                None,
                None,
            ),
        ],
    )
    def test_operation_result_args_None(
        self, k, wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the Multiplier template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(m):
            qml.BasisEmbedding(m, wires=wires)
            qml.InMultiplier(k,wires,mod,work_wires)
            return qml.sample(wires=wires)

        if mod == None:
            max = 2**len(wires)
        else:
            max=mod
        for m in range(max):
            assert np.allclose(sum(bit * (2 ** i) for i, bit in enumerate(reversed(circuit(m)))), (m*k) % max)
    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires","msg_match"),
        [
            (
                7,
                [0, 1, 2, 3, 4],
                6,
                [5, 6, 7, 8, 9, 10, 11],
                "The module mod must be larger than k."
            ),
            (
                6,
                [0, 1],
                7,
                [3, 4, 5, 6],
                "InMultiplier must have at least enough wires to represent mod."
            ),
            (
                2,
                [0, 1, 2],
                6,
                [3, 4, 5, 6, 7],
                "Since k has no inverse modulo mod, the work_wires cannot be cleaned."
            ),
        ],
    )
    def test_operation_error(self, k, wires, mod, work_wires, msg_match):
        """Test an error is raised when k or mod don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.InMultiplier(k,wires,mod,work_wires)
    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires","msg_match"),
        [
            (
                3,
                [0, 1, 2, 3, 4],
                11,
                [4,5],
                "Any wire in work_wires should not be included in wires."
            ),
            (
                3,
                [0, 1, 2, 3, 4],
                11,
                [5, 6, 7, 8, 9, 10],
                "InMultiplier needs as many work_wires as wires plus two."
            ),
        ],
    )
    def test_wires_error(self, k, wires, mod, work_wires, msg_match):
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qml.InMultiplier(k,wires,mod,work_wires)
    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                4,
                [0, 1, 2],
                7,
                [3, 4, 5, 6, 7],
            ),
            (
                3,
                [0, 1, 2, 3],
                None,
                [4, 5, 6, 7, 8, 9],
            ),
            (
                3,
                [0, 1, 2, 3],
                8,
                [4 ,5, 6, 7, 8, 9],
            ),
        ],
    )
    def test_decomposition(self, k, wires, mod, work_wires):
        """Test that compute_decomposition and decomposition work as expected."""

        multiplier_decomposition = qml.InMultiplier(
            k, wires, mod, work_wires).compute_decomposition(k, mod, work_wires, wires)
        op_list = []
        # we perform m*k modulo mod
        work_wires_aux=work_wires[0:2]
        wires_aux=work_wires[2:]
        op_list.extend(_mul_out_k_mod(k, wires, mod, work_wires_aux, wires_aux))
        for i in range(len(wires)):
            op_list.append(qml.SWAP(wires=[wires[i], wires_aux[i]]))
        inv_k=pow(k, -1, mod) 
        op_list.extend(qml.adjoint(_mul_out_k_mod)(inv_k, wires, mod, work_wires_aux, wires_aux))

        for op1, op2 in zip(multiplier_decomposition, op_list):
            qml.assert_equal(op1, op2)
    #@pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)
        m=2
        # m in binary
        m_list = [0,1,0]
        k = 6
        mod = 7
        wires=[0,1,2]
        work_wires=[4,5,6,7,8]
        dev = qml.device("default.qubit", shots=1)
        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(m_list, wires=wires)
            qml.InMultiplier(k,wires,mod,work_wires)
            return qml.sample(wires=wires)
        assert jax.numpy.allclose(sum(bit * (2 ** i) for i, bit in enumerate(reversed(circuit()))), (m*k) % mod)