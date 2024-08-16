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
    wires=[0,1,2,3]
    work_wires=[4,5]
    op = qml.Adder(k, wires=wires,mod=mod,work_wires=work_wires)
    qml.ops.functions.assert_valid(op)


class TestAdder:
    """Test the qml.Adder template."""

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                5,
                [0, 1, 2, 3],
                8,
                [4,5],
            ),
            (
                1,
                [0, 1, 2],
                3,
                [3,4],
            ),
            (
                12,
                [0, 1, 2, 3, 4 ,5, 6],
                22,
                [7,8],
            ),
        ],
    )
    def test_operation_result(
        self, k, wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the Adder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(m):
            qml.BasisEmbedding(m, wires=wires)
            qml.Adder(k,wires,mod,work_wires)
            return qml.sample(wires=wires)
        
        if mod == None:
            max = 2**len(wires)
        else:
            max=mod
        for m in range(max):
            assert np.allclose(sum(bit * (2 ** i) for i, bit in enumerate(reversed(circuit(m)))), (m+k) % max)

    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                4,
                [0, 1, 2, 3, 4],
                12,
                None,
            ),
            (
                4,
                [0, 1, 2, 3, 4],
                None,
                [5,6],
            ),
            (
                4,
                [0, 1, 2, 3, 4],
                None,
                None,
            ),
        ],
    )
    def test_operation_result_args_None(
        self, k, wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the Adder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(m):
            qml.BasisEmbedding(m, wires=wires)
            qml.Adder(k,wires,mod,work_wires)
            return qml.sample(wires=wires)

        if mod == None:
            max = 2**len(wires)
        else:
            max=mod
        for m in range(max):
            assert np.allclose(sum(bit * (2 ** i) for i, bit in enumerate(reversed(circuit(m)))), (m+k) % max)
    
    @pytest.mark.parametrize(
        ("k", "wires", "mod", "work_wires"),
        [
            (
                4,
                [0, 1, 2, 3],
                7,
                [4,5],
            ),
            (
                4,
                [0, 1, 2, 3],
                None,
                [4,5],
            ),
            (
                4,
                [0, 1, 2, 3],
                8,
                [4,5],
            ),
        ],
    )
    def test_decomposition(self, k, wires, mod, work_wires):
        """Test that compute_decomposition and decomposition work as expected."""

        Adder_decomposition = qml.Adder(
            k, wires, mod, work_wires).compute_decomposition(k, mod, work_wires, wires)
        op_list = []
        # we perform m+k modulo mod
        if (mod==2**(len(wires))):
            qft_wires=wires
        else:
            qft_wires=work_wires[:1]+wires
        # we perform m+k modulo mod
        op_list.append(qml.QFT(qft_wires))
        op_list.append(qml.PhaseAdder(k,wires,mod,work_wires))
        op_list.append(qml.adjoint(qml.QFT)(qft_wires))

        for op1, op2 in zip(Adder_decomposition, op_list):
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
        work_wires=[4,5]
        dev = qml.device("default.qubit", shots=1)
        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(m_list, wires=wires)
            qml.Adder(k,wires,mod,work_wires)
            return qml.sample(wires=wires)
        assert jax.numpy.allclose(sum(bit * (2 ** i) for i, bit in enumerate(reversed(circuit()))), (m+k) % mod)