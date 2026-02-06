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
Unit tests for the qp.is_unitary function
"""
import pytest

import pennylane as qp

unitary_ops = (
    qp.Identity(0),
    qp.Hadamard(0),
    qp.PauliX(0),
    qp.PauliY(0),
    qp.PauliZ(0),
    qp.CNOT([0, 1]),
    qp.CZ([0, 1]),
    qp.CY([0, 1]),
    qp.SWAP([0, 1]),
    qp.CSWAP([0, 1, 2]),
    qp.Toffoli([0, 1, 2]),
    qp.S(0),
    qp.T(0),
    qp.SX(0),
    qp.ISWAP([0, 1]),
    qp.SISWAP([0, 1]),
    qp.RX(1.23, 0),
    qp.RY(1.23, 0),
    qp.RZ(1.23, 0),
    qp.PhaseShift(1.23, 0),
    qp.Rot(1.23, 1.23, 1.23, 0),
    qp.U1(1.23, 0),
    qp.U2(1.23, 1.23, 0),
    qp.U3(1.23, 1.23, 1.23, 0),
    qp.CRX(1.23, [0, 1]),
    qp.CRY(1.23, [0, 1]),
    qp.CRZ(1.23, [0, 1]),
    qp.CRot(1.23, 1.23, 1.23, [0, 1]),
    qp.IsingXX(1.23, [0, 1]),
    qp.IsingYY(1.23, [0, 1]),
    qp.IsingZZ(1.23, [0, 1]),
)

ops = (
    (qp.PauliX(wires=0), qp.PauliZ(wires=0), qp.RX(1.23, 0)),
    (qp.CNOT(wires=[0, 1]), qp.RX(1.23, wires=1), qp.Identity(wires=0)),
    (
        qp.IsingXX(4.56, wires=[2, 3]),
        qp.Toffoli(wires=[1, 2, 3]),
        qp.Rot(0.34, 1.0, 0, wires=0),
    ),
)


class TestIsUnitary:
    """Tests for the qp.is_unitary function."""

    @pytest.mark.parametrize("op", unitary_ops)
    def test_unitary_ops(self, op):
        """Test that all the non-parametric ops are unitary."""
        assert qp.is_unitary(op)

    @pytest.mark.parametrize("arithmetic_ops", ops)
    def test_arithmetic_ops(self, arithmetic_ops):
        """Test unitary check with Prod and Sum arithmetic operations."""
        assert qp.is_unitary(qp.prod(*arithmetic_ops))
        assert not qp.is_unitary(qp.sum(*arithmetic_ops))

    @pytest.mark.parametrize("op", unitary_ops)
    def test_s_prod(self, op):
        """Test the hermitian check with scalar products of hermitian operators."""
        assert qp.is_unitary(qp.s_prod(1, op))
        assert not qp.is_unitary(qp.s_prod(2, op))

    @pytest.mark.all_interfaces
    def test_all_interfaces(self):
        """Test unitary check with all available interfaces."""
        import jax
        import torch

        torch_param = torch.tensor(1.23)
        jax_param = jax.numpy.array(1.23)

        for param in [torch_param, jax_param]:
            assert qp.is_unitary(qp.RX(param, 0))
