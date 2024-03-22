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
Unit tests for the qml.is_hermitian function
"""
from typing import List

import pytest

import pennylane as qml
from pennylane.operation import Operator

hermitian_ops = (
    qml.Identity(0),
    qml.Hadamard(0),
    qml.PauliX(0),
    qml.PauliY(0),
    qml.PauliZ(0),
    qml.CNOT([0, 1]),
    qml.CZ([0, 1]),
    qml.CCZ([0, 1, 2]),
    qml.CY([0, 1]),
    qml.CH([0, 1]),
    qml.SWAP([0, 1]),
    qml.CSWAP([0, 1, 2]),
    qml.Toffoli([0, 1, 2]),
    qml.MultiControlledX(wires=[0, 1, 2]),
)

non_hermitian_ops = (
    qml.S(0),
    qml.T(0),
    qml.SX(0),
    qml.ISWAP([0, 1]),
    qml.SISWAP([0, 1]),
    qml.RX(1.23, 0),
    qml.RY(1.23, 0),
    qml.RZ(1.23, 0),
    qml.PhaseShift(1.23, 0),
    qml.Rot(1.23, 1.23, 1.23, 0),
    qml.U1(1.23, 0),
    qml.U2(1.23, 1.23, 0),
    qml.U3(1.23, 1.23, 1.23, 0),
    qml.CRX(1.23, [0, 1]),
    qml.CRY(1.23, [0, 1]),
    qml.CRZ(1.23, [0, 1]),
    qml.CRot(1.23, 1.23, 1.23, [0, 1]),
    qml.IsingXX(1.23, [0, 1]),
    qml.IsingYY(1.23, [0, 1]),
    qml.IsingZZ(1.23, [0, 1]),
)

ops = (
    (qml.PauliX(wires=0), qml.PauliZ(wires=0), qml.RX(1.23, 0)),
    (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.Identity(wires=0)),
    (
        qml.IsingXX(4.56, wires=[2, 3]),
        qml.Toffoli(wires=[1, 2, 3]),
        qml.Rot(0.34, 1.0, 0, wires=0),
    ),
)


class TestIsHermitian:
    """Tests for the qml.is_hermitian function."""

    @pytest.mark.parametrize("op", hermitian_ops)
    def test_hermitian_ops(self, op: Operator):
        """Test that all the non-parametric ops are hermitian."""
        assert qml.is_hermitian(op)
        assert op.is_hermitian

    @pytest.mark.parametrize("op", non_hermitian_ops)
    def test_non_hermitian_ops(self, op: Operator):
        """Test that all the non-parametric ops are hermitian."""
        assert not qml.is_hermitian(op)
        assert not op.is_hermitian

    @pytest.mark.parametrize("arithmetic_ops", ops)
    def test_arithmetic_ops(self, arithmetic_ops: List[Operator]):
        """Test that provided arithmetic op cases are non-hermitian."""
        assert not qml.is_hermitian(qml.prod(*arithmetic_ops))
        assert not qml.is_hermitian(qml.sum(*arithmetic_ops))

    @pytest.mark.parametrize("op", hermitian_ops)
    def test_s_prod(self, op):
        """Test the hermitian check with scalar products of hermitian operators."""
        assert qml.is_hermitian(qml.s_prod(6, op))
        assert not qml.is_hermitian(qml.s_prod(1j, op))

    @pytest.mark.all_interfaces
    def test_all_interfaces(self):
        """Test hermitian check with all available interfaces."""
        import jax
        import tensorflow as tf
        import torch

        torch_param = torch.tensor(1.23)
        jax_param = jax.numpy.array(1.23)
        tf_param = tf.Variable(1.23)

        for param in [torch_param, jax_param, tf_param]:
            assert not qml.is_hermitian(qml.RX(param, 0))
