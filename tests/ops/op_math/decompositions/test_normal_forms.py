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
"""Tests for the normal forms for Clifford+T decomposition."""


import pytest

import pennylane as qml
from pennylane.ops.op_math.decompositions.normal_forms import (
    _clifford_group_to_SO3,
    _ma_normal_form,
    _parity_transforms,
)
from pennylane.ops.op_math.decompositions.rings import DyadicMatrix, SO3Matrix, ZOmega
from pennylane.ops.op_math.decompositions.solovay_kitaev import (
    _quaternion_transform,
    _SU2_transform,
)


class TestNormalForms:
    """Tests for the Matsumoto-Amano normal forms."""

    def test_clifford_group_to_SO3(self):
        """Test that the Clifford group elements are correctly mapped to SO(3) matrices."""
        clifford_matrices = _clifford_group_to_SO3()
        assert isinstance(clifford_matrices, dict)
        assert len(clifford_matrices) == 24

        for gate, so3mat in clifford_matrices.items():
            # Check that the SO(3) matrix is orthogonal and has determinant 1
            assert isinstance(so3mat, SO3Matrix), "All gates should be SO3Matrix instances"
            assert so3mat.k == 0, f"Gate {gate} should have k=0"

            su2mat = _SU2_transform(qml.matrix(gate))[0]
            w, x, y, z = _quaternion_transform(su2mat)
            assert qml.math.allclose(
                so3mat.ndarray,
                qml.math.array(
                    [
                        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
                    ]
                ),
            ), "SO(3) matrix does not match expected form"

    def test_parity_transforms(self):
        """Test that the parity transforms are correctly defined."""
        parity_transforms = _parity_transforms()
        assert isinstance(parity_transforms, dict)
        assert len(parity_transforms) == 4

        parity_vecs = [(1, 1, 1), (2, 2, 0), (0, 2, 2), (2, 0, 2)]
        for ix, (parity_vec, (so3mat, gate)) in enumerate(parity_transforms.items()):

            assert isinstance(so3mat, SO3Matrix), "All transform should have SO3Matrix instances"
            assert isinstance(
                gate, (qml.ops.Operation, qml.ops.op_math.Prod)
            ), "Each transform should have a gate"

            su2mat = _SU2_transform(qml.matrix(qml.adjoint(gate)))[0]
            w, x, y, z = _quaternion_transform(su2mat)
            so3mat2 = qml.math.array(
                [
                    [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
                ]
            )
            assert qml.math.allclose(
                so3mat.ndarray, so3mat2
            ), "SO(3) matrix does not match expected form"
            assert qml.math.allclose(parity_vecs[ix], parity_vec)

    @pytest.mark.parametrize("klen", [1, 2, 5, 6])
    def test_ma_normal_form(self, klen):
        """Test the Matsumoto-Amano normal form decomposition."""
        clifford_elements = _clifford_group_to_SO3()
        a = int(klen % 2)
        b = qml.math.random.randint(2, size=klen)
        c = qml.math.random.randint(len(clifford_elements))

        if a:
            so3mat = SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(c=1)))
            so3rep = qml.T(0)
        else:
            so3mat = SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1)))
            so3rep = qml.I(0)

        for b_ in b:
            if b_:
                so3mat @= SO3Matrix(
                    DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(b=1), ZOmega(a=-1), k=1)
                )
                so3rep @= qml.S(0) @ qml.H(0) @ qml.T(0)
            else:
                so3mat @= SO3Matrix(
                    DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(d=1), ZOmega(c=-1), k=1)
                )
                so3rep @= qml.H(0) @ qml.T(0)

        cl_list = list(clifford_elements.keys())
        so3mat @= clifford_elements[cl_list[c]]
        so3rep @= cl_list[c]

        (t_bit, rep_bits, c_bit) = _ma_normal_form(so3mat, compressed=True)
        assert t_bit == a, "T bit does not match expected value"
        assert (rep_bits == b).all(), "Representation bits do not match expected values"
        assert c_bit == c, "Clifford bit does not match expected value"

        decomposition = _ma_normal_form(so3mat, compressed=False)
        assert qml.equal(
            qml.simplify(decomposition), qml.simplify(so3rep)
        ), "Decomposition does not match expected operator"
