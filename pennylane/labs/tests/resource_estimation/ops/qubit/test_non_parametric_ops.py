# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for non parametric resource operators.
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestHadamard:
    """Tests for ResourceHadamard"""

    def test_resources(self):
        """Test that ResourceHadamard does not implement a decomposition"""
        op = re.ResourceHadamard()
        with pytest.raises(re.ResourcesNotDefined):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceHadamard()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceHadamard, {})
        assert re.ResourceHadamard.resource_rep() == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        h = re.ResourceHadamard()
        h_dag = h.adjoint_resource_decomp()

        expected = [re.GateCount(re.ResourceHadamard.resource_rep(), 1)]
        assert h_dag == expected

    pow_data = (
        (1, [re.GateCount(re.ResourceHadamard.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (3, [re.GateCount(re.ResourceHadamard.resource_rep(), 1)]),
        (4, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = re.ResourceHadamard(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestS:
    """Tests for ResourceS"""

    def test_resources(self):
        """Test that S decomposes into two Ts"""
        op = re.ResourceS(0)
        expected = [re.GateCount(re.ResourceT.resource_rep(), 2)]
        assert op.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceS(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceS, {})
        assert re.ResourceS.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""

        op = re.ResourceS(0)
        expected = [re.GateCount(re.ResourceT.resource_rep(), 2)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [
            re.GateCount(re.ResourceZ.resource_rep(), 1),
            re.GateCount(re.ResourceS.resource_rep(), 1),
        ]
        assert re.ResourceS.adjoint_resource_decomp() == expected


    pow_data = (
        (1, [re.GateCount(re.ResourceS.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourceZ.resource_rep(), 1)]),
        (
            3,
            [
                re.GateCount(re.ResourceZ.resource_rep(), 1),
                re.GateCount(re.ResourceS.resource_rep(), 1),
            ],
        ),
        (4, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (
            7,
            [
                re.GateCount(re.ResourceZ.resource_rep(), 1),
                re.GateCount(re.ResourceS.resource_rep(), 1),
            ],
        ),
        (8, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (14, [re.GateCount(re.ResourceZ.resource_rep(), 1)]),
        (
            15,
            [
                re.GateCount(re.ResourceZ.resource_rep(), 1),
                re.GateCount(re.ResourceS.resource_rep(), 1),
            ],
        ),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = re.ResourceS(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestT:
    """Tests for ResourceT"""

    def test_resources(self):
        """Test that there is no further decomposition of the T gate."""
        op = re.ResourceT(0)
        with pytest.raises(re.ResourcesNotDefined):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceT(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceT, {})
        assert re.ResourceT.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [
            re.GateCount(re.ResourceT.resource_rep(), 1),
            re.GateCount(re.ResourceS.resource_rep(), 1),
            re.GateCount(re.ResourceZ.resource_rep(), 1),
        ]
        assert re.ResourceT.adjoint_resource_decomp() == expected

    pow_data = (
        (1, [re.GateCount(re.ResourceT.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourceS.resource_rep(), 1)]),
        (
            3,
            [
                re.GateCount(re.ResourceS.resource_rep(), 1),
                re.GateCount(re.ResourceT.resource_rep(), 1),
            ],
        ),
        (
            7,
            [
                re.GateCount(re.ResourceZ.resource_rep(), 1),
                re.GateCount(re.ResourceS.resource_rep(), 1),
                re.GateCount(re.ResourceT.resource_rep(), 1),
            ],
        ),
        (8, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (
            14,
            [
                re.GateCount(re.ResourceZ.resource_rep(), 1),
                re.GateCount(re.ResourceS.resource_rep(), 1),
            ],
        ),
        (
            15,
            [
                re.GateCount(re.ResourceZ.resource_rep(), 1),
                re.GateCount(re.ResourceS.resource_rep(), 1),
                re.GateCount(re.ResourceT.resource_rep(), 1),
            ],
        ),
        (16, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = re.ResourceT
        assert op.pow_resource_decomp(z) == expected_res


class TestX:
    """Tests for the ResourceX gate"""

    def test_resources(self):
        """Tests for the ResourceX gate"""
        expected = [
            re.GateCount(re.ResourceHadamard.resource_rep(), 2),
            re.GateCount(re.ResourceS.resource_rep(), 2),
        ]
        assert re.ResourceX.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceX(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceX, {})
        assert re.ResourceX.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourceX.resource_rep(), 1)]
        assert re.ResourceX.adjoint_resource_decomp() == expected

        # x = re.ResourceX(0)
        # x_dag = re.ResourceAdjoint(x)

        # r1 = re.get_resources(x)
        # r2 = re.get_resources(x_dag)
        # assert r1 == r2

    pow_data = (
        (1, [re.GateCount(re.ResourceX.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (3, [re.GateCount(re.ResourceX.resource_rep(), 1)]),
        (4, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = re.ResourceX(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestY:
    """Tests for the ResourceY gate"""

    def test_resources(self):
        """Test that ResourceT does not implement a decomposition"""
        expected = [
            re.GateCount(re.resource_rep(re.ResourceS), 1),
            re.GateCount(re.resource_rep(re.ResourceZ), 1),
            re.GateCount(re.resource_rep(re.ResourceAdjoint, {"base_cmpr_op": re.resource_rep(re.ResourceS)})),
            re.GateCount(re.resource_rep(re.ResourceHadamard), 2)
        ]
        assert re.ResourceY.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceY(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceY, {})
        assert re.ResourceY.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourceY.resource_rep(), 1)]
        assert re.ResourceY.adjoint_resource_decomp() == expected

    pow_data = (
        (1, [re.GateCount(re.ResourceY.resource_rep())]),
        (2, [re.GateCount(re.ResourceIdentity.resource_rep())]),
        (3, [re.GateCount(re.ResourceY.resource_rep())]),
        (4, [re.GateCount(re.ResourceIdentity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = re.ResourceY(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestZ:
    """Tests for the ResourceZ gate"""

    def test_resources(self):
        """Test that ResourceZ implements the correct decomposition"""
        expected = [re.GateCount(re.ResourceS.resource_rep(), 2)]
        assert re.ResourceZ.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceZ(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceZ, {})
        assert re.ResourceZ.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourceZ.resource_rep(), 1)]
        assert re.ResourceZ.adjoint_resource_decomp() == expected

    pow_data = (
        (1, [re.GateCount(re.ResourceZ.resource_rep())]),
        (2, [re.GateCount(re.ResourceIdentity.resource_rep())]),
        (3, [re.GateCount(re.ResourceZ.resource_rep())]),
        (4, [re.GateCount(re.ResourceIdentity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = re.ResourceZ(0)
        assert op.pow_resource_decomp(z) == expected_res
