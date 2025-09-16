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
"""
Tests for non parametric resource operators.
"""
import pytest

import pennylane.labs.resource_estimation as plre

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestHadamard:
    """Tests for ResourceHadamard"""

    def test_resources(self):
        """Test that ResourceHadamard does not implement a decomposition"""
        op = plre.ResourceHadamard()
        with pytest.raises(plre.ResourcesNotDefined):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceHadamard()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = plre.CompressedResourceOp(plre.ResourceHadamard, 1, {})
        assert plre.ResourceHadamard.resource_rep() == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        h = plre.ResourceHadamard()
        h_dag = h.adjoint_resource_decomp()

        expected = [plre.GateCount(plre.ResourceHadamard.resource_rep(), 1)]
        assert h_dag == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                plre.GateCount(plre.ResourceCH.resource_rep(), 1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                plre.GateCount(plre.ResourceCH.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                plre.GateCount(plre.ResourceHadamard.resource_rep(), 2),
                plre.GateCount(plre.ResourceRY.resource_rep(), 2),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceHadamard.resource_rep(), 2),
                plre.GateCount(plre.ResourceRY.resource_rep(), 2),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 1),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceHadamard(0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceHadamard.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (3, [plre.GateCount(plre.ResourceHadamard.resource_rep(), 1)]),
        (4, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = plre.ResourceHadamard(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestSWAP:
    """Tests for ResourceSWAP"""

    def test_resources(self):
        """Test that SWAP decomposes into three CNOTs"""
        op = plre.ResourceSWAP([0, 1])
        cnot = plre.ResourceCNOT.resource_rep()
        expected = [plre.GateCount(cnot, 3)]

        assert op.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceSWAP([0, 1])
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test the compact representation"""
        expected = plre.CompressedResourceOp(plre.ResourceSWAP, 2, {})
        assert plre.ResourceSWAP.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""

        op = plre.ResourceSWAP([0, 1])
        expected = [plre.GateCount(plre.ResourceCNOT.resource_rep(), 3)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        swap = plre.ResourceSWAP([0, 1])
        expected = [plre.GateCount(swap.resource_rep(), 1)]

        assert swap.adjoint_resource_decomp() == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                plre.GateCount(plre.ResourceCSWAP.resource_rep(), 1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                plre.GateCount(plre.ResourceCSWAP.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                plre.GateCount(plre.ResourceCNOT.resource_rep(), 2),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 0), 1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceCNOT.resource_rep(), 2),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(4, 2), 1),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceSWAP([0, 1])

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceSWAP.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (3, [plre.GateCount(plre.ResourceSWAP.resource_rep(), 1)]),
        (4, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = plre.ResourceSWAP([0, 1])
        assert op.pow_resource_decomp(z) == expected_res


class TestS:
    """Tests for ResourceS"""

    def test_resources(self):
        """Test that S decomposes into two Ts"""
        op = plre.ResourceS(0)
        expected = [plre.GateCount(plre.ResourceT.resource_rep(), 2)]
        assert op.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceS(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct"""
        expected = plre.CompressedResourceOp(plre.ResourceS, 1, {})
        assert plre.ResourceS.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = plre.ResourceS(0)
        expected = [plre.GateCount(plre.ResourceT.resource_rep(), 2)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [
            plre.GateCount(plre.ResourceZ.resource_rep(), 1),
            plre.GateCount(plre.ResourceS.resource_rep(), 1),
        ]
        assert plre.ResourceS.adjoint_resource_decomp() == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
                plre.GateCount(plre.resource_rep(plre.ResourceT), 2),
                plre.GateCount(
                    plre.resource_rep(
                        plre.ResourceAdjoint, {"base_cmpr_op": plre.resource_rep(plre.ResourceT)}
                    ),
                ),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
                plre.GateCount(plre.resource_rep(plre.ResourceT), 2),
                plre.GateCount(
                    plre.resource_rep(
                        plre.ResourceAdjoint, {"base_cmpr_op": plre.resource_rep(plre.ResourceT)}
                    ),
                ),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 2),
                plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
                plre.GateCount(plre.resource_rep(plre.ResourceT), 2),
                plre.GateCount(
                    plre.resource_rep(
                        plre.ResourceAdjoint, {"base_cmpr_op": plre.resource_rep(plre.ResourceT)}
                    ),
                ),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 2),
                plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
                plre.GateCount(plre.resource_rep(plre.ResourceT), 2),
                plre.GateCount(
                    plre.resource_rep(
                        plre.ResourceAdjoint, {"base_cmpr_op": plre.resource_rep(plre.ResourceT)}
                    ),
                ),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceS(0)
        op2 = plre.ResourceControlled(
            op,
            num_ctrl_wires,
            num_ctrl_values,
        )

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceS.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceZ.resource_rep(), 1)]),
        (
            3,
            [
                plre.GateCount(plre.ResourceZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
            ],
        ),
        (4, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (
            7,
            [
                plre.GateCount(plre.ResourceZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
            ],
        ),
        (8, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (14, [plre.GateCount(plre.ResourceZ.resource_rep(), 1)]),
        (
            15,
            [
                plre.GateCount(plre.ResourceZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
            ],
        ),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = plre.ResourceS(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestT:
    """Tests for ResourceT"""

    def test_resources(self):
        """Test that there is no further decomposition of the T gate."""
        op = plre.ResourceT(0)
        with pytest.raises(plre.ResourcesNotDefined):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceT(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = plre.CompressedResourceOp(plre.ResourceT, 1, {})
        assert plre.ResourceT.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [
            plre.GateCount(plre.ResourceT.resource_rep(), 1),
            plre.GateCount(plre.ResourceS.resource_rep(), 1),
            plre.GateCount(plre.ResourceZ.resource_rep(), 1),
        ]
        assert plre.ResourceT.adjoint_resource_decomp() == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 2),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 2),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceT(0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceT.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceS.resource_rep(), 1)]),
        (
            3,
            [
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
                plre.GateCount(plre.ResourceT.resource_rep(), 1),
            ],
        ),
        (
            7,
            [
                plre.GateCount(plre.ResourceZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
                plre.GateCount(plre.ResourceT.resource_rep(), 1),
            ],
        ),
        (8, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (
            14,
            [
                plre.GateCount(plre.ResourceZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
            ],
        ),
        (
            15,
            [
                plre.GateCount(plre.ResourceZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
                plre.GateCount(plre.ResourceT.resource_rep(), 1),
            ],
        ),
        (16, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = plre.ResourceT
        assert op.pow_resource_decomp(z) == expected_res


class TestX:
    """Tests for the ResourceX gate"""

    def test_resources(self):
        """Tests for the ResourceX gate"""
        expected = [
            plre.GateCount(plre.ResourceHadamard.resource_rep(), 2),
            plre.GateCount(plre.ResourceS.resource_rep(), 2),
        ]
        assert plre.ResourceX.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceX(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = plre.CompressedResourceOp(plre.ResourceX, 1, {})
        assert plre.ResourceX.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [plre.GateCount(plre.ResourceX.resource_rep(), 1)]
        assert plre.ResourceX.adjoint_resource_decomp() == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                plre.GateCount(plre.ResourceCNOT.resource_rep(), 1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
                plre.GateCount(plre.ResourceCNOT.resource_rep(), 1),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                plre.GateCount(plre.ResourceToffoli.resource_rep(), 1),
            ],
        ),
        (
            ["c1", "c2"],
            [0, 0],
            [
                plre.GateCount(plre.ResourceX.resource_rep(), 4),
                plre.GateCount(plre.ResourceToffoli.resource_rep(), 1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 1),
            ],
        ),
        (
            ["c1", "c2", "c3", "c4"],
            [1, 0, 0, 1],
            [
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(4, 2), 1),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceX(0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceX.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (3, [plre.GateCount(plre.ResourceX.resource_rep(), 1)]),
        (4, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = plre.ResourceX(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestY:
    """Tests for the ResourceY gate"""

    def test_resources(self):
        """Test that ResourceT does not implement a decomposition"""
        expected = [
            plre.GateCount(plre.ResourceS.resource_rep(), 1),
            plre.GateCount(plre.ResourceZ.resource_rep(), 1),
            plre.GateCount(plre.ResourceAdjoint.resource_rep(plre.ResourceS.resource_rep()), 1),
            plre.GateCount(plre.ResourceHadamard.resource_rep(), 2),
        ]
        assert plre.ResourceY.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceY(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = plre.CompressedResourceOp(plre.ResourceY, 1, {})
        assert plre.ResourceY.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [plre.GateCount(plre.ResourceY.resource_rep(), 1)]
        assert plre.ResourceY.adjoint_resource_decomp() == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                plre.GateCount(plre.ResourceCY.resource_rep(), 1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                plre.GateCount(plre.ResourceCY.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
                plre.GateCount(
                    plre.resource_rep(
                        plre.ResourceAdjoint, {"base_cmpr_op": plre.ResourceS.resource_rep()}
                    ),
                    1,
                ),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 1),
            ],
        ),
        (
            ["c1", "c2"],
            [0, 0],
            [
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
                plre.GateCount(
                    plre.resource_rep(
                        plre.ResourceAdjoint, {"base_cmpr_op": plre.ResourceS.resource_rep()}
                    ),
                    1,
                ),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 2), 1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceS.resource_rep(), 1),
                plre.GateCount(
                    plre.resource_rep(
                        plre.ResourceAdjoint, {"base_cmpr_op": plre.ResourceS.resource_rep()}
                    ),
                    1,
                ),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 1),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceY(0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceY.resource_rep())]),
        (2, [plre.GateCount(plre.ResourceIdentity.resource_rep())]),
        (3, [plre.GateCount(plre.ResourceY.resource_rep())]),
        (4, [plre.GateCount(plre.ResourceIdentity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = plre.ResourceY(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestZ:
    """Tests for the ResourceZ gate"""

    def test_resources(self):
        """Test that ResourceZ implements the correct decomposition"""
        expected = [plre.GateCount(plre.ResourceS.resource_rep(), 2)]
        assert plre.ResourceZ.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceZ(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = plre.CompressedResourceOp(plre.ResourceZ, 1, {})
        assert plre.ResourceZ.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [plre.GateCount(plre.ResourceZ.resource_rep(), 1)]
        assert plre.ResourceZ.adjoint_resource_decomp() == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                plre.GateCount(plre.ResourceCZ.resource_rep(), 1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                plre.GateCount(plre.ResourceCZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                plre.GateCount(plre.ResourceCCZ.resource_rep(), 1),
            ],
        ),
        (
            ["c1", "c2"],
            [0, 0],
            [
                plre.GateCount(plre.ResourceCCZ.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 4),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceHadamard.resource_rep(), 2),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 1),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceZ(0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceZ.resource_rep())]),
        (2, [plre.GateCount(plre.ResourceIdentity.resource_rep())]),
        (3, [plre.GateCount(plre.ResourceZ.resource_rep())]),
        (4, [plre.GateCount(plre.ResourceIdentity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = plre.ResourceZ(0)
        assert op.pow_resource_decomp(z) == expected_res
