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
Tests for parametric single qubit resource operators.
"""
import copy

import pytest

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    _rotation_resources,
)

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments

params = list(zip([10e-3, 10e-4, 10e-5], [17, 21, 24]))


@pytest.mark.parametrize("epsilon, expected", params)
def test_rotation_resources(epsilon, expected):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = {}

    t = re.CompressedResourceOp(re.ResourceT, {})
    gate_types[t] = expected
    assert gate_types == _rotation_resources(epsilon=epsilon)


class TestPauliRotation:
    """Test ResourceRX, ResourceRY, and ResourceRZ"""

    params_classes = [re.ResourceRX, re.ResourceRY, re.ResourceRZ]
    params_errors = [10e-3, 10e-4, 10e-5]
    params_ctrl_res = [
        {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceRZ.resource_rep(): 2,
        },
        {
            re.ResourceRY.resource_rep(): 2,
        },
        {
            re.ResourceRZ.resource_rep(): 2,
        },
    ]

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resources(self, resource_class, epsilon):
        """Test the resources method"""

        label = "error_" + resource_class.__name__.replace("Resource", "").lower()
        config = {label: epsilon}
        op = resource_class(1.24, wires=0)
        assert op.resources(config) == _rotation_resources(epsilon=epsilon)

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resource_rep(self, resource_class, epsilon):  # pylint: disable=unused-argument
        """Test the compact representation"""
        op = resource_class(1.24, wires=0)
        expected = re.CompressedResourceOp(resource_class, {})
        assert op.resource_rep() == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resources_from_rep(self, resource_class, epsilon):
        """Test the resources can be obtained from the compact representation"""

        label = "error_" + resource_class.__name__.replace("Resource", "").lower()
        config = {label: epsilon}
        op = resource_class(1.24, wires=0)
        expected = _rotation_resources(epsilon=epsilon)

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params, config=config) == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resource_params(self, resource_class, epsilon):  # pylint: disable=unused-argument
        """Test that the resource params are correct"""
        op = resource_class(1.24, wires=0)
        assert op.resource_params == {}

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_adjoint_decomposition(self, resource_class, epsilon):
        """Test that the adjoint decompositions are correct."""

        expected = {resource_class.resource_rep(): 1}
        assert resource_class.adjoint_resource_decomp() == expected

        op = resource_class(1.24, wires=0)
        dag = re.ResourceAdjoint(op)

        label = "error_" + resource_class.__name__.replace("Resource", "").lower()
        config = {label: epsilon}

        r1 = re.get_resources(op, config=config)
        r2 = re.get_resources(dag, config=config)

        assert r1 == r2

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    @pytest.mark.parametrize("z", list(range(0, 10)))
    def test_pow_decomposition(self, resource_class, epsilon, z):
        """Test that the pow decompositions are correct."""

        expected = (
            {resource_class.resource_rep(): 1} if z else {re.ResourceIdentity.resource_rep(): 1}
        )
        assert resource_class.pow_resource_decomp(z) == expected

        op = resource_class(1.24, wires=0) if z else re.ResourceIdentity(wires=0)
        dag = re.ResourcePow(op, z)

        label = "error_" + resource_class.__name__.replace("Resource", "").lower()
        config = {label: epsilon}

        r1 = re.get_resources(op, config=config)
        r2 = re.get_resources(dag, config=config)

        assert r1 == r2

    params_ctrl_classes = (
        (re.ResourceRX, re.ResourceCRX),
        (re.ResourceRY, re.ResourceCRY),
        (re.ResourceRZ, re.ResourceCRZ),
    )

    @pytest.mark.parametrize("resource_class, controlled_class", params_ctrl_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_controlled_decomposition_single_control(
        self, resource_class, controlled_class, epsilon
    ):
        """Test that the controlled decompositions are correct."""
        expected = {controlled_class.resource_rep(): 1}
        assert resource_class.controlled_resource_decomp(1, 0, 0) == expected

        expected = {controlled_class.resource_rep(): 1, re.ResourceX.resource_rep(): 2}
        assert resource_class.controlled_resource_decomp(1, 1, 0) == expected

        op = resource_class(1.24, wires=0)
        c_op = re.ResourceControlled(op, control_wires=[1])

        c = controlled_class(1.24, wires=[0, 1])

        config = {"error_rx": epsilon, "error_ry": epsilon, "error_rz": epsilon}

        r1 = re.get_resources(c, config=config)
        r2 = re.get_resources(c_op, config=config)

        assert r1 == r2

    ctrl_res_data = (
        (
            [1, 2],
            [1, 1],
            ["w1"],
            {re.ResourceMultiControlledX.resource_rep(2, 0, 1): 2},
        ),
        (
            [1, 2],
            [1, 0],
            [],
            {re.ResourceMultiControlledX.resource_rep(2, 1, 0): 2},
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceMultiControlledX.resource_rep(3, 2, 2): 2},
        ),
    )

    @pytest.mark.parametrize("resource_class, local_res", zip(params_classes, params_ctrl_res))
    @pytest.mark.parametrize("ctrl_wires, ctrl_values, work_wires, general_res", ctrl_res_data)
    def test_controlled_decomposition_multi_controlled(
        self, resource_class, local_res, ctrl_wires, ctrl_values, work_wires, general_res
    ):
        """Test that the controlled docomposition is correct when controlled on multiple wires."""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op = resource_class(1.23, wires=0)
        op2 = re.ResourceControlled(
            op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        expected_resources = copy.copy(local_res)
        for k, v in general_res.items():
            expected_resources[k] = v

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_resources
        )
        assert op2.resources(**op2.resource_params) == expected_resources

    # pylint: disable=unused-argument, import-outside-toplevel
    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_sparse_matrix_format(self, resource_class, epsilon):
        """Test that the sparse matrix accepts the format parameter."""
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        op = resource_class(1.24, wires=0)
        assert isinstance(op.sparse_matrix(), csr_matrix)
        assert isinstance(op.sparse_matrix(format="csc"), csc_matrix)
        assert isinstance(op.sparse_matrix(format="lil"), lil_matrix)
        assert isinstance(op.sparse_matrix(format="coo"), coo_matrix)


class TestRot:
    """Test ResourceRot"""

    def test_resources(self):
        """Test the resources method"""
        op = re.ResourceRot(0.1, 0.2, 0.3, wires=0)
        ry = re.ResourceRY.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        expected = {ry: 1, rz: 2}

        assert op.resources() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = re.ResourceRot(0.1, 0.2, 0.3, wires=0)
        expected = re.CompressedResourceOp(re.ResourceRot, {})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = re.ResourceRot(0.1, 0.2, 0.3, wires=0)
        ry = re.ResourceRY.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        expected = {ry: 1, rz: 2}

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceRot(0.1, 0.2, 0.3, wires=0)
        assert op.resource_params == {}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = {re.ResourceRot.resource_rep(): 1}
        assert re.ResourceRot.adjoint_resource_decomp() == expected

        op = re.ResourceRot(1.24, 1.25, 1.26, wires=0)
        dag = re.ResourceAdjoint(op)

        r1 = re.get_resources(op)
        r2 = re.get_resources(dag)

        assert r1 == r2

    ctrl_data = (
        ([1], [1], [], {re.ResourceCRot.resource_rep(): 1}),
        (
            [1],
            [0],
            [],
            {
                re.ResourceCRot.resource_rep(): 1,
                re.ResourceX.resource_rep(): 2,
            },
        ),
        (
            [1, 2],
            [1, 1],
            ["w1"],
            {
                re.ResourceRZ.resource_rep(): 3,
                re.ResourceRY.resource_rep(): 2,
                re.ResourceMultiControlledX.resource_rep(2, 0, 1): 2,
            },
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            ["w1", "w2"],
            {
                re.ResourceRZ.resource_rep(): 3,
                re.ResourceRY.resource_rep(): 2,
                re.ResourceMultiControlledX.resource_rep(3, 2, 2): 2,
            },
        ),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, work_wires, expected_res", ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op = re.ResourceRot(1.24, 1.25, 1.26, wires=0)
        op2 = re.ResourceControlled(
            op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (0, {re.ResourceIdentity.resource_rep(): 1}),
        (1, {re.ResourceRot.resource_rep(): 1}),
        (2, {re.ResourceRot.resource_rep(): 1}),
        (5, {re.ResourceRot.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourceRot(1.24, 1.25, 1.26, wires=0)
        assert op.pow_resource_decomp(z) == expected_res

        op2 = re.ResourcePow(op, z)
        assert op2.resources(**op2.resource_params) == expected_res


class TestPhaseShift:
    """Test ResourcePhaseShift"""

    def test_resources(self):
        """Test the resources method"""
        op = re.ResourcePhaseShift(0.1, wires=0)
        rz = re.ResourceRZ.resource_rep()
        global_phase = re.ResourceGlobalPhase.resource_rep()

        expected = {rz: 1, global_phase: 1}

        assert op.resources() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = re.ResourcePhaseShift(0.1, wires=0)
        expected = re.CompressedResourceOp(re.ResourcePhaseShift, {})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = re.ResourcePhaseShift(0.1, wires=0)
        global_phase = re.ResourceGlobalPhase.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        expected = {global_phase: 1, rz: 1}

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourcePhaseShift(0.1, wires=0)
        assert op.resource_params == {}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = {re.ResourcePhaseShift.resource_rep(): 1}
        assert re.ResourcePhaseShift.adjoint_resource_decomp() == expected

        op = re.ResourcePhaseShift(0.1, wires=0)
        dag = re.ResourceAdjoint(op)

        r1 = re.get_resources(op)
        r2 = re.get_resources(dag)

        assert r1 == r2

    ctrl_data = (
        ([1], [1], [], {re.ResourceControlledPhaseShift.resource_rep(): 1}),
        (
            [1],
            [0],
            [],
            {
                re.ResourceControlledPhaseShift.resource_rep(): 1,
                re.ResourceX.resource_rep(): 2,
            },
        ),
        (
            [1, 2],
            [1, 1],
            ["w1"],
            {
                re.ResourceControlledPhaseShift.resource_rep(): 1,
                re.ResourceMultiControlledX.resource_rep(2, 0, 1): 2,
            },
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            ["w1", "w2"],
            {
                re.ResourceControlledPhaseShift.resource_rep(): 1,
                re.ResourceMultiControlledX.resource_rep(3, 2, 2): 2,
            },
        ),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, work_wires, expected_res", ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op = re.ResourcePhaseShift(0.1, wires=0)
        op2 = re.ResourceControlled(
            op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (0, {re.ResourceIdentity.resource_rep(): 1}),
        (1, {re.ResourcePhaseShift.resource_rep(): 1}),
        (2, {re.ResourcePhaseShift.resource_rep(): 1}),
        (5, {re.ResourcePhaseShift.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourcePhaseShift(0.1, wires=0)
        assert op.pow_resource_decomp(z) == expected_res

        op2 = re.ResourcePow(op, z)
        assert op2.resources(**op2.resource_params) == expected_res
