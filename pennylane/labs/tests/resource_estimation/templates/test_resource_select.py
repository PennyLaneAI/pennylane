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
Test the ResourceSelect class
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use


class TestSelect:
    """Test the ResourceSelect class"""

    op_data = (
        re.ResourceSelect(
            [re.ResourceX(0), re.ResourceY(1)],
            control=["c1"],
        ),
        re.ResourceSelect(
            [re.ResourceCNOT((0, 1)), re.ResourceHadamard(1), re.ResourceQFT(range(3))],
            control=["c1", "c2"],
        ),
        re.ResourceSelect(
            [
                re.ResourceCNOT((0, 1)),
                re.ResourceHadamard(1),
                re.ResourceT(2),
                re.ResourceS(3),
                re.ResourceZ(0),
                re.ResourceY(1),
                re.ResourceX(3),
            ],
            control=["c1", "c2", "c3"],
        ),
    )

    resource_params_data = (
        {
            "cmpr_ops": (re.ResourceX.resource_rep(), re.ResourceY.resource_rep()),
        },
        {
            "cmpr_ops": (
                re.ResourceCNOT.resource_rep(),
                re.ResourceHadamard.resource_rep(),
                re.ResourceQFT.resource_rep(num_wires=3),
            ),
        },
        {
            "cmpr_ops": (
                re.ResourceCNOT.resource_rep(),
                re.ResourceHadamard.resource_rep(),
                re.ResourceT.resource_rep(),
                re.ResourceS.resource_rep(),
                re.ResourceZ.resource_rep(),
                re.ResourceY.resource_rep(),
                re.ResourceX.resource_rep(),
            ),
        },
    )

    resource_data = (
        {
            re.ResourceX.resource_rep(): 2,
            re.ResourceControlled.resource_rep(
                re.ResourceX,
                {},
                1,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceY,
                {},
                1,
                0,
                0,
            ): 1,
        },
        {
            re.ResourceX.resource_rep(): 4,
            re.ResourceControlled.resource_rep(
                re.ResourceCNOT,
                {},
                2,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceHadamard,
                {},
                2,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceQFT,
                {"num_wires": 3},
                2,
                0,
                0,
            ): 1,
        },
        {
            re.ResourceX.resource_rep(): 8,
            re.ResourceControlled.resource_rep(
                re.ResourceCNOT,
                {},
                3,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceHadamard,
                {},
                3,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceT,
                {},
                3,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceS,
                {},
                3,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceZ,
                {},
                3,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceY,
                {},
                3,
                0,
                0,
            ): 1,
            re.ResourceControlled.resource_rep(
                re.ResourceX,
                {},
                3,
                0,
                0,
            ): 1,
        },
    )

    @pytest.mark.parametrize(
        "op, params, expected_res", zip(op_data, resource_params_data, resource_data)
    )
    def test_resources(self, op, params, expected_res):
        """Test the resources method returns the correct dictionary"""
        res_from_op = op.resources(**op.resource_params)
        res_from_func = re.ResourceSelect.resources(**params)

        assert res_from_op == expected_res
        assert res_from_func == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceSelect, expected_params)
        assert re.ResourceSelect.resource_rep(**expected_params) == expected
