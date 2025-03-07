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
Test the ResourceReflection class
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use


class TestReflection:
    """Test the ResourceReflection class"""

    op_data = (
        re.ResourceReflection(re.ResourceHadamard(0)),
        re.ResourceReflection(re.ResourceProd(re.ResourceX(0), re.ResourceY(1)), alpha=1.23),
        re.ResourceReflection(re.ResourceQFT(range(4)), reflection_wires=range(3)),
    )

    resource_data = (
        {
            re.ResourceX.resource_rep(): 2,
            re.ResourceGlobalPhase.resource_rep(): 1,
            re.ResourceHadamard.resource_rep(): 1,
            re.ResourceAdjoint.resource_rep(base_class=re.ResourceHadamard, base_params={}): 1,
            re.ResourcePhaseShift.resource_rep(): 1,
        },
        {
            re.ResourceX.resource_rep(): 2,
            re.ResourceGlobalPhase.resource_rep(): 1,
            re.ResourceProd.resource_rep(
                cmpr_factors=(re.ResourceX.resource_rep(), re.ResourceY.resource_rep())
            ): 1,
            re.ResourceAdjoint.resource_rep(
                base_class=re.ResourceProd,
                base_params={
                    "cmpr_factors": (re.ResourceX.resource_rep(), re.ResourceY.resource_rep())
                },
            ): 1,
            re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 1, 1, 0): 1,
        },
        {
            re.ResourceX.resource_rep(): 2,
            re.ResourceGlobalPhase.resource_rep(): 1,
            re.ResourceQFT.resource_rep(num_wires=4): 1,
            re.ResourceAdjoint.resource_rep(
                base_class=re.ResourceQFT, base_params={"num_wires": 4}
            ): 1,
            re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 2, 2, 0): 1,
        },
    )

    resource_params_data = (
        {
            "base_class": re.ResourceHadamard,
            "base_params": {},
            "num_ref_wires": 1,
        },
        {
            "base_class": re.ResourceProd,
            "base_params": {
                "cmpr_factors": (re.ResourceX.resource_rep(), re.ResourceY.resource_rep())
            },
            "num_ref_wires": 2,
        },
        {
            "base_class": re.ResourceQFT,
            "base_params": {"num_wires": 4},
            "num_ref_wires": 3,
        },
    )

    @pytest.mark.parametrize(
        "op, params, expected_res", zip(op_data, resource_params_data, resource_data)
    )
    def test_resources(self, op, params, expected_res):
        """Test the resources method returns the correct dictionary"""
        res_from_op = op.resources(**op.resource_params)
        res_from_func = re.ResourceReflection.resources(**params)

        assert res_from_op == expected_res
        assert res_from_func == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceReflection, expected_params)
        assert re.ResourceReflection.resource_rep(**expected_params) == expected
