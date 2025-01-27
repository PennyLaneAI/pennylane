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
Test the ResourceStatePrep class
"""
import pytest

import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane import numpy as qnp

# pylint: disable=no-self-use


class TestStatePrep:
    """Test the ResourceStatePrep class"""

    op_data = (
        re.ResourceStatePrep([1, 0], wires=[0]),
        re.ResourceStatePrep(qnp.random.rand(2**3), wires=range(3), normalize=True),
        re.ResourceStatePrep(qnp.random.rand(10), wires=range(4), normalize=True, pad_with=0),
        re.ResourceStatePrep(qnp.random.rand(2**5), wires=range(5), normalize=True),
    )

    resource_data = (
        {
            re.ResourceRZ.resource_rep(): 3,
        },
        {
            re.ResourceRZ.resource_rep(): 27,
            re.ResourceCNOT.resource_rep(): 16,
        },
        {
            re.ResourceRZ.resource_rep(): 59,
            re.ResourceCNOT.resource_rep(): 44,
        },
        {
            re.ResourceRZ.resource_rep(): 123,
            re.ResourceCNOT.resource_rep(): 104,
        },
    )

    resource_params_data = (
        {
            "num_wires": 1,
        },
        {
            "num_wires": 3,
        },
        {
            "num_wires": 4,
        },
        {
            "num_wires": 5,
        },
    )

    name_data = (
        "StatePrep(1)",
        "StatePrep(3)",
        "StatePrep(4)",
        "StatePrep(5)",
    )

    @pytest.mark.parametrize(
        "op, params, expected_res", zip(op_data, resource_params_data, resource_data)
    )
    def test_resources(self, op, params, expected_res):
        """Test the resources method returns the correct dictionary"""
        res_from_op = op.resources(**op.resource_params())
        res_from_func = re.ResourceStatePrep.resources(**params)

        assert res_from_op == expected_res
        assert res_from_func == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params() == expected_params

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceStatePrep, expected_params)
        assert re.ResourceStatePrep.resource_rep(**expected_params) == expected

    @pytest.mark.parametrize("params, expected_name", zip(resource_params_data, name_data))
    def test_tracking_name(self, params, expected_name):
        """Test that the tracking name is correct."""
        assert re.ResourceStatePrep.tracking_name(**params) == expected_name
