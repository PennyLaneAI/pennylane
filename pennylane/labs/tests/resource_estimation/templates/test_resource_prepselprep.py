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
Test the ResourcePrepSelPrep class
"""
import copy

import pytest

import pennylane.labs.resource_estimation as re
from pennylane.ops import LinearCombination

# pylint: disable=no-self-use


class TestPrepSelPrep:
    """Test the ResourcePrepSelPrep class"""

    op_data = (
        re.ResourcePrepSelPrep(
            LinearCombination([1.23, -4.5], [re.ResourceX(0), re.ResourceZ(0)]),
            control=["c1"],
        ),
        re.ResourcePrepSelPrep(
            LinearCombination(
                [1.0, 1.0, 1.0, 1.0],
                [
                    re.ResourceRX(1.2, 0),
                    re.ResourceRZ(-3.4, 1),
                    re.ResourceCNOT([0, 1]),
                    re.ResourceHadamard(0),
                ],
            ),
            control=["c1", "c2"],
        ),
        re.ResourcePrepSelPrep(
            LinearCombination(
                (0.1, -2.3, 4.5, -6, 0.78),
                (
                    re.ResourceProd(re.ResourceZ(0), re.ResourceZ(1)),
                    re.ResourceProd(re.ResourceX(0), re.ResourceX(2)),
                    re.ResourceProd(re.ResourceY(2), re.ResourceY(1)),
                    re.ResourceAdjoint(
                        re.ResourceProd(re.ResourceX(0), re.ResourceY(1), re.ResourceZ(2))
                    ),
                    re.ResourceQFT([0, 1, 2]),
                ),
            ),
            control=["c1", "c2", "c3"],
        ),
    )

    resource_data = (
        {
            re.ResourceStatePrep.resource_rep(num_wires=1): 1,
            re.ResourceSelect.resource_rep(
                cmpr_ops=(
                    re.ResourceX.resource_rep(),
                    re.ResourceZ.resource_rep(),
                ),
            ): 1,
            re.ResourceAdjoint.resource_rep(re.ResourceStatePrep, base_params={"num_wires": 1}): 1,
        },
        {
            re.ResourceStatePrep.resource_rep(num_wires=2): 1,
            re.ResourceSelect.resource_rep(
                cmpr_ops=(
                    re.ResourceRX.resource_rep(),
                    re.ResourceRZ.resource_rep(),
                    re.ResourceCNOT.resource_rep(),
                    re.ResourceHadamard.resource_rep(),
                ),
            ): 1,
            re.ResourceAdjoint.resource_rep(re.ResourceStatePrep, base_params={"num_wires": 2}): 1,
        },
        {
            re.ResourceStatePrep.resource_rep(num_wires=3): 1,
            re.ResourceSelect.resource_rep(
                cmpr_ops=(
                    re.ResourceProd.resource_rep(
                        (re.ResourceZ.resource_rep(), re.ResourceZ.resource_rep())
                    ),
                    re.ResourceProd.resource_rep(
                        (re.ResourceX.resource_rep(), re.ResourceX.resource_rep())
                    ),
                    re.ResourceProd.resource_rep(
                        (re.ResourceY.resource_rep(), re.ResourceY.resource_rep())
                    ),
                    re.ResourceAdjoint.resource_rep(
                        base_class=re.ResourceProd,
                        base_params={
                            "cmpr_factors": (
                                re.ResourceX.resource_rep(),
                                re.ResourceY.resource_rep(),
                                re.ResourceZ.resource_rep(),
                            )
                        },
                    ),
                    re.ResourceQFT.resource_rep(num_wires=3),
                ),
            ): 1,
            re.ResourceAdjoint.resource_rep(re.ResourceStatePrep, base_params={"num_wires": 3}): 1,
        },
    )

    resource_params_data = (
        {
            "cmpr_ops": (
                re.ResourceX.resource_rep(),
                re.ResourceZ.resource_rep(),
            ),
        },
        {
            "cmpr_ops": (
                re.ResourceRX.resource_rep(),
                re.ResourceRZ.resource_rep(),
                re.ResourceCNOT.resource_rep(),
                re.ResourceHadamard.resource_rep(),
            ),
        },
        {
            "cmpr_ops": (
                re.ResourceProd.resource_rep(
                    (re.ResourceZ.resource_rep(), re.ResourceZ.resource_rep())
                ),
                re.ResourceProd.resource_rep(
                    (re.ResourceX.resource_rep(), re.ResourceX.resource_rep())
                ),
                re.ResourceProd.resource_rep(
                    (re.ResourceY.resource_rep(), re.ResourceY.resource_rep())
                ),
                re.ResourceAdjoint.resource_rep(
                    base_class=re.ResourceProd,
                    base_params={
                        "cmpr_factors": (
                            re.ResourceX.resource_rep(),
                            re.ResourceY.resource_rep(),
                            re.ResourceZ.resource_rep(),
                        )
                    },
                ),
                re.ResourceQFT.resource_rep(num_wires=3),
            ),
        },
    )

    @pytest.mark.parametrize(
        "op, params, expected_res", zip(op_data, resource_params_data, resource_data)
    )
    def test_resources(self, op, params, expected_res):
        """Test the resources method returns the correct dictionary"""
        res_from_op = op.resources(**op.resource_params)
        res_from_func = re.ResourcePrepSelPrep.resources(**params)

        assert res_from_op == expected_res
        assert res_from_func == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourcePrepSelPrep, expected_params)
        assert re.ResourcePrepSelPrep.resource_rep(**expected_params) == expected

    @pytest.mark.parametrize("z", [2, 3, 5])
    @pytest.mark.parametrize("params, base_res", zip(resource_params_data, resource_data))
    def test_pow_resources(self, params, base_res, z):
        """Test the pow_resources method returns the correct dictionary"""
        pow_select = re.ResourcePow.resource_rep(re.ResourceSelect, base_params=params, z=z)

        expected_res = copy.deepcopy(base_res)
        select_res = expected_res.pop(re.ResourceSelect.resource_rep(**params))

        expected_res[pow_select] = select_res
        assert re.ResourcePrepSelPrep.pow_resource_decomp(z=z, **params) == expected_res
