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
import math

import pytest

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
            re.ResourceMottonenStatePreparation.resource_rep(1): 1,
        },
        {
            re.ResourceMottonenStatePreparation.resource_rep(3): 1,
        },
        {
            re.ResourceMottonenStatePreparation.resource_rep(4): 1,
        },
        {
            re.ResourceMottonenStatePreparation.resource_rep(5): 1,
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
        res_from_op = op.resources(**op.resource_params)
        res_from_func = re.ResourceStatePrep.resources(**params)

        assert res_from_op == expected_res
        assert res_from_func == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceStatePrep, expected_params)
        assert re.ResourceStatePrep.resource_rep(**expected_params) == expected

    @pytest.mark.parametrize("params, expected_name", zip(resource_params_data, name_data))
    def test_tracking_name(self, params, expected_name):
        """Test that the tracking name is correct."""
        assert re.ResourceStatePrep.tracking_name(**params) == expected_name


class TestResourceBasisState:
    """Test the ResourceBasisState class"""

    @pytest.mark.parametrize(
        "num_bit_flips, num_x",
        [(4, 4), (5, 5), (6, 6)],
    )
    def test_resources(self, num_bit_flips, num_x):
        """Test that the resources are correct"""
        expected = {}
        x = re.CompressedResourceOp(re.ResourceX, {})
        expected[x] = num_x

        assert re.ResourceBasisState.resources(num_bit_flips) == expected

    @pytest.mark.parametrize(
        "state, wires",
        [
            (
                [1, 1],
                range(2),
            ),
        ],
    )
    def test_resource_params(self, state, wires):
        """Test that the resource params are correct"""
        op = re.ResourceBasisState(state, wires=wires)

        assert op.resource_params == {"num_bit_flips": 2}

    @pytest.mark.parametrize(
        "num_bit_flips",
        [(4, 4), (5, 5), (6, 6)],
    )
    def test_resource_rep(self, num_bit_flips):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceBasisState,
            {"num_bit_flips": num_bit_flips},
        )
        assert expected == re.ResourceBasisState.resource_rep(num_bit_flips)

    @pytest.mark.parametrize(
        "num_bit_flips, num_x",
        [(4, 4), (5, 5), (6, 6)],
    )
    def test_resources_from_rep(self, num_bit_flips, num_x):
        """Test that computing the resources from a compressed representation works"""
        rep = re.ResourceBasisState.resource_rep(num_bit_flips)
        actual = rep.op_type.resources(**rep.params)
        expected = {}
        x = re.CompressedResourceOp(re.ResourceX, {})
        expected[x] = num_x

        assert actual == expected

    @pytest.mark.parametrize(
        "num_bit_flips",
        [(4), (5), (6)],
    )
    def test_tracking_name(self, num_bit_flips):
        """Test that the tracking name is correct."""
        assert re.ResourceBasisState.tracking_name(num_bit_flips) == f"BasisState({num_bit_flips})"


class TestResourceSuperposition:
    """Test the ResourceSuperposition class"""

    @pytest.mark.parametrize(
        "num_stateprep_wires, num_basis_states, size_basis_state",
        [(4, 2, 2), (4, 5, 2), (4, 5, 0)],
    )
    def test_resources(self, num_stateprep_wires, num_basis_states, size_basis_state):
        """Test that the resources are correct"""
        expected = {}
        msp = re.CompressedResourceOp(
            re.ResourceMottonenStatePreparation, {"num_wires": num_stateprep_wires}
        )
        expected[msp] = 1

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        num_zero_ctrls = size_basis_state // 2
        multi_x = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": size_basis_state,
                "num_ctrl_values": num_zero_ctrls,
                "num_work_wires": 0,
            },
        )

        basis_size = 2**size_basis_state
        prob_matching_basis_states = num_basis_states / basis_size
        num_permutes = round(num_basis_states * (1 - prob_matching_basis_states))

        if num_permutes:
            expected[cnot] = num_permutes * (
                size_basis_state // 2
            )  # average number of bits to flip
            expected[multi_x] = 2 * num_permutes  # for compute and uncompute

        assert (
            re.ResourceSuperposition.resources(
                num_stateprep_wires, num_basis_states, size_basis_state
            )
            == expected
        )

    @pytest.mark.parametrize(
        "coeffs, bases, wires, work_wire",
        [
            (
                qnp.sqrt(qnp.array([1 / 3, 1 / 3, 1 / 3])),
                qnp.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]]),
                [0, 1, 2],
                [3],
            ),
        ],
    )
    def test_resource_params(self, coeffs, bases, wires, work_wire):
        """Test that the resource params are correct"""
        op = re.ResourceSuperposition(coeffs, bases, wires, work_wire)

        num_basis_states = len(bases)
        size_basis_state = len(bases[0])  # assuming they are all the same size
        num_stateprep_wires = math.ceil(math.log2(len(coeffs)))

        assert op.resource_params == {
            "num_stateprep_wires": num_stateprep_wires,
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }

    @pytest.mark.parametrize(
        "num_stateprep_wires, num_basis_states, size_basis_state",
        [(4, 2, 2), (4, 5, 2), (4, 5, 0)],
    )
    def test_resource_rep(self, num_stateprep_wires, num_basis_states, size_basis_state):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceSuperposition,
            {
                "num_stateprep_wires": num_stateprep_wires,
                "num_basis_states": num_basis_states,
                "size_basis_state": size_basis_state,
            },
        )
        assert expected == re.ResourceSuperposition.resource_rep(
            num_stateprep_wires, num_basis_states, size_basis_state
        )

    @pytest.mark.parametrize(
        "num_stateprep_wires, num_basis_states, size_basis_state",
        [(4, 2, 2), (4, 5, 2), (4, 5, 0)],
    )
    def test_resources_from_rep(self, num_stateprep_wires, num_basis_states, size_basis_state):
        """Test that computing the resources from a compressed representation works"""
        expected = {}
        rep = re.ResourceSuperposition.resource_rep(
            num_stateprep_wires, num_basis_states, size_basis_state
        )
        actual = rep.op_type.resources(**rep.params)

        expected = {}
        msp = re.CompressedResourceOp(
            re.ResourceMottonenStatePreparation, {"num_wires": num_stateprep_wires}
        )
        expected[msp] = 1

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        num_zero_ctrls = size_basis_state // 2
        multi_x = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": size_basis_state,
                "num_ctrl_values": num_zero_ctrls,
                "num_work_wires": 0,
            },
        )

        basis_size = 2**size_basis_state
        prob_matching_basis_states = num_basis_states / basis_size
        num_permutes = round(num_basis_states * (1 - prob_matching_basis_states))

        if num_permutes:
            expected[cnot] = num_permutes * (
                size_basis_state // 2
            )  # average number of bits to flip
            expected[multi_x] = 2 * num_permutes  # for compute and uncompute

        assert actual == expected

    def test_tracking_name(self):
        """Test that the tracking name is correct."""
        assert re.ResourceSuperposition.tracking_name() == "Superposition"


class TestResourceMottonenStatePreparation:
    """Test the ResourceMottonenStatePreparation class"""

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources(self, num_wires):
        """Test that the resources are correct"""
        expected = {}
        rz = re.CompressedResourceOp(re.ResourceRZ, {})
        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        if r_count:
            expected[rz] = r_count

        if cnot_count:
            expected[cnot] = cnot_count
        assert re.ResourceMottonenStatePreparation.resources(num_wires) == expected

    @pytest.mark.parametrize(
        "state_vector, wires",
        [
            (
                qnp.array(
                    [
                        0.070014 + 0.0j,
                        0.0 + 0.14002801j,
                        0.21004201 + 0.0j,
                        0.0 + 0.28005602j,
                        0.35007002 + 0.0j,
                        0.0 + 0.42008403j,
                        0.49009803 + 0.0j,
                        0.0 + 0.56011203j,
                    ]
                ),
                range(3),
            ),
        ],
    )
    def test_resource_params(self, state_vector, wires):
        """Test that the resource params are correct"""
        op = re.ResourceMottonenStatePreparation(state_vector=state_vector, wires=wires)

        assert op.resource_params == {"num_wires": len(wires)}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceMottonenStatePreparation,
            {"num_wires": num_wires},
        )
        assert expected == re.ResourceMottonenStatePreparation.resource_rep(num_wires)

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources_from_rep(self, num_wires):
        """Test that computing the resources from a compressed representation works"""
        rep = re.ResourceMottonenStatePreparation.resource_rep(num_wires)
        actual = rep.op_type.resources(**rep.params)

        expected = {}
        rz = re.CompressedResourceOp(re.ResourceRZ, {})
        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        if r_count:
            expected[rz] = r_count

        if cnot_count:
            expected[cnot] = cnot_count

        assert actual == expected

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_tracking_name(self, num_wires):
        """Test that the tracking name is correct."""
        assert (
            re.ResourceMottonenStatePreparation.tracking_name(num_wires)
            == f"MottonenStatePrep({num_wires})"
        )
