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

    @pytest.mark.parametrize(
        "compact_state, wires, expected_params",
        (
            (
                re.CompactState.from_state_vector(num_qubits=3, num_coeffs=8),
                range(3),
                {"num_wires": 3},
            ),
            (
                re.CompactState.from_state_vector(num_qubits=100, num_coeffs=2**100),
                range(100),
                {"num_wires": 100},
            ),
            (
                re.CompactState.from_state_vector(num_qubits=3, num_coeffs=8, num_work_wires=5),
                range(3),
                {"num_wires": 3},
            ),
            (
                re.CompactState.from_bitstring(num_qubits=7, num_bit_flips=10),
                range(7),
                {"num_wires": 7},
            ),
        ),
    )
    def test_resource_params_with_compact_state(self, compact_state, wires, expected_params):
        """Test we can extract parameters from a compact state."""
        op = re.ResourceStatePrep(compact_state, wires)
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
        "compact_state, wires, expected_params",
        (
            (
                re.CompactState.from_bitstring(num_qubits=6, num_bit_flips=16),
                range(6),
                {"num_bit_flips": 16},
            ),
            (
                re.CompactState.from_bitstring(num_qubits=7, num_bit_flips=10),
                range(7),
                {"num_bit_flips": 10},
            ),
            (
                re.CompactState.from_bitstring(num_qubits=5, num_bit_flips=30),
                range(5),
                {"num_bit_flips": 30},
            ),
        ),
    )
    def test_resource_params_with_compact_state(self, compact_state, wires, expected_params):
        """Test we can extract parameters from a compact state."""
        op = re.ResourceBasisState(compact_state, wires)
        assert op.resource_params == expected_params

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
        "compact_state, wires, work_wire, expected_params",
        (
            (
                re.CompactState.from_state_vector(num_qubits=6, num_coeffs=16, num_work_wires=1),
                range(6),
                ["w1"],
                {
                    "num_stateprep_wires": 4,
                    "num_basis_states": 16,
                    "size_basis_state": 6,
                },
            ),
            (
                re.CompactState.from_state_vector(num_qubits=100, num_coeffs=100, num_work_wires=1),
                range(100),
                ["w1"],
                {
                    "num_stateprep_wires": 7,
                    "num_basis_states": 100,
                    "size_basis_state": 100,
                },
            ),
            (
                re.CompactState.from_state_vector(num_qubits=3, num_coeffs=8, num_work_wires=1),
                range(3),
                ["w1"],
                {
                    "num_stateprep_wires": 3,
                    "num_basis_states": 8,
                    "size_basis_state": 3,
                },
            ),
        ),
    )
    def test_resource_params_with_compact_state(
        self, compact_state, wires, work_wire, expected_params
    ):
        """Test we can extract parameters from a compact state."""
        op = re.ResourceSuperposition(state_vect=compact_state, wires=wires, work_wire=work_wire)
        assert op.resource_params == expected_params

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
        "compact_state, wires, expected_params",
        (
            (
                re.CompactState.from_state_vector(num_qubits=3, num_coeffs=8),
                range(3),
                {"num_wires": 3},
            ),
            (
                re.CompactState.from_state_vector(num_qubits=100, num_coeffs=2**100),
                range(100),
                {"num_wires": 100},
            ),
            (
                re.CompactState.from_state_vector(num_qubits=3, num_coeffs=8, num_work_wires=5),
                range(3),
                {"num_wires": 3},
            ),
            (
                re.CompactState.from_bitstring(num_qubits=7, num_bit_flips=10),
                range(7),
                {"num_wires": 7},
            ),
        ),
    )
    def test_resource_params_with_compact_state(self, compact_state, wires, expected_params):
        """Test we can extract parameters from a compact state."""
        op = re.ResourceMottonenStatePreparation(compact_state, wires)
        assert op.resource_params == expected_params

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


class TestResourceMPSPrep:
    """Test the ResourceMPSPrep class"""

    @pytest.mark.parametrize(
        "num_wires, num_work_wires, expected_res",
        (
            (
                4,
                2,
                {
                    re.ResourceQubitUnitary.resource_rep(num_wires=2): 2,
                    re.ResourceQubitUnitary.resource_rep(num_wires=3): 2,
                },
            ),
            (
                5,
                2,
                {
                    re.ResourceQubitUnitary.resource_rep(num_wires=2): 2,
                    re.ResourceQubitUnitary.resource_rep(num_wires=3): 3,
                },
            ),
            (
                6,
                4,
                {
                    re.ResourceQubitUnitary.resource_rep(num_wires=2): 2,
                    re.ResourceQubitUnitary.resource_rep(num_wires=3): 2,
                    re.ResourceQubitUnitary.resource_rep(num_wires=4): 2,
                },
            ),
        ),
    )
    def test_resources(self, num_wires, num_work_wires, expected_res):
        """Test that the resources are correct"""
        assert re.ResourceMPSPrep.resources(num_wires, num_work_wires) == expected_res

    @pytest.mark.parametrize(
        ("mps", "num_wires", "num_work_wires"),
        [
            (
                [
                    qnp.array([[0.70710678, 0.0], [0.0, 0.70710678]]),
                    qnp.array(
                        [
                            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, -0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
                        ]
                    ),
                    qnp.array(
                        [
                            [[0.00000000e00, 1.74315280e-32], [-7.07106781e-01, -7.07106781e-01]],
                            [[7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                            [[0.00000000e00, 0.00000000e00], [-7.07106781e-01, 7.07106781e-01]],
                            [[-7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                        ]
                    ),
                    qnp.array([[1.0, 0.0], [0.0, 1.0]]),
                ],
                4,
                2,
            ),
            (
                [
                    qnp.array([[0.53849604, -0.44389787], [-0.59116842, -0.40434711]]),
                    qnp.array(
                        [
                            [
                                [-6.05052107e-01, 1.34284016e-01, -2.84018989e-01, -1.12416345e-01],
                                [3.60988555e-01, 6.14571922e-01, -1.20681653e-01, -2.89527967e-04],
                            ],
                            [
                                [-1.12393068e-01, 2.11496619e-01, 3.99193070e-01, -3.44891522e-01],
                                [-5.99232567e-01, 3.76491467e-01, 3.04813277e-01, 2.65697349e-01],
                            ],
                        ]
                    ),
                    qnp.array(
                        [
                            [
                                [0.87613189, -0.34254341, -0.12704983, -0.0161698],
                                [-0.20758717, 0.1329479, -0.18184107, 0.06942658],
                            ],
                            [
                                [-0.16499137, -0.13680142, -0.18432824, 0.12950892],
                                [-0.68790868, -0.64141472, -0.12485688, -0.0556177],
                            ],
                            [
                                [0.0352582, -0.37993402, 0.26781956, -0.25935129],
                                [0.04351872, -0.27109361, 0.65111429, 0.4648453],
                            ],
                            [
                                [0.1909576, 0.25461839, -0.07463641, -0.34390477],
                                [-0.21279487, 0.0305474, 0.53420894, -0.66578494],
                            ],
                        ]
                    ),
                    qnp.array(
                        [
                            [[-0.26771292, -0.00628612], [-0.96316273, 0.02465422]],
                            [[0.96011241, 0.07601506], [-0.2663889, 0.03798452]],
                            [[-0.00727353, 0.4537835], [-0.02374101, -0.89076596]],
                            [[0.08038064, -0.88784161], [-0.02812246, -0.45220057]],
                        ]
                    ),
                    qnp.array([[-0.97855153, 0.2060022], [0.2060022, 0.97855153]]),
                ],
                5,
                2,
            ),
            (
                [
                    qnp.array([[0.0, 0.107], [0.994, 0.0]]),
                    qnp.array(
                        [
                            [[0.0, 0.0], [1.0, 0.0]],
                            [[0.0, 1.0], [0.0, 0.0]],
                        ]
                    ),
                    qnp.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                3,
                1,
            ),
        ],
    )
    def test_resource_params(self, mps, num_wires, num_work_wires):
        """Test that the resource params are correct"""
        op = re.ResourceMPSPrep(
            mps,
            wires=range(num_wires),
            work_wires=list(f"w_{j}" for j in range(num_work_wires)),
        )

        assert op.resource_params == {"num_wires": num_wires, "num_work_wires": num_work_wires}

    @pytest.mark.parametrize(
        "compact_state, wires, work_wires, expected_params",
        (
            (
                re.CompactState.from_mps(num_mps_matrices=10, max_bond_dim=4),
                range(10),
                [f"w{j}" for j in range(2)],
                {"num_wires": 10, "num_work_wires": 2},
            ),
            (
                re.CompactState.from_mps(num_mps_matrices=3, max_bond_dim=7),
                range(3),
                [f"w{j}" for j in range(3)],
                {"num_wires": 3, "num_work_wires": 3},
            ),
            (
                re.CompactState.from_mps(num_mps_matrices=12, max_bond_dim=6),
                range(12),
                [f"w{j}" for j in range(3)],
                {"num_wires": 12, "num_work_wires": 3},
            ),
        ),
    )
    def test_resource_params_with_compact_state(
        self, compact_state, wires, work_wires, expected_params
    ):
        """Test we can extract parameters from a compact state."""
        op = re.ResourceMPSPrep(compact_state, wires, work_wires=work_wires)
        assert op.resource_params == expected_params

    @pytest.mark.parametrize(
        "num_wires, num_work_wires",
        [(4, 2), (5, 2), (6, 4)],
    )
    def test_resource_rep(self, num_wires, num_work_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceMPSPrep,
            {"num_wires": num_wires, "num_work_wires": num_work_wires},
        )
        assert expected == re.ResourceMPSPrep.resource_rep(num_wires, num_work_wires)

    def test_resources_from_instance(self):
        """Test that computing the resources from the instance works"""
        mps = [
            qnp.array([[0.0, 0.107], [0.994, 0.0]]),
            qnp.array(
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ]
            ),
            qnp.array([[-1.0, -0.0], [-0.0, -1.0]]),
        ]

        op = re.ResourceMPSPrep(mps, wires=[1, 2, 3], work_wires=[0])
        res_rep = op.resource_rep_from_op()
        measured_res = res_rep.op_type.resources(**res_rep.params)

        qu = re.ResourceQubitUnitary.resource_rep(num_wires=2)
        expected_res = {qu: 3}

        assert measured_res == expected_res
