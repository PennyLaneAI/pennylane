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
Test the Resource classes for Trotterization
"""
from collections import defaultdict

import pytest

import pennylane.estimator as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.wires import Wires

# pylint: disable=no-self-use, too-many-arguments, too-many-positional-arguments


class TestResourceTrotterProduct:
    """Test the ResourceTrotterProduct class"""

    # Expected resources were obtained manually using the recursion expression
    op_data = [  # ops, num_steps, order, num_wires, expected_res
        (
            [qre.X(wires=0), qre.Y(wires=1), qre.Z(wires=0)],
            2,
            1,
            2,
            [
                GateCount(resource_rep(qre.X), 2),
                GateCount(resource_rep(qre.Y), 2),
                GateCount(resource_rep(qre.Z), 2),
            ],
        ),
        (
            [qre.X(), qre.Y()],
            10,
            2,
            1,
            [
                GateCount(resource_rep(qre.X), 11),
                GateCount(resource_rep(qre.Y), 10),
            ],
        ),
        (
            [qre.RX(precision=1e-3), qre.RY(precision=1e-3), qre.Z()],
            10,
            4,
            1,
            [
                GateCount(resource_rep(qre.RX, {"precision": 1e-3}), 51),
                GateCount(resource_rep(qre.Z), 50),
                GateCount(resource_rep(qre.RY, {"precision": 1e-3}), 100),
            ],
        ),
    ]

    @pytest.mark.parametrize("ops, num_steps, order, num_wires, _", op_data)
    def test_resource_params(self, ops, num_steps, order, num_wires, _):
        """Test that the resource params are correct"""
        trotter = qre.TrotterProduct(ops, num_steps=num_steps, order=order)
        assert trotter.resource_params == {
            "first_order_expansion": tuple(op.resource_rep_from_op() for op in ops),
            "num_steps": num_steps,
            "order": order,
            "num_wires": num_wires,
        }

    @pytest.mark.parametrize("ops, num_steps, order, num_wires, _", op_data)
    def test_resource_rep(self, ops, num_steps, order, num_wires, _):
        """Test that the resource params are correct"""
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        expected = qre.CompressedResourceOp(
            qre.TrotterProduct,
            num_wires,
            {
                "first_order_expansion": cmpr_ops,
                "num_steps": num_steps,
                "order": order,
                "num_wires": num_wires,
            },
        )
        assert qre.TrotterProduct.resource_rep(cmpr_ops, num_steps, order, num_wires) == expected

    @pytest.mark.parametrize("ops, num_steps, order, num_wires, expected_res", op_data)
    def test_resources(self, ops, num_steps, order, num_wires, expected_res):
        """Test the resources method returns the correct dictionary"""
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        computed_res = qre.TrotterProduct.resource_decomp(cmpr_ops, num_steps, order, num_wires)
        assert computed_res == expected_res

    def test_attribute_error(self):
        """Test that a AttributeError is raised for unsupported operators."""
        with pytest.raises(
            ValueError,
            match="All components of first_order_expansion must be instances of `ResourceOperator` in order to obtain resources.",
        ):
            qre.TrotterProduct(
                [qre.X(), qre.Y(), qre.Allocate(4)],
                num_steps=10,
                order=3,
            )

    def test_user_defined_wires(self):
        """Test that user-defined wires take precedence over operator wires."""
        ops = [qre.X(wires=0), qre.Y(wires=1)]
        user_wires = ["a", "b", "c"]
        trotter = qre.TrotterProduct(ops, num_steps=1, order=1, wires=user_wires)

        assert trotter.wires == Wires(user_wires)
        assert trotter.num_wires == len(user_wires)


class TestTrotterCDF:
    """Tests for Resource TrotterCDF class"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        compact_ham = qre.CDFHamiltonian(num_orbitals=8, num_fragments=4)
        with pytest.raises(ValueError, match="Expected 16 wires, got 3"):
            qre.TrotterCDF(compact_ham, num_steps=100, order=2, wires=[0, 1, 2])

    # Expected resources were obtained manually based on
    # https://arxiv.org/abs/2506.15784
    hamiltonian_data = [
        (
            8,
            4,
            100,
            2,
            {
                "zeroed": 0,
                "any_state": 0,
                "algo_wires": 16,
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 7711424,
                        "S": 201936,
                        "Z": 134624,
                        "Hadamard": 134624,
                        "CNOT": 187312,
                    },
                ),
            },
        ),
        (
            10,
            5,
            1000,
            1,
            {
                "zeroed": 0,
                "any_state": 0,
                "algo_wires": 20,
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 99920000,
                        "S": 2700000,
                        "Z": 1800000,
                        "Hadamard": 1800000,
                        "CNOT": 2420000,
                    },
                ),
            },
        ),
        (
            12,
            8,
            750,
            4,
            {
                "zeroed": 0,
                "any_state": 0,
                "algo_wires": 24,
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 1593920064,
                        "S": 41580792,
                        "Z": 27720528,
                        "Hadamard": 27720528,
                        "CNOT": 40770264,
                    },
                ),
            },
        ),
    ]

    @pytest.mark.parametrize(
        "num_orbitals, num_fragments, num_steps, order, expected_res", hamiltonian_data
    )
    def test_resource_trotter_cdf(
        self, num_orbitals, num_fragments, num_steps, order, expected_res
    ):
        """Test the Resource TrotterCDF class for correct resources"""
        compact_ham = qre.CDFHamiltonian(num_orbitals=num_orbitals, num_fragments=num_fragments)

        def circ():
            qre.TrotterCDF(compact_ham, num_steps=num_steps, order=order)

        res = qre.estimate(circ)()
        assert res.zeroed_wires == expected_res["zeroed"]
        assert res.any_state_wires == expected_res["any_state"]
        assert res.algo_wires == expected_res["algo_wires"]
        assert res.gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        r"""Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = qre.THCHamiltonian(num_orbitals=4, tensor_rank=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for TrotterCDF"
        ):
            qre.TrotterCDF(compact_ham, num_steps=100, order=2)

    @pytest.mark.parametrize(
        "num_orbitals, num_fragments, num_steps, order, num_ctrl_wires, num_zero_ctrl, gates_expected",
        (
            (
                4,
                4,
                1,
                1,
                1,
                0,
                [
                    (8, "BasisRotation(4)"),
                    (1, "Prod"),
                    (3, "Prod"),
                ],
            ),
            (
                4,
                4,
                1,
                2,
                1,
                0,
                [
                    (6, "BasisRotation(4)"),
                    (2, "Prod"),
                    (1, "Prod"),
                    (8, "BasisRotation(4)"),
                    (4, "Prod"),
                ],
            ),
        ),
    )
    def test_controlled_resource_decomp(
        self,
        num_orbitals,
        num_fragments,
        num_steps,
        order,
        num_ctrl_wires,
        num_zero_ctrl,
        gates_expected,
    ):
        """Test the controlled_resource_decomp method for TrotterCDF."""

        compact_ham = qre.CDFHamiltonian(num_orbitals=num_orbitals, num_fragments=num_fragments)
        target_resource_params = {
            "cdf_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
        }
        decomp = qre.TrotterCDF.controlled_resource_decomp(
            num_ctrl_wires, num_zero_ctrl, target_resource_params
        )

        gates_decomp = [(item.count, item.gate.name) for item in decomp]

        assert gates_decomp == gates_expected


class TestTrotterTHC:
    """Tests for Resource TrotterTHC class"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        compact_ham = qre.THCHamiltonian(num_orbitals=8, tensor_rank=20)
        with pytest.raises(ValueError, match="Expected 40 wires, got 3"):
            qre.TrotterTHC(compact_ham, num_steps=100, order=2, wires=[0, 1, 2])

    # Expected resources were obtained manually
    # based on https://arxiv.org/abs/2407.04432

    hamiltonian_data = [
        (
            8,
            20,
            100,
            2,
            {
                "zeroed": 0,
                "any_state": 0,
                "algo_wires": 40,
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 9687424,
                        "S": 261936,
                        "Z": 174624,
                        "Hadamard": 174624,
                        "CNOT": 243312,
                    },
                ),
            },
        ),
        (
            10,
            40,
            1000,
            1,
            {
                "zeroed": 0,
                "any_state": 0,
                "algo_wires": 80,
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 368720000,
                        "S": 9900000,
                        "Z": 6600000,
                        "Hadamard": 6600000,
                        "CNOT": 9620000,
                    },
                ),
            },
        ),
    ]

    @pytest.mark.parametrize(
        "num_orbitals, tensor_rank, num_steps, order, expected_res", hamiltonian_data
    )
    def test_resource_trotter_thc(self, num_orbitals, tensor_rank, num_steps, order, expected_res):
        """Test the Resource TrotterTHC class for correct resources"""
        compact_ham = qre.THCHamiltonian(num_orbitals=num_orbitals, tensor_rank=tensor_rank)

        def circ():
            qre.TrotterTHC(compact_ham, num_steps=num_steps, order=order)

        res = qre.estimate(circ)()
        assert res.zeroed_wires == expected_res["zeroed"]
        assert res.any_state_wires == expected_res["any_state"]
        assert res.algo_wires == expected_res["algo_wires"]
        assert res.gate_counts == expected_res["gate_types"]

    @pytest.mark.parametrize(
        "num_orbitals, tensor_rank, num_steps, order, num_ctrl_wires, num_zero_ctrl, gates_expected",
        (
            (
                4,
                4,
                1,
                1,
                1,
                0,
                [
                    (2, "BasisRotation(4)"),
                    (2, "BasisRotation(4)"),
                    (1, "Prod"),
                    (1, "Prod"),
                ],
            ),
            (
                4,
                4,
                1,
                2,
                1,
                0,
                [
                    (4, "BasisRotation(4)"),
                    (2, "Prod"),
                    (2, "BasisRotation(4)"),
                    (1, "Prod"),
                ],
            ),
        ),
    )
    def test_controlled_resource_decomp(
        self,
        num_orbitals,
        tensor_rank,
        num_steps,
        order,
        num_ctrl_wires,
        num_zero_ctrl,
        gates_expected,
    ):
        """Test the controlled_resource_decomp method for TrotterTHC."""

        compact_ham = qre.THCHamiltonian(num_orbitals=num_orbitals, tensor_rank=tensor_rank)

        target_resource_params = {
            "thc_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
        }
        decomp = qre.TrotterTHC.controlled_resource_decomp(
            num_ctrl_wires, num_zero_ctrl, target_resource_params
        )

        gates_decomp = [(item.count, item.gate.name) for item in decomp]

        assert gates_decomp == gates_expected

    def test_type_error(self):
        """Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = qre.CDFHamiltonian(num_orbitals=4, num_fragments=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for TrotterTHC"
        ):
            qre.TrotterTHC(compact_ham, num_steps=100, order=2)


class TestTrotterVibrational:
    """Test the Resource TrotterVibrational class"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        compact_ham = qre.VibrationalHamiltonian(num_modes=8, grid_size=4, taylor_degree=3)
        with pytest.raises(ValueError, match="Expected 32 wires, got 3"):
            qre.TrotterVibrational(compact_ham, num_steps=100, order=2, wires=[0, 1, 2])

    # Expected resources were obtained manually based on
    # https://arxiv.org/pdf/2504.10602
    hamiltonian_data = [
        (
            8,
            4,
            3,
            100,
            2,
            {
                "zeroed": 95,
                "any_state": 0,
                "algo_wires": 32,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 2,
                        "S": 2,
                        "T": 7898,
                        "X": 384064,
                        "Toffoli": 10069400,
                        "CNOT": 14300800,
                        "Hadamard": 30040840,
                    },
                ),
            },
        ),
        (
            4,
            6,
            4,
            10,
            1,
            {
                "zeroed": 127,
                "any_state": 0,
                "algo_wires": 24,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 3,
                        "S": 3,
                        "T": 2327,
                        "X": 7548,
                        "Toffoli": 398770,
                        "CNOT": 489540,
                        "Hadamard": 1159710,
                    },
                ),
            },
        ),
        (
            4,
            2,
            2,
            20,
            1,
            {
                "zeroed": 67,
                "any_state": 0,
                "algo_wires": 8,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 1,
                        "S": 1,
                        "T": 909,
                        "X": 4016,
                        "Toffoli": 37320,
                        "CNOT": 82000,
                        "Hadamard": 111140,
                    },
                ),
            },
        ),
        (
            4,
            2,
            5,
            10,
            1,
            {
                "zeroed": 127,
                "any_state": 0,
                "algo_wires": 8,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 4,
                        "S": 4,
                        "T": 3076,
                        "X": 13116,
                        "Toffoli": 242680,
                        "CNOT": 387840,
                        "Hadamard": 725510,
                    },
                ),
            },
        ),
    ]

    @pytest.mark.parametrize(
        "num_modes, grid_size, taylor_degree, num_steps, order, expected_res", hamiltonian_data
    )
    def test_resource_trotter_vibrational(
        self, num_modes, grid_size, taylor_degree, num_steps, order, expected_res
    ):
        """Test the Resource TrotterVibrational class for correct resources"""
        compact_ham = qre.VibrationalHamiltonian(
            num_modes=num_modes, grid_size=grid_size, taylor_degree=taylor_degree
        )

        def circ():
            qre.TrotterVibrational(compact_ham, num_steps=num_steps, order=order)

        res = qre.estimate(circ)()

        assert res.zeroed_wires == expected_res["zeroed"]
        assert res.any_state_wires == expected_res["any_state"]
        assert res.algo_wires == expected_res["algo_wires"]
        assert res.gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        """Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = qre.CDFHamiltonian(num_orbitals=4, num_fragments=10)
        with pytest.raises(
            TypeError,
            match="Unsupported Hamiltonian representation for TrotterVibrational",
        ):
            qre.TrotterVibrational(compact_ham, num_steps=100, order=2)


class TestTrotterVibronic:
    """Test the Resource TrotterVibronic class"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        compact_ham = qre.VibronicHamiltonian(
            num_modes=8,
            num_states=2,
            grid_size=4,
            taylor_degree=3,
        )
        with pytest.raises(ValueError, match="Expected 33 wires, got 3"):
            qre.TrotterVibronic(compact_ham, num_steps=100, order=2, wires=[0, 1, 2])

    # Expected resources were obtained manually based on
    # https://arxiv.org/abs/2411.13669
    hamiltonian_data = [
        (
            8,
            2,
            4,
            3,
            100,
            2,
            {
                "zeroed": 95,
                "any_state": 0,
                "algo_wires": 33,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 2,
                        "S": 2,
                        "T": 7898,
                        "X": 153664,
                        "Hadamard": 30271242,
                        "Toffoli": 10146200,
                        "CNOT": 15222400,
                    },
                ),
            },
        ),
        (
            4,
            3,
            6,
            4,
            10,
            1,
            {
                "zeroed": 127,
                "any_state": 0,
                "algo_wires": 26,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 3,
                        "S": 3,
                        "T": 2327,
                        "X": 6048,
                        "Hadamard": 1168714,
                        "Toffoli": 401770,
                        "CNOT": 518040,
                    },
                ),
            },
        ),
        (
            4,
            1,
            2,
            2,
            20,
            1,
            {
                "zeroed": 67,
                "any_state": 0,
                "algo_wires": 8,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 1,
                        "S": 1,
                        "T": 909,
                        "X": 16,
                        "Hadamard": 111140,
                        "Toffoli": 37320,
                        "CNOT": 86000,
                    },
                ),
            },
        ),
        (
            4,
            1,
            2,
            5,
            10,
            1,
            {
                "zeroed": 127,
                "any_state": 0,
                "algo_wires": 8,
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 4,
                        "S": 4,
                        "T": 3076,
                        "X": 16,
                        "Hadamard": 725510,
                        "Toffoli": 242680,
                        "CNOT": 400940,
                    },
                ),
            },
        ),
    ]

    @pytest.mark.parametrize(
        "num_modes, num_states, grid_size, taylor_degree, num_steps, order, expected_res",
        hamiltonian_data,
    )
    def test_resource_trotter_vibronic(
        self, num_modes, num_states, grid_size, taylor_degree, num_steps, order, expected_res
    ):
        """Test the Resource TrotterVibronic class for correct resources"""
        compact_ham = qre.VibronicHamiltonian(
            num_modes=num_modes,
            num_states=num_states,
            grid_size=grid_size,
            taylor_degree=taylor_degree,
        )

        def circ():
            qre.TrotterVibronic(compact_ham, num_steps=num_steps, order=order)

        res = qre.estimate(circ)()

        assert res.zeroed_wires == expected_res["zeroed"]
        assert res.any_state_wires == expected_res["any_state"]
        assert res.algo_wires == expected_res["algo_wires"]
        assert res.gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        """Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = qre.CDFHamiltonian(num_orbitals=4, num_fragments=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for TrotterVibronic"
        ):
            qre.TrotterVibronic(compact_ham, num_steps=100, order=2)
