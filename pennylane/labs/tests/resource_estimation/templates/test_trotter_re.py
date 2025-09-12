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

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation import GateCount, QubitManager, resource_rep

# pylint: disable=no-self-use, too-many-arguments, too-many-positional-arguments


class TestResourceTrotterProduct:
    """Test the ResourceTrotterProduct class"""

    # Expected resources were obtained manually using the recursion expression
    op_data = [  # ops, num_wires, num_steps, order, expected_res
        (
            [plre.ResourceX(wires=0), plre.ResourceY(wires=1), plre.ResourceZ(wires=0)],
            2,
            2,
            1,
            [
                GateCount(resource_rep(plre.ResourceX), 2),
                GateCount(resource_rep(plre.ResourceY), 2),
                GateCount(resource_rep(plre.ResourceZ), 2),
            ],
        ),
        (
            [plre.ResourceX(), plre.ResourceY()],
            1,
            10,
            2,
            [
                GateCount(resource_rep(plre.ResourceX), 11),
                GateCount(resource_rep(plre.ResourceY), 10),
            ],
        ),
        (
            [plre.ResourceRX(precision=1e-3), plre.ResourceRY(precision=1e-3), plre.ResourceZ()],
            1,
            10,
            4,
            [
                GateCount(resource_rep(plre.ResourceRX, {"precision": 1e-3}), 51),
                GateCount(resource_rep(plre.ResourceZ), 50),
                GateCount(resource_rep(plre.ResourceRY, {"precision": 1e-3}), 100),
            ],
        ),
    ]

    @pytest.mark.parametrize("ops, num_wires, num_steps, order, _", op_data)
    def test_resource_params(self, ops, num_wires, num_steps, order, _):
        """Test that the resource params are correct"""
        trotter = plre.ResourceTrotterProduct(ops, num_steps=num_steps, order=order)
        assert trotter.resource_params == {
            "first_order_expansion": tuple(op.resource_rep_from_op() for op in ops),
            "num_steps": num_steps,
            "order": order,
            "num_wires": num_wires,
        }

    @pytest.mark.parametrize("ops, num_wires, num_steps, order, _", op_data)
    def test_resource_rep(self, ops, num_wires, num_steps, order, _):
        """Test that the resource params are correct"""
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        expected = plre.CompressedResourceOp(
            plre.ResourceTrotterProduct,
            num_wires,
            {
                "first_order_expansion": cmpr_ops,
                "num_steps": num_steps,
                "order": order,
                "num_wires": num_wires,
            },
        )
        assert (
            plre.ResourceTrotterProduct.resource_rep(cmpr_ops, num_steps, order, num_wires)
            == expected
        )

    @pytest.mark.parametrize("ops, num_wires, num_steps, order, expected_res", op_data)
    def test_resources(self, ops, num_wires, num_steps, order, expected_res):
        """Test the resources method returns the correct dictionary"""
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        computed_res = plre.ResourceTrotterProduct.resource_decomp(
            cmpr_ops, num_steps, order, num_wires
        )
        assert computed_res == expected_res

    def test_attribute_error(self):
        """Test that a AttributeError is raised for unsupported operators."""
        with pytest.raises(
            ValueError,
            match="All components of first_order_expansion must be instances of `ResourceOperator` in order to obtain resources.",
        ):
            plre.ResourceTrotterProduct(
                [plre.ResourceX(), plre.ResourceY(), plre.AllocWires(4)],
                num_steps=10,
                order=3,
            )


class TestTrotterCDF:
    """Tests for ResourceTrotterCDF class"""

    # Expected resources were obtained manually based on
    # https://arxiv.org/abs/2506.15784
    hamiltonian_data = [
        (
            8,
            4,
            100,
            2,
            {
                "qubit_manager": QubitManager(
                    work_wires={"clean": 0, "dirty": 0}, algo_wires=16, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 7711424.0,
                        "S": 201936.0,
                        "Z": 134624.0,
                        "Hadamard": 134624.0,
                        "CNOT": 187312.0,
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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 0, "dirty": 0}, algo_wires=20, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 99920000.0,
                        "S": 2700000.0,
                        "Z": 1800000.0,
                        "Hadamard": 1800000.0,
                        "CNOT": 2420000.0,
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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 0, "dirty": 0}, algo_wires=24, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 1593920064.0,
                        "S": 41580792.0,
                        "Z": 27720528.0,
                        "Hadamard": 27720528.0,
                        "CNOT": 40770264.0,
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
        """Test the ResourceTrotterCDF class for correct resources"""

        compact_ham = plre.CompactHamiltonian.cdf(
            num_orbitals=num_orbitals, num_fragments=num_fragments
        )

        def circ():
            plre.ResourceTrotterCDF(compact_ham, num_steps=num_steps, order=order)

        res = plre.estimate(circ)()
        assert res.qubit_manager == expected_res["qubit_manager"]
        assert res.clean_gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        r"""Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = plre.CompactHamiltonian.thc(num_orbitals=4, tensor_rank=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceTrotterCDF"
        ):
            plre.ResourceTrotterCDF(compact_ham, num_steps=100, order=2)


class TestTrotterTHC:
    """Tests for ResourceTrotterCDF class"""

    # Expected resources were obtained manually
    # based on https://arxiv.org/abs/2407.04432

    hamiltonian_data = [
        (
            8,
            20,
            100,
            2,
            {
                "qubit_manager": QubitManager(
                    work_wires={"clean": 0, "dirty": 0}, algo_wires=40, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 9687424.0,
                        "S": 261936.0,
                        "Z": 174624.0,
                        "Hadamard": 174624.0,
                        "CNOT": 243312.0,
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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 0, "dirty": 0}, algo_wires=80, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "T": 368720000.0,
                        "S": 9900000.0,
                        "Z": 6600000.0,
                        "Hadamard": 6600000.0,
                        "CNOT": 9620000.0,
                    },
                ),
            },
        ),
    ]

    @pytest.mark.parametrize(
        "num_orbitals, tensor_rank, num_steps, order, expected_res", hamiltonian_data
    )
    def test_resource_trotter_thc(self, num_orbitals, tensor_rank, num_steps, order, expected_res):
        """Test the ResourceTrotterTHC class for correct resources"""
        compact_ham = plre.CompactHamiltonian.thc(
            num_orbitals=num_orbitals, tensor_rank=tensor_rank
        )

        def circ():
            plre.ResourceTrotterTHC(compact_ham, num_steps=num_steps, order=order)

        res = plre.estimate(circ)()

        assert res.qubit_manager == expected_res["qubit_manager"]
        assert res.clean_gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        """Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = plre.CompactHamiltonian.cdf(num_orbitals=4, num_fragments=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceTrotterTHC"
        ):
            plre.ResourceTrotterTHC(compact_ham, num_steps=100, order=2)


class TestTrotterVibrational:
    """Test the ResourceTrotterVibrational class"""

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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 95.0, "dirty": 0.0}, algo_wires=32, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 2,
                        "S": 2,
                        "T": 7898.0,
                        "X": 384064.0,
                        "Toffoli": 10069400,
                        "CNOT": 14300800.0,
                        "Hadamard": 30040840.0,
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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 127.0, "dirty": 0.0}, algo_wires=24, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 3,
                        "S": 3,
                        "T": 2327.0,
                        "X": 7548.0,
                        "Toffoli": 398770,
                        "CNOT": 489540.0,
                        "Hadamard": 1159710.0,
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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 67.0, "dirty": 0.0}, algo_wires=8, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 1,
                        "S": 1,
                        "T": 909.0,
                        "X": 4016.0,
                        "Toffoli": 37320,
                        "CNOT": 82000.0,
                        "Hadamard": 111140.0,
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
        """Test the ResourceTrotterVibrational class for correct resources"""
        compact_ham = plre.CompactHamiltonian.vibrational(
            num_modes=num_modes, grid_size=grid_size, taylor_degree=taylor_degree
        )

        def circ():
            plre.ResourceTrotterVibrational(compact_ham, num_steps=num_steps, order=order)

        res = plre.estimate(circ)()

        assert res.qubit_manager == expected_res["qubit_manager"]
        assert res.clean_gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        """Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = plre.CompactHamiltonian.cdf(num_orbitals=4, num_fragments=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceTrotterVibrational"
        ):
            plre.ResourceTrotterVibrational(compact_ham, num_steps=100, order=2)


class TestResourceTrotterVibronic:
    """Test the ResourceTrotterVibronic class"""

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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 95.0, "dirty": 0.0}, algo_wires=33, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 2,
                        "S": 2,
                        "T": 7898.0,
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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 127.0, "dirty": 0.0}, algo_wires=26, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 3,
                        "S": 3,
                        "T": 2327.0,
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
                "qubit_manager": QubitManager(
                    work_wires={"clean": 67.0, "dirty": 0.0}, algo_wires=8, tight_budget=False
                ),
                "gate_types": defaultdict(
                    int,
                    {
                        "Z": 1,
                        "S": 1,
                        "T": 909.0,
                        "X": 16,
                        "Hadamard": 111140,
                        "Toffoli": 37320,
                        "CNOT": 86000,
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
        """Test the ResourceTrotterVibronic class for correct resources"""
        compact_ham = plre.CompactHamiltonian.vibronic(
            num_modes=num_modes,
            num_states=num_states,
            grid_size=grid_size,
            taylor_degree=taylor_degree,
        )

        def circ():
            plre.ResourceTrotterVibronic(compact_ham, num_steps=num_steps, order=order)

        res = plre.estimate(circ)()

        assert res.qubit_manager == expected_res["qubit_manager"]
        assert res.clean_gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        """Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = plre.CompactHamiltonian.cdf(num_orbitals=4, num_fragments=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceTrotterVibronic"
        ):
            plre.ResourceTrotterVibronic(compact_ham, num_steps=100, order=2)
