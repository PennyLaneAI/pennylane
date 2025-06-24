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
from pennylane.labs.resource_estimation import QubitManager

# pylint: disable=no-self-use, too-many-arguments


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

        res = plre.estimate_resources(circ)()
        print(res, expected_res)
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

        res = plre.estimate_resources(circ)()

        assert res.qubit_manager == expected_res["qubit_manager"]
        assert res.clean_gate_counts == expected_res["gate_types"]

    def test_type_error(self):
        """Test that a TypeError is raised for unsupported Hamiltonian representations."""
        compact_ham = plre.CompactHamiltonian.cdf(num_orbitals=4, num_fragments=10)
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceTrotterTHC"
        ):
            plre.ResourceTrotterTHC(compact_ham, num_steps=100, order=2)
