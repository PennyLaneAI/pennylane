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
Test the Resource classes for Qubitization
"""
import pytest

import pennylane.labs.resource_estimation as plre
# pylint: disable=too-many-arguments

import pytest

class TestQubitizeTHC:
    # Expected resources were obtained from Pablo's code
    hamiltonian_data = [
        (
            76,
            145,
            285,
            202695
        )
    ]
    @pytest.mark.parametrize(
        "num_orbitals, tensor_rank, qubits, toffoli_count", hamiltonian_data
    )
    def test_resource_trotter_thc(self, num_orbitals, tensor_rank, qubits, toffoli_count):
        """Test the ResourceTrotterTHC class for tensor hypercontraction"""
        compact_ham = plre.CompactHamiltonian.thc(
            num_orbitals=num_orbitals, tensor_rank=tensor_rank
        )

        def circ():
            plre.ResourceQubitizeTHC(compact_ham, coeff_precision=2e-5, rotation_precision=2e-5, compare_precision=1e-2)

        res = plre.estimate_resources(circ)()

        assert res.qubit_manager.total_qubits == qubits
        assert res.clean_gate_counts["Toffoli"] == toffoli_count