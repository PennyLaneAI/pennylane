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
Contains class to represent different Hamiltonians.
"""

from typing import Any, Dict


class CompactHamiltonian:
    r"""A compact representation for the Hamiltonian of a quantum system.

    Args:
        method_name (str): The name of the method used to construct the Hamiltonian
            The methods available for constructing a `CompactHamiltonian` include:

            - :meth:`cdf`: Constructs a Hamiltonian in the compressed double factorized representation
            - :meth:`thc`: Constructs a Hamiltonian in the  tensor hypercontracted representation

        **params (Any): Keyword arguments specific to the chosen construction method,

            - For :meth:`~.CompactHamiltonian.cdf`, parameters include ``num_orbitals`` and ``num_fragments``.
            - For :meth:`~.CompactHamiltonian.thc`, parameters include ``num_orbitals`` and ``tensor_rank``.


    Returns:
        CompactHamiltonian: An instance of CompactHamiltonian

    **Example**

    The resources for trotterization of THC Hamiltonian can be extracted as:

    >>> import pennylane.labs.resource_estimation as plre
    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=8, tensor_rank=40)
    >>> trotter_thc = plre.ResourceTrotterTHC(compact_ham, num_steps=100, order=2)
    >>> res = plre.estimate_resources(trotter_thc)
    >>> print(res)
    --- Resources: ---
     Total qubits: 80
     Total gates : 3.960E+7
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 80
     Gate breakdown:
      {'T': 3.638E+7, 'S': 9.699E+5, 'Z': 6.466E+5, 'Hadamard': 6.466E+5, 'CNOT': 9.553E+5}

    """

    def __init__(self, method_name: str, **params: Dict[str, Any]):
        self.method_name = method_name
        self.params = params

    @classmethod
    def cdf(cls, num_orbitals: int, num_fragments: int):
        """Constructs a compressed double factorized Hamiltonian instance

        Args:
            num_orbitals (int): number of spatial orbitals
            num_fragments (int): number of fragments in the compressed double factorization (CDF) representation

        Returns:
            CompactHamiltonian: An instance of CompactHamiltonian initialized with CDF parameters.
        """
        return cls("cdf", num_orbitals=num_orbitals, num_fragments=num_fragments)

    @classmethod
    def thc(cls, num_orbitals: int, tensor_rank: int):
        """Constructs a tensor hypercontracted Hamiltonian instance

        Args:
            num_orbitals (int): number of spatial orbitals
            tensor_rank (int):  tensor rank of two-body integrals in the tensor hypercontracted (THC) representation

        Returns:
            CompactHamiltonian: An instance of CompactHamiltonian initialized with THC parameters.
        """
        return cls("thc", num_orbitals=num_orbitals, tensor_rank=tensor_rank)
