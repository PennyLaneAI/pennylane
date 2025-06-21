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
            (e.g., "cdf", "thc").
        **params (Any): Keyword arguments specific to the chosen construction method,
            such as ``num_orbitals``, ``num_fragments``, ``tensor_rank``, or ``num_modals``.

    .. details::
        :title: Usage Details
        The :code:`CompactHamiltonian` class is designed to be an alternative input to using the full
        Hamiltonian for resource estimation. It should be used in combination with trotterization and
        qubitization templates for more efficient state resource estimation.

        .. code-block:: python
            compact_ham = plre.CompactHamiltonian.cdf(num_orbitals=8, num_fragments=4)
            def circ():
                plre.ResourceTrotterCDF(compact_ham, num_steps=100, order=2)
                return

        The resources can then be extracted as usual:

        >>> res = re.estimate_resources(circ)()
        >>> print(res)
        --- Resources: ---
         Total qubits: 16
         Total gates : 8.370E+6
         Qubit breakdown:
          clean qubits: 0, dirty qubits: 0, algorithmic qubits: 16
         Gate breakdown:
          {'T': 7.711E+6, 'S': 2.019E+5, 'Z': 1.346E+5, 'Hadamard': 1.346E+5, 'CNOT': 1.873E+5}

        Note that the specific parameters required for each method will depend on the
        underlying Hamiltonian representation and the method used to construct it.
        The methods available for constructing a `CompactHamiltonian` include:

        - :meth:`cdf`: Saves the data for compressed double factorized Hamiltonian
        - :meth:`thc`: Saves the data for tensor hypercontracted Hamiltonian

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
