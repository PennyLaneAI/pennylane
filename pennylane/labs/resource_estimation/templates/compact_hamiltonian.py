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
            (e.g., "cdf", "thc", "vibrational", "vibronic").
        **params (Any): Keyword arguments specific to the chosen construction method,
            For example:

            - For :meth:`~.CompactHamiltonian.cdf`, parameters include ``num_orbitals`` and ``num_fragments``.
            - For :meth:`~.CompactHamiltonian.thc`, parameters include ``num_orbitals`` and ``tensor_rank``.

            Refer to the documentation of each specific constructor method for their required parameters.

    .. details::
        :title: Usage Details

        The :class:`CompactHamiltonian` class is designed to be an alternative input to using the full
        Hamiltonian for resource estimation. It should be used in combination with trotterization and
        qubitization templates for more efficient state resource estimation.

        .. code-block:: python

            import pennylane.labs.resource_estimation as plre
            compact_ham = plre.CompactHamiltonian.cdf(num_orbitals=8, num_fragments=4)
            def circ():
                plre.ResourceTrotterCDF(compact_ham, num_steps=100, order=2)
                return

        The resources can then be extracted as usual:

        >>> res = plre.estimate_resources(circ)()
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

        - :meth:`cdf`: Constructs a Hamiltonian in the compressed double factorized representation
        - :meth:`thc`: Constructs a Hamiltonian in the  tensor hypercontracted representation

    """

    def __init__(self, method_name: str, **params: Dict[str, Any]):
        self.method_name = method_name
        self.params = params

    @classmethod
    def sparsepauli(cls, num_orbitals:int, num_terms:int):
        return cls("sparsepauli", num_orbitals=num_orbitals, num_terms=num_terms)

    @classmethod
    def anticommuting(cls, num_orbitals: int, num_ac_groups:int, num_paulis:int):
        return cls("anticommuting", num_orbitals=num_orbitals, num_ac_groups=num_ac_groups, num_paulis=num_paulis)

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

    @classmethod
    def vibrational(cls, num_modes: int, grid_size: int, taylor_degree: int):
        """Constructs a vibrational Hamiltonian instance

        Args:
            num_modes (int): number of vibrational modes
            grid_size (int): number of grid points used to discretize each mode
            taylor_degree (int): degree of the Taylor expansion used in the vibrational representation

        Returns:
            CompactHamiltonian: An instance of CompactHamiltonian initialized with vibrational Hamiltonian parameters.
        """
        return cls(
            "vibrational",
            num_modes=num_modes,
            grid_size=grid_size,
            taylor_degree=taylor_degree,
        )

    @classmethod
    def vibronic(cls, num_modes: int, num_states: int, grid_size: int, taylor_degree: int):
        """Constructs a vibronic Hamiltonian instance

        Args:
            num_modes (int): number of vibronic modes
            num_states (int): number of vibronic states
            grid_size (int): number of grid points used to discretize each mode
            taylor_degree (int): degree of the Taylor expansion used in the vibronic representation

        Returns:
            CompactHamiltonian: An instance of CompactHamiltonian initialized with vibronic Hamiltonian parameters.
        """
        return cls(
            "vibronic",
            num_modes=num_modes,
            num_states=num_states,
            grid_size=grid_size,
            taylor_degree=taylor_degree,
        )

