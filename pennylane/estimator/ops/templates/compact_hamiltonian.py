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

from typing import Any


class CompactHamiltonian:
    r"""A compact representation for the Hamiltonian of a quantum system.

    Args:
        method_name (str): The name of the method used to construct the Hamiltonian.
            The available methods are cdf, thc, vibrational, and vibronic.
        **params (Any): Keyword arguments specific to the chosen construction method,

            - For :meth:`~.CompactHamiltonian.cdf`, parameters include ``num_orbitals`` and ``num_fragments``.
            - For :meth:`~.CompactHamiltonian.thc`, parameters include ``num_orbitals`` and ``tensor_rank``.
            - For :meth:`~.CompactHamiltonian.vibrational`, parameters include ``num_modes``, ``grid_size`` and ``taylor_degree``.
            - For :meth:`~.CompactHamiltonian.vibronic`, parameters include ``num_modes``, ``num_states``, ``grid_size`` and ``taylor_degree``.


    Returns:
        CompactHamiltonian: An instance of CompactHamiltonian

    **Example**

    The resources for trotterization of THC Hamiltonian can be extracted as:

    >>> import pennylane.labs.resource_estimation as plre
    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=8, tensor_rank=40)
    >>> trotter_thc = plre.ResourceTrotterTHC(compact_ham, num_steps=100, order=2)
    >>> res = plre.estimate(trotter_thc)
    >>> print(res)
    --- Resources: ---
     Total qubits: 80
     Total gates : 3.960E+7
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 80
     Gate breakdown:
      {'T': 3.638E+7, 'S': 9.699E+5, 'Z': 6.466E+5, 'Hadamard': 6.466E+5, 'CNOT': 9.553E+5}

    """

    def __init__(self, method_name: str, **params: dict[str, Any]):
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
