from typing import Callable, Any, Dict, List, Tuple

# --- 1. Mock CompactHamiltonian Class
#    This represents the actual Hamiltonian objects you are constructing.
class CompactHamiltonian:
    """
    A mock class representing a CompactHamiltonian.
    In a real scenario, this would perform actual Hamiltonian construction.
    """
    def __init__(self, method_name: str, **params: Any):
        self.method_name = method_name
        self.params = params
        print(f"CompactHamiltonian instance created via {method_name} with params: {params}")

    @classmethod
    def from_cdf(cls, num_orbitals: int, num_fragments: int):
        """Constructs a Hamiltonian from CDF data."""
        return cls("from_cdf", num_orbitals=num_orbitals, num_fragments=num_fragments)

    @classmethod
    def from_thc(cls, num_orbitals: int, tensor_rank: int):
        """Constructs a Hamiltonian from THC data."""
        return cls("from_thc", num_orbitals=num_orbitals, tensor_rank=tensor_rank)

    @classmethod
    def from_vibrational(cls, num_orbitals: int, num_fragments: int):
        """Constructs a Hamiltonian from vibrational data."""
        return cls("from_vibrational", num_orbitals=num_orbitals, num_fragments=num_fragments)