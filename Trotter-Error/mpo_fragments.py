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
"""Wrapper class for generic fragment objects"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

# Define Array type
Array = Union[np.ndarray, ArrayLike]

from pennylane.fermi import FermiSentence
from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.abstract import AbstractState
from pyblock2.algebra.core import MPO, MPS
from pyblock2.algebra.io import MPOTools

import block2
block2.SZ.__getstate__ = lambda self: getattr(self, 'data')
block2.SZ.__setstate__ = lambda self, data: setattr(self, 'data', data)


from abc import abstractmethod

def mpo_fragments(fragments: Sequence[MPO], bond_dim: int = 100, norm_values: Sequence[float] = None, n_sites: int = None, h1e_vals: Sequence[Array] = None, g2e_vals: Sequence[Array] = None) -> List[MPOFragment]:
    """Instantiates :class:`~.pennylane.labs.trotter_error.MPOFragment` objects.

    Args:
        fragments (Sequence[MPO]): A sequence of MPO objects.
        bond_dim (int): The bond dimension for the MPOFragment. Defaults to 100.
        norm_values (Sequence[float], optional): Pre-computed norm values for the fragments.
        n_sites (int, optional): Number of orbitals/sites. If None, will be inferred from MPO.
        h1e_vals (Sequence[Array], optional): One-body integral values for each fragment.
        g2e_vals (Sequence[Array], optional): Two-body integral values for each fragment.

    Returns:
        List[MPOFragment]: A list of :class:`~.pennylane.labs.trotter_error.MPOFragment` objects instantiated from `fragments`.
    """

    if len(fragments) == 0:
        return []

    if not any(isinstance(fragment, MPO) for fragment in fragments):
        raise TypeError("Fragments must be MPO objects")

    # If n_sites is not provided, try to infer it from the first fragment
    if n_sites is None and len(fragments) > 0:
        try:
            n_sites = len(fragments[0].tensors)
        except:
            n_sites = None

    # Create MPOFragment objects with corresponding h1e and g2e values
    mpo_fragment_list = []
    for i, fragment in enumerate(fragments):
        h1e = h1e_vals[i] if h1e_vals is not None and i < len(h1e_vals) else None
        g2e = g2e_vals[i] if g2e_vals is not None and i < len(g2e_vals) else None
        norm = np.sum(np.abs(h1e)) + np.sum(np.abs(g2e)) if h1e is not None and g2e is not None else (norm_values[i] if norm_values is not None and i < len(norm_values) else None)

        mpo_fragment_list.append(MPOFragment(fragment, h1e=h1e, g2e=g2e, bond_dim=bond_dim, norm=norm, n_sites=n_sites))

    return mpo_fragment_list


class MPOFragment(Fragment):
    """Abstract class used to define a generic fragment object for product formula error estimation.

    This class allows using any object implementing arithmetic dunder methods to be used
    for product formula error estimation.

    Args:
        fragment (Any): An object that implements the following arithmetic methods:
            ``__add__``, ``__mul__``, and ``__matmul__``.
        norm_fn (optional, Callable): A function used to compute the norm of ``fragment``.

    .. note:: :class:`~.pennylane.labs.trotter_error.MPOFragment` objects should be instantated through the ``generic_fragments`` function.

    **Example**

    >>> from pennylane.labs.trotter_error import generic_fragments
    >>> import numpy as np
    >>> matrices = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
    >>> generic_fragments(matrices)
    [MPOFragment(type=<class 'numpy.ndarray'>), MPOFragment(type=<class 'numpy.ndarray'>)]
    """

    def __init__(self, fragment: MPO, h1e: Array = None, g2e: Array = None, bond_dim: int = 100, norm: float = None, n_sites: int = None):
        self.fragment = fragment
        self.n_sites = n_sites
        self.h1e = h1e if h1e is not None else (np.zeros((n_sites, n_sites), dtype=np.float64) if n_sites is not None else None)
        self.g2e = g2e if g2e is not None else (np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=np.float64) if n_sites is not None else None)
        self.bond_dim = bond_dim
        self._norm_value = norm

    def __add__(self, other: MPOFragment):
        if self.h1e is not None and other.h1e is not None:
            h1e = self.h1e + other.h1e
            g2e = self.g2e + other.g2e
            # For now, use direct fragment addition when driver is not available
            # TODO: This should be properly handled with a global driver or passed as parameter
            new_fragment = self.fragment + other.fragment
            norm = np.sum(np.abs(h1e)) + np.sum(np.abs(g2e))
        else:
            h1e, g2e = None, None
            new_fragment = self.fragment + other.fragment
            norm = self._norm_value + other._norm_value if self._norm_value is not None and other._norm_value is not None else None

        new_fragment.compress(k=self.bond_dim)
        return MPOFragment(new_fragment, h1e=h1e, g2e=g2e, bond_dim=self.bond_dim, norm=norm, n_sites=self.n_sites)

    def __sub__(self, other: MPOFragment):
        new_other_h1e = self.h1e-other.h1e if (other.h1e is not None) and (self.h1e is not None) else None
        new_other_g2e = self.g2e-other.g2e if (other.g2e is not None) and (self.g2e is not None) else None
        norm = np.sum(np.abs(new_other_h1e)) + np.sum(np.abs(new_other_g2e)) if (new_other_h1e is not None) and (new_other_g2e is not None) else None

        new_other_fragment = MPOFragment(self.fragment-other.fragment, other.n_sites, h1e=new_other_h1e, g2e=new_other_g2e, norm=norm, bond_dim=other.bond_dim)
        return self.__add__(new_other_fragment)

    def __mul__(self, scalar: float):
        h1e = self.h1e * scalar if self.h1e is not None else None
        g2e = self.g2e * scalar if self.g2e is not None else None
        norm = self._norm_value * scalar if self._norm_value is not None else None

        return MPOFragment(scalar * self.fragment, h1e=h1e, g2e=g2e, norm=norm, bond_dim=self.bond_dim)

    __rmul__ = __mul__

    def __eq__(self, other: MPOFragment):
        if not isinstance(other, MPOFragment):
            raise TypeError(f"Cannot compare MPOFragment with type {type(other)}.")

        return self.fragment == other.fragment

    def __matmul__(self, other: MPOFragment):
        new_fragment = self.fragment @ other.fragment
        new_fragment.compress(k=self.bond_dim)
        h1e, g2e = None, None
        norm = self._norm_value * other._norm_value if self._norm_value is not None and other._norm_value is not None else None
        return MPOFragment(new_fragment, h1e=h1e, g2e=g2e, norm=norm, bond_dim=self.bond_dim)

    def __rmatmul__(self, other: MPOFragment):
        new_fragment = other.fragment @ self.fragment
        new_fragment.compress(k=self.bond_dim)
        h1e, g2e = None, None
        norm = self._norm_value * other._norm_value if self._norm_value is not None and other._norm_value is not None else None
        return MPOFragment(new_fragment, h1e=h1e, g2e=g2e, norm=norm, bond_dim=self.bond_dim)

    def norm(self, params: Dict = None) -> float:
        """Compute the norm of the fragment.
        
        Args:
            params (Dict, optional): A dictionary of parameters needed to compute the norm.
                                   Not used in this implementation as norm is precomputed.
        
        Returns:
            float: the norm of the MPOFragment
        """
        # Use the stored norm value if available
        if hasattr(self, '_norm_value') and self._norm_value is not None:
            return self._norm_value
        elif self.h1e is not None and self.g2e is not None:
            return float(np.sum(np.abs(self.h1e)) + np.sum(np.abs(self.g2e)))
        else:
            raise ValueError("Norm value is not set and cannot be computed from h1e and g2e values.")

    def apply(self, state: Any) -> Any:
        """Apply the fragment to a state using the underlying object's ``__matmul__`` method."""
        new_state = self.fragment @ state.mps
        new_state.compress(k=self.bond_dim)
        return MPState(new_state, bond_dim=self.bond_dim)

    def expectation(self, left: Any, right: Any) -> float:
        """Compute the expectation value using the underlying object's ``__matmul__`` method."""
        return left @ self.fragment @ right

    def dot(self, other: MPOFragment) -> float:
        return self.__matmul__(other)

    def __repr__(self):
        return self.fragment.__repr__()

class MPState(AbstractState):
    """Abstract class used to define a state object for product formula error estimation.

    A class inheriting from ``MPSState`` must implement the following dunder methods.

    * ``__add__``: implements addition
    * ``__mul__``: implements multiplication

    Additionally, it requires the following methods.

    * ``zero_state``: returns a representation of the zero state
    * ``dot``: implments the dot product of two states
    """
    def __init__(self, pyblock_mps: MPS, bond_dim: int = 10):
        """Initialize the MPSState with a specified bond dimension.

        Args:
            bond_dim (int): The bond dimension for the MPSState. Defaults to 10.
        """
        self.mps = pyblock_mps
        self.bond_dim = bond_dim

    def __add__(self, other: MPState) -> MPState:
        new_mps = self.mps + other.mps
        new_mps.compress(k=self.bond_dim)
        return MPState(new_mps, bond_dim=self.bond_dim)

    def __sub__(self, other: MPState) -> MPState:
        return self + (-1) * other

    def __mul__(self, scalar: float) -> MPState:
        return  MPState(scalar * self.mps, bond_dim=self.bond_dim)

    def __rmul__(self, scalar: float) -> MPState:
        return self.__mul__(scalar)

    @classmethod
    def zero_state(cls) -> MPState:
        """Return a representation of the zero state.

        Returns:
            MPSState: an ``MPSState`` representation of the zero state
        """
        raise NotImplementedError

    def dot(self, other) -> float:
        """Compute the dot product of two states.

        Args:
            other: the state to take the dot product with

        Returns:
        float: the dot product of self and other
        """
        # Handle _AdditiveIdentity (zero state)
        if hasattr(other, '__class__') and 'AdditiveIdentity' in other.__class__.__name__:
            return 0.0
        
        # Handle MPState objects
        if isinstance(other, MPState):
            result_mps = self.mps @ other.mps
            # For dot product, we might want to return a scalar, not another MPState
            # Depending on your MPS library, you might need to extract the scalar value
            if hasattr(result_mps, 'real'):
                return complex(result_mps.real)
            return result_mps
        
        raise TypeError(f"Cannot compute dot product between MPState and {type(other)}")
