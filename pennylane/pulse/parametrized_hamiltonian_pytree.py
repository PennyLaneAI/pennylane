# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module containing the ``JaxParametrizedHamiltonian`` class."""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.tree_util import register_pytree_node_class

import pennylane as qml

from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian


@register_pytree_node_class
@dataclass
class ParametrizedHamiltonianPytree:
    """Jax pytree class that represents a ``ParametrizedHamiltonian``."""

    mat_fixed: Optional[Union[jnp.ndarray, sparse.BCSR]]
    mats_parametrized: Tuple[Union[jnp.ndarray, sparse.BCSR], ...]
    coeffs_parametrized: Tuple[Callable]
    reorder_fn: Callable

    @staticmethod
    def from_hamiltonian(H: ParametrizedHamiltonian, *, dense: bool = False, wire_order=None):
        """Convert a ``ParametrizedHamiltonian`` into a jax pytree object.

        Args:
            H (ParametrizedHamiltonian): parametrized Hamiltonian to convert
            dense (bool, optional): Decide wether a dense/sparse matrix is used. Defaults to False.
            wire_order (list, optional): Wire order of the returned ``JaxParametrizedOperator``.
                Defaults to None.

        Returns:
            ParametrizedHamiltonianPytree: pytree object
        """
        make_array = jnp.array if dense else sparse.BCSR.fromdense

        if len(H.ops_fixed) > 0:
            mat_fixed = make_array(qml.matrix(H.H_fixed(), wire_order=wire_order))
        else:
            mat_fixed = None

        mats_parametrized = tuple(
            make_array(qml.matrix(op, wire_order=wire_order)) for op in H.ops_parametrized
        )

        if isinstance(H, HardwareHamiltonian):
            return ParametrizedHamiltonianPytree(
                mat_fixed,
                mats_parametrized,
                H.coeffs_parametrized,
                reorder_fn=H.reorder_fn,
            )

        return ParametrizedHamiltonianPytree(
            mat_fixed, mats_parametrized, H.coeffs_parametrized, reorder_fn=None
        )

    def __call__(self, pars, t):
        if self.mat_fixed is not None:
            ops = (self.mat_fixed,)
            coeffs = (1,)
        else:
            ops = ()
            coeffs = ()

        if self.reorder_fn:
            pars = self.reorder_fn(pars, self.coeffs_parametrized)
        coeffs = coeffs + tuple(f(p, t) for f, p in zip(self.coeffs_parametrized, pars))
        return LazyDotPytree(coeffs, ops + self.mats_parametrized)

    def tree_flatten(self):
        """Function used by ``jax`` to flatten the JaxParametrizedOperator.

        Returns:
            tuple: tuple containing the matrices and the parametrized coefficients defining the class
        """
        matrices = (self.mat_fixed, self.mats_parametrized)
        param_coeffs = self.coeffs_parametrized
        reorder_fn = self.reorder_fn
        return (matrices, param_coeffs, reorder_fn)

    @classmethod
    def tree_unflatten(cls, param_coeffs: tuple, matrices: tuple, reorder_fn: callable):
        """Function used by ``jax`` to unflatten the JaxParametrizedOperator.

        Args:
            param_coeffs (tuple): tuple containing the parametrized coefficients of the class
            matrices (tuple): tuple containing the matrices of the class
            reorder_fn(callable): callable or None indicating how parameters should be
                re-orderd to pass to the __call__ method

        Returns:
            JaxParametrizedOperator: a JaxParametrizedOperator instance
        """
        return cls(*matrices, param_coeffs, reorder_fn)


@register_pytree_node_class
@dataclass
class LazyDotPytree:
    """Jax pytree representing a lazy dot operation."""

    coeffs: Tuple[complex, ...]
    mats: Tuple[Union[jnp.ndarray, sparse.BCSR], ...]

    @jax.jit
    def __matmul__(self, other):
        return sum(c * (m @ other) for c, m in zip(self.coeffs, self.mats))

    def __mul__(self, other):
        if jnp.array(other).ndim == 0:
            return LazyDotPytree(tuple(other * c for c in self.coeffs), self.mats)
        return NotImplemented

    __rmul__ = __mul__

    def tree_flatten(self):
        """Function used by ``jax`` to flatten the JaxLazyDot operation.

        Returns:
            tuple: tuple containing children and the auxiliary data of the class
        """
        children = (self.coeffs, self.mats)
        aux_data = None
        return (children, aux_data)

    # pylint: disable=unused-argument
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Function used by ``jax`` to unflatten the ``JaxLazyDot`` pytree.

        Args:
            aux_data (None): empty argument
            children (tuple): tuple containing the coefficients and the matrices of the operation

        Returns:
            JaxLazyDot: JaxLazyDot instance
        """
        return cls(*children)
