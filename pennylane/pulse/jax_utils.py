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
"""Module containing the ``JaxParametrizedOperator`` and ``JaxLazyDot`` classes."""
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

import pennylane as qml

from .parametrized_hamiltonian import ParametrizedHamiltonian

has_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse
    from jax.tree_util import register_pytree_node_class
except ImportError:
    has_jax = False


@register_pytree_node_class
@dataclass
class JaxParametrizedOperator:
    """Jax pytree containing a parametrized operator."""

    O_fixed: Optional[sparse.CSR]
    O_parametrized: Tuple[sparse.CSR, ...]
    coeff_parametrized: Tuple[
        Callable[[float, float], complex], ...
    ] = field()  # pytree_node=False)

    def __call__(self, pars, t):
        if self.O_fixed is not None:
            ops = (self.O_fixed,)
            coeffs = (1,)
        else:
            ops = ()
            coeffs = ()
        coeffs = coeffs + tuple(f(p, t) for f, p in zip(self.coeff_parametrized, pars))
        return JaxLazyDot(coeffs, ops + self.O_parametrized)

    @staticmethod
    def from_parametrized_hamiltonian(
        H: ParametrizedHamiltonian, dense: bool = False, wire_order=None
    ):
        """Construct a ``JaxParametrizedOperator`` from a ``ParametrizedHamiltonian``

        Args:
            H (ParametrizedHamiltonian): parametrized Hamiltonian to convert
            dense (bool, optional): Decide wether a dense/sparse matrix is used. Defaults to False.
            wire_order (list, optional): Wire order of the returned ``JaxParametrizedOperator``.
                Defaults to None.

        Returns:
            JaxParametrizedOperator: jax-compatible parametrized operator
        """
        make_array = jnp.array if dense else sparse.CSR.fromdense

        if len(H.ops_fixed) > 0:
            fixed_mat = make_array(qml.matrix(H.H_fixed(), wire_order=wire_order))
        else:
            fixed_mat = None

        parametrized_mat = tuple(
            make_array(qml.matrix(op, wire_order=wire_order)) for op in H.ops_parametrized
        )

        return JaxParametrizedOperator(fixed_mat, parametrized_mat, tuple(H.coeffs_parametrized))

    @property
    def shape(self):
        """Returns the shape of the matrix representation of the JaxParametrizedOperator.

        Returns:
            tuple: matrix shape
        """
        return self.O_fixed.shape

    @property
    def ndim(self):
        """Returns the number of dimensions of the matrix representation of the JaxParametrizedOperator.

        Returns:
            int: number of dimensions
        """
        return self.O_fixed.ndim

    def tree_flatten(self):
        """Function used by ``jax`` to flatten the JaxParametrizedOperator.

        Returns:
            tuple: tuple containing the matrices and the parametrized coefficients defining the class
        """
        matrices = (
            self.O_fixed,
            self.O_parametrized,
        )
        param_coeffs = self.coeff_parametrized
        return (matrices, param_coeffs)

    @classmethod
    def tree_unflatten(cls, param_coeffs: tuple, matrices: tuple):
        """Function used by ``jax`` to unflatten the JaxParametrizedOperator.

        Args:
            param_coeffs (tuple): tuple containing the parametrized coefficients of the class
            matrices (tuple): tuple containing the matrices of the class

        Returns:
            JaxParametrizedOperator: a JaxParametrizedOperator instance
        """
        return cls(*matrices, param_coeffs)


@register_pytree_node_class
@dataclass
class JaxLazyDot:
    """Jax pytree representing a lazy dot operation."""

    coeffs: Tuple[complex, ...]
    ops: Tuple[sparse.CSR, ...]

    def __array__(self, dtype=None):
        res = 0
        for c, o in zip(self.coeffs, self.ops):
            if hasattr(o, "todense"):
                o = o.todense()
            res += c * o
        return np.asarray(res, dtype=dtype)

    @jax.jit
    def __matmul__(self, v):
        return sum(c * (o @ v) for c, o in zip(self.coeffs, self.ops))

    @property
    def shape(self):
        """Returns the shape of the matrix result of the operation.

        Returns:
            tuple: result shape
        """
        return self.ops[0].shape

    @property
    def ndim(self):
        """Returns the number of dimensions of the matrix result of the operation.

        Returns:
            int: number of dimensions of the result
        """
        return self.ops[0].ndim

    def tree_flatten(self):
        """Function used by ``jax`` to flatten the JaxLazyDot operation.

        Returns:
            tuple: tuple containing children and the auxiliary data of the class
        """
        children = (self.coeffs, self.ops)
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
