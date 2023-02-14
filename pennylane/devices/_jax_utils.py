from typing import Callable, Optional, Tuple
from functools import partial, reduce

# from flax import struct
import dataclasses as struct

from jax.experimental import sparse as jsparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from pennylane import matrix as qmatrix

# import pennylane as qml


def _one(*args):
    return 1


@register_pytree_node_class
@struct.dataclass
class JaxParametrisedOperator:
    O_fixed: Optional[jsparse.CSR]
    O_parametrised: Tuple[jsparse.CSR, ...]
    coeff_parametrised: Tuple[
        Callable[[float, float], complex], ...
    ] = struct.field()  # pytree_node=False)

    def __call__(self, pars, t):
        if self.O_fixed is not None:
            ops = (self.O_fixed,)
            coeffs = (1,)
        else:
            ops = ()
            coeffs = ()
        coeffs = coeffs + tuple(f(p, t) for f, p in zip(self.coeff_parametrised, pars))
        return LazyOperatorSum(coeffs, ops + self.O_parametrised)

    @staticmethod
    def from_parametrised_hamiltonian(H, *, dense: bool = False, wire_order=None):
        if wire_order is None:
            wire_order = range(len(H.wires))

        make_array = jnp.array if dense else jsparse.CSR.fromdense

        if len(H.ops_fixed) > 0:
            dense_op_fixed = make_array(qmatrix(H.H_fixed(), wire_order=wire_order))
        else:
            dense_op_fixed = None

        dense_ops_parametrized = tuple(
            make_array(qmatrix(op, wire_order=wire_order)) for op in H.ops_parametrized
        )

        return JaxParametrisedOperator(
            dense_op_fixed, dense_ops_parametrized, tuple(H.coeffs_parametrized)
        )

    @property
    def shape(self):
        return self.O_fixed.shape

    @property
    def ndim(self):
        return self.O_fixed.ndim

    def tree_flatten(self):
        children = (
            self.O_fixed,
            self.O_parametrised,
        )
        aux_data = coeff_parametrised
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)


@register_pytree_node_class
@struct.dataclass
class LazyOperatorSum:
    coeffs: Tuple[complex, ...]
    ops: Tuple[jsparse.CSR, ...]

    def __array__(self, dtype=None):
        res = 0
        for c, o in zip(self.coeffs, self.ops):
            if hasattr(o, "todense"):
                o = o.todense()
            res = res + c * o
        return np.asarray(res, dtype=dtype)

    @jax.jit
    def __matmul__(self, v):
        res = 0
        for c, o in zip(self.coeffs, self.ops):
            res = res + c * (o @ v)
        return res

    @property
    def shape(self):
        return self.ops[0].shape

    @property
    def ndim(self):
        return self.ops[0].ndim

    def tree_flatten(self):
        children = (self.coeffs, self.ops)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
