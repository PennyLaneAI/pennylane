# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Defines primitives for Operator and wires.
"""

from functools import lru_cache
from typing import Optional, Type

import pennylane as qml

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache  # construct the first time lazily
def _get_abstract_operator() -> Type["jax.core.AbstractValue"]:
    """Create an AbstractOperator once in a way protected from lack of a jax install."""
    if not has_jax:  # pragma: no-cover
        raise ImportError("Jax is required for plxpr.")

    class AbstractOperator(jax.core.AbstractValue):
        """An operator captured into plxpr."""

        # pylint: disable=missing-function-docstring
        def at_least_vspace(self):
            # TODO: investigate the proper definition of this method
            raise NotImplementedError

        # pylint: disable=missing-function-docstring
        def join(self, other):
            # TODO: investigate the proper definition of this method
            raise NotImplementedError

        # pylint: disable=missing-function-docstring
        def update(self, **kwargs):
            # TODO: investigate the proper definition of this method
            raise NotImplementedError

        def __eq__(self, other):
            return isinstance(other, AbstractOperator)

        def __hash__(self):
            return hash("AbstractOperator")

        @staticmethod
        def _matmul(*args):
            return qml.prod(*args)

        @staticmethod
        def _mul(a, b):
            return qml.s_prod(b, a)

        @staticmethod
        def _rmul(a, b):
            return qml.s_prod(b, a)

        @staticmethod
        def _add(a, b):
            return qml.sum(a, b)

        @staticmethod
        def _pow(a, b):
            return qml.pow(a, b)

    jax.core.raise_to_shaped_mappings[AbstractOperator] = lambda aval, _: aval

    return AbstractOperator


@lru_cache
def _get_abstract_wires() -> type:
    """Retrieve the AbstractWires class via a cached getter."""
    if not has_jax:  # pragma: no-cover
        raise ImportError("Jax is required for plxpr.")

    class AbstractWires(jax.core.ShapedArray):
        """Abstract wires."""

        def str_short(self, short_dtypes=False):
            return f"Wires[{self.n_wires}]"

        def __repr__(self):
            return f"Wires[{self.n_wires}]"

        def __str__(self):
            return f"Wires[{self.n_wires}]"

        def __init__(self, n_wires):
            self.n_wires = n_wires
            super().__init__((n_wires,), jax.numpy.int32)

    jax.core.raise_to_shaped_mappings[AbstractWires] = lambda aval, weak_type: AbstractWires(
        aval.shape[0]
    )

    return AbstractWires


def create_wires_primitive(wires_type):
    """Create the wires Primitive."""
    wires_p = jax.core.Primitive("Wires")

    @wires_p.def_impl
    def _(*wires):
        int_wires = tuple(int(w) for w in wires)
        return type.__call__(wires_type, int_wires)

    AbstractWires = _get_abstract_wires()

    @wires_p.def_abstract_eval
    def _(*wires):
        return AbstractWires(len(wires))

    return wires_p


def create_operator_primitive(
    operator_type: Type["qml.operation.Operator"],
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an operator type.

    Args:
        operator_type (type): a subclass of qml.operation.Operator

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive with the same name as the operator subclass.
        ``None`` is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive(operator_type.__name__)

    primitive.def_impl(operator_type._primitive_def_impl)  # pylint: disable=protected-access

    abstract_type = _get_abstract_operator()

    @primitive.def_abstract_eval
    def _(*_, **__):
        return abstract_type()

    return primitive
