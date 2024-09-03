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
This submodule defines the abstract classes and primitives for capturing operators.
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
def _get_abstract_operator() -> type:
    """Create an AbstractOperator once in a way protected from lack of a jax install."""
    if not has_jax:  # pragma: no cover
        raise ImportError("Jax is required for plxpr.")  # pragma: no cover

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


def create_operator_primitive(
    operator_type: Type["qml.operation.Operator"],
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an operator type.

    Called when defining any :class:`~.Operator` subclass, and is used to set the
    ``Operator._primitive`` class property.

    Args:
        operator_type (type): a subclass of qml.operation.Operator

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive with the same name as the operator subclass.
        ``None`` is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive(operator_type.__name__)

    @primitive.def_impl
    def _(*args, **kwargs):
        if "n_wires" not in kwargs:
            return type.__call__(operator_type, *args, **kwargs)
        n_wires = kwargs.pop("n_wires")

        split = None if n_wires == 0 else -n_wires
        # need to convert array values into integers
        # for plxpr, all wires must be integers
        wires = tuple(int(w) for w in args[split:])
        args = args[:split]
        return type.__call__(operator_type, *args, wires=wires, **kwargs)

    abstract_type = _get_abstract_operator()

    @primitive.def_abstract_eval
    def _(*_, **__):
        return abstract_type()

    def _operator_batching_rule(batched_args, batch_dims, **kwargs):

        # Extract the batched data
        batched_data = batched_args[0][0]

        # Handle non-batched arguments
        non_batched_args = batched_args[1:]

        n_wires = kwargs.pop("n_wires", 0)

        # Calculate the index where wire arguments start
        split = None if n_wires == 0 else -n_wires

        def single_op(*args):
            # Convert the wire arguments to integers
            wires = tuple(int(w) for w in args[split:])
            # Pass the batched data and static arguments to the operator
            return primitive.bind(*args[:split], *wires, **kwargs)

        # Ensure batched_data is at least 1D
        if batched_data.ndim == 0:
            batched_data = jax.numpy.expand_dims(batched_data, axis=0)

        # Use jax.vmap to vectorize the single_op over the batched data
        batched_result = jax.vmap(single_op, in_axes=(0, *[None] * len(non_batched_args)))(
            batched_data, *non_batched_args
        )

        return batched_result, batch_dims[0]

    jax.interpreters.batching.primitive_batchers[primitive] = _operator_batching_rule

    return primitive
