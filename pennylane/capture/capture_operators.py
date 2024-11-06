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
from collections.abc import Callable
from functools import lru_cache
from typing import Optional, Type

import pennylane as qml

from .capture_diff import create_non_interpreted_prim

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

        def __init__(self, abstract_eval: Callable, n_wires: Optional[int] = None):
            self._abstract_eval = abstract_eval
            self._n_wires = n_wires

        def abstract_eval(self, num_device_wires: int) -> tuple[tuple, type]:
            """Calculate the shape and dtype for an evaluation with specified number of device
            wires and shots.

            """
            return self._abstract_eval(
                n_wires=self._n_wires,
                num_device_wires=num_device_wires,
            )

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

    primitive = create_non_interpreted_prim()(operator_type.__name__)

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
        abstract_eval = operator_type._abstract_eval  # pylint: disable=protected-access

        roba = abstract_type(abstract_eval, n_wires=None)

        print(f"roba from create_operator_primitive: {roba}")

        return roba

    return primitive
