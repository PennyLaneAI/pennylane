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
Defines a metaclass for automatic integration of any ``Operator`` with plxpr program capture.

See ``explanations.md`` for technical explanations of how this works.
"""

import abc
from functools import lru_cache
from typing import Optional, Callable

import pennylane as qml

from .switches import plxpr_enabled

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache  # constrcut the first time lazily
def _get_abstract_operator() -> type:
    """Create an AbstractOperator once in a way protected from lack of a jax install."""
    if not has_jax:
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
def _get_abstract_measurement():
    if not has_jax:
        raise ImportError("Jax is required for plxpr.")

    class AbstractMeasurement(jax.core.AbstractValue):
        """An abstract measurement."""

        def __init__(self, abstract_eval: Callable, n_wires: Optional[int] = None):
            self.abstract_eval = abstract_eval
            self.n_wires = n_wires

        def __repr__(self):
            return f"AbstractMeasurement(n_wires={self.n_wires})"

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
            return isinstance(other, AbstractMeasurement)

        def __hash__(self):
            return hash("AbstractMeasurement")

    jax.core.raise_to_shaped_mappings[AbstractMeasurement] = lambda aval, _: aval

    return AbstractMeasurement


def create_operator_primitive(operator_type: type) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an operator type.

    Args:
        operator_type (type): a subclass of qml.operation.Operator

    Returns:
        Optional[jax.core.Primitive]: None is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive(operator_type.__name__)

    @primitive.def_impl
    def _(*args, **kwargs):
        if "n_wires" not in kwargs:
            return type.__call__(operator_type, *args, **kwargs)
        n_wires = kwargs.pop("n_wires")

        # need to convert array values into integers
        # for plxpr, all wires must be integers
        wires = tuple(int(w) for w in args[-n_wires:])
        args = args[:-n_wires]
        return type.__call__(operator_type, *args, wires=wires, **kwargs)

    # logic here will be extended when we make more things use this meta class
    abstract_type = _get_abstract_operator()

    @primitive.def_abstract_eval
    def _(*_, **__):
        return abstract_type()

    return primitive


def create_measurement_obs_primitive(
    measurement_type: type, name: str
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to the input type where the abstract inputs are an operator."""
    if not has_jax:
        return None

    primitive = jax.core.Primitive(name)

    @primitive.def_impl
    def _(obs, **kwargs):
        return type.__call__(measurement_type, obs=obs, **kwargs)

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _(*_, **__):
        abstract_eval = measurement_type._abstract_eval  # pylint: disable=protected-access
        return abstract_type(abstract_eval, n_wires=None)

    return primitive


def create_measurement_wires_primitive(
    measurement_type: type, name: str
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to the input type where the abstract inputs are the wires."""
    if not has_jax:
        return None

    primitive = jax.core.Primitive(name)

    @primitive.def_impl
    def _(*wires, **kwargs):
        wires = qml.wires.Wires(wires)
        return measurement_type(wires=wires, **kwargs)

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _(*wires, **_):
        abstract_eval = measurement_type._abstract_eval  # pylint: disable=protected-access
        return abstract_type(abstract_eval, n_wires=len(wires))

    return primitive


class PLXPRMeta(abc.ABCMeta):
    """A metatype that dispatches class creation to ``cls._primitve_bind_call`` instead
    of normal class creation.

    See ``pennylane/capture/explanations.md`` for more detailed information on how this technically
    works.
    """

    def _primitive_bind_call(cls, *args, **kwargs):
        raise NotImplementedError(
            "Types using PLXPRMeta must implement cls._primitive_bind_call to"
            " gain integration with plxpr program capture."
        )

    def __call__(cls, *args, **kwargs):
        # this method is called everytime we want to create an instance of the class.
        # default behavior uses __new__ then __init__

        if plxpr_enabled():
            # when tracing is enabled, we want to
            # use bind to construct the class if we want class construction to add it to the jaxpr
            return cls._primitive_bind_call(*args, **kwargs)
        return type.__call__(cls, *args, **kwargs)
