# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Defines a metaclass for automatic integration of any ``Operator`` with jaxpr program capture.

See ``explanations.md`` for technical explanations of how this works.
"""

from functools import lru_cache

import pennylane as qml

from .switches import plxpr_enabled

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache  # constrcut the first time lazily
def _get_abstract_operator():
    """Create an AbstractOperator once in a way protected from lack of a jax install."""
    if not has_jax:
        raise ImportError("Jax is required for plxpr.")

    # TODO: investigate
    # pennylane/capture/meta_type.py:37:4: W0223: Method 'at_least_vspace' is abstract in class 'AbstractValue' but is not overridden in child class 'AbstractOperator' (abstract-method)
    # pennylane/capture/meta_type.py:37:4: W0223: Method 'join' is abstract in class 'AbstractValue' but is not overridden in child class 'AbstractOperator' (abstract-method)
    # pennylane/capture/meta_type.py:37:4: W0223: Method 'update' is abstract in class 'AbstractValue' but is not overridden in child class 'AbstractOperator' (abstract-method)

    class AbstractOperator(jax.core.AbstractValue):
        """An operator captured into plxpr."""

        def __eq__(self, other):
            return isinstance(other, AbstractOperator)

        def __hash__(self):
            return hash("AbstractOperator")

        def _matmul(self, *args):
            return qml.prod(*args)

        def _mul(self, a, b):
            return qml.s_prod(b, a)

        def _rmul(self, a, b):
            return qml.s_prod(b, a)

        def _add(self, a, b):
            return qml.sum(a, b)

    jax.core.raise_to_shaped_mappings[AbstractOperator] = lambda aval, _: aval

    return AbstractOperator


class PLXPRMeta(type):
    """A metatype that:

    * automatically registers a jax primitive to ``cls._primitive``
    * Dispatches class creation to ``cls._primitive.bind`` instead of normal class
    creation when the primitive is defined and plxpr capture is enabled.

    See ``pennylane/capture/explanations.md`` for more detailed information on how this technically
    works.

    """

    def __init__(cls, *_, **__):

        # Called when constructing a new type that has this metaclass.
        # Similar to __init_subclass__ , this allows us to run this code
        # every time we define a new class

        if not has_jax:
            cls._primitive = None
            return

        cls._primitive = jax.core.Primitive(cls.__name__)

        @cls._primitive.def_impl
        def default_call(*args, **kwargs):
            if "n_wires" not in kwargs:
                return type.__call__(cls, *args, **kwargs)
            n_wires = kwargs.pop("n_wires")
            wires = args[-n_wires:]
            args = args[:-n_wires]
            return type.__call__(cls, *args, wires=wires, **kwargs)

        # logic here will be extended when we make more things use this meta class
        abstract_type = _get_abstract_operator()

        @cls._primitive.def_abstract_eval
        def abstract_init(*_, **__):
            return abstract_type()

    def _primitive_bind_call(cls, *args, **kwargs):
        if "wires" in kwargs:
            wires = kwargs.pop("wires")
            wires = (
                tuple(wires)
                if isinstance(wires, (list, tuple, qml.wires.Wires, range))
                else (wires,)
            )
            kwargs["n_wires"] = len(wires)
            args += wires
        elif args and isinstance(args[-1], (list, tuple, qml.wires.Wires, range)):
            kwargs["n_wires"] = len(args[-1])
            args = args[:-1] + tuple(args[-1])
        else:
            kwargs["n_wires"] = 1
        return cls._primitive.bind(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        # this method is called everytime we want to create an instance of the class.
        # default behavior uses __new__ then __init__
        # when tracing is enabled, we want to

        if not plxpr_enabled():
            return type.__call__(cls, *args, **kwargs)
        # use bind to construct the class if we want class construction to add it to the jaxpr
        return cls._primitive_bind_call(*args, **kwargs)
