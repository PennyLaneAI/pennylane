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

from functools import lru_cache

_USE_DEFAULT_CALL = False

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache  # constrcut the first time lazily
def _get_abstract_operator():
    if not has_jax:
        raise ImportError("Jax is required for plxpr.")

    class AbstractOperator(jax.core.AbstractValue):
        def __eq__(self, other):
            return isinstance(other, AbstractOperator)

        def __hash__(self):
            return hash("AbstractOperator")

    jax.core.raise_to_shaped_mappings[AbstractOperator] = lambda aval, _: aval

    return AbstractOperator


class PLXPRMeta(type):
    """A meta type"""

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
            return type.__call__(cls, *args, **kwargs)

        # logic here will be extended when we make more things use this meta class
        abstract_type = _get_abstract_operator()

        @cls._primitive.def_abstract_eval
        def abstract_init(*_, **__):
            return abstract_type()

    def __call__(cls, *args, **kwargs):
        # this method is called everytime we want to create an instance of the class.
        # default behavior uses __new__ then __init__
        # when tracing is enabled, we want to

        if _USE_DEFAULT_CALL or cls._primitive is None:
            return type.__call__(cls, *args, **kwargs)
        # use bind to construct the class if we want class construction to add it to the jaxpr
        if "wires" in kwargs:
            wires = kwargs.pop("wires")
            args += tuple(wires)
        if args and isinstance(args[-1], (list, tuple)):
            args = args[:-1] + tuple(args[-1])
        return cls._primitive.bind(*args, **kwargs)
