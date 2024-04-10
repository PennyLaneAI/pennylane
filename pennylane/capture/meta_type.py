# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Defines a metaclass that pennylane objects can inherit from.
"""

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


_USE_DEFAULT_CALL = True
# since this changes what happens with tracing, we need to turn the behavior
# off by default to preserve our ability to jit pennylane circuits.


def enable_plexpr():
    """Enable the capture of PlExpr."""
    if not has_jax:
        raise ImportError("plexpr requires jax.")
    global _USE_DEFAULT_CALL
    _USE_DEFAULT_CALL = False


def disable_plexpr():
    """Disable the capture of PlExpr."""
    global _USE_DEFAULT_CALL
    _USE_DEFAULT_CALL = True


ABSTRACT_TYPE_CACHE = {}


def get_abstract_type(class_type: type) -> type:
    """Construct or retrieve an abstract type corresponding to the "top parent"
    of the inheritance tree (Operator, MeasurementProcess, etc.)
    """

    if not has_jax:
        raise ImportError("Abstract types require jax to be installed.")

    # -1 is object, -2 is top parent (Operator, MeasurementProcess, etc.)
    top_parent = class_type.__mro__[-2]

    if top_parent in ABSTRACT_TYPE_CACHE:
        return ABSTRACT_TYPE_CACHE[top_parent]

    # since there's only three right now, we could just create them via the normal
    # class AbstractOperator(jax.core.AbstractValue): syntax
    # but this will extend a bit easier and makes it so we dont need to manually maintain
    # a list of things that can be abstract
    # If this proves too fragile and black-magicy, we can revert to direct construction of
    # all possible abstract types.

    type_name = f"Abstract{top_parent.__name__}"
    namespace = {
        "__eq__": lambda self, other: isinstance(other, type(self)),
        "__hash__": lambda self: hash(type_name),
    }

    AbstractType = type(type_name, (jax.core.AbstractValue,), namespace)

    # hopefully this API stays constant over jax versions... fingers crossed
    jax.core.raise_to_shaped_mappings[AbstractType] = lambda aval, _: aval

    ABSTRACT_TYPE_CACHE[type_name] = AbstractType

    return AbstractType


class JaxPRMeta(type):
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

        abstract_type = get_abstract_type(cls)

        @cls._primitive.def_abstract_eval
        def abstract_init(*_, **__):
            return abstract_type()

    def __call__(cls, *args, **kwargs):
        # this method is called everytime we want to create an instance of the class.
        # default behavior uses __new__ then __init__
        # when tracing is enabled, we want to

        if _USE_DEFAULT_CALL:
            return type.__call__(cls, *args, **kwargs)
        # use bind to construct the class if we want class construction to add it to the jaxpr
        return cls._primitive.bind(*args, **kwargs)

class JaxPRMetaCoerceWires(JaxPRMeta):

    def __call__(cls, *args, **kwargs):
        if cls._meta_coerce_wires:
            wires = kwargs.get("wires", None)
            if wires is None:
                if len(args) == 0:
                    raise ValueError("Can't create object without wires")
                kwargs['wires'] = args[-1]
                args = args[:-1]
        return JaxPRMeta.__call__(cls, *args, **kwargs)

