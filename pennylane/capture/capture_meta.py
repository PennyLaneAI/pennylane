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
from abc import ABCMeta
from inspect import Signature, signature

from .switches import enabled


def _stop_autograph(f):
    """Stop the autograph interpretation of operators by making it so that ``f`` always
    belongs to the pennylane namespace."""

    def new_f(*args, **kwargs):
        return f(*args, **kwargs)

    return new_f


class CaptureMeta(type):
    """A metatype that dispatches class creation to ``cls._primitive_bind_call`` instead
    of normal class creation.

    See ``pennylane/capture/explanations.md`` for more detailed information on how this technically
    works.

    .. code-block::

        qml.capture.enable()

        class AbstractMyObj(jax.core.AbstractValue):
            pass

        class MyObj(metaclass=qml.capture.CaptureMeta):

            primitive = jax.extend.core.Primitive("MyObj")

            @classmethod
            def _primitive_bind_call(cls, a):
                return cls.primitive.bind(a)

            def __init__(self, a):
                self.a = a

        @MyObj.primitive.def_impl
        def _(a):
            return type.__call__(MyObj, a)

        @MyObj.primitive.def_abstract_eval
        def _(a):
            return AbstractMyObj()

    >>> jaxpr = jax.make_jaxpr(MyObj)(0.1)
    >>> jaxpr
    { lambda ; a:f32[]. let b:AbstractMyObj() = MyObj a in (b,) }
    >>> jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.1)
    [<__main__.MyObj at 0x17fc3ea50>]

    """

    @property
    def __signature__(cls):
        sig = signature(cls.__init__)
        without_self = tuple(sig.parameters.values())[1:]
        return Signature(without_self)

    def _primitive_bind_call(cls, *args, **kwargs):
        raise NotImplementedError(
            "Types using CaptureMeta must implement cls._primitive_bind_call to"
            " gain integration with plxpr program capture."
        )

    @_stop_autograph
    def __call__(cls, *args, **kwargs):
        # this method is called everytime we want to create an instance of the class.
        # default behavior uses __new__ then __init__

        if enabled():
            # when tracing is enabled, we want to
            # use bind to construct the class if we want class construction to add it to the jaxpr
            return cls._primitive_bind_call(*args, **kwargs)
        return type.__call__(cls, *args, **kwargs)


# pylint: disable=abstract-method
class ABCCaptureMeta(CaptureMeta, ABCMeta):
    """A combination of the capture meta and ABCMeta"""
