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
r"""
.. currentmodule:: pennylane

This module implements PennyLane's capturing mechanism for hybrid
quantum-classical programs.

.. warning::

    This module is experimental and will change significantly in the future.

.. currentmodule:: pennylane.capture

.. autosummary::
    :toctree: api

    ~disable
    ~enable
    ~enabled
    ~pause
    ~determine_abstracted_axes
    ~expand_plxpr_transforms
    ~eval_jaxpr
    ~run_autograph
    ~disable_autograph
    ~PlxprInterpreter
    ~FlatFn
    ~make_plxpr
    ~register_custom_staging_rule

The ``primitives`` submodule offers easy access to objects with jax dependencies such as
primitives and abstract types.
It is not available with ``import pennylane``, but the contents can be accessed via manual
import ``from pennylane.capture.primitives import *``.

.. currentmodule:: pennylane.capture.primitives

.. autosummary::
    :toctree: api

    AbstractOperator
    AbstractMeasurement
    adjoint_transform_prim
    cond_prim
    ctrl_transform_prim
    for_loop_prim
    qnode_prim
    while_loop_prim

See also:

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~tape.plxpr_to_tape


To activate and deactivate the new PennyLane program capturing mechanism, use
the switches ``qml.capture.enable`` and ``qml.capture.disable``.
Whether or not the capturing mechanism is currently being used can be
queried with ``qml.capture.enabled``.
By default, the mechanism is disabled:

.. code-block:: pycon

    >>> import pennylane as qml
    >>> qml.capture.enabled()
    False
    >>> qml.capture.enable()
    >>> qml.capture.enabled()
    True
    >>> qml.capture.disable()
    >>> qml.capture.enabled()
    False

**Custom Operator Behaviour**

Any operator that inherits from :class:`~.Operator` gains a default ability to be captured
in a Jaxpr. Any positional argument is bound as a tracer, wires are processed out into individual tracers,
and any keyword arguments are passed as keyword metadata.

.. code-block:: python

    class MyOp1(qml.operation.Operator):

        def __init__(self, arg1, wires, key=None):
            super().__init__(arg1, wires=wires)

    def qfunc(a):
        MyOp1(a, wires=(0,1), key="a")

    qml.capture.enable()
    print(jax.make_jaxpr(qfunc)(0.1))

.. code-block::

    { lambda ; a:f32[]. let
        _:AbstractOperator() = MyOp1[key=a n_wires=2] a 0 1
    in () }

But an operator developer may need to override custom behavior for calling ``cls._primitive.bind``
(where ``cls`` indicates the class) if:

* The operator does not accept wires, like :class:`~.SymbolicOp` or :class:`~.CompositeOp`.
* The operator needs to enforce a data / metadata distinction, like :class:`~.PauliRot`.

In such cases, the operator developer can override ``cls._primitive_bind_call``, which
will be called when constructing a new class instance instead of ``type.__call__``.  For example,

.. code-block:: python

    class JustMetadataOp(qml.operation.Operator):

        def __init__(self, metadata):
            super().__init__(wires=[])
            self._metadata = metadata

        @classmethod
        def _primitive_bind_call(cls, metadata):
            return cls._primitive.bind(metadata=metadata)


    def qfunc():
        JustMetadataOp("Y")

    qml.capture.enable()
    print(jax.make_jaxpr(qfunc)())

.. code-block::

    { lambda ; . let _:AbstractOperator() = JustMetadataOp[metadata=Y]  in () }

As you can see, the input ``"Y"``, while being passed as a positional argument, is converted to
metadata within the custom ``_primitive_bind_call`` method.

If needed, developers can also override the implementation method of the primitive like was done with ``Controlled``.
``Controlled`` needs to do so to handle packing and unpacking the control wires.

.. code-block:: python

    class MyCustomOp(qml.operation.Operator):
        pass

    @MyCustomOp._primitive.def_impl
    def _(*args, **kwargs):
        return type.__call__(MyCustomOp, *args, **kwargs)
"""
from typing import Type
from collections.abc import Callable

from .switches import disable, enable, enabled, pause
from .capture_meta import CaptureMeta, ABCCaptureMeta
from .flatfn import FlatFn
from .make_plxpr import make_plxpr
from .autograph import run_autograph, disable_autograph
from .dynamic_shapes import determine_abstracted_axes, register_custom_staging_rule

# by defining this here, we avoid
# E0611: No name 'AbstractOperator' in module 'pennylane.capture' (no-name-in-module)
# on use of from capture import AbstractOperator
AbstractOperator: type
AbstractMeasurement: type
qnode_prim: "jax.extend.core.Primitive"
PlxprInterpreter: type
expand_plxpr_transforms: Callable[[Callable], Callable]
eval_jaxpr: Callable
QmlPrimitive: "Type[jax.extend.core.Primitive]"


# pylint: disable=import-outside-toplevel, redefined-outer-name, too-many-return-statements
def __getattr__(key):
    if key == "QmlPrimitive":
        from .custom_primitives import QmlPrimitive

        return QmlPrimitive

    if key == "AbstractOperator":
        from .primitives import _get_abstract_operator

        return _get_abstract_operator()

    if key == "AbstractMeasurement":
        from .primitives import _get_abstract_measurement

        return _get_abstract_measurement()

    if key == "qnode_prim":
        from ..workflow._capture_qnode import qnode_prim

        return qnode_prim

    if key == "PlxprInterpreter":
        from .base_interpreter import PlxprInterpreter

        return PlxprInterpreter

    if key == "eval_jaxpr":
        from .base_interpreter import eval_jaxpr

        return eval_jaxpr

    if key == "expand_plxpr_transforms":
        from .expand_transforms import expand_plxpr_transforms

        return expand_plxpr_transforms

    raise AttributeError(f"module 'pennylane.capture' has no attribute '{key}'")


__all__ = (
    "disable",
    "enable",
    "enabled",
    "eval_jaxpr",
    "CaptureMeta",
    "ABCCaptureMeta",
    "determine_abstracted_axes",
    "expand_plxpr_transforms",
    "register_custom_staging_rule",
    "AbstractOperator",
    "AbstractMeasurement",
    "qnode_prim",
    "PlxprInterpreter",
    "FlatFn",
    "run_autograph",
    "make_plxpr",
)
