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
    ~create_operator_primitive
    ~create_measurement_obs_primitive
    ~create_measurement_wires_primitive
    ~create_measurement_mcm_primitive
    ~qnode_call

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
from .switches import disable, enable, enabled
from .capture_meta import CaptureMeta, ABCCaptureMeta
from .primitives import (
    create_operator_primitive,
    create_measurement_obs_primitive,
    create_measurement_wires_primitive,
    create_measurement_mcm_primitive,
)
from .capture_qnode import qnode_call

# by defining this here, we avoid
# E0611: No name 'AbstractOperator' in module 'pennylane.capture' (no-name-in-module)
# on use of from capture import AbstractOperator
AbstractOperator: type
AbstractMeasurement: type
qnode_prim: "jax.core.Primitive"


def __getattr__(key):
    if key == "AbstractOperator":
        from .primitives import _get_abstract_operator  # pylint: disable=import-outside-toplevel

        return _get_abstract_operator()

    if key == "AbstractMeasurement":
        from .primitives import _get_abstract_measurement  # pylint: disable=import-outside-toplevel

        return _get_abstract_measurement()

    if key == "qnode_prim":
        from .capture_qnode import _get_qnode_prim  # pylint: disable=import-outside-toplevel

        return _get_qnode_prim()

    raise AttributeError(f"module 'pennylane.capture' has no attribute '{key}'")


__all__ = (
    "disable",
    "enable",
    "enabled",
    "CaptureMeta",
    "ABCCaptureMeta",
    "create_operator_primitive",
    "create_measurement_obs_primitive",
    "create_measurement_wires_primitive",
    "create_measurement_mcm_primitive",
    "qnode_call",
    "AbstractOperator",
    "AbstractMeasurement",
    "qnode_prim",
)
