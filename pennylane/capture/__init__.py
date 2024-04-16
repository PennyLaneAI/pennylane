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


To activate and deactivate the new PennyLane program capturing mechanism, use
the switches ``qml.capture.enable_plxpr`` and ``qml.capture.disable_plxpr``.
Whether or not the capturing mechanism is currently being used can be
queried with ``qml.capture.plxpr_enabled``.
By default, the mechanism is disabled:

.. code-block:: pycon

    >>> import pennylane as qml
    >>> qml.capture.plxpr_enabled()
    False
    >>> qml.capture.enable_plxpr()
    >>> qml.capture.plxpr_enabled()
    True
    >>> qml.capture.disable_plxpr()
    >>> qml.capture.plxpr_enabled()
    False

**Custom Operator Behavior**

Any operation that inherits from :class:`~.Operator` gains a default ability to be captured
by jaxpr.  Any positional argument is bound as a tracer, wires are processed out into individual tracers,
and any keyword args are passed as keyword metadata.

.. code-block:: python

    class MyOp1(qml.operation.Operator):

        def __init__(self, arg1, wires, key=None):
            super().__init__(arg1, wires=wires)

    def qfunc(a):
        MyOp1(a, wires=(0,1), key="a")

    qml.capture.enable_plxpr()
    print(jax.make_jaxpr(qfunc)(0.1))

.. code-block::

    { lambda ; a:f32[]. let
        _:AbstractOperator() = MyOp1[key=a n_wires=2] a 0 1
    in () }

But an operator developer may need to override custom behavior for calling ``cls._primitive.bind`` if:

* The operator does not accept wires like :class:`~.SymbolicOp` or :class:`~.CompositeOp`.
* The operator allows metadata to be provided positionally, like :class:`~.PauliRot`.

In such cases, the operator developer can override ``cls._primitive_bind_call``.  This is what
will be called when constructing a new class instance instead of ``type.__call__``.  For example,

.. code-block:: python

    class WeirdOp(qml.operation.Operator):

        def __init__(self, metadata="X"):
            super().__init__(wires=[])
            self._metadata = metadata

        @classmethod
        def _primitive_bind_call(cls, metadata):
            return cls._primitive.bind(metadata=metadata)
            

    def qfunc():
        WeirdOp("Y")

    qml.capture.enable_plxpr()
    print(jax.make_jaxpr(qfunc)())

.. code-block::

    { lambda ; . let _:AbstractOperator() = WeirdOp[metadata=Y]  in () }

"""
from .switches import enable_plxpr, disable_plxpr, plxpr_enabled
from .meta_type import PLXPRMeta
