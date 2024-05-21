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

    ~.disable
    ~.enable
    ~.enabled
    ~.CaptureMeta
    ~.create_operator_primitive
    ~.create_measurment_obs_primitive
    ~.create_measurement_wires_primitive
    ~.create_measurement_mcm_primitive
    ~.measure

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

**Integrated Example:**

.. code-block:: python

    def f(x):
        qml.RX(x, wires=0)
        mp1 = qml.expval(qml.Z(0))
        mp2 = qml.sample()
        res1, res2 = qml.capture.measure(mp1, mp2, shots=50, num_device_wires=4)
        return res1 * res2

    jax.make_jaxpr(f)(0.1)

.. code-block::

    { lambda ; a:f32[]. let
        _:AbstractOperator() = RX[n_wires=1] a 0
        b:AbstractOperator() = PauliZ[n_wires=1] 0
        c:AbstractMeasurement(n_wires=None) = expval b
        d:AbstractMeasurement(n_wires=0) = sample
        e:f32[] f:i32[50,4] = measure[num_device_wires=4 shots=Shots(total=50)] c d
        g:f32[50,4] = convert_element_type[new_dtype=float32 weak_type=False] f
        h:f32[50,4] = mul e g
    in (h,) }

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
* The operator needs to enforce a data/ metadata distinction, like :class:`~.PauliRot`.

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
from .capture_meta import CaptureMeta
from .primitives import (
    create_operator_primitive,
    create_measurement_obs_primitive,
    create_measurement_wires_primitive,
    create_measurement_mcm_primitive,
)
from .measure import measure


def __getattr__(key):
    if key == "AbstractOperator":
        from .primitives import _get_abstract_operator  # pylint: disable=import-outside-toplevel

        return _get_abstract_operator()
    raise AttributeError(f"module 'pennylane.capture' has no attribute '{key}'")
