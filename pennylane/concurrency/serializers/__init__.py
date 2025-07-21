# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This module contains wrapped serialization functionality for PennyLane workloads interfacing with the executor backends.

.. currentmodule:: pennylane

Serializers
***********

PennyLane's ``qml.concurrency.executor`` backends support both local and remote execution. Serialization of certain types of entities can require non-standard implementations to function correctly.

As an example, certain Python entities are inherently non-picklable. Let's take the following example:


.. code-block:: python

    >>> import pickle
    >>> pickle.dumps(qml.gradients.param_shift)
    Traceback (most recent call last):
    File "<python-input-6>", line 1, in <module>
        pickle.dumps(qml.gradients.param_shift)
        ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
    _pickle.PicklingError: Can't pickle <function param_shift at 0x7f710d84ade0>: it's not the same object as pennylane.gradients.parameter_shift.param_shift


In the above case, we tried to pickle ``qml.gradients.param_shift``, and received a failure due to the wrapped definition of the ``param_shift`` function differing from the underlying function itself. While a powerful module, Python's default pickle module can struggle with these such scenarios, where nesting of functions and decorator-based modifications can be standard in the codebase.

To aid with this, we provide access to a few externally-backed serialization utilities, such that it becomes trivial to encode objects like the above for transmission across local and remote multiprocessing environments. Let's adapt the above example to show the pickle backend, ``SerializerPickle``,  with the new interface:


.. code-block:: python

    >>> from pennylane.concurrency.serializers import SerializerPickle
    >>> SerializerPickle(qml.gradients.param_shift)
        Traceback (most recent call last):
        File "<python-input-6>", line 1, in <module>
            SerializerPickle(qml.gradients.param_shift)
            ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/pennylane/pennylane/concurrency/serializers/func_serializer.py", line 21, in __init__
            self._serialized = self._backend.dumps(entity, *args, **kwargs)
                            ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
        _pickle.PicklingError: Can't pickle <function param_shift at 0x7f000f98a700>: it's not the same object as pennylane.gradients.parameter_shift.param_shift


Now, we will install ``dill`` as ``pip install dill``, and run the same example using ``SerializerDill``:


.. code-block:: python

    >>> from pennylane.concurrency.serializers import SerializerPickle
    >>> SerializerDill(qml.gradients.param_shift)
        <pennylane.concurrency.serializers.func_serializer.SerializerDill at 0x7f01cec95fd0>



The serialized entity will now behave as the original one, and can be accessed directly using the serializer's `__call__` method. This will dynamically deserialize the the object, and pass all arguments directly to it at the point of access.
"""


from .func_serializer import SerializerABC, SerializerPickle, SerializerDill, SerializerCloudPickle
