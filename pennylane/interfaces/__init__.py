# Copyright 2018 Xanadu Quantum Technologies Inc.

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
.. raw:: html

    <h2>Classical interfaces</h2>

**Module name:** :mod:`pennylane.interfaces`

.. currentmodule:: pennylane.interfaces

PennyLane now provides support for additional classical
machine learning interfaces, specifically PyTorch and TensorFlow eager execution mode.

Depending on the classical machine learning interface chosen,
you may be able to offload the classical portion of your hybrid model
onto an accelerator, such as a GPU or TPU.

By default, when constructing a :ref:`QNode <qnode_decorator>`, PennyLane allows
the underlying quantum function to accept any default Python types (for example,
floats, ints, lists) as well as NumPy array arguments, and will always return
NumPy arrays representing the returned expectation values. To enable the QNode
to then be used in arbitrary hybrid classical-quantum computation, you can
make use of the wrapped version of NumPy provided by PennyLane
(via `autograd <https://github.com/HIPS/autograd>`_):

.. code-block:: python

    from pennylane import numpy as np

Any classical computation in the model can then make use of arbitrary NumPy
functions, while retaining support for automatic differentiation. For an example,
see the :ref:`hybrid computation tutorial <plugins_hybrid>`.

However, PennyLane has the ability to contruct quantum nodes can also be used in conjunction
with other classical machine learning libraries; in such a case, the QNode is modified such that

1. It accepts and returns the correct object types expected by the classical
   machine learning library (i.e., Python default types and NumPy array for
   the PennyLane-provided wrapped NumPy, ``torch.tensor`` for PyTorch, and
   ``tf.Tensor`` or ``tfe.Variable`` for TensorFlow), and

2. It correctly passes the quantum analytic gradient to the classical machine
   learning library during backpropagation.


.. toctree::
    :maxdepth: 1
    :hidden:

    Creating QNodes <self>

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    interfaces/numpy
    interfaces/torch
    interfaces/tfe
"""
