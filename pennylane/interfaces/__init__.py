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
Classical interfaces overview
=============================

**Module name:** :mod:`pennylane.interfaces`

.. currentmodule:: pennylane.interfaces

PennyLane now provides experimental support for additional classical
automatic differentiation interfaces, beginning with PyTorch.

Background
----------

By default, when constructing a :ref:`QNode <qnode_decorator>`, PennyLane allows
the underlying quantum function to accept any default Python types (for example,
floats, ints, lists) as well as NumPy array arguments, and will always return
NumPy arrays representing the returned expectation values. To enable the QNode
to then be used in arbitrary hybrid classical-quantum computation, you can then
make use of the patched version of NumPy provided by PennyLane
(via `autograd <https://github.com/HIPS/autograd>`_):

.. code-block:: python

    from pennylane import numpy as np

Any classical computation in the model can then make use of arbitrary NumPy
functions, while retaining support for automatic differentiation. For an example,
see the :ref:`hybrid computation tutorial <plugins_hybrid>`.

However, there is no reason why PennyLane's quantum nodes cannot be used in conjunction
with other classical machine learning libraries; all that is required is that
the QNode is modified such that

1. It accepts and returns the correct object types expected by the classical
   machine learning library (i.e., Python default types and NumPy array for
   the PennyLane-provided wrapped NumPy, and ``torch.tensor`` for PyTorch), and

2. It correctly passes the quantum analytic gradient to the classical machine
   learning library during backprogation.

To that end, we will begin supporting additional classical interfaces in PennyLane,
beginning with PyTorch.

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    interfaces/numpy
    interfaces/torch
"""
