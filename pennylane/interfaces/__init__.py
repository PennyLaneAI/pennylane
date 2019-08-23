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
.. currentmodule:: pennylane.interfaces

PennyLane provides support for various classical
machine learning interfaces, including Autograd/NumPy, PyTorch,
and TensorFlow. The interfaces have access to gradients of a QNode, and
can therefore integrate quantum computations into a larger machine learning
or optimization pipeline.

Depending on the classical machine learning interface chosen,
you may be able to offload the classical portion of your hybrid model
onto an accelerator, such as a GPU or TPU.

By default, QNodes make use of the wrapped version of NumPy provided
by PennyLane (via `autograd <https://github.com/HIPS/autograd>`_). By
importing NumPy from PennyLane,

.. code-block:: python

    from pennylane import numpy as np

any classical computation in the model can then make use of arbitrary NumPy
functions, while retaining support for automatic differentiation. For an example,
see the :ref:`hybrid computation tutorial <plugins_hybrid>`.

However, PennyLane has the ability to contruct quantum nodes that can also be used in conjunction
with other classical machine learning libraries. Such QNodes will accept and return the correct
object types expected by the machine learning library (i.e., Python default types and NumPy array
for the PennyLane-provided wrapped NumPy, ``torch.tensor`` for PyTorch, and
``tf.Tensor`` or ``tfe.Variable`` for TensorFlow). Furthermore, PennyLane will correctly pass
the quantum analytic gradient to the machine learning library during backpropagation.

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    interfaces/numpy
    interfaces/torch
    interfaces/tfe
"""
