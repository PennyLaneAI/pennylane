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
r"""Submodule containing PennyLane optimizers.

.. currentmodule:: pennylane.optimize

.. warning::

  The built-in optimizers only support the default NumPy-interfacing QNode.

  If using the :ref:`PennyLane PyTorch <torch_qnode>`
  or the :ref:`PennyLane TensorFlow <tf_qnode>` interfaces,
  `PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ and
  TensorFlow optimizers (available in ``tf.train``) should be used respectively.

In PennyLane, an optimizer is a procedure that executes one weight
update step along (some function of) the negative gradient of the cost.
This update depends in general on:

* The function :math:`f(x)`, from which we calculate a gradient :math:`\nabla f(x)`.
  If :math:`x` is a vector, the gradient is also a vector whose entries are
  the partial derivatives of :math:`f` with respect to the elements of :math:`x`.
* the current weights :math:`x`
* the (initial) step size :math:`\eta`

The different optimizers can also depend on additional hyperparameters.

In the following, recursive definitions assume that :math:`x^{(0)}` is some
initial value in the optimization landscape, and all other step-dependent
values are initialized to zero at :math:`t=0`.

Available optimizers
--------------------

.. autosummary::
   AdagradOptimizer
   AdamOptimizer
   GradientDescentOptimizer
   MomentumOptimizer
   NesterovMomentumOptimizer
   RMSPropOptimizer
   QGTOptimizer

Code details
~~~~~~~~~~~~
"""

# Python optimizers that are available in PennyLane
# listed in alphabetical order to avoid circular imports
from .adagrad import AdagradOptimizer
from .adam import AdamOptimizer
from .gradient_descent import GradientDescentOptimizer
from .momentum import MomentumOptimizer
from .nesterov_momentum import NesterovMomentumOptimizer
from .rms_prop import RMSPropOptimizer
from .qgt import QGTOptimizer


# Optimizers to display in the docs
__all__ = [
    'AdagradOptimizer',
    'AdamOptimizer',
    'GradientDescentOptimizer',
    'MomentumOptimizer',
    'NesterovMomentumOptimizer',
    'RMSPropOptimizer',
    'QGTOptimizer'
]
