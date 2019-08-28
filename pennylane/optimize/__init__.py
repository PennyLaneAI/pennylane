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
r"""
.. _optimization_methods:

Optimization methods
====================

**Module name:** :mod:`pennylane.optimize`

.. currentmodule:: pennylane.optimize

This module contains optimizers for the standard :mod:`QNode` class, which uses the NumPy interface.

An optimizer is a procedure that computes a new candidate for the trainable parameters in every step,
following the negative gradient of a cost function.

Since NumPy does not specifically focus on machine learning and optimization, this collection of
optimizers is provided for convenience.


.. warning::

  If using the :ref:`PennyLane PyTorch <torch_qnode>`
  or the :ref:`PennyLane TensorFlow <tf_qnode>` interfaces,
  `PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ and
  TensorFlow optimizers (found in the module ``tf.train``) should be used respectively.


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
