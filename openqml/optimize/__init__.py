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
Optimization methods
====================

**Module name:** :mod:`openqml.optimize`

.. currentmodule:: openqml.optimize

.. todo:: Add more details here and in the optimizers docstrings.
    Also, might we want to add more fine-grained section?
    i.e., talk about 'base' optimizer classes, like GradientDescent,
    followed by all inheriting classes?

Available optimizers include:

.. autosummary::
   AdagradOptimizer
   AdamOptimizer
   GradientDescentOptimizer
   NesterovMomentumOptimizer
   RMSPropOptimizer

Code details
------------
"""

# Python optimizers that are available in OpenQML
# listed in alphabetical order to avoid circular imports
from .adagrad import AdagradOptimizer
from .adam import AdamOptimizer
from .gradient_descent import GradientDescentOptimizer
from .momentum import MomentumOptimizer
from .nesterov_momentum import NesterovMomentumOptimizer
from .rms_prop import RMSPropOptimizer

# Optimizers to display in the docs
__all__ = [
    'AdagradOptimizer',
    'AdamOptimizer',
    'GradientDescentOptimizer',
    'MomentumOptimizer',
    'NesterovMomentumOptimizer',
    'RMSPropOptimizer'
]
