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

interfaces
==========

**Module name**: pennylane.interfaces

.. currentmodule:: pennylane.interfaces

This module defines quantum nodes that are compatible with different :ref:`interfaces <intro_interfaces>`.

However, PennyLane has the ability to construct quantum nodes that can also be used in conjunction
with other classical machine learning libraries. Such QNodes will accept and return the correct
object types expected by the machine learning library (i.e., Python default types and NumPy array
for the PennyLane-provided wrapped NumPy, ``torch.tensor`` for PyTorch, and
``tf.Tensor`` or ``tf.Variable`` for TensorFlow). Furthermore, PennyLane will correctly pass
the quantum analytic gradient to the machine learning library during backpropagation.

"""
