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
"""Submodule containing core quantum operations supported by PennyLane.

.. currentmodule:: pennylane.ops

PennyLane supports a collection of built-in quantum operations,
including both discrete-variable (DV) gates as used in the qubit model,
and continuous-variable (CV) gates as used in the qumode model of quantum
computation.

Here, we summarize the built-in operations supported by PennyLane, as well
as the conventions chosen for their implementation.

.. note::

    When writing a plugin device for PennyLane, make sure that your plugin
    supports as many of the PennyLane built-in operations defined here as possible.

    If the convention differs between the built-in PennyLane operation
    and the corresponding operation in the targeted framework, ensure that the
    conversion between the two conventions takes places automatically
    by the plugin device.


.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    ops/qubit
    ops/cv
"""

from .cv import *
from .qubit import *
