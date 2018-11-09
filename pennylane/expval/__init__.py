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
"""Submodule containing core quantum expectation values supported by PennyLane.

.. currentmodule:: pennylane.expval

PennyLane also supports a collection of built-in quantum **expectations**,
including both discrete-variable (DV) expectations as used in the qubit model,
and continuous-variable (CV) expectations as used in the qumode model of quantum
computation.

Here, we summarize the built-in expectations supported by PennyLane, as well
as the conventions chosen for their implementation.

.. note::

    All quantum operations in PennyLane are top level; they can be accessed
    via ``qml.OperationName``. Expectation values, however, are contained within
    the :mod:`pennylane.expval`, and are thus accessed via ``qml.expval.ExpectationName``.


.. note::

    When writing a plugin device for PennyLane, make sure that your plugin
    supports as many of the PennyLane built-in expectations defined here as possible.

    If the convention differs between the built-in PennyLane expectation
    and the corresponding expectation in the targeted framework, ensure that the
    conversion between the two conventions takes places automatically
    by the plugin device.


.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    expval/qubit
    expval/cv
"""

from .qubit import *
from .cv import *
