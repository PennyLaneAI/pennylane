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
Reference plugins
=================

**Module name:** :mod:`pennylane.plugins`

.. currentmodule:: pennylane.plugins

PennyLane supports a collection of built-in quantum operations and observables,
including both discrete-variable (DV) operations as used in the qubit model,
and continuous-variable (CV) operations as used in the qumode model of quantum
computation.

Here, we provide two reference plugin implementations; one supporting qubit
operations and observables (:mod:`'default.qubit' <pennylane.plugins.default_qubit>`)
and  one supporting CV operations and observables
(:mod:`'default.gaussian' <pennylane.plugins.default_gaussian>`).

These reference plugins provide basic built-in qubit and CV circuit
simulators that can be used with PennyLane without the need for additional
dependencies. They may also be used in the PennyLane test suite in order
to verify and test quantum gradient computations.

.. note::

    When writing a plugin device for PennyLane, make sure that your plugin
    supports as many of the PennyLane built-in operations defined here as possible.

    If the convention differs between the built-in PennyLane operation
    and the corresponding operation in the targeted framework, ensure that the
    conversion between the two conventions takes places automatically
    by the plugin device.


Architecture-specific operations
--------------------------------

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    plugins/default_qubit
    plugins/default_gaussian
"""
from .default_qubit import DefaultQubit
from .default_gaussian import DefaultGaussian
