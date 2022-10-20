# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This subpackage provides default devices for PennyLane, which do not need external plugins to be installed.
The default devices provide basic built-in qubit
and CV circuit simulators that can be used with PennyLane without the need for additional
dependencies. They may also be used in the PennyLane test suite in order
to verify and test quantum gradient computations.

.. currentmodule:: pennylane.devices
.. autosummary::
    :toctree: api

    default_qubit
    default_qubit_jax
    default_qubit_torch
    default_qubit_tf
    default_qubit_autograd
    default_gaussian
    default_mixed
    default_qutrit
    tests
"""
# DefaultQubitTF and DefaultQubitAutograd not imported here since this
# would lead to an automatic import of tensorflow and autograd, which are
# not PennyLane core dependencies
from .default_qubit import DefaultQubit
from .default_gaussian import DefaultGaussian
from .default_mixed import DefaultMixed
from .experimental import *
