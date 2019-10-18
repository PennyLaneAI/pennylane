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
This module provides reference plugin implementations. The reference plugins provide basic built-in qubit
and CV circuit simulators that can be used with PennyLane without the need for additional
dependencies. They may also be used in the PennyLane test suite in order
to verify and test quantum gradient computations.

PennyLane comes with two reference plugins:

+-------------------------------------------+---------------------------------------------------------+
| :mod:`'default.qubit'                     | Reference plugin for qubit architectures.               |
| <pennylane.plugins.default_qubit>`        |                                                         |
+-------------------------------------------+---------------------------------------------------------+
| :mod:`'default.gaussian'                  | Reference plugin for continuous-variable architectures. |
| <pennylane.plugins.default_gaussian>`     |                                                         |
+-------------------------------------------+---------------------------------------------------------+
"""
from .default_qubit import DefaultQubit
from .default_gaussian import DefaultGaussian
