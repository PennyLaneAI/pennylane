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
r"""
This subpackage defines functions and classes for Pauli-word partitioning
functionality used in measurement optimization.
"""
import warnings
import pennylane as qml


def __getattr__(name):
    warnings.warn(
        f"\nThe grouping module is deprecated!"
        f"\nPlease use the pauli module:"
        f"\npennylane.pauli.{name} or pennylane.pauli.grouping.{name}"
    )
    return getattr(qml.pauli, name)
