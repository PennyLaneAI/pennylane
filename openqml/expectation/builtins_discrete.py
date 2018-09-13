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
"""This module contains operations for the default qubit observables"""

from openqml.operation import Expectation


__all__ = ['PauliX', 'PauliY', 'PauliZ', 'Hermitian']


class PauliX(Expectation):
    r"""Returns the Pauli-X expectation value.

    """


class PauliY(Expectation):
    r"""Returns the Pauli-Y expectation value.

    """


class PauliZ(Expectation):
    r"""Returns the Pauli-Z expectation value.

    """


class Hermitian(Expectation):
    r"""Returns the expectation value of an arbitrary Hermitian observable.

    Args:
        A (array): square hermitian matrix.
    """
    n_params = 1
