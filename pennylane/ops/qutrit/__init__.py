# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the discrete-variable qutrit quantum operations.

The operations are in one file:

* ``matrix_ops.py``: Generalized operations that accept a matrix parameter,
  either unitary or hermitian depending.
"""

from .matrix_ops import *
from .observables import *
from .non_parametric_ops import *
from ..identity import Identity

# TODO: Change `qml.Identity` for qutrit support or add `qml.TIdentity` for qutrits
ops = {
    "Identity",
    "QutritUnitary",
    "TShift",
    "TClock",
}

obs = {
    "THermitian",
}

__all__ = list(ops | obs)
