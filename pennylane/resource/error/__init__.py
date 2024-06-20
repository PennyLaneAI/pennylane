# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
The ``error`` module provides classes and functionality to track and propagate the
algorithmic error from advanced quantum algorithms.
"""

from .error import AlgorithmicError, ErrorOperation, SpectralNormError, _compute_algo_error
from .trotter_error import _commutator_error, _one_norm_error
