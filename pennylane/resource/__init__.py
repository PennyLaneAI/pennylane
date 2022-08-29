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
The ``resource`` module provides functionality to estimate the number of non-Clifford gates and
logical qubits required to implement advanced quantum algorithms.
"""
from .first_quantization import FirstQuantization
from .second_quantization import DoubleFactorization
from .measurement import estimate_error, estimate_samples
