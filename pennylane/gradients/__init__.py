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
"""This subpackage contains quantum gradient transforms."""
import pennylane as qml

from . import finite_difference
from . import parameter_shift
from . import parameter_shift_cv

from .finite_difference import finite_diff, finite_diff_coeffs, generate_shifted_tapes
from .parameter_shift import param_shift
from .parameter_shift_cv import param_shift_cv
