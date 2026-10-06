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
"""This module contains the functionality to work with noise models in PennyLane."""

from .conditionals import wires_in, wires_eq, op_in, op_eq, meas_eq, partial_wires
from .noise_model import NoiseModel
from .add_noise import add_noise
from .insert_ops import insert
from .mitigate import (
    mitigate_with_zne,
    fold_global,
    poly_extrapolate,
    richardson_extrapolate,
    exponential_extrapolate,
)
