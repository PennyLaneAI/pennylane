# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aliases for pauli-based computation passes from Catalyst's passes module."""

from pennylane.transforms.core import transform

import catalyst


def to_ppr_setup_inputs():
    return (), {}


to_ppr = transform(pass_name="to-ppr", setup_inputs=to_ppr_setup_inputs)


def commute_ppr_setup_inputs(max_pauli_size: int = 0):
    if not isinstance(max_pauli_size, int) or max_pauli_size < 0:
        raise ValueError(f"max_pauli_size must be an int and >= 0. Got {max_pauli_size}")
    return (), {"max_pauli_size": max_pauli_size}


commute_ppr = transform(pass_name="commute-ppr", setup_inputs=commute_ppr_setup_inputs)


def merge_ppr_ppm_setup_inputs(max_pauli_size: int = 0):
    if not isinstance(max_pauli_size, int) or max_pauli_size < 0:
        raise ValueError(f"max_pauli_size must be an int and >= 0. Got {max_pauli_size}")
    return (), {"max_pauli_size": max_pauli_size}


merge_ppr_ppm = transform(pass_name="merge-ppr-ppm", setup_inputs=merge_ppr_ppm_setup_inputs)


def ppr_to_ppm_setup_inputs(decompose_method="pauli-corrected", avoid_y_measure=False):
    return (), {"decompose_method": decompose_method, "avoid_y_measure": avoid_y_measure}


ppr_to_ppm = transform(pass_name="ppr-to-ppm", setup_inputs=ppr_to_ppm_setup_inputs)


def ppm_compilation_setup_inputs(
    decompose_method="pauli-corrected", avoid_y_measure=False, max_pauli_size=0
):
    if not isinstance(max_pauli_size, int) or max_pauli_size < 0:
        raise ValueError(f"max_pauli_size must be an int and >= 0. Got {max_pauli_size}")
    return (), {
        "decompose_method": decompose_method,
        "avoid_y_measure": avoid_y_measure,
        "max_pauli_size": max_pauli_size,
    }


ppm_compilation = transform(pass_name="ppm-compilation", setup_inputs=ppm_compilation_setup_inputs)


def reduce_t_depth_setup_inputs():
    return (), {}


reduce_t_depth = transform(pass_name="reduce-t-depth", setup_inputs=reduce_t_depth_setup_inputs)


def decompose_arbitrary_ppr_setup_inputs():
    return (), {}


decompose_arbitrary_ppr = transform(
    pass_name="decompose-arbitrary-ppr", setup_inputs=decompose_arbitrary_ppr_setup_inputs
)
