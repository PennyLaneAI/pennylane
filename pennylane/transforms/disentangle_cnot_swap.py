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
"""Aliases for disentangle_cnot and disentangle_swap from Catalyst's passes module."""

from pennylane.transforms.core import transform


# pylint: disable=missing-function-docstring
def disentangle_cnot_setup_inputs():
    return (), {}


disentangle_cnot = transform(pass_name="disentangle-cnot", setup_inputs=disentangle_cnot_setup_inputs)

# pylint: disable=missing-function-docstring
def disentangle_swap_setup_inputs():
    return (), {}


disentangle_swap = transform(pass_name="disentangle-swap", setup_inputs=disentangle_swap_setup_inputs)
