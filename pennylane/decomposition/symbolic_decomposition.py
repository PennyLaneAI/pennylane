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

"""Compatibility exports for the shared symbolic decomposition rules."""

from importlib import import_module

import numpy as np


def is_integer(x):
    """Checks if x is an integer."""
    return isinstance(x, int) or np.issubdtype(getattr(x, "dtype", None), np.integer)


_RULE_MODULES = {
    "adjoint_rotation": "pennylane.decomposition_rules.op_math.adjoint",
    "cancel_adjoint": "pennylane.decomposition_rules.op_math.adjoint",
    "ctrl_single_work_wire": "pennylane.decomposition_rules.op_math.controlled",
    "decompose_to_base": "pennylane.decomposition_rules.op_math.symbolicop2",
    "decompose_to_base_legacy": "pennylane.decomposition_rules.op_math.symbolicop",
    "flip_control_adjoint": "pennylane.decomposition_rules.op_math.controlled",
    "flip_pow_adjoint": "pennylane.decomposition_rules.op_math.pow",
    "flip_zero_control": "pennylane.decomposition_rules.op_math.controlled",
    "make_adjoint_decomp": "pennylane.decomposition_rules.op_math.adjoint",
    "make_controlled_decomp": "pennylane.decomposition_rules.op_math.controlled",
    "make_pow_decomp_with_period": "pennylane.decomposition_rules.op_math.pow",
    "merge_powers": "pennylane.decomposition_rules.op_math.pow",
    "pow_involutory": "pennylane.decomposition_rules.op_math.pow",
    "pow_rotation": "pennylane.decomposition_rules.op_math.pow",
    "repeat_pow_base": "pennylane.decomposition_rules.op_math.pow",
    "self_adjoint": "pennylane.decomposition_rules.op_math.symbolicop2",
    "self_adjoint_legacy": "pennylane.decomposition_rules.op_math.symbolicop",
    "to_controlled_qubit_unitary": "pennylane.decomposition_rules.op_math.controlled",
}


def __getattr__(name):
    """Resolve the existing public names of relocated symbolic rules."""
    if name in _RULE_MODULES:
        return getattr(import_module(_RULE_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
