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

"""This module contains special logic of decomposing symbolic operations."""

from __future__ import annotations

from importlib import import_module as _import_module

import numpy as np


def is_integer(x):
    """Checks if x is an integer."""
    return isinstance(x, int) or np.issubdtype(getattr(x, "dtype", None), np.integer)


_RULE_MODULES = {
    "make_adjoint_decomp": "pennylane.decomposition_rules.op_math.adjoint",
    "_adjoint_cancel_resources": "pennylane.decomposition_rules.op_math.adjoint",
    "_adjoint_cancel": "pennylane.decomposition_rules.op_math.adjoint",
    "_adjoint_negate_angle_resources": "pennylane.decomposition_rules.op_math.adjoint",
    "_adjoint_negate_angle": "pennylane.decomposition_rules.op_math.adjoint",
    "_pow_repeat_base": "pennylane.decomposition_rules.op_math.pow",
    "_pow_merge_powers_resources": "pennylane.decomposition_rules.op_math.pow",
    "_pow_merge_powers": "pennylane.decomposition_rules.op_math.pow",
    "_pow_flip_adjoint_resources": "pennylane.decomposition_rules.op_math.pow",
    "_pow_flip_adjoint": "pennylane.decomposition_rules.op_math.pow",
    "make_pow_decomp_with_period": "pennylane.decomposition_rules.op_math.pow",
    "pow_involutory": "pennylane.decomposition_rules.op_math.pow",
    "_pow_scale_angle_resources": "pennylane.decomposition_rules.op_math.pow",
    "_pow_scale_angle": "pennylane.decomposition_rules.op_math.pow",
    "_operator1_to_base_resources": "pennylane.decomposition_rules.op_math.symbolicop",
    "_operator1_to_base": "pennylane.decomposition_rules.op_math.symbolicop",
    "self_adjoint_legacy": "pennylane.decomposition_rules.op_math.symbolicop",
    "_operator_to_base_resources": "pennylane.decomposition_rules.op_math.symbolicop2",
    "_operator_to_base": "pennylane.decomposition_rules.op_math.symbolicop2",
    "self_adjoint": "pennylane.decomposition_rules.op_math.symbolicop2",
    "make_controlled_decomp": "pennylane.decomposition_rules.op_math.controlled",
    "flip_zero_control": "pennylane.decomposition_rules.op_math.controlled",
    "_controlled_flip_adjoint_resources": "pennylane.decomposition_rules.op_math.controlled",
    "_controlled_flip_adjoint": "pennylane.decomposition_rules.op_math.controlled",
    "_controlled_single_work_wire_resources": "pennylane.decomposition_rules.op_math.controlled",
    "_controlled_single_work_wire": "pennylane.decomposition_rules.op_math.controlled",
    "ctrl_single_work_wire": "pennylane.decomposition_rules.op_math.controlled",
    "_to_controlled_qu_condition": "pennylane.decomposition_rules.op_math.controlled",
    "_controlled_to_unitary_resources": "pennylane.decomposition_rules.op_math.controlled",
    "_controlled_to_unitary": "pennylane.decomposition_rules.op_math.controlled",
}


def __getattr__(name):
    """Resolve compatibility exports for relocated decomposition rules."""
    if name in _RULE_MODULES:
        return getattr(_import_module(_RULE_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
