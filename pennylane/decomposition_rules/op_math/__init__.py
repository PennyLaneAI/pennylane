# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Operator2 symbolic decomposition rules and factories.

Import Operator2 symbolic rules from this package. Legacy symbolic rules are
available from the ``adjoint``, ``controlled``, ``pow`` and ``symbolicop`` modules.

.. currentmodule:: pennylane.decomposition_rules.op_math

.. autosummary::

    adjoint_cancel
    adjoint_negate_angle
    controlled_flip_adjoint
    controlled_many_zeroed_work_wires
    controlled_single_work_wire
    controlled_to_unitary
    ctrl_many_zeroed_work_wires
    ctrl_single_work_wire
    flip_zero_control
    make_adjoint_decomp
    make_controlled_decomp
    make_pow_decomp_with_period
    operator_to_base
    pow_flip_adjoint
    pow_involutory
    pow_merge_powers
    pow_repeat_base
    pow_scale_angle
    self_adjoint

Concrete controlled operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    controlled_ops
    controlled_qubit_unitary
    multi_controlled_x
"""

from .symbolicop2 import operator_to_base, self_adjoint
from .adjoint2 import adjoint_cancel, adjoint_negate_angle, make_adjoint_decomp
from .controlled2 import (
    controlled_flip_adjoint,
    controlled_many_zeroed_work_wires,
    controlled_single_work_wire,
    controlled_to_unitary,
    ctrl_many_zeroed_work_wires,
    ctrl_single_work_wire,
    flip_zero_control,
    make_controlled_decomp,
)
from .pow2 import (
    make_pow_decomp_with_period,
    pow_flip_adjoint,
    pow_involutory,
    pow_merge_powers,
    pow_repeat_base,
    pow_scale_angle,
)

__all__ = [
    "adjoint_cancel",
    "adjoint_negate_angle",
    "controlled_flip_adjoint",
    "controlled_many_zeroed_work_wires",
    "controlled_single_work_wire",
    "controlled_to_unitary",
    "ctrl_many_zeroed_work_wires",
    "ctrl_single_work_wire",
    "flip_zero_control",
    "make_adjoint_decomp",
    "make_controlled_decomp",
    "make_pow_decomp_with_period",
    "operator_to_base",
    "pow_flip_adjoint",
    "pow_involutory",
    "pow_merge_powers",
    "pow_repeat_base",
    "pow_scale_angle",
    "self_adjoint",
]
