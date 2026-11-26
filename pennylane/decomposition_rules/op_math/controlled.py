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

"""Numerical decomposition helper for legacy controlled operators."""

from __future__ import annotations

from pennylane import math, ops
from pennylane.core.queuing import QueuingManager
from pennylane.core.operator import Operator
from pennylane.ops.op_math.controlled import (
    Controlled,
    _decompose_custom_ops,
    _is_single_qubit_special_unitary,
)
from .controlled_qubit_unitary import ctrl_decomp_bisect, ctrl_decomp_zyz


def _decompose_no_control_values(op: Controlled) -> list[Operator] | None:
    """Decompose without considering control values. Returns None if no decomposition."""

    decomp = _decompose_custom_ops(op)
    if decomp is not None:
        return decomp

    is_su2 = _is_single_qubit_special_unitary(op.base)
    if not math.is_tracer(is_su2) and is_su2:
        if len(op.control_wires) >= 2 and math.get_interface(*op.data) == "numpy":
            return ctrl_decomp_bisect(op.base, op.control_wires)
        return ctrl_decomp_zyz(
            op.base,
            control_wires=op.control_wires,
            work_wires=op.work_wires,
            work_wire_type=op.work_wire_type,
        )

    if not op.base.has_decomposition:
        return None

    with QueuingManager.stop_recording():
        base_decomp = op.base.decomposition()

    return [
        ops.ctrl(newop, op.control_wires, work_wires=op.work_wires, work_wire_type=op.work_wire_type)
        for newop in base_decomp
    ]
