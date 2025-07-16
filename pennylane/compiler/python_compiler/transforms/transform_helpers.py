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

"""This module contains helper functions and utilities for common usage in xDSL compiler passes."""

from xdsl.ir import Operation

from pennylane.compiler.python_compiler import quantum_dialect as quantum


def get_gate_wires(gate_op: quantum.CustomOp, separate_control: bool = False):
    """_summary_

    Args:
        gate_op (quantum.CustomOp): _description_
        separate_control (bool, optional): _description_
    """
    wires = []
    ctrl_wires = []

    for qubit in gate_op.in_qubits:
        owner_op = qubit.owner
        if isinstance(owner_op, quantum.ExtractOp):
            idx_attr = owner_op.properties.get("idx_attr")
            assert idx_attr is not None, f"Unable to determine wire index from extract op; missing property 'idx_attr'"
            idx_attr_value = idx_attr.value
            wires.append(idx_attr_value)

    for ctrl_qubit in gate_op.in_ctrl_qubits:
        pass

    if separate_control:
        return wires, ctrl_wires
    else:
        return wires + ctrl_wires
