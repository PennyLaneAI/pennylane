# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This submodule offers all the non-operator/ measurement custom primitives
created in pennylane.

It has a jax dependency and should be located in a standard import path.
"""

from pennylane._grad.grad import jacobian_prim
from pennylane._grad.jvp import jvp_prim
from pennylane._grad.value_and_grad import value_and_grad_prim
from pennylane._grad.vjp import vjp_prim
from pennylane.control_flow.for_loop import for_loop_prim
from pennylane.control_flow.while_loop import while_loop_prim
from pennylane.core._capture_measurements import AbstractMeasurement  # tach-ignore
from pennylane.core.operator.base import AbstractOperator  # tach-ignore
from pennylane.core.operator.operator2 import operator_p  # tach-ignore
from pennylane.core.transforms.transform import transform_prim  # tach-ignore
from pennylane.ops.mid_measure.mid_measure import measure_prim
from pennylane.ops.mid_measure.pauli_measure import pauli_measure_prim
from pennylane.ops.op_math.adjoint import adjoint_transform_prim
from pennylane.ops.op_math.condition import cond_prim
from pennylane.ops.op_math.controlled import ctrl_transform_prim
from pennylane.workflow._capture_qnode import qnode_prim

from .subroutine import quantum_subroutine_prim
from .symbolic_array import symbolic_array_p

symbolic_array_prim = symbolic_array_p

__all__ = [
    "AbstractOperator",
    "AbstractMeasurement",
    "adjoint_transform_prim",
    "ctrl_transform_prim",
    "symbolic_array_prim",
    "jacobian_prim",
    "vjp_prim",
    "jvp_prim",
    "value_and_grad_prim",
    "qnode_prim",
    "cond_prim",
    "for_loop_prim",
    "while_loop_prim",
    "measure_prim",
    "operator_p",
    "quantum_subroutine_prim",
    "pauli_measure_prim",
    "transform_prim",
]
