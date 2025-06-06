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
from pennylane._grad import _get_grad_prim, _get_jacobian_prim
from pennylane.control_flow.for_loop import _get_for_loop_qfunc_prim
from pennylane.control_flow.while_loop import _get_while_loop_qfunc_prim
from pennylane.measurements.capture_measurements import _get_abstract_measurement
from pennylane.measurements.mid_measure import _create_mid_measure_primitive
from pennylane.operation import _get_abstract_operator
from pennylane.ops.op_math.adjoint import _get_adjoint_qfunc_prim
from pennylane.ops.op_math.condition import _get_cond_qfunc_prim
from pennylane.ops.op_math.controlled import _get_ctrl_qfunc_prim
from pennylane.workflow._capture_qnode import qnode_prim

AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()
adjoint_transform_prim = _get_adjoint_qfunc_prim()
ctrl_transform_prim = _get_ctrl_qfunc_prim()
grad_prim = _get_grad_prim()
jacobian_prim = _get_jacobian_prim()
cond_prim = _get_cond_qfunc_prim()
for_loop_prim = _get_for_loop_qfunc_prim()
while_loop_prim = _get_while_loop_qfunc_prim()
measure_prim = _create_mid_measure_primitive()

__all__ = [
    "AbstractOperator",
    "AbstractMeasurement",
    "adjoint_transform_prim",
    "ctrl_transform_prim",
    "grad_prim",
    "jacobian_prim",
    "qnode_prim",
    "cond_prim",
    "for_loop_prim",
    "while_loop_prim",
    "measure_prim",
]
